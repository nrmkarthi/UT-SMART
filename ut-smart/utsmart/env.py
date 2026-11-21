"""Discrete-time simulator of the connected-AUV edge-intelligence network.

Models (Sec. II of the paper): 3-D sensor field, patrolling AUVs with current drift,
Thorp-Urick acoustic links with AR(1) shadowing and Markov outages, short-range
optical links, Poisson semantic tasks of three missions, FIFO queues (cluster
channel, AUV outbound modem, AUV CPU, gateway CPU) served at time-varying rates,
propagation delay, deadline drops, compromised AUVs, a Kalman digital twin,
Beta trust, on-demand / periodic twin sync and event-triggered / periodic state
sharing. Every acoustic bit (data, retransmissions, results, spot checks,
probes, state broadcasts) is counted.
"""
from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from . import channel as ch
from .config import MISSIONS, SimConfig, Variant
from .trust import CONSISTENCY, RELIABILITY, RESULT, TrustManager
from .twin import LinkTwin

N_LEVELS = 4
N_EXEC = 3
N_ACTIONS = N_LEVELS * N_EXEC          # a = k * 3 + e
EXEC_OWN, EXEC_NEIGH, EXEC_GW = 0, 1, 2
GW = -1
G_DIM, PHI_DIM, CRITIC_DIM = 15, 12, 19
BIG = 1e4


# --------------------------------------------------------------------------
# Tasks and jobs
# --------------------------------------------------------------------------
class Task:
    __slots__ = ("id", "sensor", "cluster", "mission", "size", "arrival", "dmax", "deadline",
                 "voi", "a_min", "sens", "k", "e", "executor", "accuracy", "payload",
                 "energy", "air_bits", "corrupt", "spot", "silent_drop", "done", "rec",
                 "recv_exec", "adv_at_decision")

    def __init__(self, tid, sensor, cluster, mission_idx, size, arrival):
        m = MISSIONS[mission_idx]
        self.id, self.sensor, self.cluster, self.mission = tid, sensor, cluster, mission_idx
        self.size, self.arrival = size, arrival
        self.dmax, self.voi, self.a_min, self.sens = m.deadline, m.voi, m.a_min, m.sens
        self.deadline = arrival + m.deadline
        self.k = self.e = self.executor = 0
        self.accuracy = self.payload = self.energy = self.air_bits = 0.0
        self.corrupt = self.spot = self.silent_drop = self.done = False
        self.rec = None
        self.recv_exec = 0.0
        self.adv_at_decision = 0.0


class Job:
    __slots__ = ("task", "kind", "work", "link", "ready", "seq", "prio", "on_done", "owner",
                 "power")

    def __init__(self, task, kind, work, link, ready, seq, prio, on_done, owner, power=0.0):
        self.task, self.kind, self.work, self.link = task, kind, work, link
        self.ready, self.seq, self.prio, self.on_done = ready, seq, prio, on_done
        self.owner, self.power = owner, power


class Server:
    """FIFO server (priority jobs first) with per-job, per-slot service rate.

    Comm servers serve bits at the current goodput of the job's link (jobs on
    links in outage are skipped, others behind them proceed); CPU servers serve
    cycles at a fixed rate.
    """

    def __init__(self, env: "UnderwaterEnv", cpu_rate: float | None = None):
        self.env = env
        self.cpu_rate = cpu_rate
        self.jobs: list[Job] = []
        self.keys: list[tuple] = []

    def push(self, job: Job) -> None:
        key = (0 if job.prio else 1, job.ready, job.seq)
        i = bisect.bisect(self.keys, key)
        self.keys.insert(i, key)
        self.jobs.insert(i, job)

    def _remove(self, i: int) -> None:
        del self.jobs[i]
        del self.keys[i]

    def rate(self, job: Job) -> float:
        return self.cpu_rate if self.cpu_rate is not None else self.env.rate_now[job.link]

    def __len__(self):
        return len(self.jobs)

    def backlog(self, rate_fn=None) -> float:
        """Seconds of queued work (rate_fn(job) -> estimated rate)."""
        tot = 0.0
        for j in self.jobs:
            if j.task is not None and j.task.done:
                continue
            r = self.cpu_rate if self.cpu_rate is not None else (
                rate_fn(j) if rate_fn else self.env.rate_now[j.link])
            tot += j.work / max(r, 50.0)
        return min(tot, BIG)

    def serve(self, t0: float, dt: float) -> None:
        env = self.env
        t, end = t0, t0 + dt
        while t < end - 1e-12:
            chosen, next_ready = -1, math.inf
            i = 0
            while i < len(self.jobs):
                j = self.jobs[i]
                if j.task is not None and j.task.done:      # expired / resolved
                    self._remove(i)
                    continue
                if j.ready <= t + 1e-9:
                    if self.rate(j) > 0:
                        chosen = i
                        break
                elif j.ready < next_ready:
                    next_ready = j.ready
                i += 1
            if chosen < 0:
                if next_ready < end:
                    t = next_ready
                    continue
                break
            job = self.jobs[chosen]
            r = self.rate(job)
            need = job.work / r
            used = min(need, end - t)
            if self.cpu_rate is None:
                env.account_air(job, used)
            if need <= end - t + 1e-12:
                t += need
                self._remove(chosen)
                job.on_done(job, t)
            else:
                job.work -= used * r
                t = end


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    tasks: int = 0
    success: int = 0
    voi_total: float = 0.0
    voi_success: float = 0.0
    miss: int = 0
    acc_fail: int = 0
    corrupt_accepted: int = 0
    corrupt_detected: int = 0
    ontime: int = 0
    delay_ontime: float = 0.0
    energy: float = 0.0
    data_bits: float = 0.0
    ctrl_bits: float = 0.0
    probes: int = 0
    twin_sq_err: float = 0.0
    twin_n: int = 0
    reward: float = 0.0
    exec_counts: list = field(default_factory=lambda: [0, 0, 0])
    level_counts: list = field(default_factory=lambda: [0, 0, 0, 0])
    to_malicious: int = 0

    def summary(self) -> dict:
        n = max(self.tasks, 1)
        return dict(
            tasks=self.tasks,
            success_rate=self.success / n,
            voi_utility=self.voi_success / max(self.voi_total, 1e-9),
            miss_rate=self.miss / n,
            acc_fail_rate=self.acc_fail / n,
            corrupt_rate=self.corrupt_accepted / n,
            mean_delay=self.delay_ontime / max(self.ontime, 1),
            energy_per_task=self.energy / n,
            traffic_kbit_per_task=(self.data_bits + self.ctrl_bits) / n / 1e3,
            control_kbit_per_task=self.ctrl_bits / n / 1e3,
            twin_rmse=math.sqrt(self.twin_sq_err / max(self.twin_n, 1)),
            probes_per_task=self.probes / n,
            reward_per_task=self.reward / n,
            frac_own=self.exec_counts[0] / n,
            frac_neigh=self.exec_counts[1] / n,
            frac_gw=self.exec_counts[2] / n,
            frac_to_malicious=self.to_malicious / n,
        )


class UnderwaterEnv:
    def __init__(self, cfg: SimConfig, variant: Variant, policy, seed: int,
                 lambdas=(0.0, 0.0), record: bool = False):
        self.cfg, self.var, self.policy = cfg, variant, policy
        self.rng = np.random.default_rng(seed)
        self.record = record
        self.decisions: list[dict] = []
        self.stats = EpisodeStats()
        self.lambdas = lambdas
        self._seq = itertools.count()
        self._tid = itertools.count()
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        c, rng = self.cfg, self.rng
        M, N, L = c.n_sensors, c.n_auvs, c.area
        self.M, self.N = M, N
        xy = rng.uniform(0, L, (M, 2))
        z = rng.uniform(*c.sensor_depth, M)
        self.sensor_pos = np.column_stack([xy, z])
        self.cluster, centers = _kmeans(xy, N, rng)
        self.centers = centers
        self.cluster_sensors = [np.flatnonzero(self.cluster == k) for k in range(N)]
        self.radius = 0.3 * L / math.ceil(math.sqrt(N))
        self.phase = rng.uniform(0, 2 * math.pi, N)
        self.omega = c.auv_speed / self.radius
        self.auv_depth = rng.uniform(*c.auv_depth, N)
        self.offset = np.zeros((N, 2))
        self.cur_phase = rng.uniform(0, 2 * math.pi, 2)
        self.gw_pos = np.array([L / 2, L / 2, 0.0])
        n_mal = int(round(c.malicious_frac * N))
        self.malicious = np.zeros(N, dtype=bool)
        if n_mal:
            self.malicious[rng.choice(N, n_mal, replace=False)] = True

        # links: [0, M) sensor j -> its AUV; [M, M+N) AUV -> GW; then AUV-AUV (symmetric)
        self.n_links = M + N + N * N
        self.sl = np.empty(self.n_links)
        self.sl[:M] = c.sl_sensor
        self.sl[M:] = c.sl_auv
        self.noise_band = ch.urick_noise_db(c.freq_khz, c.shipping, c.wind) + 10 * math.log10(c.bandwidth)
        self.shadow = ch.ShadowingProcess(self.n_links, c.shadow_sigma, c.shadow_rho, c.p_off, c.p_on, rng)
        self.twin = LinkTwin(N, self.n_links, c.shadow_sigma, c.shadow_rho, c.meas_noise_var)
        self.trust = TrustManager(N, c.trust_prior, c.trust_weights)

        self.chan = [Server(self) for _ in range(N)]
        self.out = [Server(self) for _ in range(N)]
        self.cpu = [Server(self, c.f_auv) for _ in range(N)]
        self.gw_cpu = Server(self, c.f_gw)
        self.servers = self.chan + self.out + self.cpu + [self.gw_cpu]

        self.adv_load = np.zeros(N)
        self.adv_time = np.zeros(N)
        self.last_adv_val = np.zeros(N)
        self.last_adv_t = np.full(N, -1e9)
        self.next_sync = rng.uniform(0, c.sync_period, N)
        self.active: dict[int, Task] = {}
        self.now = 0.0
        self._update_geometry()

    def gw_link(self, i: int) -> int:
        return self.M + i

    def aa_link(self, i: int, k: int) -> int:
        a, b = (i, k) if i < k else (k, i)
        return self.M + self.N + a * self.N + b

    # -------------------------------------------------------------- dynamics
    def _update_geometry(self) -> None:
        c, t, N, M = self.cfg, self.now, self.N, self.M
        th = self.phase + self.omega * t
        nominal = self.centers + self.radius * np.column_stack([np.cos(th), np.sin(th)])
        L = c.area
        u = (c.current_max / math.sqrt(2)) * np.column_stack([
            np.sin(math.pi * nominal[:, 1] / L + self.cur_phase[0]),
            np.cos(math.pi * nominal[:, 0] / L + self.cur_phase[1])])
        self.offset = 0.97 * self.offset + u * c.slot
        xy = np.clip(nominal + self.offset, 0, L)
        self.auv_pos = np.column_stack([xy, self.auv_depth])

        dist = np.empty(self.n_links)
        dist[:M] = np.linalg.norm(self.sensor_pos - self.auv_pos[self.cluster], axis=1)
        dist[M:M + N] = np.linalg.norm(self.auv_pos - self.gw_pos, axis=1)
        dd = np.linalg.norm(self.auv_pos[:, None, :] - self.auv_pos[None, :, :], axis=2)
        dist[M + N:] = dd.reshape(-1)
        self.dist = np.maximum(dist, 1.0)
        self.det_snr = self.sl - ch.path_loss_db(self.dist, c.freq_khz, c.spreading_k) - self.noise_band
        self.optical = np.zeros(self.n_links, dtype=bool)
        self.optical[:M] = self.dist[:M] <= c.optical_range

    def _update_rates(self) -> None:
        c = self.cfg
        snr = self.det_snr + self.shadow.chi
        self.snr_now = snr
        r = ch.goodput(snr, c.bandwidth, c.eta, c.min_snr)
        r[self.shadow.outage] = 0.0
        self.raw_now = ch.raw_rate(snr, c.bandwidth, c.eta)
        r[self.optical] = c.optical_rate
        self.rate_now = r

    def est_rate(self, c: int, link: int, lcb: bool = False) -> float:
        if self.optical[link]:
            return self.cfg.optical_rate
        m = self.twin.m[c, link]
        if lcb:
            m -= self.cfg.lcb_z * math.sqrt(self.twin.P[c, link])
        return ch.goodput_scalar(self.det_snr[link] + m, self.cfg.bandwidth, self.cfg.eta, self.cfg.min_snr)

    def account_air(self, job: Job, seconds: float) -> None:
        if self.optical[job.link]:
            air = 0.0
        else:
            air = seconds * self.raw_now[job.link]
        if job.task is not None:
            job.task.air_bits += air
            job.task.energy += job.power * seconds

    def _measure(self, c: int, link: int) -> None:
        """Piggy-backed / probe SNR measurement of the shadowing on `link`."""
        if self.optical[link]:
            return
        z = self.shadow.chi[link] + self.rng.normal(0.0, math.sqrt(self.cfg.meas_noise_var))
        self.twin.update(c, link, z)

    def _job(self, task, kind, work, link, ready, on_done, owner, prio=False, power=0.0) -> Job:
        return Job(task, kind, work, link, ready, next(self._seq), prio, on_done, owner, power)

    # ------------------------------------------------------------- control
    def _probe(self, c: int, link: int, server: Server, immediate: bool) -> None:
        self.stats.ctrl_bits += self.cfg.probe_bits
        self.stats.probes += 1
        if immediate:
            if self.rate_now[link] > 0:
                self._measure(c, link)
            done = lambda job, t: None
        else:
            done = lambda job, t, c=c, link=link: self._measure(c, link)
        server.push(self._job(None, "probe", self.cfg.probe_bits, link, self.now, done, c, prio=True))

    def _periodic_sync(self) -> None:
        for c in range(self.N):
            if self.now + 1e-9 < self.next_sync[c]:
                continue
            self.next_sync[c] += self.cfg.sync_period
            for j in self.cluster_sensors[c]:
                if not self.optical[j]:
                    self._probe(c, int(j), self.chan[c], immediate=False)
            self._probe(c, self.gw_link(c), self.out[c], immediate=False)
            for k in range(self.N):
                if k != c:
                    self._probe(c, self.aa_link(c, k), self.out[c], immediate=False)

    def load(self, i: int) -> float:
        return self.cpu[i].backlog() + self.out[i].backlog()

    def _state_sharing(self) -> None:
        c = self.cfg
        for i in range(self.N):
            load = self.load(i)
            if self.var.share == "event":
                fire = (abs(load - self.last_adv_val[i]) > c.share_delta
                        or self.now - self.last_adv_t[i] >= c.share_heartbeat)
            else:
                fire = self.now - self.last_adv_t[i] >= c.share_period
            if not fire:
                continue
            self.last_adv_val[i], self.last_adv_t[i] = load, self.now
            value = 0.0 if self.malicious[i] else load      # sinkhole: advertise no backlog
            self.stats.ctrl_bits += c.state_bits

            def done(job, t, i=i, value=value):
                self.adv_load[i], self.adv_time[i] = value, t
            self.out[i].push(self._job(None, "state", c.state_bits, self.gw_link(i), self.now, done, i,
                                       prio=True))

    # ----------------------------------------------------------- decisions
    def _neighbour(self, c: int) -> int:
        best, best_s = -1, math.inf
        T = self.trust.scores()
        for i in range(self.N):
            if i == c:
                continue
            if self.var.trust and T[i] < self.cfg.trust_threshold:
                continue
            s = self.adv_load[i] + self.dist[self.aa_link(c, i)] / 400.0
            if s < best_s:
                best, best_s = i, s
        return best

    def _whatif(self, task: Task, c: int, n: int):
        """Twin what-if evaluation of all 12 actions (mean and LCB link rates)."""
        cfg = self.cfg
        j, S, v = task.sensor, task.size, cfg.sound_speed
        rate_fn = lambda job: self.est_rate(c, job.link)
        w_ch = self.chan[c].backlog(rate_fn)
        w_out = self.out[c].backlog(rate_fn)
        w_cpu = self.cpu[c].backlog()
        w_gw = self.gw_cpu.backlog()
        use_lcb = self.var.uncertainty
        gl_c = self.gw_link(c)
        d_s = self.dist[j] / v
        d_gw_c = self.dist[gl_c] / v
        if n >= 0:
            al = self.aa_link(c, n)
            gl_n = self.gw_link(n)
        D = np.full(N_ACTIONS, BIG)
        DL = np.full(N_ACTIONS, BIG)
        A = np.zeros(N_ACTIONS)
        E = np.full(N_ACTIONS, 100.0)
        bits = np.zeros(N_ACTIONS)
        mask = np.zeros(N_ACTIONS, dtype=bool)

        def rt(link, lcb):
            return max(self.est_rate(c, link, lcb), 1e-6)

        for lcb_pass in ((False, True) if use_lcb else (False,)):
            r_s = rt(j, lcb_pass)
            r_gc = rt(gl_c, lcb_pass)
            if n >= 0:
                r_aa, r_gn = rt(al, lcb_pass), rt(gl_n, lcb_pass)
            for k in range(N_LEVELS):
                if not self.var.semantic and k > 0:
                    continue
                b = S / cfg.kappa[k]
                t_enc = cfg.enc_cycles_per_bit[k] * S / cfg.f_sensor
                C = cfg.cycles_raw if k == 0 else cfg.cycles_feat
                up = t_enc + w_ch + b / r_s + d_s
                e_up = cfg.enc_cycles_per_bit[k] * S * cfg.e_cycle_sensor + cfg.p_tx_sensor * b / r_s
                for e in range(N_EXEC):
                    a = k * N_EXEC + e
                    if e == EXEC_OWN:
                        d = up + w_cpu + C / cfg.f_auv + w_out + cfg.result_bits / r_gc + d_gw_c
                        en = e_up + C * cfg.e_cycle_auv + cfg.p_tx_auv * cfg.result_bits / r_gc
                        acc = cfg.acc_auv
                        bt = b + cfg.result_bits
                    elif e == EXEC_NEIGH:
                        if n < 0:
                            continue
                        d = (up + w_out + b / r_aa + self.dist[al] / v + self.adv_load[n]
                             + C / cfg.f_auv + cfg.result_bits / r_gn + self.dist[gl_n] / v)
                        en = (e_up + cfg.p_tx_auv * b / r_aa + C * cfg.e_cycle_auv
                              + cfg.p_tx_auv * cfg.result_bits / r_gn)
                        acc = cfg.acc_auv
                        bt = 2 * b + cfg.result_bits
                    else:
                        if not cfg.gateway_compute:
                            continue
                        Cg = C * cfg.gw_cycle_factor
                        d = up + w_out + b / r_gc + d_gw_c + w_gw + Cg / cfg.f_gw
                        en = e_up + cfg.p_tx_auv * b / r_gc + Cg * cfg.e_cycle_gw
                        acc = cfg.acc_gw
                        bt = 2 * b
                    mask[a] = True
                    if lcb_pass:
                        DL[a] = min(d, BIG)
                    else:
                        D[a] = min(d, BIG)
                        E[a] = en
                        bits[a] = bt
                        A[a] = acc - task.sens * cfg.acc_drop[k]
        if not use_lcb:
            DL = D.copy()
        return dict(D=D, DL=DL, A=A, E=E, bits=bits, mask=mask, w_ch=w_ch, w_out=w_out,
                    w_cpu=w_cpu, w_gw=w_gw)

    def _features(self, task: Task, c: int, n: int, p: dict):
        cfg = self.cfg
        T = self.trust.scores() if self.var.trust else np.ones(self.N)
        unc = self.twin.std(c, task.sensor) / 8.0 if self.var.uncertainty else 0.0
        oh = [0.0, 0.0, 0.0]
        oh[task.mission] = 1.0
        g = np.array([task.size / 600e3, task.dmax / 480.0, task.voi / 3.0, *oh, task.a_min,
                      p["w_ch"] / 60, p["w_out"] / 60, p["w_cpu"] / 60, T[c], unc,
                      self.N / 10.0, float(cfg.gateway_compute), p["w_gw"] / 60], dtype=np.float64)
        g = np.clip(g, -5, 5)
        phi = np.zeros((N_ACTIONS, PHI_DIM))
        for a in np.flatnonzero(p["mask"]):
            k, e = divmod(int(a), N_EXEC)
            if e == EXEC_OWN:
                t_ex, stale = T[c], 0.0
            elif e == EXEC_NEIGH:
                t_ex, stale = T[n], min((self.now - self.adv_time[n]) / 120.0, 2.0)
            else:
                t_ex, stale = 1.0, 0.0
            dn, dl = p["D"][a] / task.dmax, p["DL"][a] / task.dmax
            phi[a] = [math.log2(cfg.kappa[k]) / 6.0, e == 0, e == 1, e == 2, min(dn, 3.0), min(dl, 3.0),
                      float(dl > 1.0), float(np.clip(10 * (p["A"][a] - task.a_min), -2, 2)),
                      min(p["E"][a] / 20.0, 5.0), p["bits"][a] / 300e3, t_ex, stale]
        loads = np.array([self.load(i) for i in range(self.N)])
        chans = np.array([s.backlog() for s in self.chan])
        crit = np.concatenate([g, np.clip([loads.mean() / 60, loads.max() / 60, chans.mean() / 60,
                                           p["w_gw"] / 60], 0, 5)])
        return g, phi, crit

    def _decide(self, task: Task) -> None:
        cfg, c, j = self.cfg, task.cluster, task.sensor
        if self.var.sync == "ondemand" and not self.optical[j]:
            if self.twin.std(c, j) > cfg.sync_threshold:
                self._probe(c, j, self.chan[c], immediate=True)
        if not self.optical[j]:
            err = self.twin.m[c, j] - self.shadow.chi[j]
            self.stats.twin_sq_err += err * err
            self.stats.twin_n += 1
        n = self._neighbour(c)
        p = self._whatif(task, c, n)
        obs = dict(task=task, pred=p, mask=p["mask"])
        if self.var.policy == "learned":
            obs["g"], obs["phi"], obs["critic"] = self._features(task, c, n, p)
        a, logp = self.policy.act(obs)
        if self.record and self.var.policy == "learned":
            task.rec = dict(g=obs["g"], phi=obs["phi"], mask=p["mask"].copy(), a=a, logp=logp,
                            critic=obs["critic"], r=None, miss=0.0, accfail=0.0)
            self.decisions.append(task.rec)
        self._apply(task, a, n)

    def _apply(self, task: Task, a: int, n: int) -> None:
        cfg, c, j = self.cfg, task.cluster, task.sensor
        k, e = divmod(int(a), N_EXEC)
        task.k, task.e = k, e
        task.executor = c if e == EXEC_OWN else (n if e == EXEC_NEIGH else GW)
        task.accuracy = (cfg.acc_gw if e == EXEC_GW else cfg.acc_auv) - task.sens * cfg.acc_drop[k]
        task.payload = task.size / cfg.kappa[k]
        if e == EXEC_NEIGH:
            task.adv_at_decision = self.adv_load[n]
        self.stats.exec_counts[e] += 1
        self.stats.level_counts[k] += 1
        if task.executor >= 0 and self.malicious[task.executor]:
            self.stats.to_malicious += 1
        t_enc = cfg.enc_cycles_per_bit[k] * task.size / cfg.f_sensor
        task.energy += cfg.enc_cycles_per_bit[k] * task.size * cfg.e_cycle_sensor
        self.chan[c].push(self._job(task, "up", task.payload, j, task.arrival + t_enc, self._up_done, c,
                                    power=cfg.p_tx_sensor))

    # ------------------------------------------------------------ pipeline
    def _up_done(self, job: Job, t: float) -> None:
        task, c = job.task, job.owner
        self._measure(c, task.sensor)
        t_arr = t + self.dist[task.sensor] / self.cfg.sound_speed
        cfg = self.cfg
        if task.e == EXEC_OWN:
            self._to_cpu(c, task, t_arr)
        elif task.e == EXEC_NEIGH:
            self.out[c].push(self._job(task, "fwd", task.payload, self.aa_link(c, task.executor), t_arr,
                                       self._fwd_done, c, power=cfg.p_tx_auv))
        else:
            self.out[c].push(self._job(task, "fwd_gw", task.payload, self.gw_link(c), t_arr,
                                       self._fwd_gw_done, c, power=cfg.p_tx_auv))

    def _fwd_done(self, job: Job, t: float) -> None:
        task, c = job.task, job.owner
        self._measure(c, job.link)
        self._to_cpu(task.executor, task, t + self.dist[job.link] / self.cfg.sound_speed)

    def _fwd_gw_done(self, job: Job, t: float) -> None:
        task, c, cfg = job.task, job.owner, self.cfg
        self._measure(c, job.link)
        t_arr = t + self.dist[job.link] / cfg.sound_speed
        C = (cfg.cycles_raw if task.k == 0 else cfg.cycles_feat) * cfg.gw_cycle_factor
        task.energy += C * cfg.e_cycle_gw
        self.gw_cpu.push(self._job(task, "cpu", C, None, t_arr, lambda jb, tt: self._finalize(jb.task, tt),
                                   GW))

    def _to_cpu(self, x: int, task: Task, t_ready: float) -> None:
        cfg = self.cfg
        if self.malicious[x] and self.rng.random() < cfg.drop_prob:
            task.silent_drop = True        # never reported; resolved at deadline
            return
        task.recv_exec = t_ready
        C = cfg.cycles_raw if task.k == 0 else cfg.cycles_feat
        task.energy += C * cfg.e_cycle_auv
        self.cpu[x].push(self._job(task, "cpu", C, None, t_ready, self._cpu_done, x))

    def _cpu_done(self, job: Job, t: float) -> None:
        task, x, cfg = job.task, job.owner, self.cfg
        task.corrupt = bool(self.malicious[x] and self.rng.random() < cfg.corrupt_prob)
        bits = cfg.result_bits
        if self.var.trust and self.rng.random() < cfg.spot_check_prob:
            task.spot = True
            bits += min(task.payload, cfg.spot_check_max_bits)
        self.out[x].push(self._job(task, "result", bits, self.gw_link(x), t, self._result_done, x,
                                   power=cfg.p_tx_auv))

    def _result_done(self, job: Job, t: float) -> None:
        task, x = job.task, job.owner
        self._measure(x, job.link)
        if self.var.trust and task.e == EXEC_NEIGH:
            residence = t - task.recv_exec
            self.trust.observe(CONSISTENCY, x,
                               residence <= task.adv_at_decision + self.cfg.consistency_slack)
        self._finalize(task, t + self.dist[job.link] / self.cfg.sound_speed)

    # ---------------------------------------------------------- resolution
    def _finalize(self, task: Task, t_arr: float) -> None:
        if task.done:
            return
        delay = t_arr - task.arrival
        on_time = delay <= task.dmax + 1e-9
        acc_ok = task.accuracy >= task.a_min - 1e-9
        detected = False
        if task.executor >= 0 and self.var.trust:
            self.trust.observe(RELIABILITY, task.executor, True)
            if task.spot:
                self.trust.observe(RESULT, task.executor, not task.corrupt)
                detected = task.corrupt
        accepted_corrupt = task.corrupt and not detected and on_time
        success = on_time and acc_ok and not accepted_corrupt and not detected
        if detected:
            self.stats.corrupt_detected += 1
        self._resolve(task, success, on_time, acc_ok, accepted_corrupt, min(delay, task.dmax))
        if on_time:
            self.stats.ontime += 1
            self.stats.delay_ontime += delay

    def _expire(self, task: Task) -> None:
        if task.silent_drop and self.var.trust and task.executor >= 0:
            self.trust.observe(RELIABILITY, task.executor, False)
        acc_ok = task.accuracy >= task.a_min - 1e-9
        self._resolve(task, False, False, acc_ok, False, task.dmax)

    def _resolve(self, task, success, on_time, acc_ok, accepted_corrupt, delay) -> None:
        cfg, s = self.cfg, self.stats
        task.done = True
        self.active.pop(task.id, None)
        s.tasks += 1
        s.voi_total += task.voi
        s.energy += task.energy
        s.data_bits += task.air_bits
        if success:
            s.success += 1
            s.voi_success += task.voi
        s.miss += int(not on_time)
        s.acc_fail += int(not acc_ok)
        s.corrupt_accepted += int(accepted_corrupt)
        r = (task.voi / cfg.v_max * task.accuracy * float(success) * (1 - cfg.w1 * min(delay / task.dmax, 1.0))
             - cfg.w2 * min(task.energy / cfg.e0, cfg.e_cap)
             - cfg.w3 * task.air_bits / cfg.b0
             - cfg.w5 * float(accepted_corrupt))
        s.reward += r
        if task.rec is not None:
            task.rec["r"] = r
            task.rec["miss"] = float(not on_time)
            task.rec["accfail"] = float(not acc_ok)

    # ----------------------------------------------------------------- run
    def _arrivals(self) -> list[Task]:
        cfg = self.cfg
        counts = self.rng.poisson(cfg.task_rate * cfg.slot, self.M)
        out = []
        shares = np.array([m.share for m in MISSIONS])
        for j in np.flatnonzero(counts):
            for _ in range(counts[j]):
                mi = int(self.rng.choice(3, p=shares))
                m = MISSIONS[mi]
                size = self.rng.uniform(m.size_lo, m.size_hi)
                t = self.now + self.rng.uniform(0, cfg.slot)
                out.append(Task(next(self._tid), int(j), int(self.cluster[j]), mi, size, t))
        out.sort(key=lambda x: x.arrival)
        return out

    def run(self) -> dict:
        cfg = self.cfg
        max_drain = int(max(m.deadline for m in MISSIONS) / cfg.slot) + 2
        for slot in range(cfg.n_slots + max_drain):
            self.now = slot * cfg.slot
            if slot > 0:
                self.shadow.step()
                self._update_geometry()
                self.twin.propagate()
            self._update_rates()
            if slot < cfg.n_slots:
                for task in self._arrivals():
                    self.active[task.id] = task
                    self._decide(task)
            elif not self.active:
                break
            if self.var.sync == "periodic":
                self._periodic_sync()
            self._state_sharing()
            for s in self.servers:
                if s.jobs:
                    s.serve(self.now, cfg.slot)
            end = self.now + cfg.slot
            for task in [t for t in self.active.values() if t.deadline <= end]:
                self._expire(task)
        return self.stats.summary()


def _kmeans(xy: np.ndarray, k: int, rng: np.random.Generator, iters: int = 20):
    centers = xy[rng.choice(len(xy), k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin(((xy[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        for i in range(k):
            pts = xy[lab == i]
            centers[i] = pts.mean(0) if len(pts) else xy[rng.integers(len(xy))]
    lab = np.argmin(((xy[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    return lab, centers
