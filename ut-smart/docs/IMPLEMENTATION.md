# Paper → code map

| Paper (section / equation) | Code |
|---|---|
| Thorp absorption α(f), path loss A(d,f) (Eq. 1–2) | `utsmart/channel.py: thorp_absorption, path_loss_db` |
| Urick noise N(f): turbulence, shipping, wind, thermal | `channel.urick_noise_db` |
| SNR γ_l(t), AR(1) shadowing χ_l, Markov outage (Eq. 3–4) | `channel.ShadowingProcess`, `env.UnderwaterEnv._update_rates` |
| ARQ goodput R_l, PER(γ) (Eq. 5–6), 3 dB floor, optical 1 Mb/s ≤ 60 m | `channel.goodput`, `env._update_rates` |
| Missions (Table II), semantic levels κ_k, c_k, δ_k, A(k,e) (Eq. 7) | `config.MISSIONS`, `SimConfig.kappa/enc_cycles_per_bit/acc_drop`, `env._apply` |
| Pipeline & delay D (Eq. 8): encode → cluster channel → forward → CPU → result | `env._apply → _up_done → _fwd_done / _fwd_gw_done → _to_cpu → _cpu_done → _result_done → _finalize` |
| FIFO queues (cluster channel, AUV outbound, AUV CPU, gateway CPU) | `env.Server` (priority-then-FIFO, time-varying rate, sub-slot timing) |
| Energy E (Eq. 9) | accumulated per task in `env.account_air`, `_apply`, `_to_cpu`, `_fwd_gw_done` |
| Threat model: 70 % corruption, 10 % silent drops, zero-backlog sinkhole advertisement | `env._to_cpu`, `env._cpu_done`, `env._state_sharing` |
| Success indicator, reward r_τ (Eq. 11–12) | `env._finalize`, `env._resolve` |
| CMDP constraints ε_D, ε_A; Lagrangian (Eq. 13–16) | `mappo.ConstrainedMAPPO.update` |
| Kalman twin: propagation and update (Eq. 17–18) | `twin.LinkTwin` |
| On-demand sync (√P_j > τ_s) vs periodic sync (every T_s) | `env._decide`, `env._periodic_sync`, `env._probe` |
| What-if evaluation with mean and LCB rates (z = 1) | `env._whatif` |
| Beta trust R, B, S and T_i = 0.3R + 0.3B + 0.4S (Eq. 19); spot checks p_c = 0.2 | `trust.TrustManager`, `env._finalize`, `env._result_done`, `env._expire` |
| Trust threshold τ_T for neighbour selection | `env._neighbour` |
| Shared actor π_θ(a\|o) ∝ exp f_θ(g, φ_a), 27-64-64-1 tanh (Eq. 20) | `policies.ActionScorer`, features in `env._features` |
| Centralized critic (19-D), shaped reward, one-step advantage (Eq. 21–22) | `mappo.ConstrainedMAPPO.update` |
| PPO-clip 0.2, entropy 0.005, 6 epochs, minibatch 128, Adam 1e-3, grad-clip 1 | `mappo.PPOConfig` |
| Projected dual ascent η_λ = 0.5 (Eq. 23–24) | end of `ConstrainedMAPPO.update` |
| Domain randomization over M, N, ρ_m, σ, regime | `mappo.random_scenario` |
| Event-triggered (Δ > 15 s or 120 s heartbeat) vs periodic (20 s) state sharing | `env._state_sharing` |
| Algorithm 1 | `env._decide` + `env._apply` |
| Baselines Raw-Local, DE-BR, Greedy-Sem, MAPPO | `policies.RawLocal / DelayEnergyBR / GreedySemantic`, `config.VARIANTS["mappo"]` |
| Ablations (Table IV) | `config.VARIANTS["wo_*"]` |
| Sweeps (Figs. 3–7) | `utsmart/experiments.py: sweeps()` |

## Action and feature layout

* Action index `a = k * 3 + e`, with `k ∈ {0,1,2,3}` the compression level (1×, 4×, 16×, 64×) and
  `e ∈ {0: own AUV, 1: best-ranked neighbour, 2: gateway}`. Invalid actions are masked: the gateway in
  the AUV-only regime, the neighbour when none passes the trust threshold, and k > 0 for `wo_semantic`.
* `g` (15): size, deadline, VoI, mission one-hot (3), A_min, own channel / outbound / CPU backlog, own
  trust, twin σ of the sensor link, fleet size, regime flag, gateway CPU backlog.
* `φ_a` (12): log₂κ/6, executor one-hot (3), D̂/D_max, D̂_LCB/D_max, 1[D̂_LCB > D_max],
  10(A − A_min), Ê/20 J, acoustic bits/300 kbit, executor trust, staleness of the neighbour's advertisement.
* Critic (19): `g` plus mean/max AUV load, mean cluster-channel backlog and gateway CPU backlog.

## Modelling choices not pinned down in the paper

These are documented so the results can be interpreted correctly; each is a single constant or a few lines.

1. **Advertised load** = AUV CPU backlog + outbound-modem backlog (seconds). CPU backlog alone is
   almost always ~0 s at the paper's loads, so it could not drive the 15 s event trigger.
2. **Trust prior** Beta(2, 1) per component (T = 0.67 initially). With a uniform Beta(1, 1) prior,
   T = 0.5 < τ_T = 0.55 would exclude every neighbour before any evidence exists.
3. **On-demand probe**: the 512-bit exchange is queued with priority on the cluster channel (it costs
   airtime ahead of the data). Its measurement is applied at decision time if the link is not in outage.
4. **Periodic sync** probes every non-optical sensor link of the cluster, the AUV–gateway link and all
   AUV–AUV links of the AUV every T_s.
5. **State broadcasts** are sent on the AUV's outbound modem at the rate of its gateway link.
6. **Spot checks** add min(payload, 20 kbit) to the result message. A detected corrupted result is
   rejected (the task fails, but the corruption is not accepted), and the S evidence is updated.
   Gateway re-inference is not queued on the gateway CPU.
7. **Consistency evidence B** compares a neighbour's residence time (arrival → result sent) with the
   load it advertised at decision time plus 30 s.
8. **Silent drops** are noticed when the deadline expires and count as R-evidence against the executor.
   Expired tasks dropped by honest congested queues do not count as R-evidence.
9. **Gateway CPU backlog** is assumed known to the AUVs (its downlink cost is not modelled), as is the
   trust vector broadcast by the gateway.
10. **Gateway compute energy** uses 5×10⁻¹⁰ J/cycle. All AUV transmissions (forwarding, results and
    spot checks) are charged to the task's energy.
11. Episodes stop generating tasks after 500 slots and then **drain** until every task is resolved, so
    late tasks are not censored.
12. **AUV drift**: the offset follows o ← 0.97·o + u(x)·Δ, with a sinusoidal current field |u| ≤ 0.3 m/s.

## Performance

One 1000 s episode with M = 100 and N = 5 takes about 0.2 s of CPU time (pure NumPy/Python). One
600-episode training run takes about 2.5–3 min on a single core.
