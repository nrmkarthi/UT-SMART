| Variant | Success, gateway (%) | Success, AUV-only (%) | p vs UT-SMART (AUV-only) | Corrupted, AUV-only (%) | Control (kbit/task) | Twin RMSE (dB) | Energy, AUV-only (J/task) |
|---|---|---|---|---|---|---|---|
| UT-SMART | 99.3 ± 0.6 | 94.2 ± 1.8 | — | 2.4 | 0.68 | 1.65 | 32.3 |
| w/o uncertainty | 99.5 ± 0.4 | 89.6 ± 4.9 | 0.0546 | 6.0 | 7.08 | 3.13 | 27.6 |
| w/o on-demand sync | 99.8 ± 0.3 | 91.8 ± 3.0 | 0.0743 | 4.0 | 6.96 | 3.03 | 29.2 |
| w/o trust | 99.6 ± 0.5 | 84.8 ± 2.6 | 1.5e-05 | 12.1 | 0.67 | 1.51 | 22.3 |
| w/o event-triggered sharing | 99.7 ± 0.3 | 91.5 ± 2.9 | 0.0835 | 4.2 | 1.32 | 1.50 | 30.3 |
| w/o semantic computing | 45.1 ± 2.6 | 44.6 ± 3.2 | 1.28e-11 | 4.1 | 0.72 | 1.76 | 73.1 |
| MAPPO | 99.9 ± 0.2 | 85.3 ± 2.7 | 7.59e-06 | 11.0 | 7.41 | 3.10 | 22.7 |
