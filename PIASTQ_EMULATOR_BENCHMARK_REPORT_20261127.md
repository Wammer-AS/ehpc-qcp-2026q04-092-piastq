# PIAST-Q Emulator Benchmark Report and Revised 25.0 QPU-Hour Execution Plan

- **Proposal ID:** `EHPC-QCP-2026Q04-092`
- **Partition:** `Piast-Q - 20Qubit Ion-Trap` (PCSS / PSNC, Poznań)
- **Applicant:** Wammer AS (Norway), PI Vetle Wammer
- **Execution environment:** Four CPU compute nodes on the PSNC PIAST-Q emulator cluster (AlmaLinux 9.4, Slurm production jobs `2116`–`2120` using `1.91` node-hours [`1.908 h`], `2.12` node-hours [`2.118 h`] cumulative including cancelled thread-configuration test job `2110`, out of `20.0` allocated; `651 MB` storage used out of `20.0 GB`)
- **Software versions:** `piastq-bench` `0.2.0` (`Apache-2.0`), `Python 3.11.0`, `qiskit==2.4.1`, `qiskit-aqt-provider==2.0.1`, `qiskit-aer==0.18.4`, `aqt-connector==0.7.0`, `ortools==9.15.6755`, `z3-solver==5.4.1.0`, `dwave-neal==0.6.0`, `numpy==2.4.6`, `scipy==1.17.1`
- **Protocol and tests:** [`PREREGISTRATION.md`](PREREGISTRATION.md), [`preregistration_seeds.json`](preregistration_seeds.json), `26/26` unit and integration tests passing (`pytest -v`)

## Summary

This report documents the emulator benchmark and revised `25.0 QPU-hour` execution plan requested in the PSNC technical assessment (`27 November 2026`), covering the candidate/conflict-graph QUBO/HUBO encoding, classical certification, and gate-based QAOA subroutines of `EHPC-QCP-2026Q04-092`:

1. **Hardware path (Item 1):** Implemented in [`src/piastq_bench/piastq_adapter.py`](src/piastq_bench/piastq_adapter.py) on `qiskit==2.4.1` and `qiskit-aqt-provider==2.0.1` (`AQTProvider().offline.ideal()`, `AQTProvider().offline.noisy()`, and `AQTProvider().direct_access()`), with Qiskit `QuantumCircuit` and OpenQASM 2.0 (`fmt="qasm2"`) ingestion (`ingest_external_circuit`) for external CUDA-Q or PennyLane circuits. Each submission is checked structurally and serialized through `aqt_connector.models.circuits.QuantumCircuit(repetitions=shots, quantum_circuit=qiskit_to_aqt_circuit(circuit), number_of_qubits=N)`: native basis `{"rz", "r", "rxx"}` plus one terminal projective measurement of all $N$ qubits, no mid-circuit measurement/`reset`/`delay`/feed-forward, `1` circuit per submission, `<= 200` shots per submission, `<= 2,000` operations per circuit, and `<= 400` `RXX` gates (`0` constraint violations across all emulator runs; `0` physical QPU jobs submitted during the emulator benchmark; output in `artifacts_20261127/point1_hardware_path_demo.json`).
2. **Instance-to-gate mapping and pruning (Item 2):** Compiled `RXX` and wire-operation counts were measured across `72` two-body `(N, density, p)` cells and `36` three-body `(N, density_2body, |E_3|, p)` cells (`108` cells × `3` seeded instances = `324` compiled circuits in `artifacts_20261127/point2_gate_mapping.json`):
   - At $N = 20, \rho = 1.00$, circuits compile to `190 RXX` ($p=1$) and `380 RXX` ($p=2$, `481` wire ops), while $p=3$ requires `570 RXX` and is excluded;
   - At $N = 20, p = 3$, the `<= 400 RXX` ceiling requires $\rho \le 0.70$ (`133` edges give `399 RXX`; $\rho = 0.65$ gives `372 RXX`; $\rho = 0.85$ requires `486 RXX` and is excluded);
   - At $N = 16$, all densities up to $\rho = 1.00$ remain within the gate ceiling through $p = 3$ (`360 RXX`, `473` wire ops);
   - In the ancilla-free three-body (`ZZZ`) HUBO extension, each `ZZZ` phase gadget uses `4 RXX` unmerged plus induced pairwise $Z_a Z_b$ terms from $x_i x_j x_k = \frac{1}{8}(1 - Z_i - Z_j - Z_k + Z_i Z_j + Z_i Z_k + Z_j Z_k - Z_i Z_j Z_k)$. Qiskit `optimization_level=3` KAK synthesis reduces two-qubit gate counts by 15–35%, retaining sparse $N=20, \rho_2=0.30, |E_3|=12$ instances (`118–126 RXX` at $p=1$; `236–252 RXX` / `631` wire ops across the `3` gate-map seeds and `<= 258 RXX` / `627` wire ops across the `8` development seeds at $p=2$) and excluding dense 3-body cells ($|E_3|=120$ compiles to `446–1,338 RXX` and up to `3,858` wire ops).
3. **Shot scaling and cross-platform statistics (Item 3):** Shot-scaling curves across $S \in \{200, 400, 600, 1000, 1200, 2200, 4000\}$ (`1` to `20` independent `200`-shot single-circuit submissions per circuit, `8` instances per cell, discrete resolution `1/8 = 12.5%`) on `AQTProvider().offline.noisy()` (`1%` two-qubit `RXX` and `0.3%` single-qubit `R` depolarizing noise) give the following solved fractions:
   - At $S = 200$ (`1` submission), noisy QAOA solves `12.5%` (`1/8`) at $N=12, \rho=0.65$, `0.0%` (`0/8`) at $N=16, \rho=0.65$, `12.5%` (`1/8`) at $N=16, \rho=1.00$, and `0.0%` (`0/8`) at $N=20, \rho \in \{0.65, 1.00\}$.
   - At $S = 2,200$ shots (`11 x 200` submissions), noisy QAOA solves `87.5%–100.0%` at $N=12$ (versus `50.0%` for uniform random at $N=12$), `87.5%` (`7/8`, mean gap `0.25`) at $N=16, \rho=1.00$ (`240 RXX`), `12.5%` (`1/8`, mean gap reduced from `27.88` to `3.50`, `Gap_CI95 = [-34.62, -5.50]`) at $N=16, \rho=0.65$ (`156 RXX`), and `50.0%` (`4/8`) at $N=20$ ($\rho=0.65$ and $\rho=1.00$) versus `0.0%` (`0/8`) for uniform random at $N \in \{16, 20\}$. At $S = 4,000$ shots (`20 x 200` submissions), $N=12, \rho=0.65$ reaches `100.0%` (`8/8`) and $N=16, \rho=0.65$ reaches mean gap `2.38`. At $N=8$ ($2^8 = 256$ states), uniform random solves `100.0%` (`8/8`) by $S \ge 600$.
   - Joint comparisons across the three gate-based platforms (`Euro-Q-Exa`, `VLQ`, `PIAST-Q`; `0` `Euro-Q-Exa` or `VLQ` QPU runs are included in this emulator package) are fixed at `1,000` shots in development (`5 x 200` on PIAST-Q vs `1,024` subsampled to `1,000` on Euro-Q-Exa), `2,200` shots in intermediate held-out runs (`11 x 200` on PIAST-Q vs `2,250` subsampled to `2,200` on VLQ), and `4,000` shots in final held-out runs (`20 x 200` on PIAST-Q vs `4,096` subsampled to `4,000` on Euro-Q-Exa and VLQ).
4. **Classical baselines and stopping rule (Item 4):** Across all `224` development instances (`192` two-body and `32` three-body), exhaustive $2^N$ enumeration, OR-Tools CP-SAT (`natural` and `QUBO/HUBO`), and `pilot.anneal` ran on `224/224` instances (`100.0%` optimal, maximum `pilot.anneal` runtime `0.0759 s` 2-body and `0.0819 s` 3-body), `dwave-neal` ran on `192/192` 2-body instances (`100.0%` optimal), and Z3 (`natural` and `QUBO`) ran on `144/192` 2-body instances at $N \in \{8, 12, 16\}$ (`100.0%` optimal). Operationalizing the proposal stopping rule (`2026-10-27`) as a dual two-sided 95% paired bootstrap CI criterion (`MeanE_CI95_high < 0` and `Gap_CI95_high < 0` against matched-shot uniform random at $S_{\text{dev}} = 1,000$) excludes all $N=8$ cells, `5` sparse/shallow 2-body cells, and all $N \in \{8, 12, 16\}$ 3-body cells, advancing `47 / 72` two-body cells and `2 / 10` evaluated three-body cells (`artifacts_20261127/piastq_benchmark_summary.json`).
5. **Revised `25.0 QPU-hour` execution plan (Item 5):** Allocating multi-submission runs (`5`, `11`, and `20` submissions of `200` shots per circuit, `3` seeds) onto advancing cells (`<= 380 RXX`, `MeanE_CI95_high < 0`, `Gap_CI95_high < 0`), plus `54` session readout and `RXX`-mirror calibration submissions (`Block E1`), gives `612` distinct circuits and `4,860` single-circuit `200`-shot submissions (`972,000` shots). At the `18.5 s` per-submission baseline floor, this equals `89,910.0 s = 24.9750 QPU hours` (`25.0 QPU hours` requested). Whole-instance plan rescaling (`rescale_rebudget_to_observed_rate`) keeps planned execution within `90,000 s` for any observed Session-1 rate $\tau_{\text{obs}} \ge 18.5\text{ s}$, and `execute_single_submission` enforces client-side pre-call reservation via `QPUBudgetGuard` on non-emulator backends; server-side QPU accounting and per-job hardware timeout configuration with PSNC site support remain pre-QPU operational steps.

## 1. Hardware Path and Constraint Verification (Item 1)

| Constraint | Site Requirement | Adapter Enforcement (`piastq_adapter.py`) | Emulator Verification |
| :--- | :--- | :--- | :--- |
| Qiskit and provider version | Qiskit `2.3–2.4.1`, site AQT provider | `qiskit==2.4.1`, `qiskit-aqt-provider==2.0.1`, `aqt-connector==0.7.0` | Verified on four PIAST-Q CPU nodes (`0` physical QPU runs in emulator phase) |
| External CUDA-Q / PennyLane bridge | Reach client as Qiskit circuits | `ingest_external_circuit(qasm2_or_qc)` (`OpenQASM 2.0` + `QuantumCircuit`) -> `transpile_for_piastq` | Verified in unit tests and `point1_hardware_path_demo.json` |
| Native gate basis | `RZ`, `R(θ,φ)`, `RXX` only | Rejects any op outside `{"rz", "r", "rxx", "measure"}`; strips barriers | `0` violations across all runs |
| Measurement and control flow | Single terminal projective full-register `measure`; no mid-circuit measure, `reset`, `delay`, feed-forward | Topological scan + `q[i]->c[i]` full-register check + `aqt_connector` `QuantumCircuit` wire model | `8` negative unit tests and `100%` wire pass |
| Per-submission limits | `<= 200` shots, `<= 2,000` ops, `1` circuit/submission | `PIASTQ_MAX_SHOTS = 200`, `PIASTQ_MAX_OPS = 2000`, `PIASTQ_MAX_RXX = 400`, single `QuantumCircuit` check | Multi-shot runs split into $K = \lceil S/200 \rceil$ single-circuit submissions |

## 2. Compiled Instance-to-Gate Mapping and Pruning Matrix (Item 2)

### 2.1 Two-Body QUBO Matrix (`72 / 72` Cells After `transpile(..., optimization_level=3)`)

Each pairwise conflict edge compiles to `1` `RXX` gate per QAOA layer (`RXX = p * |E_2|`), zero routing SWAPs under all-to-all ion connectivity, $N \cdot p$ `R` gates, $N(p+1)$ virtual `RZ` gates, and `1` terminal wire `MEASURE` (`wire_ops = p*|E_2| + N(2p+1) + 1 <= 711`, below the `2,000` limit).

| $N$ | Target $\rho$ | Edges $|E_2|$ | $p=1$ `RXX` (wire ops) | $p=2$ `RXX` (wire ops) | $p=3$ `RXX` (wire ops) | Gate Ceiling Status (`<= 400 RXX`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 8 | `0.30` | `8` | `8` (`33`) | `16` (`57`) | `24` (`81`) | `p = 1, 2, 3` within ceiling (excluded in Item 4 by stopping rule) |
| 8 | `0.50` | `14` | `14` (`39`) | `28` (`69`) | `42` (`99`) | `p = 1, 2, 3` within ceiling (excluded in Item 4 by stopping rule) |
| 8 | `0.65` | `18` | `18` (`43`) | `36` (`77`) | `54` (`111`) | `p = 1, 2, 3` within ceiling (excluded in Item 4 by stopping rule) |
| 8 | `0.70` | `20` | `20` (`45`) | `40` (`81`) | `60` (`117`) | `p = 1, 2, 3` within ceiling (excluded in Item 4 by stopping rule) |
| 8 | `0.85` | `24` | `24` (`49`) | `48` (`89`) | `72` (`129`) | `p = 1, 2, 3` within ceiling (excluded in Item 4 by stopping rule) |
| 8 | `1.00` | `28` | `28` (`53`) | `56` (`97`) | `84` (`141`) | `p = 1, 2, 3` within ceiling (excluded in Item 4 by stopping rule) |
| 12 | `0.30` | `20` | `20` (`57`) | `40` (`101`) | `60` (`145`) | `p = 1, 2, 3` within ceiling (`p = 3` excluded in Item 4) |
| 12 | `0.50` | `33` | `33` (`70`) | `66` (`127`) | `99` (`184`) | `p = 1, 2, 3` within ceiling (`p = 1` excluded in Item 4) |
| 12 | `0.65` | `43` | `43` (`80`) | `86` (`147`) | `129` (`214`) | `p = 1, 2, 3` within ceiling (`p = 3` excluded in Item 4) |
| 12 | `0.70` | `46` | `46` (`83`) | `92` (`153`) | `138` (`223`) | `p = 1, 2, 3` within ceiling (`p = 1` excluded in Item 4) |
| 12 | `0.85` | `56` | `56` (`93`) | `112` (`173`) | `168` (`253`) | `p = 1, 2, 3` within ceiling |
| 12 | `1.00` | `66` | `66` (`103`) | `132` (`193`) | `198` (`283`) | `p = 1, 2, 3` within ceiling |
| 16 | `0.30` | `36` | `36` (`85`) | `72` (`153`) | `108` (`221`) | `p = 1, 2, 3` within ceiling (`p = 1` excluded in Item 4) |
| 16 | `0.50` | `60` | `60` (`109`) | `120` (`201`) | `180` (`293`) | `p = 1, 2, 3` within ceiling |
| 16 | `0.65` | `78` | `78` (`127`) | `156` (`237`) | `234` (`347`) | `p = 1, 2, 3` within ceiling |
| 16 | `0.70` | `84` | `84` (`133`) | `168` (`249`) | `252` (`365`) | `p = 1, 2, 3` within ceiling |
| 16 | `0.85` | `102` | `102` (`151`) | `204` (`285`) | `306` (`419`) | `p = 1, 2, 3` within ceiling |
| 16 | `1.00` | `120` | `120` (`169`) | `240` (`321`) | `360` (`473`) | `p = 1, 2, 3` within ceiling (`<= 360 RXX`, `473` wire ops) |
| 20 | `0.30` | `57` | `57` (`118`) | `114` (`215`) | `171` (`312`) | `p = 1, 2, 3` within ceiling |
| 20 | `0.50` | `95` | `95` (`156`) | `190` (`291`) | `285` (`426`) | `p = 1, 2, 3` within ceiling |
| 20 | `0.65` | `124` | `124` (`185`) | `248` (`349`) | `372` (`513`) | `p = 1, 2, 3` within ceiling (`< 0.70` density) |
| 20 | `0.70` | `133` | `133` (`194`) | `266` (`367`) | `399` (`540`) | Boundary (`133` edges = `399 RXX`; `>= 134` edges excluded) |
| 20 | `0.85` | `162` | `162` (`223`) | `324` (`425`) | `486` (`627`) | `p = 1, 2` within ceiling; `p = 3` excluded (`486 > 400`) |
| 20 | `1.00` | `190` | `190` (`251`) | `380` (`481`) | `570` (`711`) | `p = 1, 2` within ceiling; `p = 3` excluded (`570 > 400`) |

### 2.2 Three-Body (`ZZZ`) HUBO Matrix (`36 / 36` Cells, `3` Gate-Map Seeds per Cell)

| $N$ | Base $\rho_2$ | $|E_2|$ | Triples $|E_3|$ | Unmerged `RXX` ($p=1/2/3$) | Compiled Max `RXX` ($p=1 / 2 / 3$) | Max Wire Ops ($p=1 / 2 / 3$) | Gate Ceiling and Advancement Status (`<= 400 RXX`, `<= 2,000` ops) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 8 | `0.30` | `8` | `6` | `46 / 92 / 138` | `40 / 80 / 120` | `130 / 253 / 377` | Within gate ceiling; excluded in Item 4 ($N=8$ saturated) |
| 8 | `0.50` | `14` | `10` | `66 / 132 / 198` | `52 / 104 / 156` | `162 / 321 / 471` | Within gate ceiling; not advanced ($N=8$ saturated) |
| 12 | `0.30` | `20` | `10` | `83 / 166 / 249` | `73 / 146 / 219` | `212 / 423 / 642` | Within gate ceiling; excluded in Item 4 (`Gap_CI95_high >= 0`) |
| 12 | `0.30` | `20` | `20` | `139 / 278 / 417` | `111 / 222 / 333` | `324 / 638 / 958` | Within gate ceiling; not advanced (sparse $N=12$ saturation) |
| 12 | `0.50` | `33` | `20` | `144 / 288 / 432` | `113 / 226 / 339` | `323 / 620 / 942` | Within gate ceiling; not advanced ($N=12$ 2-body / depth cost) |
| 16 | `0.30` | `36` | `12` | `115 / 230 / 345` | `104 / 208 / 312` | `278 / 543 / 813` | Within gate ceiling; excluded in Item 4 (`Gap_CI95_high >= 0`) |
| 16 | `0.30` | `36` | `25` | `188 / 376 / 564` | `160 / 320 / 480` | `437 / 879 / 1,291` | $p = 3$ excluded (`480 > 400 RXX`) |
| 16 | `0.50` | `60` | `40` | `274 / 548 / 822` | `212 / 424 / 636` | `617 / 1,179 / 1,743` | $p \ge 2$ excluded (`> 400 RXX`) |
| 20 | `0.30` | `57` | `12` | `138 / 276 / 414` | `126 / 252 / 378` | `316 / 631 / 924` | `p = 1, 2` advance (`<= 252 RXX` gate-map; `<= 258 RXX` / `627` ops across `8` dev seeds) |
| 20 | `0.30` | `57` | `25` | `214 / 428 / 642` | `190 / 380 / 570` | `491 / 972 / 1,421` | $p = 3$ excluded (`570 > 400 RXX`) |
| 20 | `0.50` | `95` | `35` | `301 / 602 / 903` | `254 / 508 / 762` | `626 / 1,242 / 1,922` | $p \ge 2$ excluded (`> 400 RXX`) |
| 20 | `0.50` | `95` | `120` | `670 / 1,340 / 2,010` | `446 / 892 / 1,338` | `1,283 / 2,649 / 3,858` | All $p$ excluded (`> 400 RXX`; $p \ge 2$ also `> 2,000` ops) |

## 3. Noisy-Emulator Shot Scaling and Cross-Platform Statistics (Item 3)

Measured on `AQTProvider().offline.noisy()` (`M = 8` development instances per cell, discrete solved-fraction step `1/8 = 0.125`, $p=2$, accumulated via $1 \dots 20$ independent `200`-shot single-circuit submissions; raw outputs in `point3_shot_scaling_n8_12_16.json` and `point3_shot_scaling_n20.json`):

| Regime ($p=2$, $M=8$) | `RXX` | $S=200$ (`1x200`) Solved (Gap) | $S=1,000$ (`5x200`) Solved (Gap) | $S=2,200$ (`11x200`) Solved (Gap) | Random @ $S=2,200$ Solved (Gap) | $S=4,000$ (`20x200`) Solved (Gap) | Random @ $S=4,000$ Solved (Gap) | Gap Diff CI95 @ $S=2,200$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $N=8, \rho=0.65$ | `36` | `0.750` (`1.00`) | `0.875` (`0.25`) | `0.875` (`0.25`) | `1.000` (`0.00`) | `0.875` (`0.25`) | `1.000` (`0.00`) | `[+0.00, +0.75]` |
| $N=8, \rho=1.00$ | `56` | `1.000` (`0.00`) | `1.000` (`0.00`) | `1.000` (`0.00`) | `1.000` (`0.00`) | `1.000` (`0.00`) | `1.000` (`0.00`) | `[+0.00, +0.00]` |
| $N=12, \rho=0.65$ | `86` | `0.125` (`3.62`) | `0.625` (`2.00`) | `0.875` (`0.62`) | `0.500` (`0.88`) | `1.000` (`0.00`) | `0.750` (`0.25`) | `[-1.50, +1.50]` |
| $N=12, \rho=1.00$ | `132` | `0.750` (`0.75`) | `1.000` (`0.00`) | `1.000` (`0.00`) | `0.500` (`1.00`) | `1.000` (`0.00`) | `0.750` (`0.38`) | `[-1.75, -0.25]` |
| $N=16, \rho=0.65$ | `156` | `0.000` (`27.88`) | `0.000` (`5.50`) | `0.125` (`3.50`) | `0.000` (`19.62`) | `0.125` (`2.38`) | `0.000` (`8.38`) | `[-34.62, -5.50]` |
| $N=16, \rho=1.00$ | `240` | `0.125` (`53.50`) | `0.375` (`2.50`) | `0.875` (`0.25`) | `0.000` (`64.25`) | `0.875` (`0.25`) | `0.000` (`34.88`) | `[-77.50, -44.25]` |
| $N=20, \rho=0.65$ | `248` | `0.000` (`22.12`) | `0.000` (`8.62`) | `0.500` (`3.25`) | `0.000` (`112.50`) | `0.500` (`2.62`) | `0.000` (`85.62`) | `[-142.00, -73.37]` |
| $N=20, \rho=1.00$ | `380` | `0.000` (`94.50`) | `0.000` (`79.00`) | `0.500` (`25.88`) | `0.000` (`173.25`) | `0.500` (`25.88`) | `0.000` (`108.25`) | `[-218.50, -73.75]` |

### Cross-Platform Equal-Statistics Protocol (`Euro-Q-Exa`, `VLQ`, `PIAST-Q`)

1. **Development comparison (`1,000` shots):** `5 x 200 = 1,000` shots on PIAST-Q vs `1,024` shots planned on `Euro-Q-Exa` (joint analysis subsampled without replacement to `1,000` shots per circuit).
2. **Held-out comparison (`2,200` shots):** `11 x 200 = 2,200` shots on PIAST-Q vs `2,250` shots planned on `VLQ` (subsampled to `2,200` shots per circuit). At $N=16, \rho=1.00$ and $N=20, \rho \in \{0.65, 1.00\}$, `2,200` shots increases the solved fraction from `0.0%–37.5%` (`<= 1,000` shots) to `50.0%–87.5%`, while at $N=16, \rho=0.65$ (`1/8 = 12.5%` solved at $M=8$) `2,200–4,000` shots separates the objective gap (`Gap_CI95 = [-34.62, -5.50]`) and motivates larger held-out instance counts ($M=10\text{–}16$ in Blocks `B2` and `C1`).
3. **Final held-out comparison (`4,000` shots):** `20 x 200 = 4,000` shots on PIAST-Q vs `4,096` shots planned on `Euro-Q-Exa` and `VLQ` (subsampled to `4,000` shots per circuit). No physical QPU runs from `Euro-Q-Exa` or `VLQ` are included in this emulator report.

## 4. Classical Reference Coverage and Stopping Rule Application (Item 4)

On the `224` development instances (`192` two-body and `32` three-body; `8` instances per cell, `S_dev = 1,000` shots = `5 x 200` submissions):

- **Classical solver coverage and runtimes:**
  - Exhaustive $2^N$ enumeration, OR-Tools CP-SAT (`natural` and `QUBO/HUBO`), and Simulated Annealing (`pilot.anneal`, `1,000` reads × `50` sweeps) ran on `224 / 224` development instances (`192/192` 2-body and `32/32` 3-body), reaching the certified global optimum $E^*$ on `224 / 224` (`100.0%`) with maximum `pilot.anneal` runtime `0.0759 s` (2-body) and `0.0819 s` (3-body).
  - `dwave-neal` (`1,000` reads) ran on `192 / 192` two-body instances and reached $E^*$ on `192 / 192` (`100.0%`) (not run on 3-body HUBO).
  - Z3 Optimize (`natural` and `QUBO`) ran on `144 / 192` two-body instances with $N \in \{8, 12, 16\}$ and matched $E^*$ on `144 / 144` (`100.0%`) (omitted on $N=20$ and 3-body where exhaustive $2^N$ enumeration and CP-SAT certify $E^*$).
- **Stopping rule specification and results:**
  - The baseline stopping rule in `EHPC-QCP-2026Q04-092` (`2026-10-27`) specifies that regimes indistinguishable from uniform random sampling on the noisy emulator do not advance to QPU execution.
  - Prior to physical QPU execution (`2026-11-27`), [`PREREGISTRATION.md`](PREREGISTRATION.md) §4.4, `assemble_summary`, `piastq_benchmark_summary.json`, and `build_piastq_25h_rebudget()` apply the dual criterion: a gate-feasible cell (`RXX <= 400`, `ops <= 2,000`) sets `advances_to_qpu = True` if and only if its 10,000-resample two-sided 95% paired bootstrap CI upper bound against `1,000`-shot uniform random sampling is `< 0` on both mean sample energy ($\mathbb{E}[E(x)]$) and selected best-of-$S$ objective gap ($E_{\text{sel}} - E^*$).
  - **Excluded (`advances_to_qpu == False`):**
    1. All `18 / 18` two-body $N = 8$ cells and `2 / 2` three-body $N = 8$ cells (`Gap_CI95_high = 0.00` at $S=1,000$; $2^8 = 256$ states are saturated by random sampling, so `0` QPU submissions are allocated to $N=8$);
    2. Five sparse/shallow two-body cells: $N=12, \rho=0.30, p=3$ (`+2.12`), $N=12, \rho=0.50, p=1$ (`0.00`), $N=12, \rho=0.65, p=3$ (`+0.12`), $N=12, \rho=0.70, p=1$ (`+0.88`), and $N=16, \rho=0.30, p=1$ (`0.00`), in addition to the `2` gate-ceiling exclusions ($N=20, \rho \in \{0.85, 1.00\}, p=3$);
    3. All `6` evaluated $N \in \{12, 16\}$ sparse three-body cells (`Gap_CI95_high = +0.38` to `+3.25`).
  - **Advancing to QPU (`47 / 72` two-body cells and `2 / 10` evaluated three-body cells):**
    - 2-body $N = 12$ (`14` advancing cells, including $\rho \in \{0.50, 0.70, 0.85, 1.00\}$ at $p \in \{2, 3\}$ and $\rho \in \{0.65, 1.00\}$ at $p \in \{1, 2\}$): `43–198 RXX`, `Gap_CI95_high` `-0.25` to `-2.00`, `MeanE_CI95_high` `-290` to `-603`.
    - 2-body $N = 16$ (`17` advancing cells, including $\rho \in \{0.50, 0.65, 0.70, 0.85, 1.00\}$ at $p \in \{1, 2, 3\}$): `60–360 RXX`, `Gap_CI95_high` `-0.75` to `-70.62`, `MeanE_CI95_high` `-506` to `-1,033`.
    - 2-body $N = 20$ (`16` advancing cells, including $\rho \in \{0.50, 0.65\}$ at $p \in \{1, 2, 3\}$ and $\rho \in \{0.85, 1.00\}$ at $p \in \{1, 2\}$): `95–380 RXX`, `Gap_CI95_high` `-15.12` to `-202.50`, `MeanE_CI95_high` `-914` to `-1,600`.
    - 3-body (`ZZZ`) $N = 20, \rho_2 = 0.30, |E_3| = 12, p \in \{1, 2\}$ (`2` advancing cells): `118–258 RXX` (`<= 631` wire ops), `Gap_CI95_high = -12.37` and `-6.75`, `MeanE_CI95_high = -566.6` and `-644.7`.

## 5. Revised 25.0 QPU-Hour Execution Plan and Rescaling Protocol (Item 5)

Validated programmatically in `build_piastq_25h_rebudget()` and cross-checked against `piastq_benchmark_summary.json` by `test_official_25h_rebudget_matches_measured_artifacts`:

| Block | Regime (`<= 380 RXX`, `MeanE_CI95 < 0` and `Gap_CI95 < 0`) | Instances | Depths | Seeds | Shots / Circuit | `200`-Shot Subs / Circuit | Distinct Circuits | Total `200`-Shot Submissions | Total Shots | Hours @ `18.5 s` Floor |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A2 | Dev 2-body $N=12$, $\rho \in \{0.5, 0.7, 0.85, 1.0\}$, $p \in \{2,3\}$ (`<=198 RXX`) | `16` | `2` | `3` | `1,000` | `5` | `96` | `480` | `96,000` | `2.47 h` |
| A3 | Dev 2-body $N=16$, $\rho \in \{0.5, 0.65, 0.85, 1.0\}$, $p \in \{1,2,3\}$ (`<=360 RXX`) | `12` | `3` | `3` | `1,000` | `5` | `108` | `540` | `108,000` | `2.78 h` |
| A4a | Dev 2-body $N=20$, $\rho \in \{0.5, 0.65\}$, $p \in \{1,2,3\}$ (`<=372 RXX`) | `6` | `3` | `3` | `1,000` | `5` | `54` | `270` | `54,000` | `1.39 h` |
| A4b | Dev 2-body $N=20$, $\rho \in \{0.85, 1.0\}$, $p \in \{1,2\}$ (`<=380 RXX`) | `6` | `2` | `3` | `1,000` | `5` | `36` | `180` | `36,000` | `0.92 h` |
| B1 | Held-out 2-body $N=12$, $\rho \in \{0.65, 1.0\}$, $p \in \{1,2\}$ (`<=132 RXX`, VLQ parity) | `10` | `2` | `3` | `2,200` | `11` | `60` | `660` | `132,000` | `3.39 h` |
| B2 | Held-out 2-body $N=16$, $\rho \in \{0.65, 1.0\}$, $p \in \{2,3\}$ (`<=360 RXX`, VLQ parity) | `10` | `2` | `3` | `2,200` | `11` | `60` | `660` | `132,000` | `3.39 h` |
| B3 | Held-out 2-body $N=20$, $\rho \in \{0.65, 1.0\}$, $p \in \{1,2\}$ (`<=380 RXX`, VLQ parity) | `10` | `2` | `3` | `2,200` | `11` | `60` | `660` | `132,000` | `3.39 h` |
| C1 | Final held-out 2-body $N \in \{16, 20\}$, $\rho \in \{0.65, 1.0\}$, $p=2$ (`<=380 RXX`, `4k` parity) | `16` | `1` | `3` | `4,000` | `20` | `48` | `960` | `192,000` | `4.93 h` |
| D1 | Held-out 3-body `ZZZ` $N=20$, $\rho_2=0.30, |E_3|=12$, $p \in \{1,2\}$ (`<=258 RXX`) | `6` | `2` | `3` | `2,200` | `11` | `36` | `396` | `79,200` | `2.04 h` |
| E1 | Calibration reference runs (`6` sessions × `3` circuits: `\|0>^20` `0 RXX`, `\|1>^20` `0 RXX`, `95`-pair `RXX` mirror `190 RXX`) | `18` | `1` | `3` | `200` | `1` | `54` | `54` | `10,800` | `0.28 h` |
| Total | Advancing problem cells (`<= 380 RXX`) plus session calibration (`E1`) | `108` | `1–3` | `3` | `200–4,000` | `1–20` | `612` | `4,860` | `972,000` | `24.98 h` (`89,910 s`) |

### Requested Allocation, Rescaling, and Pre-QPU Accounting Scope

- **Requested QPU allocation:** `25.0 QPU hours` (`90,000.0 s`, call maximum), corresponding to `4,860 x 18.5 s = 89,910.0 s = 24.9750 h` at the site baseline floor (depth-weighted pre-truncation estimate across `19.5–22.5 s` block rates: `103,776.0 s = 28.8267 h`).
- **Whole-instance plan rescaling and operational scope:** In Session 1 (first `<= 1.0` QPU hour, coordinated with PSNC site support), observed mean seconds per `200`-shot submission $\tau_{\text{obs}} \ge 18.5\text{ s}$ sets the planned submission cap $N_{\text{cap}} = \lfloor 90,000 / \tau_{\text{obs}} \rfloor$. Subtracting fixed blocks (`E1 + A2..A4b + D1 = 1,920` submissions) first, remaining capacity $N_{\text{cap}} - 1,920$ is allocated to whole-instance packages in `B1, B2, B3` (`66` submissions per instance) and `C1` (`60` submissions per instance) in locked seed order (`rescale_rebudget_to_observed_rate`). On the client side, `execute_single_submission` requires `QPUBudgetGuard` on non-emulator backends and blocks `backend.run` whenever remaining budget is below the reserved per-submission upper bound (`0` backend calls when exhausted); server-side physical QPU accounting and per-job hardware timeout configuration with PSNC site support remain operational steps prior to physical QPU execution:

| Observed $\tau_{\text{obs}}$ | Max Allowed $N_{\text{cap}}$ | Allocated Submissions | Allocated Shots | Whole Instances `(B1, B2, B3, C1)` | Planned Time (`Subs × τ_obs`) | Within `25.0 h` (`90,000 s`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `18.5 s` | `4,864` | `4,860` | `972,000` | `(10, 10, 10, 16)` | `89,910.0 s` (`24.98 h`) | Yes |
| `20.0 s` | `4,500` | `4,476` | `895,200` | `(9, 9, 8, 14)` | `89,520.0 s` (`24.87 h`) | Yes |
| `22.5 s` | `4,000` | `3,966` | `793,200` | `(7, 7, 7, 11)` | `89,235.0 s` (`24.79 h`) | Yes |
| `30.0 s` | `3,000` | `2,946` | `589,200` | `(4, 4, 3, 5)` | `88,380.0 s` (`24.55 h`) | Yes |

## 6. Artifact Manifest and SHA-256 Checksums

| File | SHA-256 |
| :--- | :--- |
| `pilot.py` (v0.1 baseline) | `d852a72a202eb49a605d5a040a75bd97d8838d1b6b112e46740479c0fe88e057` |
| `artifacts_20261127/point1_hardware_path_demo.json` | `605d25f13aa79a69bf1c8d777c3ada70053ce255b89ea3031fc3d748eca8d2e5` |
| `artifacts_20261127/point2_gate_mapping.json` | `9677f386ebc9ba3323dce14dfd50102a66108f150f2c81361db6a5f891825725` |
| `artifacts_20261127/point3_shot_scaling_n8_12_16.json` | `f414621b780aa7991400cc61025a05a3f1c2c9a13dd824d6e894c883f3ff2ce4` |
| `artifacts_20261127/point3_shot_scaling_n20.json` | `7e756731a2bd798338105e2cc0910b37a6d31c33280d2ece6241b5147b48ae97` |
| `artifacts_20261127/point4_dev_2body_n8_12.json` | `b386c68cd21fe5760aa16fc2fc082643b3fd2aede3eb157e43d698ad05ecef16` |
| `artifacts_20261127/point4_dev_2body_n16.json` | `4658269177aa15141636ce6fa7148053231b9e5b31607523d4fb91c4b8b36e99` |
| `artifacts_20261127/point4_dev_2body_n20_part1.json` | `1089969d489055f27638f6b03e305506c50ed37fc93af58b8a64b9048b2a7ddb` |
| `artifacts_20261127/point4_dev_2body_n20_part2.json` | `90c0d28d60c5119c13640f79e2fcbf5fe8fd3afee93d3c11e79a5de0499cd05c` |
| `artifacts_20261127/point4_dev_3body.json` | `27d1e7ac1bf552d5bfeee68937b85d2f94c87b3d1e168aa7c262575e6f7d8959` |
| `artifacts_20261127/piastq_benchmark_summary.json` | `a0f720e49979e3a7cb5c8de8e6c40ed68a9a3c176451afd1aa583ec1b1c0beeb` |
| `artifacts_20261127/point5_rebudget_25h.json` | `0035420d1eadf61dde673cce04ccd9de5a0483ed70b6011382ebfe344417f448` |
