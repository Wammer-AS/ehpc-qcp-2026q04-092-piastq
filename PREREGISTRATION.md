# Pre-Registration: EHPC-QCP-2026Q04-092 (PIAST-Q 20-Qubit Ion-Trap Partition)

- **Proposal ID:** `EHPC-QCP-2026Q04-092`
- **Applicant:** Wammer AS (Norway), PI Vetle Wammer
- **Partition:** `Piast-Q - 20Qubit Ion-Trap` (PCSS / PSNC, Poznań)
- **Software:** `piastq-bench` `0.2.0` (`Apache-2.0`), `qiskit==2.4.1`, `qiskit-aqt-provider==2.0.1`
- **Pre-registration date:** `2026-11-27` (deposited prior to physical QPU execution; baseline protocol defined in the `EHPC-QCP-2026Q04-092` feasibility study, `2026-10-27`)

## 1. Objective Formulation

Given $N \in \{8, 12, 16, 20\}$ binary decision variables with integer weights $w_i \in \{1, \dots, 9\}$, pairwise conflict graph $G = (V, E_2)$, and (in the higher-order extension) 3-body mutual-exclusion hyperedges $E_3 \subseteq \binom{V}{3}$, all solvers minimize the common unconstrained binary objective

$$E(x) = -\sum_{i=0}^{N-1} w_i x_i + P \sum_{(i,j)\in E_2} x_i x_j + P \sum_{(i,j,k)\in E_3} x_i x_j x_k, \qquad P = \sum_{i=0}^{N-1} w_i + 1, \quad x \in \{0, 1\}^N.$$

Every feasible assignment satisfies $E(x) \le 0$ and every infeasible assignment satisfies $E(x) \ge 1$. Invalid samples are never repaired post hoc. Classical solvers are evaluated on both this penalized objective (`CP-SAT QUBO/HUBO`, `Z3 QUBO/HUBO`, `pilot.anneal`, `dwave-neal`) and the natural hard-constraint formulation (`CP-SAT natural`, `Z3 natural`), alongside exhaustive $2^N$ enumeration.

## 2. Hardware Interface and Gate Ceilings (Items 1 and 2)

1. **Client and native basis:** Qiskit `2.4.1` and `qiskit-aqt-provider` `2.0.1` (`AQTProvider`), native operations `{"rz", "r", "rxx"}` plus a single terminal projective measurement of the full $N$-qubit register (`q[i] -> c[i]`).
2. **Excluded operations:** No mid-circuit measurements, no `reset`, no `delay`, no classical feed-forward or control flow, and no ancilla qubits.
3. **Submission limits:** One circuit per submission and at most `200` shots per submission, serialized and checked via `aqt_connector.models.circuits.QuantumCircuit` (`<= 2,000` operations per circuit).
4. **Two-qubit gate ceiling:** Any 2-body `(N, density, p)` or 3-body `(N, density_2body, |E_3|, p)` cell whose compiled `RXX` count exceeds `400` (or `2,000` total wire operations) is excluded before QPU allocation.

## 3. Development and Held-Out Split

- **Development set (`base_seed_dev = 20261100`):** Used for (a) ideal-statevector bounded angle selection (`24` seeded candidates per instance and depth, matching `pilot.py`), (b) noisy-emulator shot-scaling evaluation (`S in {200, 400, 600, 1000, 1200, 2200, 4000}` via `1..20` independent `200`-shot submissions), and (c) evaluating the stopping rule.
- **Held-out set (`base_seed_heldout = 20270000`):** Disjoint deterministic seed namespace (`>= +8,900` seed offset; zero instance overlap verified in `test_dev_and_heldout_seed_namespaces_are_disjoint`, including archived 3-body development seeds `20781220..20781227` versus held-out seeds `20790120..20790125`). Held-out instances are not used to select angles, shot budgets, or regime boundaries.

## 4. Outcomes, Cross-Platform Shot Counts, and Stopping Rule (Items 3 and 4)

1. **Primary outcome:** Fraction of instances in which the lowest-energy sampled bitstring (deterministic lexicographic tie-break, no post-hoc repair) attains the certified global optimum ($E_{\text{sel}} = E^*$ and feasible set) at shot budget $S$.
2. **Secondary outcomes:** Mean objective gap ($E_{\text{sel}} - E^*$), mean sample energy $\mathbb{E}[E(x)]$, valid-sample fraction, optimum-sample fraction, feasible weight ratio $E_{\text{sel}} / E^*$, and QPU or emulator execution time reported separately from classical pre- and post-processing.
3. **Cross-platform equal-statistics shot counts (gate-based platforms):**
   - **Development comparison:** `S_dev = 1,000` shots (`5 x 200`-shot submissions on PIAST-Q) matched against `1,024` shots on `Euro-Q-Exa` (with `1,000`-shot subsampling without replacement in joint analysis).
   - **Intermediate held-out comparison:** `S_mid = 2,200` shots (`11 x 200`-shot submissions on PIAST-Q) matched against `2,250` shots on `VLQ` (with `2,200`-shot subsampling).
   - **Final held-out comparison:** `S_final = 4,000` shots (`20 x 200`-shot submissions on PIAST-Q) matched against `4,096` shots on `Euro-Q-Exa` and `VLQ` (with `4,000`-shot subsampling).
4. **Stopping rule against uniform random sampling:**
   - **Proposal baseline (`2026-10-27`):** Regimes where the noisy emulator does not outperform uniform random sampling do not advance to QPU execution.
   - **Operational dual criterion (`2026-11-27`, fixed prior to QPU execution):** Implemented identically in `assemble_summary`, `piastq_benchmark_summary.json`, and `build_piastq_25h_rebudget()`. A gate-feasible problem cell (`RXX <= 400`, `ops <= 2,000`) advances (`advances_to_qpu == True`) if and only if, on the `1,000`-shot development set, `AQTProvider().offline.noisy()` achieves a `10,000`-resample paired bootstrap two-sided 95% confidence interval upper bound `< 0` relative to matched-shot uniform random sampling on both mean sample energy ($\mathbb{E}[E(x)]$) and selected best-of-$S$ objective gap ($E_{\text{sel}} - E^*$). Under this criterion, `47 / 72` two-body cells and `2 / 10` evaluated three-body cells advance; all $N = 8$ cells ($2^8 = 256$ states) and `5` sparse/shallow two-body cells at $N \in \{12, 16\}$ are excluded. Session readout and `RXX` mirror drift circuits (`Block E1`, `0` and `190 RXX`) are budgeted separately as calibration reference runs.

## 5. QPU Execution Budget (`25.0 QPU Hours`) and Session-1 Rescaling (Item 5)

1. **Baseline plan (`18.5 s` per submission floor):** `612` distinct circuits (`4,860` single-circuit `200`-shot submissions, `972,000` total shots) across `A2`–`A4b` (`1,000` shots), `B1`–`B3` and `D1` (`2,200` shots), `C1` (`4,000` shots), and `E1` (`200` shots) total `4,860 x 18.5 s = 89,910.0 s = 24.9750 QPU hours` (`25.0 QPU hours` requested).
2. **Plan-level rescaling and accounting scope:** Given observed Session-1 mean submission duration $\tau_{\text{obs}} \ge 18.5\text{ s}$, the planned submission count is $N_{\text{cap}} = \lfloor 90,000 / \tau_{\text{obs}} \rfloor$. Fixed blocks (`E1 -> A2..A4b -> D1`) are allocated first in whole-instance packages, and remaining capacity $N_{\text{cap}} - N_{\text{fixed}}$ is distributed across whole-instance packages in `B1`–`B3` (`66` submissions per instance) and `C1` (`60` submissions per instance) via `rescale_rebudget_to_observed_rate`. On the client side, `execute_single_submission` requires `QPUBudgetGuard` on non-emulator backends and blocks `backend.run` whenever remaining budget is below the reserved per-submission upper bound; server-side QPU accounting and per-job hardware timeout configuration with PSNC site support remain operational steps prior to physical QPU execution.
