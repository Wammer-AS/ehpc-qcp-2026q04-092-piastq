# EHPC-QCP-2026Q04-092: PIAST-Q Emulator Benchmark (`v0.2.0`)

Emulator benchmark, native-gate compilation, classical reference solvers, and QPU execution plan for EuroHPC Quantum Access Pilot Call project `EHPC-QCP-2026Q04-092`, partition `Piast-Q - 20Qubit Ion-Trap` (PCSS / PSNC).

Code and benchmark artifacts are released under Apache-2.0 (`LICENSE`). All benchmark inputs are synthetic integer-weighted conflict graphs and 3-uniform hypergraphs generated deterministically from the seed namespaces in [`preregistration_seeds.json`](preregistration_seeds.json).

## Repository Layout

- `pilot.py`: preserved v0.1 ideal-statevector feasibility probe from the initial proposal (`2026-10-27`).
- `src/piastq_bench/piastq_adapter.py`: Qiskit `2.4.1` and `qiskit-aqt-provider` `2.0.1` adapter (`AQTProvider().offline.ideal()`, `AQTProvider().offline.noisy()`, and `AQTProvider().direct_access()`), OpenQASM 2.0 ingestion (`ingest_external_circuit`) for circuits exported from CUDA-Q or PennyLane, structural and `aqt_connector` wire-payload validation (`{"rz", "r", "rxx"}` plus a single terminal full-register measurement, `1` circuit and `<= 200` shots per submission, `<= 2,000` operations, `<= 400` `RXX` gates), multi-submission shot accumulation (`execute_accumulated_submissions`), and client-side QPU budget reservation (`QPUBudgetGuard`).
- `src/piastq_bench/core.py` and `gate_mapping.py`: 2-body QUBO and ancilla-free 3-body HUBO (`ZZZ` phase gadgets) with penalty $P = \sum_i w_i + 1$, statevector QAOA angle selection, and compiled `RXX` / wire-operation sweeps across $N \in \{8, 12, 16, 20\}$, edge density $\rho \in [0.30, 1.00]$, and QAOA depth $p \in \{1, 2, 3\}$.
- `src/piastq_bench/solvers.py`: classical reference solvers on both natural hard constraints and penalized QUBO/HUBO objectives (exhaustive $2^N$ enumeration, OR-Tools CP-SAT, Z3 Optimize, `pilot.anneal`, `dwave-neal`, and matched-shot uniform random sampling).
- `src/piastq_bench/statistics.py`: 10,000-resample paired bootstrap confidence intervals, dual stopping-rule evaluation against uniform random sampling, and `25.0 QPU-hour` execution plan validation and Session-1 rate rescaling.
- `PREREGISTRATION.md` and `preregistration_seeds.json`: pre-registered protocol, seed split (`base_seed_dev = 20261100`, `base_seed_heldout = 20270000`), stopping rule, and cross-platform shot counts (`1,000`, `2,200`, `4,000`).
- `PIASTQ_EMULATOR_BENCHMARK_REPORT_20261127.md` and `artifacts_20261127/`: emulator benchmark report and JSON measurement outputs.

## Reproduction and Tests

```bash
pip install -e ".[dev]"
pytest -v
python scripts/run_piastq_benchmark.py --mode gate_map --output-dir artifacts
```
