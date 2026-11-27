"""Unit and integration tests for the PIAST-Q emulator benchmark."""
import math
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from piastq_bench.core import (
    make_density_instance,
    make_3body_instance,
    energy_table,
    energy_table_3body,
    valid_mask,
    valid_mask_3body,
    qaoa_state,
    sample_metrics,
)
from piastq_bench.piastq_adapter import (
    PIASTQ_MAX_SHOTS,
    PIASTQ_MAX_OPS,
    PIASTQ_MAX_RXX,
    get_piastq_backends,
    build_qaoa_circuit,
    transpile_for_piastq,
    validate_piastq_submission,
    ingest_external_circuit,
    execute_single_submission,
    execute_accumulated_submissions,
)
from piastq_bench.gate_mapping import (
    compile_and_count_2body,
    compile_and_count_3body,
    compile_calibration_reference_circuits,
)
from piastq_bench.solvers import (
    solve_cpsat_natural,
    solve_cpsat_qubo,
    solve_z3_natural,
    solve_z3_qubo,
    solve_neal_qubo,
    uniform_random_samples,
)
from piastq_bench.statistics import (
    QPUBudgetGuard,
    build_piastq_25h_rebudget,
    paired_bootstrap_stopping_rule,
    rescale_rebudget_to_observed_rate,
    validate_rebudget_plan,
)


class TestPoint1HardwarePathAndConstraints:
    def test_site_client_versions_and_offline_backends(self):
        b_ideal, b_noisy = get_piastq_backends()
        assert b_ideal.name == "offline_simulator_no_noise"
        assert b_noisy.name == "offline_simulator_noise"
        assert b_ideal.num_qubits == 20
        assert b_noisy.num_qubits == 20
        assert set(b_noisy.operation_names) == {"rz", "r", "rxx", "measure"}

    def test_validator_rejects_all_forbidden_constructs(self):
        _, b_noisy = get_piastq_backends()
        w, edges = make_density_instance(4, 0.5, seed=101)
        qc = build_qaoa_circuit(w, edges, [0.3], [0.2])
        tqc = transpile_for_piastq(qc, b_noisy, seed_transpiler=42)

        # Valid baseline passes
        info = validate_piastq_submission(tqc, shots=200)
        assert info["valid"] is True
        assert info["rxx_count"] == len(edges)
        assert info["wire_ops"] <= PIASTQ_MAX_OPS

        # 1. > 200 shots rejected
        with pytest.raises(ValueError, match="shots"):
            validate_piastq_submission(tqc, shots=201)

        # 2. Multi-circuit batch rejected (one circuit per submission)
        with pytest.raises(ValueError, match="single QuantumCircuit"):
            validate_piastq_submission([tqc, tqc], shots=100)  # type: ignore[arg-type]

        # 3. Non-native gate rejected
        bad_gate = QuantumCircuit(2, 2)
        bad_gate.h(0)
        bad_gate.cx(0, 1)
        bad_gate.measure([0, 1], [0, 1])
        with pytest.raises(ValueError, match="non-native"):
            validate_piastq_submission(bad_gate, shots=100)

        # 4. Mid-circuit measurement rejected
        mid_meas = QuantumCircuit(2, 2)
        mid_meas.rz(0.2, 0)
        mid_meas.measure(0, 0)
        mid_meas.rz(0.3, 0)
        mid_meas.measure(1, 1)
        with pytest.raises(ValueError, match="(?i)terminal|mid-circuit"):
            validate_piastq_submission(mid_meas, shots=100)

        # 4b. Classical feed-forward (if_test) rejected
        ff_qc = QuantumCircuit(2, 2)
        ff_qc.rz(0.2, 0)
        with ff_qc.if_test((ff_qc.clbits[0], 1)):
            ff_qc.rz(0.4, 1)
        ff_qc.measure([0, 1], [0, 1])
        with pytest.raises(ValueError, match="(?i)feed-forward|control flow"):
            validate_piastq_submission(ff_qc, shots=100)

        # 4c. > 400 RXX rejected when max_rxx=400
        w20, e20 = make_density_instance(20, 1.0, seed=102)
        qc20_p3 = build_qaoa_circuit(w20, e20, [0.2, 0.3, 0.4], [0.1, 0.2, 0.3])
        tqc20_p3 = transpile_for_piastq(qc20_p3, b_noisy, seed_transpiler=42)
        with pytest.raises(ValueError, match="(?i)RXX count 570 exceeds ceiling 400"):
            validate_piastq_submission(tqc20_p3, shots=200, max_rxx=PIASTQ_MAX_RXX)

        # 5. Reset and delay rejected
        has_reset = QuantumCircuit(2, 2)
        has_reset.rz(0.1, 0)
        has_reset.reset(0)
        has_reset.measure([0, 1], [0, 1])
        with pytest.raises(ValueError, match="non-native|reset"):
            validate_piastq_submission(has_reset, shots=100)

        has_delay = QuantumCircuit(2, 2)
        has_delay.delay(10, 0)
        has_delay.measure([0, 1], [0, 1])
        with pytest.raises(ValueError, match="non-native|delay"):
            validate_piastq_submission(has_delay, shots=100)

        # 6. Partial register measurement rejected (must measure full register)
        partial_meas = QuantumCircuit(3, 3)
        partial_meas.rz(0.2, 0)
        partial_meas.measure([0, 1], [0, 1])
        with pytest.raises(ValueError, match="full register"):
            validate_piastq_submission(partial_meas, shots=100)

        # 7. > 2,000 operations rejected
        huge_ops = QuantumCircuit(2, 2)
        for k in range(2005):
            huge_ops.rz(0.01 * ((k % 7) + 1), k % 2)
        huge_ops.measure([0, 1], [0, 1])
        with pytest.raises(ValueError, match="2000|operations"):
            validate_piastq_submission(huge_ops, shots=100)

    def test_qasm_external_ingestion_for_cudaq_and_pennylane(self):
        _, b_noisy = get_piastq_backends()
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
h q[2];
cx q[0],q[1];
rz(0.4) q[1];
cx q[0],q[1];
rx(0.5) q[0];
rx(0.5) q[1];
rx(0.5) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
"""
        qc = ingest_external_circuit(qasm_str, fmt="qasm2")
        tqc = transpile_for_piastq(qc, b_noisy, seed_transpiler=7)
        info = validate_piastq_submission(tqc, shots=200)
        assert info["valid"] is True
        assert info["rxx_count"] == 1

    def test_multi_submission_accumulator_respects_200_shot_cap(self):
        b_ideal, b_noisy = get_piastq_backends()
        w, edges = make_density_instance(6, 0.5, seed=202)
        qc = build_qaoa_circuit(w, edges, [0.35], [0.28])
        tqc = transpile_for_piastq(qc, b_noisy, seed_transpiler=11)

        res = execute_accumulated_submissions(
            b_noisy, tqc, total_shots=450, max_shots_per_submission=PIASTQ_MAX_SHOTS, base_seed=900
        )
        assert res["submissions_count"] == 3
        assert res["shot_chunks"] == [200, 200, 50]
        assert len(res["samples"]) == 450
        assert np.all((res["samples"] >= 0) & (res["samples"] < 64))


class TestExactCircuitStatevectorEquivalence:
    def test_2body_compiled_aqt_circuit_matches_pilot_statevector(self):
        b_ideal, _ = get_piastq_backends()
        w, edges = make_density_instance(5, 0.6, seed=303)
        energy = energy_table(w, edges)
        scale = float(np.max(np.abs(energy)))
        normalized = energy / scale
        gammas = [0.41, -0.27]
        betas = [0.33, 0.52]

        expected_probs = np.abs(qaoa_state(normalized, gammas, betas)) ** 2
        qc = build_qaoa_circuit(w, edges, gammas, betas, phase_scale=scale)
        tqc = transpile_for_piastq(qc, b_ideal, seed_transpiler=19)
        validate_piastq_submission(tqc, shots=200)

        qc_no_meas = tqc.remove_final_measurements(inplace=False)
        actual_probs = Statevector.from_instruction(qc_no_meas).probabilities()
        np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-10)

    def test_3body_compiled_aqt_circuit_matches_3body_statevector(self):
        b_ideal, _ = get_piastq_backends()
        w, edges2, triples3 = make_3body_instance(5, density_2body=0.4, num_triples=3, seed=404)
        energy = energy_table_3body(w, edges2, triples3)
        valid = valid_mask_3body(5, edges2, triples3)
        # Verify penalty theorem on 3-body HUBO
        assert np.all(energy[valid] <= 0)
        assert np.all(energy[~valid] >= 1)

        scale = float(np.max(np.abs(energy)))
        normalized = energy / scale
        gammas = [0.38, -0.22]
        betas = [0.29, 0.47]

        expected_probs = np.abs(qaoa_state(normalized, gammas, betas)) ** 2
        qc = build_qaoa_circuit(w, edges2, gammas, betas, triples=triples3, phase_scale=scale)
        tqc = transpile_for_piastq(qc, b_ideal, seed_transpiler=23)
        validate_piastq_submission(tqc, shots=200)

        qc_no_meas = tqc.remove_final_measurements(inplace=False)
        actual_probs = Statevector.from_instruction(qc_no_meas).probabilities()
        np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-10)


class TestPoint2InstanceGateMappingAndPruning:
    def test_psnc_gate_pruning_boundaries(self):
        _, b_noisy = get_piastq_backends()
        # N=20, density=1.0: p=2 <= 400 (380 RXX), p=3 > 400 (570 RXX)
        r_20_10_p2 = compile_and_count_2body(20, 1.0, depth=2, seed=501, backend=b_noisy)
        r_20_10_p3 = compile_and_count_2body(20, 1.0, depth=3, seed=501, backend=b_noisy)
        assert r_20_10_p2["rxx_count"] == 380
        assert r_20_10_p2["passes_ceiling"] is True
        assert r_20_10_p3["rxx_count"] == 570
        assert r_20_10_p3["passes_ceiling"] is False

        # N=20, p=3 requires density < ~0.70
        r_20_065_p3 = compile_and_count_2body(20, 0.65, depth=3, seed=502, backend=b_noisy)
        r_20_075_p3 = compile_and_count_2body(20, 0.75, depth=3, seed=502, backend=b_noisy)
        assert r_20_065_p3["rxx_count"] < PIASTQ_MAX_RXX
        assert r_20_065_p3["passes_ceiling"] is True
        assert r_20_075_p3["rxx_count"] > PIASTQ_MAX_RXX
        assert r_20_075_p3["passes_ceiling"] is False

        # N=16 runs to p=3 even at density=1.0 (360 RXX <= 400)
        r_16_10_p3 = compile_and_count_2body(16, 1.0, depth=3, seed=503, backend=b_noisy)
        assert r_16_10_p3["rxx_count"] <= 360
        assert r_16_10_p3["passes_ceiling"] is True

    def test_3body_gate_cost_and_pruning(self):
        _, b_noisy = get_piastq_backends()
        r_small = compile_and_count_3body(12, density_2body=0.3, num_triples=10, depth=1, seed=601, backend=b_noisy)
        assert r_small["passes_ceiling"] is True
        assert r_small["rxx_count"] <= PIASTQ_MAX_RXX

        # Dense 3-body at N=20 exceeds ceiling and is pruned
        r_dense = compile_and_count_3body(20, density_2body=0.5, num_triples=120, depth=2, seed=602, backend=b_noisy)
        assert r_dense["passes_ceiling"] is False


class TestClassicalSolversAndStoppingRule:
    def test_cpsat_z3_and_neal_match_exact_optimum_2body_and_3body(self):
        w2, e2 = make_density_instance(10, 0.45, seed=701)
        en2 = energy_table(w2, e2)
        opt2 = int(np.min(en2))

        assert solve_cpsat_natural(w2, e2)["energy"] == opt2
        assert solve_cpsat_qubo(w2, e2)["energy"] == opt2
        assert solve_z3_natural(w2, e2)["energy"] == opt2
        assert solve_z3_qubo(w2, e2)["energy"] == opt2
        neal_res = solve_neal_qubo(w2, e2, num_reads=200, seed=71)
        assert neal_res["selected_energy"] == opt2

        # 3-body exact agreement
        w3, e3_2, e3_3 = make_3body_instance(9, density_2body=0.35, num_triples=6, seed=702)
        en3 = energy_table_3body(w3, e3_2, e3_3)
        opt3 = int(np.min(en3))
        assert solve_cpsat_natural(w3, e3_2, triples=e3_3)["energy"] == opt3
        assert solve_cpsat_qubo(w3, e3_2, triples=e3_3)["energy"] == opt3
        assert solve_z3_natural(w3, e3_2, triples=e3_3)["energy"] == opt3
        assert solve_z3_qubo(w3, e3_2, triples=e3_3)["energy"] == opt3

    def test_stopping_rule_and_rebudget_validator(self):
        rng = np.random.default_rng(801)
        rand_energies = rng.normal(loc=50.0, scale=5.0, size=12)
        same_energies = rand_energies + rng.normal(loc=0.1, scale=0.5, size=12)
        better_energies = rand_energies - rng.uniform(8.0, 15.0, size=12)

        res_fail = paired_bootstrap_stopping_rule(same_energies, rand_energies, seed=1)
        assert res_fail["advances"] is False

        res_pass = paired_bootstrap_stopping_rule(better_energies, rand_energies, seed=1)
        assert res_pass["advances"] is True
        assert res_pass["ci95_high"] < 0.0

        plan = [
            {
                "block": "test_block",
                "instances": 60,
                "depths": 2,
                "seeds": 3,
                "shots_per_circuit": 2200,
                "submissions_per_circuit": 11,
                "max_rxx": 360,
                "advances": True,
            },
            {
                "block": "heldout_final",
                "instances": 15,
                "depths": 1,
                "seeds": 3,
                "shots_per_circuit": 4000,
                "submissions_per_circuit": 20,
                "max_rxx": 380,
                "advances": True,
            },
        ]
        summary = validate_rebudget_plan(plan, seconds_per_submission=18.5)
        assert summary["total_submissions"] == 60 * 2 * 3 * 11 + 15 * 1 * 3 * 20
        assert math.isclose(summary["total_hours_floor"], summary["total_submissions"] * 18.5 / 3600.0)

    def test_official_25h_rebudget_matches_measured_artifacts(self):
        from pathlib import Path
        import json

        rebudget = build_piastq_25h_rebudget()
        assert rebudget["total_distinct_circuits"] == 612
        assert rebudget["total_submissions"] == 4860
        assert rebudget["total_shots"] == 972000
        assert math.isclose(rebudget["total_seconds_floor"], 89910.0)
        assert math.isclose(rebudget["total_hours_floor"], 24.975)
        assert math.isclose(rebudget["total_seconds_depth_adjusted"], 103776.0)
        assert math.isclose(rebudget["total_hours_depth_adjusted"], 28.8267)
        assert rebudget["requested_qpu_hours"] == 25.0

        # Verify E1 compiled calibration reference circuits (0 RXX readout, 190 RXX mirror drift)
        cal = compile_calibration_reference_circuits(20)
        assert cal["readout_all_zero"]["rxx_count"] == 0
        assert cal["readout_all_one"]["rxx_count"] == 0
        assert cal["rxx_mirror_95pairs"]["rxx_count"] == 190

        # Verify deterministic rescaling stays within 90,000 s (25.0 h) across observed rates
        for tau in (18.5, 20.0, 22.5, 30.0, 60.0):
            scaled = rescale_rebudget_to_observed_rate(tau, budget_seconds=90000.0)
            assert scaled["within_budget"] is True
            assert scaled["total_seconds"] <= 90000.0

        # Verify live QPUBudgetGuard blocks any overrun
        guard = QPUBudgetGuard(budget_seconds=50.0)
        guard.check_can_submit(20.0)
        guard.record_submission(20.0)
        guard.check_can_submit(20.0)
        guard.record_submission(20.0)
        with pytest.raises(RuntimeError, match="blocked"):
            guard.check_can_submit(18.5)

        summary_path = Path(__file__).resolve().parents[1] / "artifacts_20261127" / "piastq_benchmark_summary.json"
        if summary_path.exists():
            bench = json.loads(summary_path.read_text(encoding="utf-8"))
            map_2b = {(c["n"], c["density"], c["depth_p"]): c for c in bench["cells_2body"]}
            map_3b = {
                (c["n"], c["density_2body"], c["num_triples_3body"], c["depth_p"]): c
                for c in bench["cells_3body"]
            }
            for blk in rebudget["blocks"]:
                assert blk["max_rxx"] <= PIASTQ_MAX_RXX
                if blk["tier"] == "calibration_and_drift":
                    continue
                if blk["tier"] == "heldout_3body_extension":
                    for n in blk["n_values"]:
                        for d in blk["densities"]:
                            for p in blk["p_values"]:
                                cell = map_3b[(n, d, blk["num_triples"], p)]
                                assert not cell["pruned_by_gate_ceiling"]
                                assert cell["advances_to_qpu"] is True
                                assert cell["rxx_max"] <= PIASTQ_MAX_RXX
                                assert cell["stopping_test_mean_energy"]["ci95_high"] < 0.0
                                assert cell["stopping_test_selected_gap"]["ci95_high"] < 0.0
                else:
                    for n in blk["n_values"]:
                        for d in blk["densities"]:
                            for p in blk["p_values"]:
                                cell = map_2b[(n, d, p)]
                                assert not cell["pruned_by_gate_ceiling"]
                                assert cell["advances_to_qpu"] is True
                                assert cell["rxx_max"] <= PIASTQ_MAX_RXX
                                assert cell["stopping_test_mean_energy"]["ci95_high"] < 0.0
                                assert cell["stopping_test_selected_gap"]["ci95_high"] < 0.0

    def test_dev_and_heldout_seed_namespaces_are_disjoint(self):
        import hashlib, json

        dev_hashes = set()
        heldout_hashes = set()
        for n in (8, 12, 16, 20):
            for d in (0.30, 0.50, 0.65, 0.70, 0.85, 1.00):
                for inst in range(16):
                    w_d, e_d = make_density_instance(n, d, 20261100 + n * 1000 + int(round(d * 100)) * 10 + inst)
                    w_h, e_h = make_density_instance(n, d, 20270000 + n * 1000 + int(round(d * 100)) * 10 + inst)
                    dev_hashes.add(hashlib.sha256(json.dumps([w_d.tolist(), e_d]).encode()).hexdigest())
                    heldout_hashes.add(hashlib.sha256(json.dumps([w_h.tolist(), e_h]).encode()).hexdigest())
        assert dev_hashes.isdisjoint(heldout_hashes)

        # Verify 3-body D1 (N=20, d2=0.30, num_triples=12): archived 8 dev seeds vs locked 6 held-out seeds
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        seeds_cfg = json.loads((root / "preregistration_seeds.json").read_text(encoding="utf-8"))["seeds"]
        locked_heldout_seeds = [int(s) for s in seeds_cfg["d1_heldout_3body_seeds"]]
        assert locked_heldout_seeds == [20270000 + 500000 + 20 * 1000 + 12 * 10 + i for i in range(6)]
        assert len(locked_heldout_seeds) == 6

        raw_3b_path = root / "artifacts_20261127" / "point4_dev_3body.json"
        if raw_3b_path.exists():
            raw_3b = json.loads(raw_3b_path.read_text(encoding="utf-8"))["instances"]
            d1_dev_rows = sorted(
                [
                    r
                    for r in raw_3b
                    if int(r["n"]) == 20
                    and math.isclose(float(r["density_2body"]), 0.30)
                    and int(r["num_triples_3body"]) == 12
                ],
                key=lambda r: int(r["instance"]),
            )
            assert len(d1_dev_rows) == 8
            archived_dev_seeds = [int(r["seed"]) for r in d1_dev_rows]
            assert archived_dev_seeds == list(range(20781220, 20781228))
        else:
            archived_dev_seeds = [int(s) for s in seeds_cfg["d1_dev_3body_seeds"]]
            assert len(archived_dev_seeds) == 8

        dev_3b = {
            hashlib.sha256(json.dumps([w.tolist(), e2, e3]).encode()).hexdigest()
            for w, e2, e3 in (make_3body_instance(20, 0.30, 12, s) for s in archived_dev_seeds)
        }
        heldout_3b = {
            hashlib.sha256(json.dumps([w.tolist(), e2, e3]).encode()).hexdigest()
            for w, e2, e3 in (make_3body_instance(20, 0.30, 12, s) for s in locked_heldout_seeds)
        }
        assert len(dev_3b) == 8
        assert len(heldout_3b) == 6
        assert dev_3b.isdisjoint(heldout_3b)

    def test_exhausted_budget_makes_zero_backend_calls(self):
        _, b_noisy = get_piastq_backends()
        w, edges = make_density_instance(4, 0.5, seed=909)
        qc = build_qaoa_circuit(w, edges, [0.3], [0.2])
        tqc = transpile_for_piastq(qc, b_noisy, seed_transpiler=42)

        class _RecordingHardwareBackend:
            def __init__(self) -> None:
                self.calls = 0

            def run(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("backend.run should never have been reached")

        rec = _RecordingHardwareBackend()
        # 1. Non-emulator backend without budget_guard is rejected before backend.run
        with pytest.raises(ValueError, match="budget_guard is required"):
            execute_single_submission(rec, tqc, shots=200)  # type: ignore[arg-type]
        assert rec.calls == 0

        # 2. Exhausted QPUBudgetGuard blocks both single and accumulated submissions with 0 backend calls
        exhausted = QPUBudgetGuard(budget_seconds=90000.0, max_submission_seconds=120.0)
        exhausted.record_submission(89950.0)
        with pytest.raises(RuntimeError, match="blocked"):
            execute_single_submission(rec, tqc, shots=200, budget_guard=exhausted)  # type: ignore[arg-type]
        assert rec.calls == 0

        with pytest.raises(RuntimeError, match="blocked"):
            execute_accumulated_submissions(rec, tqc, total_shots=1000, budget_guard=exhausted)  # type: ignore[arg-type]
        assert rec.calls == 0

    def test_report_table3_matches_summary_json(self):
        from pathlib import Path
        import json, re

        root = Path(__file__).resolve().parents[1]
        summary_path = root / "artifacts_20261127" / "piastq_benchmark_summary.json"
        report_path = root / "PIASTQ_EMULATOR_BENCHMARK_REPORT_20261127.md"
        if not (summary_path.exists() and report_path.exists()):
            pytest.skip("Artifacts or report not present")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {(int(r["n"]), round(float(r["density"]), 2)): r for r in summary["shot_scaling"]}
        text = report_path.read_text(encoding="utf-8")

        row_pat = re.compile(
            r"^\|\s*\$N=(\d+),\s*\\rho=([0-9.]+)\$\s*\|\s*`(\d+)`\s*\|"
            r"\s*(?:\*\*)?`([0-9.]+)`(?:\*\*)?\s*\(`([0-9.]+)`\)\s*\|"
            r"\s*(?:\*\*)?`([0-9.]+)`(?:\*\*)?\s*\(`([0-9.]+)`\)\s*\|"
            r"\s*(?:\*\*)?`([0-9.]+)`(?:\*\*)?\s*\(`([0-9.]+)`\)\s*\|"
            r"\s*(?:\*\*)?`([0-9.]+)`(?:\*\*)?\s*\(`([0-9.]+)`\)\s*\|"
            r"\s*(?:\*\*)?`([0-9.]+)`(?:\*\*)?\s*\(`([0-9.]+)`\)\s*\|"
            r"\s*(?:\*\*)?`([0-9.]+)`(?:\*\*)?\s*\(`([0-9.]+)`\)\s*\|"
            r"\s*(?:\*\*)?`\[([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\]`(?:\*\*)?",
            re.MULTILINE,
        )
        matches = row_pat.findall(text)
        assert len(matches) == 8
        for m in matches:
            n = int(m[0])
            d = round(float(m[1]), 2)
            rxx = int(m[2])
            row = expected[(n, d)]
            assert rxx == int(row["rxx_count"])
            by_s = {int(c["shots"]): c for c in row["curve"]}
            assert f"{by_s[200]['noisy_solved_fraction']:.3f}" == m[3]
            assert f"{by_s[200]['noisy_mean_gap']:.2f}" == m[4]
            assert f"{by_s[1000]['noisy_solved_fraction']:.3f}" == m[5]
            assert f"{by_s[1000]['noisy_mean_gap']:.2f}" == m[6]
            assert f"{by_s[2200]['noisy_solved_fraction']:.3f}" == m[7]
            assert f"{by_s[2200]['noisy_mean_gap']:.2f}" == m[8]
            assert f"{by_s[2200]['random_solved_fraction']:.3f}" == m[9]
            assert f"{by_s[2200]['random_mean_gap']:.2f}" == m[10]
            assert f"{by_s[4000]['noisy_solved_fraction']:.3f}" == m[11]
            assert f"{by_s[4000]['noisy_mean_gap']:.2f}" == m[12]
            assert f"{by_s[4000]['random_solved_fraction']:.3f}" == m[13]
            assert f"{by_s[4000]['random_mean_gap']:.2f}" == m[14]
            lo, hi = by_s[2200]["gap_diff_vs_random_ci95"]
            assert math.isclose(float(m[15]), round(lo, 2), abs_tol=1e-6)
            assert math.isclose(float(m[16]), round(hi, 2), abs_tol=1e-6)
