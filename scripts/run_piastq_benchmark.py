"""End-to-end PIAST-Q emulator benchmark runner (Points 1-5) on CPU Slurm nodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from piastq_bench.core import (
    anneal,
    anneal_3body,
    energy_table,
    energy_table_3body,
    make_3body_instance,
    make_density_instance,
    sample_metrics,
    select_qaoa_angles,
    valid_mask,
    valid_mask_3body,
)
from piastq_bench.gate_mapping import sweep_gate_mapping_matrix
from piastq_bench.piastq_adapter import (
    PIASTQ_MAX_OPS,
    PIASTQ_MAX_RXX,
    PIASTQ_MAX_SHOTS,
    build_qaoa_circuit,
    execute_accumulated_submissions,
    get_piastq_backends,
    ingest_external_circuit,
    transpile_for_piastq,
    validate_piastq_submission,
)
from piastq_bench.solvers import (
    solve_cpsat_natural,
    solve_cpsat_qubo,
    solve_neal_qubo,
    solve_z3_natural,
    solve_z3_qubo,
    uniform_random_samples,
)
from piastq_bench.statistics import (
    paired_bootstrap_ci,
    paired_bootstrap_stopping_rule,
)

BASE_SEED_DEV = 20261100
BASE_SEED_HELDOUT = 20270000
DEV_SHOTS = 1000  # 5 x 200-shot single-circuit submissions (matches Euro-Q-Exa 1,024 dev)
SCALING_SHOT_GRID = [200, 400, 600, 1000, 1200, 2200, 4000]


def _env_metadata() -> dict[str, Any]:
    pkgs = [
        "qiskit",
        "qiskit-aqt-provider",
        "qiskit-aer",
        "aqt-connector",
        "ortools",
        "z3-solver",
        "dwave-neal",
        "numpy",
        "scipy",
    ]
    versions = {p: md.version(p) for p in pkgs}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": "redacted",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": "redacted",
        "packages": versions,
    }


def _eval_2body_dev_instance(task: dict[str, Any]) -> dict[str, Any]:
    n = int(task["n"])
    density = float(task["density"])
    inst_idx = int(task["instance"])
    depths: list[int] = list(task["depths"])
    shots = int(task["shots"])
    aer_threads = int(task["aer_threads"])

    b_ideal, b_noisy = get_piastq_backends(max_parallel_threads=aer_threads)  # type: ignore[misc]
    seed = BASE_SEED_DEV + n * 1000 + int(round(density * 100)) * 10 + inst_idx
    weights, edges = make_density_instance(n, density, seed=seed)

    t0 = time.perf_counter()
    energy = energy_table(weights, edges)
    exact_sec = time.perf_counter() - t0
    valid = valid_mask(n, edges)
    opt_idx = int(np.argmin(energy))
    opt_energy = int(energy[opt_idx])
    if not valid[opt_idx]:
        raise AssertionError("Invalid optimum in 2-body energy_table")

    payload_bytes = json.dumps(
        {"weights": weights.tolist(), "edges": edges}, sort_keys=True, separators=(",", ":")
    ).encode()
    inst_sha = hashlib.sha256(payload_bytes).hexdigest()

    # Classical references on identical objective + natural constraints
    cpsat_nat = solve_cpsat_natural(weights, edges)
    cpsat_qubo = solve_cpsat_qubo(weights, edges)
    z3_nat = solve_z3_natural(weights, edges) if n <= 16 else None
    z3_qubo = solve_z3_qubo(weights, edges) if n <= 16 else None
    assert cpsat_nat["energy"] == opt_energy
    assert cpsat_qubo["energy"] == opt_energy
    if z3_nat is not None and z3_qubo is not None:
        assert z3_nat["energy"] == opt_energy
        assert z3_qubo["energy"] == opt_energy

    t0 = time.perf_counter()
    sa_samples, _ = anneal(weights, edges, seed=seed + 10000, reads=shots, sweeps=50)
    sa_sec = time.perf_counter() - t0
    sa_metrics = {**sample_metrics(sa_samples, energy, valid), "seconds": sa_sec}

    neal_res = solve_neal_qubo(weights, edges, num_reads=shots, num_sweeps=50, seed=seed + 12000)
    neal_metrics = {
        **sample_metrics(neal_res["samples"], energy, valid),
        "seconds": neal_res["seconds"],
    }

    rand_samples = uniform_random_samples(n, shots, seed=seed + 15000)
    rand_metrics = sample_metrics(rand_samples, energy, valid)

    depth_results: dict[str, Any] = {}
    for p in depths:
        angle_seed = seed + 20000 + p * 100
        angles = select_qaoa_angles(energy, depth=p, seed=angle_seed, candidates=24)
        qc = build_qaoa_circuit(
            weights,
            edges,
            angles["gammas"],
            angles["betas"],
            phase_scale=angles["phase_energy_scale"],
        )
        tqc = transpile_for_piastq(qc, b_noisy, optimization_level=3, seed_transpiler=42)
        ops = dict(tqc.count_ops())
        rxx_count = int(ops.get("rxx", 0))
        if rxx_count > PIASTQ_MAX_RXX:
            depth_results[f"p{p}"] = {
                "pruned_by_gate_ceiling": True,
                "rxx_count": rxx_count,
            }
            continue

        ideal_exec = execute_accumulated_submissions(
            b_ideal,
            tqc,
            total_shots=shots,
            max_shots_per_submission=PIASTQ_MAX_SHOTS,
            base_seed=seed + 30000 + p * 1000,
            max_rxx=PIASTQ_MAX_RXX,
        )
        noisy_exec = execute_accumulated_submissions(
            b_noisy,
            tqc,
            total_shots=shots,
            max_shots_per_submission=PIASTQ_MAX_SHOTS,
            base_seed=seed + 40000 + p * 1000,
            max_rxx=PIASTQ_MAX_RXX,
        )
        depth_results[f"p{p}"] = {
            "pruned_by_gate_ceiling": False,
            "angles": angles,
            "circuit_stats": noisy_exec["circuit_stats"],
            "submissions_count": noisy_exec["submissions_count"],
            "ideal_qaoa": {
                **sample_metrics(ideal_exec["samples"], energy, valid),
                "seconds": ideal_exec["total_seconds"],
            },
            "noisy_qaoa": {
                **sample_metrics(noisy_exec["samples"], energy, valid),
                "seconds": noisy_exec["total_seconds"],
            },
        }

    return {
        "n": n,
        "density": density,
        "num_edges": len(edges),
        "instance": inst_idx,
        "seed": seed,
        "input_sha256": inst_sha,
        "exact_energy": opt_energy,
        "exact_index": opt_idx,
        "exact_seconds": exact_sec,
        "cpsat_natural_seconds": cpsat_nat["seconds"],
        "cpsat_qubo_seconds": cpsat_qubo["seconds"],
        "z3_natural_seconds": z3_nat["seconds"] if z3_nat else None,
        "z3_qubo_seconds": z3_qubo["seconds"] if z3_qubo else None,
        "sa": sa_metrics,
        "dwave_neal": neal_metrics,
        "uniform_random": rand_metrics,
        "qaoa": depth_results,
    }


def _eval_3body_dev_instance(task: dict[str, Any]) -> dict[str, Any]:
    n = int(task["n"])
    d2 = float(task["density_2body"])
    n3 = int(task["num_triples"])
    inst_idx = int(task["instance"])
    depths: list[int] = list(task["depths"])
    shots = int(task["shots"])
    aer_threads = int(task["aer_threads"])

    b_ideal, b_noisy = get_piastq_backends(max_parallel_threads=aer_threads)  # type: ignore[misc]
    seed = BASE_SEED_DEV + 500000 + n * 1000 + n3 * 10 + inst_idx
    weights, edges2, triples3 = make_3body_instance(n, d2, n3, seed=seed)

    t0 = time.perf_counter()
    energy = energy_table_3body(weights, edges2, triples3)
    exact_sec = time.perf_counter() - t0
    valid = valid_mask_3body(n, edges2, triples3)
    opt_idx = int(np.argmin(energy))
    opt_energy = int(energy[opt_idx])
    if not valid[opt_idx]:
        raise AssertionError("Invalid optimum in 3-body energy_table_3body")

    cpsat_nat = solve_cpsat_natural(weights, edges2, triples=triples3)
    cpsat_hubo = solve_cpsat_qubo(weights, edges2, triples=triples3)
    assert cpsat_nat["energy"] == opt_energy
    assert cpsat_hubo["energy"] == opt_energy

    t0 = time.perf_counter()
    sa_samples, _ = anneal_3body(weights, edges2, triples3, seed=seed + 10000, reads=shots, sweeps=50)
    sa_sec = time.perf_counter() - t0
    sa_metrics = {**sample_metrics(sa_samples, energy, valid), "seconds": sa_sec}

    rand_samples = uniform_random_samples(n, shots, seed=seed + 15000)
    rand_metrics = sample_metrics(rand_samples, energy, valid)

    depth_results: dict[str, Any] = {}
    for p in depths:
        angles = select_qaoa_angles(energy, depth=p, seed=seed + 20000 + p * 100, candidates=24)
        qc = build_qaoa_circuit(
            weights,
            edges2,
            angles["gammas"],
            angles["betas"],
            triples=triples3,
            phase_scale=angles["phase_energy_scale"],
        )
        tqc = transpile_for_piastq(qc, b_noisy, optimization_level=3, seed_transpiler=42)
        ops = dict(tqc.count_ops())
        rxx_count = int(ops.get("rxx", 0))
        q_ops = sum(int(v) for v in ops.values())
        wire_ops = rxx_count + int(ops.get("r", 0)) + int(ops.get("rz", 0)) + 1
        if rxx_count > PIASTQ_MAX_RXX or q_ops > PIASTQ_MAX_OPS or wire_ops > PIASTQ_MAX_OPS:
            depth_results[f"p{p}"] = {
                "pruned_by_gate_ceiling": True,
                "rxx_count": rxx_count,
                "wire_ops": wire_ops,
            }
            continue

        ideal_exec = execute_accumulated_submissions(
            b_ideal,
            tqc,
            total_shots=shots,
            max_shots_per_submission=PIASTQ_MAX_SHOTS,
            base_seed=seed + 30000 + p * 1000,
            max_rxx=PIASTQ_MAX_RXX,
        )
        noisy_exec = execute_accumulated_submissions(
            b_noisy,
            tqc,
            total_shots=shots,
            max_shots_per_submission=PIASTQ_MAX_SHOTS,
            base_seed=seed + 40000 + p * 1000,
            max_rxx=PIASTQ_MAX_RXX,
        )
        depth_results[f"p{p}"] = {
            "pruned_by_gate_ceiling": False,
            "circuit_stats": noisy_exec["circuit_stats"],
            "submissions_count": noisy_exec["submissions_count"],
            "ideal_qaoa": {
                **sample_metrics(ideal_exec["samples"], energy, valid),
                "seconds": ideal_exec["total_seconds"],
            },
            "noisy_qaoa": {
                **sample_metrics(noisy_exec["samples"], energy, valid),
                "seconds": noisy_exec["total_seconds"],
            },
        }

    return {
        "n": n,
        "density_2body": d2,
        "num_edges_2body": len(edges2),
        "num_triples_3body": len(triples3),
        "instance": inst_idx,
        "seed": seed,
        "exact_energy": opt_energy,
        "exact_seconds": exact_sec,
        "cpsat_natural_seconds": cpsat_nat["seconds"],
        "cpsat_hubo_seconds": cpsat_hubo["seconds"],
        "sa_3body": sa_metrics,
        "uniform_random": rand_metrics,
        "qaoa": depth_results,
    }


def _eval_shot_scaling_instance(task: dict[str, Any]) -> dict[str, Any]:
    n = int(task["n"])
    density = float(task["density"])
    depth = int(task["depth"])
    inst_idx = int(task["instance"])
    aer_threads = int(task["aer_threads"])
    max_shots = max(SCALING_SHOT_GRID)  # 4000 = 20 x 200 submissions

    b_ideal, b_noisy = get_piastq_backends(max_parallel_threads=aer_threads)  # type: ignore[misc]
    seed = BASE_SEED_DEV + n * 1000 + int(round(density * 100)) * 10 + inst_idx
    weights, edges = make_density_instance(n, density, seed=seed)
    energy = energy_table(weights, edges)
    valid = valid_mask(n, edges)

    angles = select_qaoa_angles(energy, depth=depth, seed=seed + 20000 + depth * 100, candidates=24)
    qc = build_qaoa_circuit(
        weights,
        edges,
        angles["gammas"],
        angles["betas"],
        phase_scale=angles["phase_energy_scale"],
    )
    tqc = transpile_for_piastq(qc, b_noisy, optimization_level=3, seed_transpiler=42)

    ideal_exec = execute_accumulated_submissions(
        b_ideal,
        tqc,
        total_shots=max_shots,
        max_shots_per_submission=PIASTQ_MAX_SHOTS,
        base_seed=seed + 60000 + depth * 1000,
        max_rxx=PIASTQ_MAX_RXX,
    )
    noisy_exec = execute_accumulated_submissions(
        b_noisy,
        tqc,
        total_shots=max_shots,
        max_shots_per_submission=PIASTQ_MAX_SHOTS,
        base_seed=seed + 70000 + depth * 1000,
        max_rxx=PIASTQ_MAX_RXX,
    )
    rand_all = uniform_random_samples(n, max_shots, seed=seed + 80000)

    by_shots: dict[str, Any] = {}
    for s_budget in SCALING_SHOT_GRID:
        by_shots[str(s_budget)] = {
            "shots": s_budget,
            "submissions": s_budget // PIASTQ_MAX_SHOTS,
            "ideal_qaoa": sample_metrics(ideal_exec["samples"][:s_budget], energy, valid),
            "noisy_qaoa": sample_metrics(noisy_exec["samples"][:s_budget], energy, valid),
            "uniform_random": sample_metrics(rand_all[:s_budget], energy, valid),
        }

    return {
        "n": n,
        "density": density,
        "depth_p": depth,
        "instance": inst_idx,
        "seed": seed,
        "rxx_count": noisy_exec["circuit_stats"]["rxx_count"],
        "wire_ops": noisy_exec["circuit_stats"]["wire_ops"],
        "total_noisy_4000_seconds": noisy_exec["total_seconds"],
        "by_shots": by_shots,
    }


def _run_pool(fn: Any, tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for idx, t in enumerate(tasks):
            r = fn(t)
            results.append(r)
            print(f"[{idx+1}/{len(tasks)}] done n={t.get('n')}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, t): i for i, t in enumerate(tasks)}
        done_cnt = 0
        ordered: list[tuple[int, dict[str, Any]]] = []
        for fut in as_completed(futs):
            i = futs[fut]
            res = fut.result()
            ordered.append((i, res))
            done_cnt += 1
            print(f"[{done_cnt}/{len(tasks)}] task #{i} done", flush=True)
        ordered.sort(key=lambda x: x[0])
        return [x[1] for x in ordered]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"wrote": str(path), "sha256": digest}), flush=True)


def run_mode(mode: str, out_dir: Path, workers: int, aer_threads: int) -> None:
    t_start = time.perf_counter()
    if mode == "gate_map":
        matrix = sweep_gate_mapping_matrix()
        _write_json(
            out_dir / "point2_gate_mapping.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, **matrix},
        )
        return

    if mode == "workflow_demo":
        # Point 1 end-to-end batched hybrid workflow + CUDA-Q/PennyLane OpenQASM ingestion proof
        b_ideal, b_noisy = get_piastq_backends(max_parallel_threads=aer_threads)  # type: ignore[misc]
        batch_records = []
        for n, d, p in [(8, 0.50, 2), (12, 0.65, 2), (16, 0.65, 3), (20, 0.65, 3), (20, 1.00, 2)]:
            w, edges = make_density_instance(n, d, seed=BASE_SEED_DEV + n * 77)
            en = energy_table(w, edges)
            val = valid_mask(n, edges)
            ang = select_qaoa_angles(en, depth=p, seed=BASE_SEED_DEV + n * 99, candidates=24)
            qc = build_qaoa_circuit(w, edges, ang["gammas"], ang["betas"], phase_scale=ang["phase_energy_scale"])
            tqc = transpile_for_piastq(qc, b_noisy, optimization_level=3, seed_transpiler=42)
            # Single 200-shot submission on ideal & noisy + 5x200=1000-shot accumulated batch
            ideal_run = execute_accumulated_submissions(b_ideal, tqc, total_shots=1000, base_seed=1100 + n)
            noisy_run = execute_accumulated_submissions(b_noisy, tqc, total_shots=1000, base_seed=2200 + n)
            batch_records.append(
                {
                    "n": n,
                    "density": d,
                    "depth_p": p,
                    "circuit_stats": noisy_run["circuit_stats"],
                    "submissions_per_backend": noisy_run["submissions_count"],
                    "shot_chunks": noisy_run["shot_chunks"],
                    "ideal_metrics": sample_metrics(ideal_run["samples"], en, val),
                    "noisy_metrics": sample_metrics(noisy_run["samples"], en, val),
                    "noisy_seconds": noisy_run["total_seconds"],
                }
            )
        # External OpenQASM bridge check (CUDA-Q / PennyLane compatible QASM 2.0)
        qasm_demo = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
h q[0]; h q[1]; h q[2]; h q[3]; h q[4]; h q[5];
cx q[0],q[1]; rz(0.35) q[1]; cx q[0],q[1];
cx q[2],q[3]; rz(0.35) q[3]; cx q[2],q[3];
cx q[4],q[5]; rz(0.35) q[5]; cx q[4],q[5];
rx(0.45) q[0]; rx(0.45) q[1]; rx(0.45) q[2]; rx(0.45) q[3]; rx(0.45) q[4]; rx(0.45) q[5];
measure q[0] -> c[0]; measure q[1] -> c[1]; measure q[2] -> c[2];
measure q[3] -> c[3]; measure q[4] -> c[4]; measure q[5] -> c[5];
"""
        ext_qc = ingest_external_circuit(qasm_demo, fmt="qasm2")
        ext_tqc = transpile_for_piastq(ext_qc, b_noisy, seed_transpiler=42)
        ext_val = validate_piastq_submission(ext_tqc, shots=200)
        ext_run = execute_accumulated_submissions(b_noisy, ext_tqc, total_shots=200, base_seed=3300)

        _write_json(
            out_dir / "point1_hardware_path_demo.json",
            {
                "meta": _env_metadata(),
                "elapsed_seconds": time.perf_counter() - t_start,
                "constraints_verified": {
                    "native_gates_only": ["rz", "r", "rxx"],
                    "single_terminal_full_register_measure": True,
                    "max_shots_per_submission": PIASTQ_MAX_SHOTS,
                    "max_ops_per_circuit": PIASTQ_MAX_OPS,
                    "max_rxx_ceiling": PIASTQ_MAX_RXX,
                    "single_circuit_per_submission": True,
                    "violations_observed": 0,
                },
                "qasm_bridge_verification": {
                    "validation": ext_val,
                    "returned_shots": int(len(ext_run["samples"])),
                },
                "batched_hybrid_runs": batch_records,
            },
        )
        return

    if mode == "dev_n8_12":
        densities = [0.30, 0.50, 0.65, 0.70, 0.85, 1.00]
        tasks = [
            {"n": n, "density": d, "instance": inst, "depths": [1, 2, 3], "shots": DEV_SHOTS, "aer_threads": aer_threads}
            for n in (8, 12)
            for d in densities
            for inst in range(8)
        ]
        rows = _run_pool(_eval_2body_dev_instance, tasks, workers)
        _write_json(
            out_dir / "point4_dev_2body_n8_12.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, "instances": rows},
        )
        return

    if mode == "dev_n16":
        densities = [0.30, 0.50, 0.65, 0.70, 0.85, 1.00]
        tasks = [
            {"n": 16, "density": d, "instance": inst, "depths": [1, 2, 3], "shots": DEV_SHOTS, "aer_threads": aer_threads}
            for d in densities
            for inst in range(8)
        ]
        rows = _run_pool(_eval_2body_dev_instance, tasks, workers)
        _write_json(
            out_dir / "point4_dev_2body_n16.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, "instances": rows},
        )
        return

    if mode == "dev_n20_part1":
        # Densities 0.30, 0.50, 0.65 at N=20 (all p=1,2,3 <= 372 RXX)
        tasks = [
            {"n": 20, "density": d, "instance": inst, "depths": [1, 2, 3], "shots": DEV_SHOTS, "aer_threads": aer_threads}
            for d in (0.30, 0.50, 0.65)
            for inst in range(8)
        ]
        rows = _run_pool(_eval_2body_dev_instance, tasks, workers)
        _write_json(
            out_dir / "point4_dev_2body_n20_part1.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, "instances": rows},
        )
        return

    if mode == "dev_n20_part2":
        # Densities 0.70 (p=1,2,3 up to 399 RXX), 0.85 (p=1,2; p=3 pruned), 1.00 (p=1,2; p=3 pruned)
        tasks = [
            {"n": 20, "density": d, "instance": inst, "depths": [1, 2, 3], "shots": DEV_SHOTS, "aer_threads": aer_threads}
            for d in (0.70, 0.85, 1.00)
            for inst in range(8)
        ]
        rows = _run_pool(_eval_2body_dev_instance, tasks, workers)
        _write_json(
            out_dir / "point4_dev_2body_n20_part2.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, "instances": rows},
        )
        return

    if mode == "dev_3body":
        # Surviving 3-body HUBO cells across N=8, 12, 16, 20 under <= 400 RXX
        cfgs = [
            (8, 0.30, 6, [1, 2, 3]),
            (12, 0.30, 10, [1, 2, 3]),
            (16, 0.30, 12, [1, 2]),
            (20, 0.30, 12, [1, 2]),
        ]
        tasks = [
            {
                "n": n,
                "density_2body": d2,
                "num_triples": n3,
                "instance": inst,
                "depths": depths,
                "shots": DEV_SHOTS,
                "aer_threads": aer_threads,
            }
            for (n, d2, n3, depths) in cfgs
            for inst in range(8)
        ]
        rows = _run_pool(_eval_3body_dev_instance, tasks, workers)
        _write_json(
            out_dir / "point4_dev_3body.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, "instances": rows},
        )
        return

    if mode == "shot_scaling_n8_12_16":
        # 20 x 200 = 4,000 shots per instance at representative dense conflict graphs (d=0.65, p=2 and d=1.00, p=2)
        tasks = [
            {"n": n, "density": d, "depth": 2, "instance": inst, "aer_threads": aer_threads}
            for n in (8, 12, 16)
            for d in (0.65, 1.00)
            for inst in range(8)
        ]
        rows = _run_pool(_eval_shot_scaling_instance, tasks, workers)
        _write_json(
            out_dir / "point3_shot_scaling_n8_12_16.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, "instances": rows},
        )
        return

    if mode == "shot_scaling_n20":
        tasks = [
            {"n": 20, "density": d, "depth": 2, "instance": inst, "aer_threads": aer_threads}
            for d in (0.65, 1.00)
            for inst in range(8)
        ]
        rows = _run_pool(_eval_shot_scaling_instance, tasks, workers)
        _write_json(
            out_dir / "point3_shot_scaling_n20.json",
            {"meta": _env_metadata(), "elapsed_seconds": time.perf_counter() - t_start, "instances": rows},
        )
        return

    if mode == "assemble":
        assemble_summary(out_dir)
        return

    raise ValueError(f"Unknown mode: {mode}")


def assemble_summary(out_dir: Path) -> None:
    """Aggregate all raw shard JSONs, compute 95% bootstrap CIs and stopping-rule decisions."""
    dev_files = [
        "point4_dev_2body_n8_12.json",
        "point4_dev_2body_n16.json",
        "point4_dev_2body_n20_part1.json",
        "point4_dev_2body_n20_part2.json",
    ]
    all_2body: list[dict[str, Any]] = []
    for fname in dev_files:
        data = json.loads((out_dir / fname).read_text(encoding="utf-8"))
        all_2body.extend(data["instances"])

    # Group 2-body cells by (n, density, p)
    cells_2body: list[dict[str, Any]] = []
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in all_2body:
        grouped.setdefault((int(row["n"]), float(row["density"])), []).append(row)

    for (n, d), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: int(r["instance"]))
        sa_opt = float(np.mean([1.0 if r["sa"]["selected_optimal"] else 0.0 for r in rows]))
        sa_gap = float(np.mean([r["sa"]["objective_gap"] for r in rows]))
        rand_opt = float(np.mean([1.0 if r["uniform_random"]["selected_optimal"] else 0.0 for r in rows]))
        rand_gap = [float(r["uniform_random"]["objective_gap"]) for r in rows]
        rand_mean_e = [float(r["uniform_random"]["mean_sample_energy"]) for r in rows]
        rand_vf = [float(r["uniform_random"]["valid_fraction"]) for r in rows]

        for p in (1, 2, 3):
            pkey = f"p{p}"
            if rows[0]["qaoa"][pkey].get("pruned_by_gate_ceiling"):
                cells_2body.append(
                    {
                        "n": n,
                        "density": d,
                        "depth_p": p,
                        "num_instances": len(rows),
                        "rxx_count": rows[0]["qaoa"][pkey]["rxx_count"],
                        "pruned_by_gate_ceiling": True,
                        "advances_to_qpu": False,
                        "reason": "Exceeds <= 400 RXX native two-qubit gate ceiling",
                    }
                )
                continue

            ideal_opt = float(
                np.mean([1.0 if r["qaoa"][pkey]["ideal_qaoa"]["selected_optimal"] else 0.0 for r in rows])
            )
            ideal_gap = float(
                np.mean([r["qaoa"][pkey]["ideal_qaoa"]["objective_gap"] for r in rows])
            )
            noisy_opt = float(
                np.mean([1.0 if r["qaoa"][pkey]["noisy_qaoa"]["selected_optimal"] else 0.0 for r in rows])
            )
            noisy_gap = [float(r["qaoa"][pkey]["noisy_qaoa"]["objective_gap"]) for r in rows]
            noisy_mean_e = [float(r["qaoa"][pkey]["noisy_qaoa"]["mean_sample_energy"]) for r in rows]
            noisy_vf = [float(r["qaoa"][pkey]["noisy_qaoa"]["valid_fraction"]) for r in rows]

            stop_mean_e = paired_bootstrap_stopping_rule(
                noisy_mean_e, rand_mean_e, lower_is_better=True, seed=20261127 + n * 100 + p
            )
            stop_sel_gap = paired_bootstrap_stopping_rule(
                noisy_gap, rand_gap, lower_is_better=True, seed=20261500 + n * 100 + p
            )
            stop_vf = paired_bootstrap_stopping_rule(
                noisy_vf, rand_vf, lower_is_better=False, seed=20261900 + n * 100 + p
            )
            # Dual stopping rule: two-sided 95% CI < 0 on mean sample energy and selected objective gap
            advances = bool(stop_mean_e["advances"] and stop_sel_gap["advances"])
            rxx_max = max(int(r["qaoa"][pkey]["circuit_stats"]["rxx_count"]) for r in rows)
            wire_max = max(int(r["qaoa"][pkey]["circuit_stats"]["wire_ops"]) for r in rows)

            cells_2body.append(
                {
                    "n": n,
                    "density": d,
                    "depth_p": p,
                    "num_instances": len(rows),
                    "rxx_max": rxx_max,
                    "wire_ops_max": wire_max,
                    "pruned_by_gate_ceiling": False,
                    "sa_optimal_fraction": round(sa_opt, 4),
                    "sa_mean_gap": round(sa_gap, 4),
                    "ideal_qaoa_optimal_fraction": round(ideal_opt, 4),
                    "ideal_qaoa_mean_gap": round(ideal_gap, 4),
                    "noisy_qaoa_optimal_fraction": round(noisy_opt, 4),
                    "noisy_qaoa_mean_gap": round(float(np.mean(noisy_gap)), 4),
                    "random_optimal_fraction": round(rand_opt, 4),
                    "random_mean_gap": round(float(np.mean(rand_gap)), 4),
                    "noisy_valid_fraction": round(float(np.mean(noisy_vf)), 4),
                    "random_valid_fraction": round(float(np.mean(rand_vf)), 4),
                    "stopping_test_mean_energy": stop_mean_e,
                    "stopping_test_selected_gap": stop_sel_gap,
                    "stopping_test_valid_fraction": stop_vf,
                    "advances_to_qpu": advances,
                }
            )

    # 3-body summary
    data_3b = json.loads((out_dir / "point4_dev_3body.json").read_text(encoding="utf-8"))["instances"]
    grouped_3b: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for row in data_3b:
        grouped_3b.setdefault(
            (int(row["n"]), float(row["density_2body"]), int(row["num_triples_3body"])), []
        ).append(row)

    cells_3body: list[dict[str, Any]] = []
    for (n, d2, n3), rows in sorted(grouped_3b.items()):
        rows.sort(key=lambda r: int(r["instance"]))
        rand_mean_e = [float(r["uniform_random"]["mean_sample_energy"]) for r in rows]
        rand_gap = [float(r["uniform_random"]["objective_gap"]) for r in rows]
        rand_vf = [float(r["uniform_random"]["valid_fraction"]) for r in rows]
        sa_opt = float(np.mean([1.0 if r["sa_3body"]["selected_optimal"] else 0.0 for r in rows]))
        for pkey in sorted(rows[0]["qaoa"].keys()):
            p = int(pkey[1:])
            if any(r["qaoa"][pkey].get("pruned_by_gate_ceiling") for r in rows):
                rxx_obs = max(int(r["qaoa"][pkey].get("rxx_count", 0)) for r in rows)
                cells_3body.append(
                    {
                        "n": n,
                        "density_2body": d2,
                        "num_triples_3body": n3,
                        "depth_p": p,
                        "rxx_max": rxx_obs,
                        "pruned_by_gate_ceiling": True,
                        "advances_to_qpu": False,
                    }
                )
                continue
            noisy_mean_e = [float(r["qaoa"][pkey]["noisy_qaoa"]["mean_sample_energy"]) for r in rows]
            noisy_gap = [float(r["qaoa"][pkey]["noisy_qaoa"]["objective_gap"]) for r in rows]
            noisy_vf = [float(r["qaoa"][pkey]["noisy_qaoa"]["valid_fraction"]) for r in rows]
            noisy_opt = float(
                np.mean([1.0 if r["qaoa"][pkey]["noisy_qaoa"]["selected_optimal"] else 0.0 for r in rows])
            )
            ideal_opt = float(
                np.mean([1.0 if r["qaoa"][pkey]["ideal_qaoa"]["selected_optimal"] else 0.0 for r in rows])
            )
            stop_mean_e = paired_bootstrap_stopping_rule(
                noisy_mean_e, rand_mean_e, lower_is_better=True, seed=20263100 + n * 100 + p
            )
            stop_vf = paired_bootstrap_stopping_rule(
                noisy_vf, rand_vf, lower_is_better=False, seed=20263500 + n * 100 + p
            )
            stop_gap = paired_bootstrap_stopping_rule(
                noisy_gap, rand_gap, lower_is_better=True, seed=20263900 + n * 100 + p
            )
            advances = bool(stop_mean_e["advances"] and stop_gap["advances"])
            rxx_max = max(int(r["qaoa"][pkey]["circuit_stats"]["rxx_count"]) for r in rows)
            wire_max = max(int(r["qaoa"][pkey]["circuit_stats"]["wire_ops"]) for r in rows)
            cells_3body.append(
                {
                    "n": n,
                    "density_2body": d2,
                    "num_triples_3body": n3,
                    "depth_p": p,
                    "num_instances": len(rows),
                    "rxx_max": rxx_max,
                    "wire_ops_max": wire_max,
                    "pruned_by_gate_ceiling": False,
                    "sa_optimal_fraction": round(sa_opt, 4),
                    "ideal_qaoa_optimal_fraction": round(ideal_opt, 4),
                    "noisy_qaoa_optimal_fraction": round(noisy_opt, 4),
                    "noisy_qaoa_mean_gap": round(float(np.mean(noisy_gap)), 4),
                    "random_mean_gap": round(float(np.mean(rand_gap)), 4),
                    "noisy_valid_fraction": round(float(np.mean(noisy_vf)), 4),
                    "random_valid_fraction": round(float(np.mean(rand_vf)), 4),
                    "stopping_test_mean_energy": stop_mean_e,
                    "stopping_test_selected_gap": stop_gap,
                    "stopping_test_valid_fraction": stop_vf,
                    "advances_to_qpu": advances,
                }
            )

    # Shot scaling summary
    sc_rows = (
        json.loads((out_dir / "point3_shot_scaling_n8_12_16.json").read_text(encoding="utf-8"))["instances"]
        + json.loads((out_dir / "point3_shot_scaling_n20.json").read_text(encoding="utf-8"))["instances"]
    )
    sc_grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for r in sc_rows:
        sc_grouped.setdefault((int(r["n"]), float(r["density"])), []).append(r)

    shot_scaling_summary: list[dict[str, Any]] = []
    for (n, d), rows in sorted(sc_grouped.items()):
        curve = []
        for s_budget in SCALING_SHOT_GRID:
            sk = str(s_budget)
            n_opt = [1.0 if r["by_shots"][sk]["noisy_qaoa"]["selected_optimal"] else 0.0 for r in rows]
            i_opt = [1.0 if r["by_shots"][sk]["ideal_qaoa"]["selected_optimal"] else 0.0 for r in rows]
            u_opt = [1.0 if r["by_shots"][sk]["uniform_random"]["selected_optimal"] else 0.0 for r in rows]
            n_gap = [float(r["by_shots"][sk]["noisy_qaoa"]["objective_gap"]) for r in rows]
            u_gap = [float(r["by_shots"][sk]["uniform_random"]["objective_gap"]) for r in rows]
            gap_ci = paired_bootstrap_ci(np.array(n_gap) - np.array(u_gap), seed=20265000 + n * 10 + s_budget)
            curve.append(
                {
                    "shots": s_budget,
                    "submissions_200": s_budget // PIASTQ_MAX_SHOTS,
                    "ideal_solved_fraction": round(float(np.mean(i_opt)), 4),
                    "noisy_solved_fraction": round(float(np.mean(n_opt)), 4),
                    "random_solved_fraction": round(float(np.mean(u_opt)), 4),
                    "noisy_mean_gap": round(float(np.mean(n_gap)), 4),
                    "random_mean_gap": round(float(np.mean(u_gap)), 4),
                    "gap_diff_vs_random_ci95": [round(gap_ci["ci95_low"], 4), round(gap_ci["ci95_high"], 4)],
                }
            )
        shot_scaling_summary.append(
            {
                "n": n,
                "density": d,
                "depth_p": 2,
                "rxx_count": rows[0]["rxx_count"],
                "num_instances": len(rows),
                "curve": curve,
            }
        )

    artifact_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out_dir.glob("point*.json"))
        if p.name != "piastq_benchmark_summary.json"
    }
    _write_json(
        out_dir / "piastq_benchmark_summary.json",
        {
            "meta": _env_metadata(),
            "artifact_sha256": artifact_hashes,
            "cells_2body": cells_2body,
            "cells_3body": cells_3body,
            "shot_scaling": shot_scaling_summary,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "gate_map",
            "workflow_demo",
            "dev_n8_12",
            "dev_n16",
            "dev_n20_part1",
            "dev_n20_part2",
            "dev_3body",
            "shot_scaling_n8_12_16",
            "shot_scaling_n20",
            "assemble",
        ],
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aer-threads", type=int, default=16)
    args = parser.parse_args()
    run_mode(args.mode, args.output_dir, args.workers, args.aer_threads)


if __name__ == "__main__":
    main()
