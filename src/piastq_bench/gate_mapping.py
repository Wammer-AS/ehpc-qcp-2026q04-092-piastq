"""Point 2: Compiled native gate mapping (RXX, R, RZ, wire ops) and ceiling pruning."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from qiskit.providers import BackendV2

from .core import make_3body_instance, make_density_instance
from .piastq_adapter import (
    PIASTQ_MAX_OPS,
    PIASTQ_MAX_RXX,
    PIASTQ_MAX_SHOTS,
    build_qaoa_circuit,
    get_piastq_backends,
    transpile_for_piastq,
    validate_piastq_submission,
)


def compile_and_count_2body(
    n: int,
    density: float,
    depth: int,
    *,
    seed: int,
    backend: BackendV2 | None = None,
    max_rxx: int = PIASTQ_MAX_RXX,
    optimization_level: int = 3,
) -> dict[str, Any]:
    """Compile a 2-body QUBO QAOA circuit for PIAST-Q and evaluate gate-ceiling pruning."""
    if backend is None:
        _, backend = get_piastq_backends()  # type: ignore[misc]
    weights, edges = make_density_instance(n, density, seed=seed)
    rng = np.random.default_rng(seed + 31337 + depth)
    gammas = rng.uniform(0.15, 1.35, size=depth).tolist()
    betas = rng.uniform(0.15, 0.95, size=depth).tolist()
    scale = float(int(weights.sum()) + 1) * max(1, len(edges))

    qc = build_qaoa_circuit(weights, edges, gammas, betas, phase_scale=scale)
    tqc = transpile_for_piastq(
        qc, backend, optimization_level=optimization_level, seed_transpiler=42
    )
    ops = dict(tqc.count_ops())
    rxx_count = int(ops.get("rxx", 0))
    r_count = int(ops.get("r", 0))
    rz_count = int(ops.get("rz", 0))
    measure_count = int(ops.get("measure", 0))
    qiskit_total_ops = sum(int(v) for v in ops.values())
    # Wire ops: each gate + 1 single terminal MEASURE instruction
    wire_ops = rxx_count + r_count + rz_count + 1

    passes_ops = qiskit_total_ops <= PIASTQ_MAX_OPS and wire_ops <= PIASTQ_MAX_OPS
    passes_rxx = rxx_count <= max_rxx
    passes_ceiling = bool(passes_ops and passes_rxx)

    if passes_ops:
        # Also run full wire-serializer check when within 2000-op wire limit
        validate_piastq_submission(tqc, shots=PIASTQ_MAX_SHOTS, max_rxx=None)

    return {
        "order": 2,
        "n": n,
        "target_density": density,
        "num_edges": len(edges),
        "actual_density": round(len(edges) / (n * (n - 1) / 2), 4),
        "depth_p": depth,
        "seed": seed,
        "rxx_count": rxx_count,
        "r_count": r_count,
        "rz_count": rz_count,
        "measure_count": measure_count,
        "qiskit_total_ops": qiskit_total_ops,
        "wire_ops": wire_ops,
        "compiled_depth": tqc.depth(),
        "passes_rxx_ceiling": passes_rxx,
        "passes_ops_ceiling": passes_ops,
        "passes_ceiling": passes_ceiling,
    }


def compile_and_count_3body(
    n: int,
    density_2body: float,
    num_triples: int,
    depth: int,
    *,
    seed: int,
    backend: BackendV2 | None = None,
    max_rxx: int = PIASTQ_MAX_RXX,
    optimization_level: int = 3,
) -> dict[str, Any]:
    """Compile an ancilla-free 3-body HUBO QAOA circuit and evaluate gate-ceiling pruning."""
    if backend is None:
        _, backend = get_piastq_backends()  # type: ignore[misc]
    weights, edges2, triples3 = make_3body_instance(n, density_2body, num_triples, seed=seed)
    rng = np.random.default_rng(seed + 51515 + depth)
    gammas = rng.uniform(0.15, 1.35, size=depth).tolist()
    betas = rng.uniform(0.15, 0.95, size=depth).tolist()
    scale = float(int(weights.sum()) + 1) * max(1, len(edges2) + len(triples3))

    qc = build_qaoa_circuit(
        weights, edges2, gammas, betas, triples=triples3, phase_scale=scale
    )
    tqc = transpile_for_piastq(
        qc, backend, optimization_level=optimization_level, seed_transpiler=42
    )
    ops = dict(tqc.count_ops())
    rxx_count = int(ops.get("rxx", 0))
    r_count = int(ops.get("r", 0))
    rz_count = int(ops.get("rz", 0))
    measure_count = int(ops.get("measure", 0))
    qiskit_total_ops = sum(int(v) for v in ops.values())
    wire_ops = rxx_count + r_count + rz_count + 1

    passes_ops = qiskit_total_ops <= PIASTQ_MAX_OPS and wire_ops <= PIASTQ_MAX_OPS
    passes_rxx = rxx_count <= max_rxx
    passes_ceiling = bool(passes_ops and passes_rxx)

    if passes_ops:
        validate_piastq_submission(tqc, shots=PIASTQ_MAX_SHOTS, max_rxx=None)

    # Count distinct induced pairs from triples + edges2
    induced_pairs = set(edges2)
    for i, j, k in triples3:
        induced_pairs.add((i, j))
        induced_pairs.add((i, k))
        induced_pairs.add((j, k))

    return {
        "order": 3,
        "n": n,
        "density_2body": density_2body,
        "num_edges_2body": len(edges2),
        "num_triples_3body": len(triples3),
        "distinct_2body_zz_pairs": len(induced_pairs),
        "unmerged_rxx_formula": depth * (len(induced_pairs) + 4 * len(triples3)),
        "depth_p": depth,
        "seed": seed,
        "rxx_count": rxx_count,
        "r_count": r_count,
        "rz_count": rz_count,
        "measure_count": measure_count,
        "qiskit_total_ops": qiskit_total_ops,
        "wire_ops": wire_ops,
        "compiled_depth": tqc.depth(),
        "passes_rxx_ceiling": passes_rxx,
        "passes_ops_ceiling": passes_ops,
        "passes_ceiling": passes_ceiling,
    }


def sweep_gate_mapping_matrix(
    sizes: Sequence[int] = (8, 12, 16, 20),
    densities: Sequence[float] = (0.30, 0.50, 0.65, 0.70, 0.85, 1.00),
    depths: Sequence[int] = (1, 2, 3),
    instances_per_cell: int = 3,
    base_seed: int = 20261127,
) -> dict[str, Any]:
    """Sweep full 2-body and 3-body instance-gate mapping matrices on PIAST-Q Target."""
    _, b_noisy = get_piastq_backends()  # type: ignore[misc]
    rows_2body: list[dict[str, Any]] = []
    for n in sizes:
        for d in densities:
            for p in depths:
                rxx_vals = []
                wire_vals = []
                depth_vals = []
                sample_row = None
                for inst in range(instances_per_cell):
                    s = base_seed + n * 10000 + int(round(d * 100)) * 100 + inst
                    row = compile_and_count_2body(n, d, p, seed=s, backend=b_noisy)
                    rxx_vals.append(row["rxx_count"])
                    wire_vals.append(row["wire_ops"])
                    depth_vals.append(row["compiled_depth"])
                    if sample_row is None:
                        sample_row = row
                assert sample_row is not None
                rows_2body.append(
                    {
                        **sample_row,
                        "instances_checked": instances_per_cell,
                        "rxx_min": int(min(rxx_vals)),
                        "rxx_max": int(max(rxx_vals)),
                        "rxx_mean": float(np.mean(rxx_vals)),
                        "wire_ops_max": int(max(wire_vals)),
                        "compiled_depth_mean": float(np.mean(depth_vals)),
                        "passes_ceiling": bool(
                            max(rxx_vals) <= PIASTQ_MAX_RXX and max(wire_vals) <= PIASTQ_MAX_OPS
                        ),
                    }
                )

    rows_3body: list[dict[str, Any]] = []
    triple_grid = [
        (8, 0.30, 6),
        (8, 0.50, 10),
        (12, 0.30, 10),
        (12, 0.30, 20),
        (12, 0.50, 20),
        (16, 0.30, 12),
        (16, 0.30, 25),
        (16, 0.50, 40),
        (20, 0.30, 12),
        (20, 0.30, 25),
        (20, 0.50, 35),
        (20, 0.50, 120),
    ]
    for n, d2, n3 in triple_grid:
        for p in (1, 2, 3):
            rxx_vals = []
            wire_vals = []
            depth_vals = []
            sample_row = None
            for inst in range(instances_per_cell):
                s = base_seed + 500000 + n * 10000 + n3 * 10 + inst
                row = compile_and_count_3body(n, d2, n3, p, seed=s, backend=b_noisy)
                rxx_vals.append(row["rxx_count"])
                wire_vals.append(row["wire_ops"])
                depth_vals.append(row["compiled_depth"])
                if sample_row is None:
                    sample_row = row
            assert sample_row is not None
            rows_3body.append(
                {
                    **sample_row,
                    "instances_checked": instances_per_cell,
                    "rxx_min": int(min(rxx_vals)),
                    "rxx_max": int(max(rxx_vals)),
                    "rxx_mean": float(np.mean(rxx_vals)),
                    "wire_ops_max": int(max(wire_vals)),
                    "compiled_depth_mean": float(np.mean(depth_vals)),
                    "passes_ceiling": bool(
                        max(rxx_vals) <= PIASTQ_MAX_RXX and max(wire_vals) <= PIASTQ_MAX_OPS
                    ),
                }
            )
    return {"two_body": rows_2body, "three_body": rows_3body}


def compile_calibration_reference_circuits(
    n: int = 20, *, backend: BackendV2 | None = None
) -> dict[str, dict[str, Any]]:
    """Build and compile Block E1 readout confusion and RXX mirror-drift reference circuits."""
    from qiskit import QuantumCircuit

    if backend is None:
        _, backend = get_piastq_backends()  # type: ignore[misc]

    # 1. All-zero computational basis readout calibration
    qc_zero = QuantumCircuit(n, n)
    qc_zero.measure(range(n), range(n))
    t_zero = transpile_for_piastq(qc_zero, backend, optimization_level=1, seed_transpiler=42)
    v_zero = validate_piastq_submission(t_zero, shots=PIASTQ_MAX_SHOTS, max_rxx=PIASTQ_MAX_RXX)

    # 2. All-one computational basis readout calibration
    qc_one = QuantumCircuit(n, n)
    qc_one.x(range(n))
    qc_one.measure(range(n), range(n))
    t_one = transpile_for_piastq(qc_one, backend, optimization_level=1, seed_transpiler=42)
    v_one = validate_piastq_submission(t_one, shots=PIASTQ_MAX_SHOTS, max_rxx=PIASTQ_MAX_RXX)

    # 3. 95-pair forward + barrier-separated inverse RXX mirror drift circuit (190 RXX <= 400)
    # Constructed directly in native RXX basis so transpilation preserves the 190 physical gates
    qc_mirror = QuantumCircuit(n, n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:95]
    for i, j in pairs:
        qc_mirror.rxx(0.25, i, j)
    qc_mirror.rz(0.01, range(n))
    for i, j in reversed(pairs):
        qc_mirror.rxx(0.25, i, j)
    qc_mirror.measure(range(n), range(n))
    v_mirror = validate_piastq_submission(qc_mirror, shots=PIASTQ_MAX_SHOTS, max_rxx=PIASTQ_MAX_RXX)

    return {
        "readout_all_zero": v_zero,
        "readout_all_one": v_one,
        "rxx_mirror_95pairs": v_mirror,
    }
