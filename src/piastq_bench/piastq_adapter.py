"""PIAST-Q site-client adapter (Qiskit 2.3-2.4.1, qiskit-aqt-provider 2.0.1)."""
from __future__ import annotations

from collections import defaultdict
import time
from typing import Any, Sequence

from aqt_connector.models.circuits import QuantumCircuit as AQTWireQuantumCircuit
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.providers import BackendV2
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider._transformers import qiskit_to_aqt_circuit

from .core import _validate, _validate_triples
from .statistics import QPUBudgetGuard

PIASTQ_NATIVE_GATES = frozenset({"rz", "r", "rxx"})
PIASTQ_ALLOWED_OPS = frozenset({"rz", "r", "rxx", "measure"})
PIASTQ_MAX_QUBITS = 20
PIASTQ_MAX_SHOTS = 200
PIASTQ_MAX_OPS = 2000
PIASTQ_MAX_RXX = 400


def get_piastq_backends(
    *,
    direct_base_url: str | None = None,
    access_token: str | None = None,
    max_parallel_threads: int | None = None,
) -> tuple[BackendV2, BackendV2] | BackendV2:
    """Return PIAST-Q offline (ideal, noisy) backends or DirectAccess hardware resource."""
    provider = AQTProvider()
    if direct_base_url is not None:
        if not access_token:
            raise ValueError("access_token is required when direct_base_url is set")
        return provider.direct_access().get_resource(direct_base_url, access_token)

    b_ideal = provider.offline.ideal()
    b_noisy = provider.offline.noisy()
    if max_parallel_threads is not None and max_parallel_threads > 0:
        b_ideal.simulator.set_options(max_parallel_threads=max_parallel_threads)
        b_noisy.simulator.set_options(
            max_parallel_threads=max_parallel_threads,
            max_parallel_shots=max_parallel_threads,
        )
    return b_ideal, b_noisy


def ising_decomposition(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    triples: Sequence[tuple[int, int, int]] | None = None,
    *,
    phase_scale: float = 1.0,
) -> tuple[dict[int, float], dict[tuple[int, int], float], dict[tuple[int, int, int], float]]:
    """Exact mapping of E(x)/phase_scale to Z_i, Z_i Z_j, and Z_i Z_j Z_k coefficients.

    Uses x_i = (1 - Z_i) / 2 so computational basis |0> (z=+1) maps to x=0 and |1> (z=-1) to x=1.
    """
    w = _validate(weights, edges)
    t_list = _validate_triples(len(w), triples or [])
    if phase_scale <= 0 or not np.isfinite(phase_scale):
        raise ValueError("phase_scale must be finite and positive")

    penalty = float(int(w.sum()) + 1) / float(phase_scale)
    c1: dict[int, float] = defaultdict(float)
    c2: dict[tuple[int, int], float] = defaultdict(float)
    c3: dict[tuple[int, int, int], float] = defaultdict(float)

    for i, weight in enumerate(w):
        w_scaled = float(weight) / float(phase_scale)
        # -w_i * (1 - Z_i)/2 = -w_i/2 + (w_i/2) Z_i
        c1[i] += 0.5 * w_scaled

    for i, j in edges:
        # P * (1 - Z_i)(1 - Z_j)/4 = P/4 * (1 - Z_i - Z_j + Z_i Z_j)
        c1[i] -= 0.25 * penalty
        c1[j] -= 0.25 * penalty
        c2[(i, j)] += 0.25 * penalty

    for i, j, k in t_list:
        # P * (1 - Z_i)(1 - Z_j)(1 - Z_k)/8
        # = P/8 * (1 - Z_i - Z_j - Z_k + Z_i Z_j + Z_i Z_k + Z_j Z_k - Z_i Z_j Z_k)
        c1[i] -= 0.125 * penalty
        c1[j] -= 0.125 * penalty
        c1[k] -= 0.125 * penalty
        c2[(i, j)] += 0.125 * penalty
        c2[(i, k)] += 0.125 * penalty
        c2[(j, k)] += 0.125 * penalty
        c3[(i, j, k)] -= 0.125 * penalty

    return (
        {k: v for k, v in sorted(c1.items()) if abs(v) > 1e-15},
        {k: v for k, v in sorted(c2.items()) if abs(v) > 1e-15},
        {k: v for k, v in sorted(c3.items()) if abs(v) > 1e-15},
    )


def build_qaoa_circuit(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    triples: Sequence[tuple[int, int, int]] | None = None,
    phase_scale: float = 1.0,
) -> QuantumCircuit:
    """Build ancilla-free QAOA circuit with single terminal projective full-register measurement."""
    if len(gammas) != len(betas) or not len(gammas):
        raise ValueError("gammas and betas must have equal positive length")
    n = len(weights)
    c1, c2, c3 = ising_decomposition(weights, edges, triples, phase_scale=phase_scale)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        g = float(gamma)
        b = float(beta)
        # 2-body ZZ phase gadgets
        for (i, j), coeff in c2.items():
            qc.rzz(2.0 * g * coeff, i, j)
        # 3-body ZZZ ancilla-free phase gadgets (4 CNOTs -> 4 RXX unmerged)
        for (i, j, k), coeff in c3.items():
            qc.cx(i, j)
            qc.cx(j, k)
            qc.rz(2.0 * g * coeff, k)
            qc.cx(j, k)
            qc.cx(i, j)
        # 1-body Z rotations
        for i, coeff in c1.items():
            qc.rz(2.0 * g * coeff, i)
        # Transverse-field mixer exp(-i * beta * X_i) = RX(2 * beta)
        for i in range(n):
            qc.rx(2.0 * b, i)
    qc.measure(range(n), range(n))
    return qc


def ingest_external_circuit(source: str | QuantumCircuit, *, fmt: str = "qasm2") -> QuantumCircuit:
    """Ingest external CUDA-Q / PennyLane OpenQASM 2.0 or Qiskit circuits into the PIAST-Q pipeline."""
    if isinstance(source, QuantumCircuit):
        return source.copy()
    if fmt == "qasm2":
        return qasm2.loads(source)
    raise ValueError(f"Unsupported external circuit format: {fmt} (supported: 'qasm2' or QuantumCircuit)")


def transpile_for_piastq(
    circuit: QuantumCircuit,
    backend: BackendV2,
    *,
    optimization_level: int = 3,
    seed_transpiler: int = 42,
) -> QuantumCircuit:
    """Transpile to AQT native {rz, r, rxx} + single terminal full-register measurement without barriers."""
    tqc = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    # Strip any cosmetic barriers introduced by upstream circuit builders
    if "barrier" in tqc.count_ops():
        clean = QuantumCircuit(*tqc.qregs, *tqc.cregs, name=tqc.name)
        clean.global_phase = tqc.global_phase
        for inst in tqc.data:
            if inst.operation.name != "barrier":
                clean.append(inst.operation, inst.qubits, inst.clbits)
        tqc = clean
    return tqc


def validate_piastq_submission(
    circuit: QuantumCircuit,
    *,
    shots: int = PIASTQ_MAX_SHOTS,
    max_rxx: int | None = None,
) -> dict[str, Any]:
    """Enforce all PSNC PIAST-Q hardware interface and wire-payload constraints."""
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("PIAST-Q requires a single QuantumCircuit per submission")
    if not isinstance(shots, (int, np.integer)) or not (1 <= int(shots) <= PIASTQ_MAX_SHOTS):
        raise ValueError(f"PIAST-Q requires 1 <= shots <= {PIASTQ_MAX_SHOTS} per submission (got {shots})")
    n_qubits = circuit.num_qubits
    if not (1 <= n_qubits <= PIASTQ_MAX_QUBITS):
        raise ValueError(f"PIAST-Q requires 1..{PIASTQ_MAX_QUBITS} qubits (got {n_qubits})")
    if circuit.num_clbits != n_qubits:
        raise ValueError(
            f"PIAST-Q requires full-register classical readout ({n_qubits} clbits, got {circuit.num_clbits})"
        )

    seen_measure = False
    measured_qubits: set[int] = set()
    measured_map: dict[int, int] = {}
    counts: dict[str, int] = {"rz": 0, "r": 0, "rxx": 0, "measure": 0}

    for inst in circuit.data:
        op_name = inst.operation.name
        if getattr(inst.operation, "condition", None) is not None or inst.is_control_flow():
            raise ValueError(f"Classical feed-forward / control flow is forbidden on PIAST-Q ({op_name})")
        if op_name not in PIASTQ_ALLOWED_OPS:
            raise ValueError(
                f"Operation '{op_name}' is non-native / forbidden on PIAST-Q (allowed: rz, r, rxx, terminal measure)"
            )
        if op_name == "measure":
            seen_measure = True
            q_idx = circuit.find_bit(inst.qubits[0]).index
            c_idx = circuit.find_bit(inst.clbits[0]).index
            if q_idx in measured_qubits:
                raise ValueError(f"Duplicate measurement on qubit {q_idx}")
            measured_qubits.add(q_idx)
            measured_map[q_idx] = c_idx
            counts["measure"] += 1
        else:
            if seen_measure:
                raise ValueError(
                    f"Mid-circuit measurement forbidden: gate '{op_name}' appears after measurement"
                )
            counts[op_name] += 1

    if measured_qubits != set(range(n_qubits)) or any(measured_map[q] != q for q in range(n_qubits)):
        raise ValueError(
            "PIAST-Q requires a single terminal projective measurement of the full register (q[i] -> c[i])"
        )

    qiskit_total_ops = sum(counts.values())
    if qiskit_total_ops > PIASTQ_MAX_OPS:
        raise ValueError(
            f"Circuit exceeds {PIASTQ_MAX_OPS} operations limit (qiskit_total_ops={qiskit_total_ops})"
        )

    # Validate wire-level serialization via site client's aqt_connector model (also enforces <= 2000 wire ops)
    try:
        wire_payload = AQTWireQuantumCircuit(
            repetitions=int(shots),
            quantum_circuit=qiskit_to_aqt_circuit(circuit),
            number_of_qubits=n_qubits,
        )
    except Exception as exc:
        raise ValueError(f"AQT wire serialization failed: {exc}") from exc

    wire_ops = len(wire_payload.quantum_circuit.root)
    if wire_ops > PIASTQ_MAX_OPS:
        raise ValueError(f"Wire payload exceeds {PIASTQ_MAX_OPS} operations (wire_ops={wire_ops})")

    if max_rxx is not None and counts["rxx"] > max_rxx:
        raise ValueError(f"Compiled RXX count {counts['rxx']} exceeds ceiling {max_rxx}")

    return {
        "valid": True,
        "num_qubits": n_qubits,
        "shots": int(shots),
        "rxx_count": counts["rxx"],
        "r_count": counts["r"],
        "rz_count": counts["rz"],
        "measure_count": counts["measure"],
        "qiskit_total_ops": qiskit_total_ops,
        "wire_ops": wire_ops,
        "depth": circuit.depth(),
    }


def _counts_to_samples(counts: dict[str, int], seed: int) -> np.ndarray:
    """Convert Qiskit bitstring count dictionary into a deterministically shuffled 1D array of state indices."""
    items = sorted((int(bitstr.replace(" ", ""), 2), int(cnt)) for bitstr, cnt in counts.items())
    expanded = np.repeat(
        np.array([idx for idx, _ in items], dtype=np.int64),
        np.array([cnt for _, cnt in items], dtype=np.int64),
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(expanded)
    return expanded


def execute_single_submission(
    backend: BackendV2,
    circuit: QuantumCircuit,
    *,
    shots: int = PIASTQ_MAX_SHOTS,
    seed_simulator: int | None = None,
    max_rxx: int | None = PIASTQ_MAX_RXX,
    budget_guard: QPUBudgetGuard | None = None,
    reserved_submission_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate and submit a single circuit with <= 200 shots to the PIAST-Q backend."""
    is_offline_emulator = hasattr(backend, "simulator")
    if not is_offline_emulator and budget_guard is None:
        raise ValueError("budget_guard is required when submitting to a non-emulator / QPU backend")
    if budget_guard is not None:
        budget_guard.check_can_submit(reserved_submission_seconds)

    validation = validate_piastq_submission(circuit, shots=shots, max_rxx=max_rxx)
    run_kwargs: dict[str, Any] = {"shots": int(shots)}
    if seed_simulator is not None and is_offline_emulator:
        run_kwargs["seed_simulator"] = int(seed_simulator)
    t0 = time.perf_counter()
    job = backend.run(circuit, **run_kwargs)
    result = job.result()
    elapsed = time.perf_counter() - t0
    if budget_guard is not None:
        budget_guard.record_submission(elapsed)
    counts = result.get_counts()
    total_returned = sum(int(v) for v in counts.values())
    if total_returned != int(shots):
        raise RuntimeError(f"Expected {shots} returned shots, got {total_returned}")
    samples = _counts_to_samples(counts, seed=int(seed_simulator or 0) + 17)
    return {
        "validation": validation,
        "counts": counts,
        "samples": samples,
        "elapsed_seconds": elapsed,
    }


def execute_accumulated_submissions(
    backend: BackendV2,
    circuit: QuantumCircuit,
    *,
    total_shots: int,
    max_shots_per_submission: int = PIASTQ_MAX_SHOTS,
    base_seed: int = 1000,
    max_rxx: int | None = PIASTQ_MAX_RXX,
    budget_guard: QPUBudgetGuard | None = None,
    reserved_submission_seconds: float | None = None,
) -> dict[str, Any]:
    """Accumulate `total_shots` via K independent single-circuit `<= 200`-shot submissions."""
    if total_shots < 1:
        raise ValueError("total_shots must be >= 1")
    if not (1 <= max_shots_per_submission <= PIASTQ_MAX_SHOTS):
        raise ValueError(f"max_shots_per_submission must be in 1..{PIASTQ_MAX_SHOTS}")

    chunks: list[int] = []
    rem = int(total_shots)
    while rem > 0:
        c = min(rem, int(max_shots_per_submission))
        chunks.append(c)
        rem -= c

    all_samples: list[np.ndarray] = []
    submission_times: list[float] = []
    validation_summary: dict[str, Any] | None = None
    for sub_idx, chunk_shots in enumerate(chunks):
        sub_res = execute_single_submission(
            backend,
            circuit,
            shots=chunk_shots,
            seed_simulator=base_seed + sub_idx * 9973,
            max_rxx=max_rxx,
            budget_guard=budget_guard,
            reserved_submission_seconds=reserved_submission_seconds,
        )
        if validation_summary is None:
            validation_summary = sub_res["validation"]
        all_samples.append(sub_res["samples"])
        submission_times.append(sub_res["elapsed_seconds"])

    assert validation_summary is not None
    concat_samples = np.concatenate(all_samples)
    return {
        "total_shots": int(total_shots),
        "submissions_count": len(chunks),
        "shot_chunks": chunks,
        "samples": concat_samples,
        "submission_seconds": submission_times,
        "total_seconds": float(sum(submission_times)),
        "circuit_stats": validation_summary,
    }
