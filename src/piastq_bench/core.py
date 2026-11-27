"""Core QUBO / 3-body HUBO encoding, exact certification, SA, and statevector QAOA."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def make_instance(n: int, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Original v0.1 cycle + 0.25 random edge instance generator."""
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 10, size=n, dtype=np.int64)
    edges = {(i, i + 1) for i in range(n - 1)}
    if n > 2:
        edges.add((0, n - 1))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.25:
                edges.add((i, j))
    return weights, sorted(edges)


def make_density_instance(
    n: int, density: float, seed: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate an N-variable conflict graph with exact edge density round(density * C(N, 2))."""
    if not 2 <= n <= 20:
        raise ValueError("Expected 2 <= n <= 20")
    if not 0.0 <= density <= 1.0:
        raise ValueError("Expected 0.0 <= density <= 1.0")
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 10, size=n, dtype=np.int64)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    target_m = int(round(density * len(all_pairs)))
    if target_m == 0:
        return weights, []
    if target_m == len(all_pairs):
        return weights, all_pairs

    chosen: set[tuple[int, int]] = set()
    if n > 2 and target_m >= n:
        perm = rng.permutation(n).tolist()
        for idx in range(n):
            u, v = perm[idx], perm[(idx + 1) % n]
            chosen.add((min(u, v), max(u, v)))
    remaining = [p for p in all_pairs if p not in chosen]
    need = target_m - len(chosen)
    if need > 0:
        picks = rng.choice(len(remaining), size=need, replace=False)
        for p_idx in picks:
            chosen.add(remaining[int(p_idx)])
    return weights, sorted(chosen)


def make_3body_instance(
    n: int, density_2body: float, num_triples: int, seed: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Generate an N-variable instance with pairwise conflicts and 3-body hyperedges (i, j, k)."""
    if not 3 <= n <= 20:
        raise ValueError("Expected 3 <= n <= 20 for 3-body instances")
    max_triples = math.comb(n, 3)
    if not 0 <= num_triples <= max_triples:
        raise ValueError(f"Expected 0 <= num_triples <= {max_triples}")

    weights, edges2 = make_density_instance(n, density_2body, seed)
    if num_triples == 0:
        return weights, edges2, []

    edge_set = set(edges2)
    all_triples = [
        (i, j, k)
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    ]
    # Prefer non-redundant triples (where no pairwise conflict edge already forbids {i,j,k})
    pure_triples = [
        (i, j, k)
        for (i, j, k) in all_triples
        if (i, j) not in edge_set and (i, k) not in edge_set and (j, k) not in edge_set
    ]
    other_triples = [t for t in all_triples if t not in set(pure_triples)]

    rng3 = np.random.default_rng(seed + 7919)
    chosen_triples: list[tuple[int, int, int]] = []
    if len(pure_triples) >= num_triples:
        idxs = rng3.choice(len(pure_triples), size=num_triples, replace=False)
        chosen_triples = [pure_triples[int(x)] for x in idxs]
    else:
        chosen_triples.extend(pure_triples)
        rem = num_triples - len(chosen_triples)
        idxs = rng3.choice(len(other_triples), size=rem, replace=False)
        chosen_triples.extend(other_triples[int(x)] for x in idxs)

    return weights, edges2, sorted(chosen_triples)


def _validate(weights: Sequence[int] | np.ndarray, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(weights)
    if (
        arr.ndim != 1
        or not 1 <= len(arr) <= 20
        or not np.issubdtype(arr.dtype, np.integer)
        or np.any(arr < 1)
        or np.any(arr > 1000)
    ):
        raise ValueError("Expected 1..20 positive integer weights, each at most 1000")
    if len(edges) != len(set(edges)) or any(not 0 <= i < j < len(arr) for i, j in edges):
        raise ValueError("Edges must be unique ordered vertex pairs")
    return arr.astype(np.int64)


def _validate_triples(n: int, triples: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    if len(triples) != len(set(triples)) or any(
        not 0 <= i < j < k < n for i, j, k in triples
    ):
        raise ValueError("Triples must be unique ordered vertex triples")
    return [(int(i), int(j), int(k)) for i, j, k in triples]


def energy_table(
    weights: Sequence[int] | np.ndarray, edges: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Exact 2-body QUBO energy table over all 2^N assignments."""
    w = _validate(weights, edges)
    n = len(w)
    indices = np.arange(1 << n, dtype=np.int64)
    energy = np.zeros(len(indices), dtype=np.int64)
    for i, weight in enumerate(w):
        energy -= int(weight) * ((indices >> i) & 1)
    penalty = int(w.sum()) + 1
    for i, j in edges:
        energy += penalty * ((indices >> i) & 1) * ((indices >> j) & 1)
    return energy


def energy_table_3body(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    triples: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Exact 3-body HUBO energy table over all 2^N assignments (ancilla-free)."""
    w = _validate(weights, edges)
    t_list = _validate_triples(len(w), triples)
    energy = energy_table(w, edges)
    if not t_list:
        return energy
    indices = np.arange(1 << len(w), dtype=np.int64)
    penalty = int(w.sum()) + 1
    for i, j, k in t_list:
        energy += penalty * ((indices >> i) & 1) * ((indices >> j) & 1) * ((indices >> k) & 1)
    return energy


def valid_mask(n: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    indices = np.arange(1 << n, dtype=np.int64)
    valid = np.ones(len(indices), dtype=bool)
    for i, j in edges:
        valid &= ~(((indices >> i) & 1).astype(bool) & ((indices >> j) & 1).astype(bool))
    return valid


def valid_mask_3body(
    n: int,
    edges: Sequence[tuple[int, int]],
    triples: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    t_list = _validate_triples(n, triples)
    valid = valid_mask(n, edges)
    if not t_list:
        return valid
    indices = np.arange(1 << n, dtype=np.int64)
    for i, j, k in t_list:
        valid &= ~(
            ((indices >> i) & 1).astype(bool)
            & ((indices >> j) & 1).astype(bool)
            & ((indices >> k) & 1).astype(bool)
        )
    return valid


def qaoa_state(
    energy: np.ndarray, gammas: Sequence[float], betas: Sequence[float]
) -> np.ndarray:
    energy_f = np.asarray(energy, dtype=float)
    size = len(energy_f)
    if energy_f.ndim != 1 or size < 2 or size & (size - 1):
        raise ValueError("Energy must have power-of-two length, at least two")
    if len(gammas) != len(betas) or not len(gammas):
        raise ValueError("Gamma and beta must have the same positive depth")
    if not (
        np.all(np.isfinite(energy_f))
        and np.all(np.isfinite(gammas))
        and np.all(np.isfinite(betas))
    ):
        raise ValueError("Statevector inputs must be finite")
    state = np.full(size, 1.0 / math.sqrt(size), dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * float(gamma) * energy_f)
        c, s = math.cos(float(beta)), -1j * math.sin(float(beta))
        for qubit in range(size.bit_length() - 1):
            step = 1 << qubit
            blocks = state.reshape(-1, 2 * step)
            lo, hi = blocks[:, :step].copy(), blocks[:, step:].copy()
            blocks[:, :step] = c * lo + s * hi
            blocks[:, step:] = s * lo + c * hi
    return state


def select_qaoa_angles(
    energy: np.ndarray, depth: int, seed: int, candidates: int = 24
) -> dict:
    """Pre-registered bounded angle selection on the ideal statevector emulator."""
    rng = np.random.default_rng(seed)
    scale = float(np.max(np.abs(energy)))
    normalized = energy / scale if scale else energy.astype(float)
    best_expectation = math.inf
    best_gammas: np.ndarray | None = None
    best_betas: np.ndarray | None = None
    for trial in range(candidates):
        gammas = np.zeros(depth) if trial == 0 else rng.uniform(0.0, 2.0 * math.pi, depth)
        betas = np.zeros(depth) if trial == 0 else rng.uniform(0.0, math.pi, depth)
        state = qaoa_state(normalized, gammas, betas)
        expectation = float(np.dot(np.abs(state) ** 2, energy))
        if expectation < best_expectation:
            best_expectation = expectation
            best_gammas, best_betas = gammas, betas
    assert best_gammas is not None and best_betas is not None
    return {
        "gammas": best_gammas.tolist(),
        "betas": best_betas.tolist(),
        "phase_energy_scale": scale,
        "exact_expected_energy": best_expectation,
        "expectation_evaluations": candidates,
    }


def anneal(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    seed: int,
    reads: int = 1024,
    sweeps: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    w = _validate(weights, edges)
    if reads < 1 or sweeps < 1:
        raise ValueError("Positive reads and sweeps required")
    rng = np.random.default_rng(seed)
    n, penalty = len(w), int(w.sum()) + 1
    coupling = np.zeros((n, n), dtype=np.int64)
    for i, j in edges:
        coupling[i, j] = coupling[j, i] = penalty
    bits = rng.integers(0, 2, size=(reads, n), dtype=np.int64)
    energy = -(bits @ w)
    for i, j in edges:
        energy += penalty * bits[:, i] * bits[:, j]
    for temperature in np.geomspace(float(penalty), 0.05, sweeps):
        for bit in rng.permutation(n):
            delta = (1 - 2 * bits[:, bit]) * (-w[bit] + bits @ coupling[:, bit])
            accept = (delta <= 0) | (
                rng.random(reads) < np.exp(-np.maximum(delta, 0) / temperature)
            )
            bits[accept, bit] = 1 - bits[accept, bit]
            energy[accept] += delta[accept]
    indices = bits @ (1 << np.arange(n, dtype=np.int64))
    return indices, energy


def anneal_3body(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    triples: Sequence[tuple[int, int, int]],
    *,
    seed: int,
    reads: int = 1024,
    sweeps: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated annealing directly on the identical 3-body HUBO objective."""
    w = _validate(weights, edges)
    t_list = _validate_triples(len(w), triples)
    if not t_list:
        return anneal(w, edges, seed=seed, reads=reads, sweeps=sweeps)
    if reads < 1 or sweeps < 1:
        raise ValueError("Positive reads and sweeps required")
    rng = np.random.default_rng(seed)
    n, penalty = len(w), int(w.sum()) + 1
    coupling = np.zeros((n, n), dtype=np.int64)
    for i, j in edges:
        coupling[i, j] = coupling[j, i] = penalty
    adj_triples: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i, j, k in t_list:
        adj_triples[i].append((j, k))
        adj_triples[j].append((i, k))
        adj_triples[k].append((i, j))

    bits = rng.integers(0, 2, size=(reads, n), dtype=np.int64)
    energy = -(bits @ w)
    for i, j in edges:
        energy += penalty * bits[:, i] * bits[:, j]
    for i, j, k in t_list:
        energy += penalty * bits[:, i] * bits[:, j] * bits[:, k]

    for temperature in np.geomspace(float(penalty), 0.05, sweeps):
        for bit in rng.permutation(n):
            cubic_sum = np.zeros(reads, dtype=np.int64)
            for u, v in adj_triples[int(bit)]:
                cubic_sum += bits[:, u] * bits[:, v]
            delta = (1 - 2 * bits[:, bit]) * (
                -w[bit] + bits @ coupling[:, bit] + penalty * cubic_sum
            )
            accept = (delta <= 0) | (
                rng.random(reads) < np.exp(-np.maximum(delta, 0) / temperature)
            )
            bits[accept, bit] = 1 - bits[accept, bit]
            energy[accept] += delta[accept]
    indices = bits @ (1 << np.arange(n, dtype=np.int64))
    return indices, energy


def sample_metrics(samples: np.ndarray, energy: np.ndarray, valid: np.ndarray) -> dict:
    s = np.asarray(samples, dtype=np.int64)
    if s.ndim != 1 or not len(s) or np.any(s < 0) or np.any(s >= len(energy)):
        raise ValueError("Samples must be nonempty indices into the energy table")
    optimum = int(np.min(energy))
    chosen = int(s[np.lexsort((s, energy[s]))[0]])
    chosen_energy = int(energy[chosen])
    feasible = bool(valid[chosen])
    return {
        "selected_index": chosen,
        "selected_energy": chosen_energy,
        "objective_gap": chosen_energy - optimum,
        "selected_optimal": bool(feasible and chosen_energy == optimum),
        "selected_valid": feasible,
        "mean_sample_energy": float(np.mean(energy[s])),
        "valid_fraction": float(np.mean(valid[s])),
        "optimum_fraction": float(np.mean(energy[s] == optimum)),
        "feasible_weight_ratio": (
            float(chosen_energy / optimum) if feasible and optimum < 0 else None
        ),
    }
