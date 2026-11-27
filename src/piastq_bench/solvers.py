"""Classical reference solvers on natural constraint and QUBO/HUBO representations."""
from __future__ import annotations

import time
from typing import Any, Sequence

import neal
import numpy as np
from ortools.sat.python import cp_model
import z3

from .core import _validate, _validate_triples


def solve_cpsat_natural(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    triples: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """OR-Tools CP-SAT on natural hard independent-set / hypergraph exclusion constraints."""
    w = _validate(weights, edges)
    t_list = _validate_triples(len(w), triples or [])
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(w))]
    for i, j in edges:
        model.Add(x[i] + x[j] <= 1)
    for i, j, k in t_list:
        model.Add(x[i] + x[j] + x[k] <= 2)
    model.Maximize(sum(int(w[i]) * x[i] for i in range(len(w))))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT natural failed with status {status}")
    bits = [int(solver.Value(x[i])) for i in range(len(w))]
    idx = sum(b << i for i, b in enumerate(bits))
    energy = -sum(int(w[i]) * bits[i] for i in range(len(w)))
    return {"index": idx, "energy": energy, "valid": True, "seconds": elapsed}


def solve_cpsat_qubo(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    triples: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """OR-Tools CP-SAT minimizing the penalized unconstrained QUBO / HUBO objective directly."""
    w = _validate(weights, edges)
    t_list = _validate_triples(len(w), triples or [])
    penalty = int(w.sum()) + 1
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(w))]
    pair_vars = []
    for idx_e, (i, j) in enumerate(edges):
        y = model.NewBoolVar(f"y_{idx_e}")
        model.AddMultiplicationEquality(y, [x[i], x[j]])
        pair_vars.append(y)
    triple_vars = []
    for idx_t, (i, j, k) in enumerate(t_list):
        z = model.NewBoolVar(f"z_{idx_t}")
        model.AddBoolOr([x[i].Not(), x[j].Not(), x[k].Not(), z])
        model.AddImplication(z, x[i])
        model.AddImplication(z, x[j])
        model.AddImplication(z, x[k])
        triple_vars.append(z)

    model.Minimize(
        -sum(int(w[i]) * x[i] for i in range(len(w)))
        + penalty * sum(pair_vars)
        + penalty * sum(triple_vars)
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT QUBO failed with status {status}")
    bits = [int(solver.Value(x[i])) for i in range(len(w))]
    idx = sum(b << i for i, b in enumerate(bits))
    energy = int(round(solver.ObjectiveValue()))
    return {"index": idx, "energy": energy, "seconds": elapsed}


def solve_z3_natural(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    triples: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Z3 Optimize on natural hard boolean conflict constraints."""
    w = _validate(weights, edges)
    t_list = _validate_triples(len(w), triples or [])
    t0 = time.perf_counter()
    opt = z3.Optimize()
    bvars = [z3.Bool(f"x_{i}") for i in range(len(w))]
    for i, j in edges:
        opt.add(z3.Not(z3.And(bvars[i], bvars[j])))
    for i, j, k in t_list:
        opt.add(z3.Not(z3.And(bvars[i], bvars[j], bvars[k])))
    total_w = z3.Sum([z3.If(bvars[i], int(w[i]), 0) for i in range(len(w))])
    opt.maximize(total_w)
    if opt.check() != z3.sat:
        raise RuntimeError("Z3 natural returned unsat")
    m = opt.model()
    elapsed = time.perf_counter() - t0
    bits = [1 if z3.is_true(m.evaluate(bvars[i], model_completion=True)) else 0 for i in range(len(w))]
    idx = sum(b << i for i, b in enumerate(bits))
    energy = -sum(int(w[i]) * bits[i] for i in range(len(w)))
    return {"index": idx, "energy": energy, "valid": True, "seconds": elapsed}


def solve_z3_qubo(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    triples: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Z3 Optimize on the unconstrained penalized QUBO / HUBO objective."""
    w = _validate(weights, edges)
    t_list = _validate_triples(len(w), triples or [])
    penalty = int(w.sum()) + 1
    t0 = time.perf_counter()
    opt = z3.Optimize()
    bvars = [z3.Bool(f"x_{i}") for i in range(len(w))]
    terms = [z3.If(bvars[i], -int(w[i]), 0) for i in range(len(w))]
    for i, j in edges:
        terms.append(z3.If(z3.And(bvars[i], bvars[j]), penalty, 0))
    for i, j, k in t_list:
        terms.append(z3.If(z3.And(bvars[i], bvars[j], bvars[k]), penalty, 0))
    obj = z3.Sum(terms)
    opt.minimize(obj)
    if opt.check() != z3.sat:
        raise RuntimeError("Z3 QUBO returned unsat")
    m = opt.model()
    elapsed = time.perf_counter() - t0
    bits = [1 if z3.is_true(m.evaluate(bvars[i], model_completion=True)) else 0 for i in range(len(w))]
    idx = sum(b << i for i, b in enumerate(bits))
    energy = int(m.evaluate(obj, model_completion=True).as_long())
    return {"index": idx, "energy": energy, "seconds": elapsed}


def solve_neal_qubo(
    weights: Sequence[int] | np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    num_reads: int = 1000,
    num_sweeps: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """dwave-neal SimulatedAnnealingSampler on the exact 2-body QUBO."""
    w = _validate(weights, edges)
    penalty = int(w.sum()) + 1
    Q: dict[tuple[int, int], float] = {(i, i): -float(w[i]) for i in range(len(w))}
    for i, j in edges:
        Q[(i, j)] = float(penalty)
    sampler = neal.SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    elapsed = time.perf_counter() - t0
    rec = sampleset.record
    # Convert variable columns [0..n-1] to integer bitmasks
    var_order = list(sampleset.variables)
    bits = np.zeros((len(rec), len(w)), dtype=np.int64)
    for col_idx, var_id in enumerate(var_order):
        bits[:, int(var_id)] = rec.sample[:, col_idx]
    indices = bits @ (1 << np.arange(len(w), dtype=np.int64))
    energies = np.round(rec.energy).astype(np.int64)
    best_pos = int(np.lexsort((indices, energies))[0])
    return {
        "samples": indices,
        "selected_index": int(indices[best_pos]),
        "selected_energy": int(energies[best_pos]),
        "seconds": elapsed,
    }


def uniform_random_samples(n: int, num_shots: int, *, seed: int) -> np.ndarray:
    """Matched-shot uniform random bitstring baseline for the stopping rule."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << n, size=num_shots, dtype=np.int64)
