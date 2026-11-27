"""Ideal-statevector feasibility probe; no hardware calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np


def make_instance(n, seed):
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


def _validate(weights, edges):
    weights = np.asarray(weights)
    if (weights.ndim != 1 or not 1 <= len(weights) <= 20
            or not np.issubdtype(weights.dtype, np.integer)
            or np.any(weights < 1) or np.any(weights > 1000)):
        raise ValueError('Expected 1..20 positive integer weights, each at most 1000')
    if len(edges) != len(set(edges)) or any(
            not 0 <= i < j < len(weights) for i, j in edges):
        raise ValueError('Edges must be unique ordered vertex pairs')
    return weights.astype(np.int64)


def energy_table(weights, edges):
    weights = _validate(weights, edges)
    indices = np.arange(1 << len(weights), dtype=np.int64)
    energy = np.zeros(len(indices), dtype=np.int64)
    for i, weight in enumerate(weights):
        energy -= weight * ((indices >> i) & 1)
    penalty = int(weights.sum()) + 1
    for i, j in edges:
        energy += penalty * ((indices >> i) & 1) * ((indices >> j) & 1)
    return energy


def valid_mask(n, edges):
    indices = np.arange(1 << n, dtype=np.int64)
    valid = np.ones(len(indices), dtype=bool)
    for i, j in edges:
        valid &= ~(((indices >> i) & 1).astype(bool)
                   & ((indices >> j) & 1).astype(bool))
    return valid


def qaoa_state(energy, gammas, betas):
    energy = np.asarray(energy, dtype=float)
    size = len(energy)
    if energy.ndim != 1 or size < 2 or size & (size - 1):
        raise ValueError('Energy must have power-of-two length, at least two')
    if len(gammas) != len(betas) or not len(gammas):
        raise ValueError('Gamma and beta must have the same positive depth')
    if not (np.all(np.isfinite(energy)) and np.all(np.isfinite(gammas))
            and np.all(np.isfinite(betas))):
        raise ValueError('Statevector inputs must be finite')
    state = np.full(size, 1 / math.sqrt(size), dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * energy)
        c, s = math.cos(beta), -1j * math.sin(beta)
        for qubit in range(size.bit_length() - 1):
            step = 1 << qubit
            blocks = state.reshape(-1, 2 * step)
            lo, hi = blocks[:, :step].copy(), blocks[:, step:].copy()
            blocks[:, :step] = c * lo + s * hi
            blocks[:, step:] = s * lo + c * hi
    return state


def anneal(weights, edges, *, seed, reads=1024, sweeps=50):
    weights = _validate(weights, edges)
    if reads < 1 or sweeps < 1:
        raise ValueError('Positive reads and sweeps required')
    rng = np.random.default_rng(seed)
    n, penalty = len(weights), int(weights.sum()) + 1
    coupling = np.zeros((n, n), dtype=np.int64)
    for i, j in edges:
        coupling[i, j] = coupling[j, i] = penalty
    bits = rng.integers(0, 2, size=(reads, n), dtype=np.int64)
    energy = -(bits @ weights)
    for i, j in edges:
        energy += penalty * bits[:, i] * bits[:, j]
    for temperature in np.geomspace(float(penalty), 0.05, sweeps):
        for bit in rng.permutation(n):
            delta = (1 - 2 * bits[:, bit]) * (-weights[bit] + bits @ coupling[:, bit])
            accept = ((delta <= 0)
                      | (rng.random(reads) < np.exp(-np.maximum(delta, 0) / temperature)))
            bits[accept, bit] = 1 - bits[accept, bit]
            energy[accept] += delta[accept]
    indices = bits @ (1 << np.arange(n, dtype=np.int64))
    return indices, energy


def sample_metrics(samples, energy, valid):
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim != 1 or not len(samples) or np.any(samples < 0) or np.any(samples >= len(energy)):
        raise ValueError('Samples must be nonempty indices into the energy table')
    optimum = int(np.min(energy))
    chosen = int(samples[np.lexsort((samples, energy[samples]))[0]])
    chosen_energy = int(energy[chosen])
    feasible = bool(valid[chosen])
    return {
        'selected_index': chosen,
        'selected_energy': chosen_energy,
        'objective_gap': chosen_energy - optimum,
        'selected_valid': feasible,
        'valid_fraction': float(np.mean(valid[samples])),
        'optimum_fraction': float(np.mean(energy[samples] == optimum)),
        'feasible_weight_ratio': float(chosen_energy / optimum) if feasible and optimum < 0 else None,
    }


def qaoa_samples(energy, depth, seed, reads, candidates):
    rng = np.random.default_rng(seed)
    scale = float(np.max(np.abs(energy)))
    normalized = energy / scale if scale else energy.astype(float)
    best_expectation = math.inf
    best_state, best_gammas, best_betas = None, None, None
    for trial in range(candidates):
        gammas = np.zeros(depth) if trial == 0 else rng.uniform(0, 2 * math.pi, depth)
        betas = np.zeros(depth) if trial == 0 else rng.uniform(0, math.pi, depth)
        state = qaoa_state(normalized, gammas, betas)
        expectation = float(np.dot(np.abs(state) ** 2, energy))
        if expectation < best_expectation:
            best_expectation = expectation
            best_state, best_gammas, best_betas = state, gammas, betas
    probabilities = np.abs(best_state) ** 2
    probabilities /= probabilities.sum()
    samples = rng.choice(len(energy), size=reads, p=probabilities)
    return samples, {
        'gammas': best_gammas.tolist(), 'betas': best_betas.tolist(),
        'phase_energy_scale': scale, 'exact_expected_energy': best_expectation,
        'expectation_evaluations': candidates, 'state_count_per_evaluation': len(energy),
    }


def _histogram(samples):
    indices, counts = np.unique(samples, return_counts=True)
    return [[int(i), int(c)] for i, c in zip(indices, counts)]


def experiment():
    protocol = {'sizes': [8, 12, 16], 'instances_per_size': 10,
                'depths': [1, 2], 'reads': 1024, 'angle_candidates': 24,
                'sa_sweeps': 50, 'seed': 20261027}
    results = []
    for n in protocol['sizes']:
        for instance in range(protocol['instances_per_size']):
            seed = protocol['seed'] + n * 100 + instance
            weights, edges = make_instance(n, seed)
            start = time.perf_counter()
            energy = energy_table(weights, edges)
            exact_seconds = time.perf_counter() - start
            valid = valid_mask(n, edges)
            optimum_index = int(np.argmin(energy))
            if not valid[optimum_index]:
                raise AssertionError('Penalty encoding produced an invalid global optimum')
            inputs = {'weights': weights.tolist(), 'edges': edges}
            input_bytes = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
            row = {'n': n, 'instance': instance, 'seed': seed, **inputs,
                   'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
                   'exact': {'energy': int(energy[optimum_index]), 'index': optimum_index,
                             'assignments': len(energy), 'seconds': exact_seconds}, 'methods': {}}
            start = time.perf_counter()
            samples, terminal = anneal(weights, edges, seed=seed + 10000,
                                       reads=protocol['reads'], sweeps=protocol['sa_sweeps'])
            seconds = time.perf_counter() - start
            if not np.array_equal(terminal, energy[samples]):
                raise AssertionError('Annealing incremental energy drift')
            row['methods']['sa'] = {
                **sample_metrics(samples, energy, valid), 'seconds': seconds,
                'seed': seed + 10000, 'attempted_flips': protocol['reads'] * protocol['sa_sweeps'] * n,
                'histogram': _histogram(samples)}
            for depth in protocol['depths']:
                solver_seed = seed + 20000 + depth * 100
                start = time.perf_counter()
                samples, metadata = qaoa_samples(energy, depth, solver_seed,
                                                protocol['reads'], protocol['angle_candidates'])
                seconds = time.perf_counter() - start
                row['methods'][f'qaoa_p{depth}'] = {
                    **sample_metrics(samples, energy, valid), **metadata,
                    'seed': solver_seed, 'seconds': seconds, 'histogram': _histogram(samples)}
            results.append(row)
            print(json.dumps({'n': n, 'instance': instance, 'exact': row['exact']['energy'],
                              'gaps': {k: v['objective_gap'] for k, v in row['methods'].items()}}), flush=True)
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ('pilot.py', 'test_pilot.py', 'README.md')}
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
            'scope': 'Synthetic ideal-simulator feasibility only; no QPU, noise, held-out generalisation or quantum-advantage claim.',
            'budget_warning': 'Parameter search uses exact statevector expectations. SA and QAOA do not have equal compute budgets. Method times exclude shared energy-table construction, reported under exact.',
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'system': platform.system(), 'machine': platform.machine(),
                            'cpu_count': os.cpu_count(), 'thread_env': {k: os.environ.get(k) for k in
                                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
            'source_sha256': sources, 'instances': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; refusing to overwrite results')
    result = experiment()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'instances': len(result['instances'])}), flush=True)


if __name__ == '__main__':
    main()
