"""Points 3, 4 & 5: Bootstrap CIs, stopping rule vs random sampling, shot scaling, and re-budget."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

PIASTQ_MAX_SHOTS = 200
PIASTQ_MAX_RXX = 400


def paired_bootstrap_ci(
    diffs: Sequence[float] | np.ndarray,
    *,
    num_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 20261127,
) -> dict[str, float]:
    """Two-sided (1 - alpha) paired bootstrap confidence interval on instance-level differences."""
    arr = np.asarray(diffs, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("diffs must be a non-empty 1D array")
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(arr), size=(num_resamples, len(arr)))
    boot_means = np.mean(arr[boot_idx], axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    hi_one_sided_95 = float(np.quantile(boot_means, 1.0 - alpha))
    return {
        "mean_diff": float(np.mean(arr)),
        "ci95_low": lo,
        "ci95_high": hi,
        "ci95_one_sided_high": hi_one_sided_95,
    }


def paired_bootstrap_stopping_rule(
    noisy_metric: Sequence[float] | np.ndarray,
    random_metric: Sequence[float] | np.ndarray,
    *,
    lower_is_better: bool = True,
    num_resamples: int = 10000,
    seed: int = 20261127,
) -> dict[str, Any]:
    """Pre-registered stopping rule: noisy emulator must beat uniform random sampling with 95% CI < 0."""
    n_arr = np.asarray(noisy_metric, dtype=float)
    r_arr = np.asarray(random_metric, dtype=float)
    if n_arr.shape != r_arr.shape or n_arr.ndim != 1 or len(n_arr) == 0:
        raise ValueError("noisy_metric and random_metric must have identical non-empty 1D shape")

    diffs = (n_arr - r_arr) if lower_is_better else (r_arr - n_arr)
    ci = paired_bootstrap_ci(diffs, num_resamples=num_resamples, alpha=0.05, seed=seed)
    # Require two-sided 95% CI upper bound below zero
    advances = bool(ci["ci95_high"] < 0.0)
    return {
        **ci,
        "num_instances": int(len(n_arr)),
        "noisy_mean": float(np.mean(n_arr)),
        "random_mean": float(np.mean(r_arr)),
        "advances": advances,
    }


def validate_rebudget_plan(
    blocks: Sequence[dict[str, Any]],
    *,
    seconds_per_submission: float = 18.5,
    max_rxx: int = PIASTQ_MAX_RXX,
    max_shots_per_submission: int = PIASTQ_MAX_SHOTS,
) -> dict[str, Any]:
    """Validate Point 5 re-budget plan (`instances x depths x seeds x shots x submissions`)."""
    if seconds_per_submission < 18.5:
        raise ValueError("PSNC planning figure requires at least 18.5 s per 200-shot submission")

    total_circuits = 0
    total_submissions = 0
    total_shots = 0
    validated_blocks: list[dict[str, Any]] = []

    for blk in blocks:
        instances = int(blk["instances"])
        depths = int(blk["depths"])
        seeds = int(blk["seeds"])
        shots = int(blk["shots_per_circuit"])
        subs_per_circuit = int(blk["submissions_per_circuit"])
        blk_rxx = int(blk["max_rxx"])
        advances = bool(blk.get("advances", True))

        if not advances:
            raise ValueError(f"Block '{blk['block']}' failed stopping rule and cannot advance to QPU")
        if blk_rxx > max_rxx:
            raise ValueError(f"Block '{blk['block']}' max_rxx={blk_rxx} exceeds {max_rxx}")
        expected_subs = (shots + max_shots_per_submission - 1) // max_shots_per_submission
        if subs_per_circuit != expected_subs or shots > subs_per_circuit * max_shots_per_submission:
            raise ValueError(
                f"Block '{blk['block']}' shots={shots} inconsistent with {subs_per_circuit} x <=200 submissions"
            )

        n_circuits = instances * depths * seeds
        n_subs = n_circuits * subs_per_circuit
        n_shots = n_circuits * shots
        sec_floor = n_subs * seconds_per_submission
        sec_est = n_subs * float(blk.get("estimated_seconds_per_submission", seconds_per_submission))

        total_circuits += n_circuits
        total_submissions += n_subs
        total_shots += n_shots
        validated_blocks.append(
            {
                **blk,
                "distinct_circuits": n_circuits,
                "total_submissions": n_subs,
                "total_shots": n_shots,
                "seconds_at_18_5s_floor": round(sec_floor, 1),
                "hours_at_18_5s_floor": round(sec_floor / 3600.0, 4),
                "seconds_depth_adjusted": round(sec_est, 1),
                "hours_depth_adjusted": round(sec_est / 3600.0, 4),
            }
        )

    total_sec_floor = total_submissions * seconds_per_submission
    total_sec_adj = sum(b["seconds_depth_adjusted"] for b in validated_blocks)
    return {
        "seconds_per_submission_floor": seconds_per_submission,
        "total_distinct_circuits": total_circuits,
        "total_submissions": total_submissions,
        "total_shots": total_shots,
        "total_seconds_floor": round(total_sec_floor, 1),
        "total_hours_floor": round(total_sec_floor / 3600.0, 4),
        "total_seconds_depth_adjusted": round(total_sec_adj, 1),
        "total_hours_depth_adjusted": round(total_sec_adj / 3600.0, 4),
        "blocks": validated_blocks,
    }


def build_piastq_25h_rebudget() -> dict[str, Any]:
    """Return the validated 25.0 QPU-hour (4,860 x 200-shot submissions) PIAST-Q execution plan."""
    blocks = [
        {
            "block": "A2_dev_2body_N12_dense_d050_070_085_100_p2_3",
            "tier": "development_euroqexa_parity",
            "n_values": [12],
            "densities": [0.50, 0.70, 0.85, 1.00],
            "p_values": [2, 3],
            "instances": 16,
            "depths": 2,
            "seeds": 3,
            "shots_per_circuit": 1000,
            "submissions_per_circuit": 5,
            "max_rxx": 198,
            "estimated_seconds_per_submission": 19.5,
            "advances": True,
            "rationale": "Matches Euro-Q-Exa 1,024-shot dev statistics (5x200=1,000) across Gap_CI95<0 cells (N=8 pruned by stopping rule).",
        },
        {
            "block": "A3_dev_2body_N16_dense_d050_065_085_100_p1_2_3",
            "tier": "development_euroqexa_parity",
            "n_values": [16],
            "densities": [0.50, 0.65, 0.85, 1.00],
            "p_values": [1, 2, 3],
            "instances": 12,
            "depths": 3,
            "seeds": 3,
            "shots_per_circuit": 1000,
            "submissions_per_circuit": 5,
            "max_rxx": 360,
            "estimated_seconds_per_submission": 21.0,
            "advances": True,
            "rationale": "Full p=1..3 depth sweep at N=16 (<=360 RXX) at 1,000-shot Euro-Q-Exa dev parity.",
        },
        {
            "block": "A4a_dev_2body_N20_dense_d050_065_p1_2_3",
            "tier": "development_euroqexa_parity",
            "n_values": [20],
            "densities": [0.50, 0.65],
            "p_values": [1, 2, 3],
            "instances": 6,
            "depths": 3,
            "seeds": 3,
            "shots_per_circuit": 1000,
            "submissions_per_circuit": 5,
            "max_rxx": 372,
            "estimated_seconds_per_submission": 21.5,
            "advances": True,
            "rationale": "N=20 with density <0.70 permits p=1..3 up to 372 RXX <= 400 with strong signal over random.",
        },
        {
            "block": "A4b_dev_2body_N20_dense_d085_100_p1_2",
            "tier": "development_euroqexa_parity",
            "n_values": [20],
            "densities": [0.85, 1.00],
            "p_values": [1, 2],
            "instances": 6,
            "depths": 2,
            "seeds": 3,
            "shots_per_circuit": 1000,
            "submissions_per_circuit": 5,
            "max_rxx": 380,
            "estimated_seconds_per_submission": 22.5,
            "advances": True,
            "rationale": "N=20 at density 0.85-1.00 pruned at p=2 (324-380 RXX; p=3 at 486-570 RXX excluded).",
        },
        {
            "block": "B1_heldout_2body_N12_d065_100_p1_2_vlq_parity",
            "tier": "heldout_vlq_parity",
            "scalable": True,
            "n_values": [12],
            "densities": [0.65, 1.00],
            "p_values": [1, 2],
            "instances": 10,
            "depths": 2,
            "seeds": 3,
            "shots_per_circuit": 2200,
            "submissions_per_circuit": 11,
            "max_rxx": 132,
            "estimated_seconds_per_submission": 19.5,
            "advances": True,
            "rationale": "Held-out evaluation at 2,200 shots (11x200) matching VLQ 2,250-shot statistics (87.5-100% solved).",
        },
        {
            "block": "B2_heldout_2body_N16_d065_100_p2_3_vlq_parity",
            "tier": "heldout_vlq_parity",
            "scalable": True,
            "n_values": [16],
            "densities": [0.65, 1.00],
            "p_values": [2, 3],
            "instances": 10,
            "depths": 2,
            "seeds": 3,
            "shots_per_circuit": 2200,
            "submissions_per_circuit": 11,
            "max_rxx": 360,
            "estimated_seconds_per_submission": 21.5,
            "advances": True,
            "rationale": "Held-out N=16 at 2,200 shots (11x200), where noisy emulator solved rate reaches 87.5% at d=1.0.",
        },
        {
            "block": "B3_heldout_2body_N20_d065_100_p1_2_vlq_parity",
            "tier": "heldout_vlq_parity",
            "scalable": True,
            "n_values": [20],
            "densities": [0.65, 1.00],
            "p_values": [1, 2],
            "instances": 10,
            "depths": 2,
            "seeds": 3,
            "shots_per_circuit": 2200,
            "submissions_per_circuit": 11,
            "max_rxx": 380,
            "estimated_seconds_per_submission": 22.5,
            "advances": True,
            "rationale": "Held-out N=20 at 2,200 shots (11x200), required to resolve 50% exact optima on noisy emulator.",
        },
        {
            "block": "C1_final_heldout_2body_N16_N20_d065_100_p2_4000shots",
            "tier": "final_heldout_superconducting_parity",
            "scalable": True,
            "n_values": [16, 20],
            "densities": [0.65, 1.00],
            "p_values": [2],
            "instances": 16,
            "depths": 1,
            "seeds": 3,
            "shots_per_circuit": 4000,
            "submissions_per_circuit": 20,
            "max_rxx": 380,
            "estimated_seconds_per_submission": 22.5,
            "advances": True,
            "rationale": "Final held-out comparison at 4,000 shots (20x200) matching Euro-Q-Exa & VLQ 4,096-shot final runs.",
        },
        {
            "block": "D1_heldout_3body_ZZZ_N20_d030_triples12_p1_2",
            "tier": "heldout_3body_extension",
            "n_values": [20],
            "densities": [0.30],
            "num_triples": 12,
            "p_values": [1, 2],
            "instances": 6,
            "depths": 2,
            "seeds": 3,
            "shots_per_circuit": 2200,
            "submissions_per_circuit": 11,
            "max_rxx": 258,
            "estimated_seconds_per_submission": 22.0,
            "advances": True,
            "rationale": "Pruned ancilla-free 3-body ZZZ HUBO (<=258 RXX across dev seeds, <=252 RXX across gate-map seeds, <=631 wire ops, Gap_CI95<0).",
        },
        {
            "block": "E1_session_readout_and_rxx_drift_references",
            "tier": "calibration_and_drift",
            "n_values": [20],
            "instances": 18,
            "depths": 1,
            "seeds": 3,
            "shots_per_circuit": 200,
            "submissions_per_circuit": 1,
            "max_rxx": 190,
            "estimated_seconds_per_submission": 18.5,
            "advances": True,
            "rationale": "Operational session overhead (6 sessions x 3 calibration reference circuits x 3 repeats): readout |0>^20 (0 RXX), readout |1>^20 (0 RXX), and 95-pair RXX mirror identity (190 RXX).",
        },
    ]
    summary = validate_rebudget_plan(blocks, seconds_per_submission=18.5)
    summary["requested_qpu_hours"] = 25.0
    summary["rescaling_rule"] = (
        "Plan-level deterministic rescaling: for B_max = 90,000 s (25.0 QPU hours) and Session-1 mean rate "
        "tau_obs >= 18.5 s per 200-shot submission, N_cap = floor(B_max / tau_obs). Fixed blocks "
        "(E1 -> A2..A4b -> D1) are allocated first in whole instances, and remaining capacity max(0, N_cap - N_fixed) "
        "is allocated to whole instance packages in B1, B2, B3 (66 submissions/instance) and C1 "
        "(60 submissions/instance) in locked seed order. Client-side execute_single_submission enforces "
        "QPUBudgetGuard pre-call reservation (blocking backend.run if remaining_seconds < max_submission_seconds); "
        "server-side QPU accounting and per-job hardware timeout configuration with PSNC remain pre-QPU operational steps."
    )
    return summary


def rescale_rebudget_to_observed_rate(
    tau_obs_seconds: float,
    *,
    budget_seconds: float = 90000.0,
) -> dict[str, Any]:
    """Deterministically select whole instance packages so total time does not exceed budget_seconds."""
    if tau_obs_seconds < 18.5:
        raise ValueError("tau_obs_seconds must be >= 18.5 s")
    base = build_piastq_25h_rebudget()
    n_cap = int(budget_seconds // tau_obs_seconds)

    # Priority order: E1 (calibration), then A2..A4b (dev), then D1 (3-body), then scalable B1..B3 & C1
    priority_fixed = [
        "E1_session_readout_and_rxx_drift_references",
        "A2_dev_2body_N12_dense_d050_070_085_100_p2_3",
        "A3_dev_2body_N16_dense_d050_065_085_100_p1_2_3",
        "A4a_dev_2body_N20_dense_d050_065_p1_2_3",
        "A4b_dev_2body_N20_dense_d085_100_p1_2",
        "D1_heldout_3body_ZZZ_N20_d030_triples12_p1_2",
    ]
    scalable_names = [
        "B1_heldout_2body_N12_d065_100_p1_2_vlq_parity",
        "B2_heldout_2body_N16_d065_100_p2_3_vlq_parity",
        "B3_heldout_2body_N20_d065_100_p1_2_vlq_parity",
        "C1_final_heldout_2body_N16_N20_d065_100_p2_4000shots",
    ]
    blk_map = {b["block"]: dict(b) for b in base["blocks"]}

    used_subs = 0
    allocated_instances: dict[str, int] = {k: 0 for k in blk_map}

    # 1. Allocate fixed blocks in priority order, whole instances only
    for name in priority_fixed:
        b = blk_map[name]
        subs_per_inst = int(b["depths"]) * int(b["seeds"]) * int(b["submissions_per_circuit"])
        max_inst = int(b["instances"])
        fit = min(max_inst, (n_cap - used_subs) // subs_per_inst)
        allocated_instances[name] = int(fit)
        used_subs += int(fit) * subs_per_inst

    # 2. Allocate remaining capacity proportionally across scalable held-out blocks (whole instances)
    rem_subs = n_cap - used_subs
    base_scalable_subs = sum(int(blk_map[k]["total_submissions"]) for k in scalable_names)
    if rem_subs >= base_scalable_subs:
        for k in scalable_names:
            allocated_instances[k] = int(blk_map[k]["instances"])
        used_subs += base_scalable_subs
    elif rem_subs > 0:
        frac = rem_subs / float(base_scalable_subs)
        for k in scalable_names:
            b = blk_map[k]
            subs_per_inst = int(b["depths"]) * int(b["seeds"]) * int(b["submissions_per_circuit"])
            target_inst = min(int(b["instances"]), int(b["instances"] * frac))
            fit = min(target_inst, (n_cap - used_subs) // subs_per_inst)
            allocated_instances[k] = int(fit)
            used_subs += int(fit) * subs_per_inst
        # Greedily top up one instance at a time in round-robin order B1 -> B2 -> B3 -> C1 while capacity remains
        made_progress = True
        while made_progress:
            made_progress = False
            for k in scalable_names:
                b = blk_map[k]
                subs_per_inst = int(b["depths"]) * int(b["seeds"]) * int(b["submissions_per_circuit"])
                if allocated_instances[k] < int(b["instances"]) and (used_subs + subs_per_inst) <= n_cap:
                    allocated_instances[k] += 1
                    used_subs += subs_per_inst
                    made_progress = True

    total_seconds = used_subs * tau_obs_seconds
    return {
        "tau_obs_seconds": tau_obs_seconds,
        "budget_seconds": budget_seconds,
        "max_allowed_submissions": n_cap,
        "allocated_submissions": used_subs,
        "allocated_shots": used_subs * PIASTQ_MAX_SHOTS,
        "total_seconds": round(total_seconds, 2),
        "total_hours": round(total_seconds / 3600.0, 4),
        "within_budget": bool(total_seconds <= budget_seconds + 1e-9),
        "allocated_instances": allocated_instances,
    }


class QPUBudgetGuard:
    """Client-side pre-submission QPU budget guard reserving an upper-bound duration before backend.run."""

    def __init__(self, budget_seconds: float = 90000.0, max_submission_seconds: float = 120.0) -> None:
        if budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        if max_submission_seconds <= 0:
            raise ValueError("max_submission_seconds must be positive")
        self.budget_seconds = float(budget_seconds)
        self.max_submission_seconds = float(max_submission_seconds)
        self.consumed_seconds = 0.0
        self.completed_submissions = 0

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget_seconds - self.consumed_seconds)

    def check_can_submit(self, expected_submission_seconds: float | None = None) -> None:
        reserve = self.max_submission_seconds if expected_submission_seconds is None else float(expected_submission_seconds)
        if reserve <= 0:
            raise ValueError("expected_submission_seconds must be positive")
        if self.consumed_seconds + reserve > self.budget_seconds:
            raise RuntimeError(
                f"QPU budget guard blocked submission: consumed={self.consumed_seconds:.1f}s + "
                f"reserved={reserve:.1f}s > cap={self.budget_seconds:.1f}s"
            )

    def record_submission(self, billed_seconds: float) -> None:
        if billed_seconds < 0:
            raise ValueError("billed_seconds must be non-negative")
        self.consumed_seconds += float(billed_seconds)
        self.completed_submissions += 1

