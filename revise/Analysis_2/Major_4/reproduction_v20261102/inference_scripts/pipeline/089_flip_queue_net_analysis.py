#!/usr/bin/env python3
"""Post-hoc paired flip-direction and queue-net-benefit analysis.

Reads immutable 075 GT-joined caches.  It does not run inference, mutate caches,
or calculate hashes.  Queue benefit is always evaluated against INT4 errors so
BF16- and INT4-ranked queues share the same outcome denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


CELLS = {
    "internvl3_2b_mmstar": {"model": "internvl3_2b", "dataset": "MMStar", "n": 1500, "joined": "04_gt_join/internvl3_2b/mmstar"},
    "internvl3_2b_ai2d": {"model": "internvl3_2b", "dataset": "AI2D", "n": 3088, "joined": "04_gt_join/internvl3_2b/ai2d"},
    "qwen2_5_vl_7b_mmstar": {"model": "qwen2_5_vl_7b", "dataset": "MMStar", "n": 1500, "joined": "04_gt_join/qwen2_5_vl_7b/mmstar"},
    "qwen2_5_vl_7b_ai2d": {"model": "qwen2_5_vl_7b", "dataset": "AI2D", "n": 3088, "joined": "04_gt_join/qwen2_5_vl_7b/ai2d"},
}
POPULATIONS = ("full_clean", "strict_unseen")
BUDGETS = (0.05, 0.10, 0.20, 0.30)
FLIP_CATEGORIES = (
    "correct_to_incorrect",
    "incorrect_to_correct",
    "incorrect_to_incorrect_changed_answer",
)
ALL_CATEGORIES = FLIP_CATEGORIES + (
    "stable_correct",
    "stable_incorrect_same_answer",
)


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object row: {path}:{line_no}")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    os.replace(temporary, path)


def load_pair(root: Path, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base = root / "artifacts/eswa_revision_v1" / config["joined"]
    bf16 = read_jsonl(base / "bf16/predictions_with_gt.jsonl")
    int4 = read_jsonl(base / "int4_g128/predictions_with_gt.jsonl")
    if len(bf16) != config["n"] or len(int4) != config["n"]:
        raise RuntimeError(f"joined row count mismatch: {base}")
    fields = ("sample_id", "source_row", "dataset_revision", "ground_truth", "provenance_status")
    for field in fields:
        if [r.get(field) for r in bf16] != [r.get(field) for r in int4]:
            raise RuntimeError(f"pair mismatch for {field}: {base}")
    for rows in (bf16, int4):
        for row in rows:
            if row.get("ground_truth") not in {"A", "B", "C", "D"}:
                raise RuntimeError(f"invalid ground truth: {base}")
            if row.get("normalized_prediction") not in {"A", "B", "C", "D"}:
                raise RuntimeError(f"invalid normalized prediction: {base}")
            if row.get("GT_accessed") is not True or row.get("scientific_metrics_computed") is not False:
                raise RuntimeError(f"joined provenance flags mismatch: {base}")
            if not math.isfinite(float(row.get("risk_primary"))):
                raise RuntimeError(f"invalid primary risk: {base}")
    audit = read_json(base / "075_join_audit.json")
    if audit.get("status") != "PASS" or audit.get("metrics_computed") is not False:
        raise RuntimeError(f"075 join audit failed: {base}")
    return bf16, int4


def select_population(left, right, population):
    keep = np.array([population == "full_clean" or r.get("provenance_status") == "strict_unseen" for r in left])
    return [r for r, k in zip(left, keep) if k], [r for r, k in zip(right, keep) if k]


def percentile(values: np.ndarray) -> List[float]:
    lo, hi = np.percentile(values[np.isfinite(values)], [2.5, 97.5])
    return [float(lo), float(hi)]


def rank_order(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.array(sorted(range(len(rows)), key=lambda i: (-float(rows[i]["risk_primary"]), int(rows[i]["source_row"]))), dtype=np.int64)


def point_analysis(left, right):
    n = len(left)
    pred_b = np.array([r["normalized_prediction"] for r in left])
    pred_i = np.array([r["normalized_prediction"] for r in right])
    gt = np.array([r["ground_truth"] for r in left])
    correct_b = pred_b == gt
    correct_i = pred_i == gt
    changed = pred_b != pred_i
    masks = {
        "correct_to_incorrect": correct_b & ~correct_i,
        "incorrect_to_correct": ~correct_b & correct_i,
        "incorrect_to_incorrect_changed_answer": ~correct_b & ~correct_i & changed,
        "stable_correct": correct_b & correct_i,
        "stable_incorrect_same_answer": ~correct_b & ~correct_i & ~changed,
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    flip_count = sum(counts[name] for name in FLIP_CATEGORIES)
    flip = {
        "n": n,
        "counts": counts,
        "rates": {name: counts[name] / n for name in ALL_CATEGORIES},
        "answer_flip_count": flip_count,
        "answer_flip_rate": flip_count / n,
        "among_flips_fraction": {name: counts[name] / flip_count for name in FLIP_CATEGORIES},
        "net_correct_change_count": counts["incorrect_to_correct"] - counts["correct_to_incorrect"],
        "delta_accuracy_identity": (counts["incorrect_to_correct"] - counts["correct_to_incorrect"]) / n,
    }
    order_b, order_i = rank_order(left), rank_order(right)
    int4_error = ~correct_i
    queue = {}
    for q in BUDGETS:
        k = max(1, math.ceil(q * n))
        qb, qi = set(order_b[:k].tolist()), set(order_i[:k].tolist())
        shared, exited, entered = qb & qi, qb - qi, qi - qb
        e_shared = int(int4_error[list(shared)].sum()) if shared else 0
        e_exited = int(int4_error[list(exited)].sum()) if exited else 0
        e_entered = int(int4_error[list(entered)].sum()) if entered else 0
        selected_b, selected_i = e_shared + e_exited, e_shared + e_entered
        total_errors = int(int4_error.sum())
        net = selected_i - selected_b
        queue[str(q)] = {
            "q": q,
            "k": k,
            "shared_slots": len(shared),
            "exited_slots": len(exited),
            "entered_slots": len(entered),
            "int4_errors_in_shared_slots": e_shared,
            "int4_errors_in_exited_slots": e_exited,
            "int4_errors_in_entered_slots": e_entered,
            "int4_errors_selected_by_bf16_queue": selected_b,
            "int4_errors_selected_by_int4_queue": selected_i,
            "total_int4_errors": total_errors,
            "net_additional_int4_errors_found": net,
            "net_errors_per_100_review_slots": 100.0 * net / k,
            "bf16_queue_error_precision": selected_b / k,
            "int4_queue_error_precision": selected_i / k,
            "precision_delta": net / k,
            "bf16_queue_error_recall": selected_b / total_errors,
            "int4_queue_error_recall": selected_i / total_errors,
            "recall_delta": net / total_errors,
        }
    arrays = {name: mask.astype(np.int8) for name, mask in masks.items()}
    return flip, queue, arrays, int4_error.astype(np.int8), order_b, order_i


def bootstrap(arrays, int4_error, order_b, order_i, budgets, B, seed, batch=250):
    n = len(int4_error)
    rng = np.random.default_rng(seed)
    names = list(ALL_CATEGORIES)
    flip_rates = {name: np.empty(B, dtype=float) for name in names}
    delta_accuracy = np.empty(B, dtype=float)
    qstats = {str(q): {name: np.empty(B, dtype=float) for name in ("net", "per100", "precision_delta", "recall_delta")} for q in budgets}
    probability = np.full(n, 1.0 / n)
    cursor = 0
    while cursor < B:
        take = min(batch, B - cursor)
        counts = rng.multinomial(n, probability, size=take)
        for name in names:
            flip_rates[name][cursor:cursor + take] = counts @ arrays[name] / n
        delta_accuracy[cursor:cursor + take] = flip_rates["incorrect_to_correct"][cursor:cursor + take] - flip_rates["correct_to_incorrect"][cursor:cursor + take]
        total_errors = counts @ int4_error
        for order, label in ((order_b, "bf16"), (order_i, "int4")):
            ordered_counts = counts[:, order]
            previous = np.cumsum(ordered_counts, axis=1) - ordered_counts
            ordered_error = int4_error[order]
            for q in budgets:
                k = max(1, math.ceil(q * n))
                selected_occurrences = np.minimum(ordered_counts, np.maximum(k - previous, 0))
                selected_errors = selected_occurrences @ ordered_error
                qstats[str(q)][label] = selected_errors
        for q in budgets:
            k = max(1, math.ceil(q * n))
            net = qstats[str(q)]["int4"] - qstats[str(q)]["bf16"]
            qstats[str(q)]["net"][cursor:cursor + take] = net
            qstats[str(q)]["per100"][cursor:cursor + take] = 100.0 * net / k
            qstats[str(q)]["precision_delta"][cursor:cursor + take] = net / k
            qstats[str(q)]["recall_delta"][cursor:cursor + take] = np.divide(net, total_errors, out=np.full(take, np.nan), where=total_errors > 0)
        cursor += take
    return {
        "B": B,
        "seed": seed,
        "resampling_unit": "paired sample row; shared multinomial counts for both backend queues",
        "queue_bootstrap_method": "sample-with-replacement occurrence counts; each backend order is reconstituted from its frozen rank order and top-k is applied to occurrences",
        "flip_rate_95_percentile_intervals": {name: percentile(values) for name, values in flip_rates.items()},
        "delta_accuracy_95_percentile_interval": percentile(delta_accuracy),
        "queue_95_percentile_intervals": {
            q: {
                "net_additional_int4_errors_found": percentile(values["net"]),
                "net_errors_per_100_review_slots": percentile(values["per100"]),
                "precision_delta": percentile(values["precision_delta"]),
                "recall_delta": percentile(values["recall_delta"]),
            }
            for q, values in qstats.items()
        },
    }


def cross_artifact_gate(root: Path, cells: Dict[str, Any]) -> Dict[str, Any]:
    p078 = read_json(root / "artifacts/eswa_revision_v1/05_primary_metrics/078_primary_metrics_v1_3_corrected.json")
    p081 = read_json(root / "artifacts/eswa_revision_v1/08_queue_utility/081_queue_metrics_v1_3_corrected.json")
    checks = []
    for cell in CELLS:
        for pop in POPULATIONS:
            ours = cells[cell][pop]
            ref78 = p078["cells"][cell][pop]
            checks.extend([
                math.isclose(ours["flip_direction"]["answer_flip_rate"], ref78["answer_flip_rate"], rel_tol=0.0, abs_tol=1e-12),
                math.isclose(ours["flip_direction"]["delta_accuracy_identity"], ref78["delta_accuracy"], rel_tol=0.0, abs_tol=1e-12),
            ])
            for q in BUDGETS:
                ours_q = ours["queue_net_benefit"][str(q)]
                ref_q = p081["cells"][cell][pop]["queue_budgets"][str(q)]
                checks.extend([
                    ours_q["k"] == ref_q["k"],
                    ours_q["shared_slots"] == ref_q["shared_slots"],
                    ours_q["int4_errors_selected_by_bf16_queue"] == ref_q["int4_errors_selected_by_bf16_queue"],
                    ours_q["int4_errors_selected_by_int4_queue"] == ref_q["int4_errors_selected_by_int4_queue"],
                ])
    return {"all_exact": all(checks), "checks_passed": sum(checks), "checks_total": len(checks)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    cells = {}
    summary_rows = []
    for cell_index, (cell, config) in enumerate(CELLS.items()):
        left_all, right_all = load_pair(args.root, config)
        cells[cell] = {}
        for pop_index, population in enumerate(POPULATIONS):
            left, right = select_population(left_all, right_all, population)
            flip, queue, arrays, int4_error, order_b, order_i = point_analysis(left, right)
            boot = bootstrap(arrays, int4_error, order_b, order_i, BUDGETS, args.bootstrap, args.seed + cell_index * 10 + pop_index)
            cells[cell][population] = {
                "model": config["model"],
                "dataset": config["dataset"],
                "population": population,
                "n": len(left),
                "flip_direction": flip,
                "queue_net_benefit": queue,
                "paired_bootstrap": boot,
            }
            for q in BUDGETS:
                v = queue[str(q)]
                ci = boot["queue_95_percentile_intervals"][str(q)]
                summary_rows.append([
                    cell, population, len(left), q, v["k"], flip["counts"]["correct_to_incorrect"],
                    flip["counts"]["incorrect_to_correct"], flip["counts"]["incorrect_to_incorrect_changed_answer"],
                    v["exited_slots"], v["entered_slots"], v["int4_errors_in_exited_slots"],
                    v["int4_errors_in_entered_slots"], v["net_additional_int4_errors_found"],
                    *ci["net_additional_int4_errors_found"], v["net_errors_per_100_review_slots"],
                    *ci["net_errors_per_100_review_slots"], v["recall_delta"], *ci["recall_delta"],
                ])
    gate = cross_artifact_gate(args.root, cells)
    if not gate["all_exact"]:
        raise RuntimeError(f"canonical cross-artifact gate failed: {gate}")
    artifact = {
        "stage": "POST_HOC_FLIP_DIRECTION_AND_QUEUE_NET_BENEFIT",
        "protocol_version": "vlm_eswa_revision_v1_3_089",
        "status": "PASS",
        "GT_accessed": True,
        "scientific_metrics_computed": True,
        "inference_performed": False,
        "hashes_computed": False,
        "analysis_status": "post_hoc_descriptive_with_paired_percentile_intervals",
        "delta_convention": "INT4 minus BF16",
        "queue_outcome_definition": "INT4 prediction error for both BF16-risk and INT4-risk queues",
        "budgets": list(BUDGETS),
        "cross_artifact_gate": gate,
        "cells": cells,
    }
    out = args.root / "artifacts/eswa_revision_v1/12_flip_queue_net"
    atomic_json(out / "089_flip_queue_net_v1.json", artifact)
    atomic_csv(out / "089_flip_queue_net_summary.csv", [
        "cell", "population", "n", "q", "k", "correct_to_incorrect", "incorrect_to_correct",
        "incorrect_to_incorrect_changed_answer", "exited_slots", "entered_slots", "errors_in_exited_slots",
        "errors_in_entered_slots", "net_additional_errors", "net_errors_ci_low", "net_errors_ci_high",
        "net_errors_per_100_slots", "per100_ci_low", "per100_ci_high", "recall_delta", "recall_delta_ci_low",
        "recall_delta_ci_high",
    ], summary_rows)
    atomic_json(out / "089_validity_audit.json", {
        "status": "PASS", "row_pairing": "exact", "cross_artifact_gate": gate,
        "bootstrap_B": args.bootstrap, "hashes_computed": False,
    })
    print(json.dumps({"status": "PASS", "output_dir": str(out), "cross_artifact_gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
