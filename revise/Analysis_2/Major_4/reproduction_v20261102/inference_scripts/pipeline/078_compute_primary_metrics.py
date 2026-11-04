#!/usr/bin/env python3
"""Compute the authorized 078 deterministic primary point metrics.

This script reads the separate 075 joined caches and never mutates them.  It
reports Full-clean and Strict-unseen populations separately and does not run
bootstrap or any later sensitivity unit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


CELLS = {
    "internvl3_2b_mmstar": {
        "model": "internvl3_2b",
        "dataset": "MMStar",
        "dataset_key": "mmstar",
        "n": 1500,
        "joined": "04_gt_join/internvl3_2b/mmstar",
    },
    "internvl3_2b_ai2d": {
        "model": "internvl3_2b",
        "dataset": "AI2D",
        "dataset_key": "ai2d",
        "n": 3088,
        "joined": "04_gt_join/internvl3_2b/ai2d",
    },
    "qwen2_5_vl_7b_mmstar": {
        "model": "qwen2_5_vl_7b",
        "dataset": "MMStar",
        "dataset_key": "mmstar",
        "n": 1500,
        "joined": "04_gt_join/qwen2_5_vl_7b/mmstar",
    },
    "qwen2_5_vl_7b_ai2d": {
        "model": "qwen2_5_vl_7b",
        "dataset": "AI2D",
        "dataset_key": "ai2d",
        "n": 3088,
        "joined": "04_gt_join/qwen2_5_vl_7b/ai2d",
    },
}

MUTABLE_FIELDS = {
    "correct",
}


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object row at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        value = indexed[position][1]
        while end < len(indexed) and indexed[end][1] == value:
            end += 1
        rank = (position + 1 + end) / 2.0
        for cursor in range(position, end):
            ranks[indexed[cursor][0]] = rank
        position = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return 1.0
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    mean_left = sum(left_rank) / len(left_rank)
    mean_right = sum(right_rank) / len(right_rank)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left_rank, right_rank))
    denom_left = sum((a - mean_left) ** 2 for a in left_rank)
    denom_right = sum((b - mean_right) ** 2 for b in right_rank)
    if denom_left == 0.0 or denom_right == 0.0:
        return 1.0 if left_rank == right_rank else 0.0
    return numerator / math.sqrt(denom_left * denom_right)


class Fenwick:
    def __init__(self, size: int) -> None:
        self.values = [0] * (size + 1)

    def add(self, index: int, amount: int = 1) -> None:
        while index < len(self.values):
            self.values[index] += amount
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self.values[index]
            index -= index & -index
        return total


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Tie-corrected Kendall tau-b without a SciPy dependency."""
    n = len(left)
    if n < 2:
        return 1.0
    n0 = n * (n - 1) // 2
    left_counts = Counter(left)
    right_counts = Counter(right)
    both_counts = Counter(zip(left, right))
    tie_left = sum(count * (count - 1) // 2 for count in left_counts.values())
    tie_right = sum(count * (count - 1) // 2 for count in right_counts.values())
    tie_both = sum(count * (count - 1) // 2 for count in both_counts.values())
    y_values = {value: index for index, value in enumerate(sorted(set(right)), 1)}
    ordered = sorted(zip(left, right), key=lambda pair: (pair[0], pair[1]))
    fenwick = Fenwick(len(y_values))
    inversions = 0
    processed = 0
    cursor = 0
    while cursor < n:
        end = cursor + 1
        x_value = ordered[cursor][0]
        while end < n and ordered[end][0] == x_value:
            end += 1
        # Within an x-tie group, y is already ascending, so these pairs are
        # excluded from inversions as required by tau-b.
        for _, y_value in ordered[cursor:end]:
            rank = y_values[y_value]
            inversions += processed - fenwick.prefix_sum(rank)
        for _, y_value in ordered[cursor:end]:
            fenwick.add(y_values[y_value])
            processed += 1
        cursor = end
    comparable = n0 - tie_left - tie_right + tie_both
    numerator = comparable - 2 * inversions
    denominator = math.sqrt((n0 - tie_left) * (n0 - tie_right))
    if denominator == 0.0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def auroc_from_scores(error: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(int(value) for value in error)
    negatives = len(error) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, error) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def joined_cache(root: Path, config: Dict[str, Any], backend: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base = root / "artifacts" / "eswa_revision_v1" / config["joined"]
    rows = read_jsonl(base / backend / "predictions_with_gt.jsonl")
    if len(rows) != config["n"]:
        raise RuntimeError(f"joined row count mismatch: {base}/{backend}")
    for index, row in enumerate(rows):
        if row.get("ground_truth") not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"invalid joined ground_truth at {base}/{backend}:{index + 1}")
        if row.get("correct") is not None:
            raise RuntimeError(f"joined cache already contains correctness at {base}/{backend}:{index + 1}")
        if row.get("GT_accessed") is not True or row.get("scientific_metrics_computed") is not False:
            raise RuntimeError(f"joined cache provenance flags mismatch: {base}/{backend}")
        if not isinstance(row.get("risk_primary"), (int, float)) or not finite(row["risk_primary"]):
            raise RuntimeError(f"invalid risk_primary at {base}/{backend}:{index + 1}")
    audit = read_json(base / "075_join_audit.json")
    if audit.get("status") != "PASS" or audit.get("metrics_computed") is not False:
        raise RuntimeError(f"075 audit unlock failed: {base}")
    return rows, audit


def assert_pair(rows_bf16: Sequence[Dict[str, Any]], rows_int4: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> None:
    if [row["sample_id"] for row in rows_bf16] != [row["sample_id"] for row in rows_int4]:
        raise RuntimeError(f"joined BF16/INT4 sample order mismatch: {config['dataset_key']}")
    if [row["source_row"] for row in rows_bf16] != [row["source_row"] for row in rows_int4]:
        raise RuntimeError(f"joined BF16/INT4 source_row mismatch: {config['dataset_key']}")
    if any(row["dataset_revision"] != other["dataset_revision"] for row, other in zip(rows_bf16, rows_int4)):
        raise RuntimeError(f"joined BF16/INT4 dataset_revision mismatch: {config['dataset_key']}")
    if any(row.get("ground_truth") != other.get("ground_truth") for row, other in zip(rows_bf16, rows_int4)):
        raise RuntimeError(f"joined BF16/INT4 ground_truth mismatch: {config['dataset_key']}")
    if any(row.get("provenance_status") != other.get("provenance_status") for row, other in zip(rows_bf16, rows_int4)):
        raise RuntimeError(f"joined BF16/INT4 provenance mismatch: {config['dataset_key']}")


def ranked_indices(risks: Sequence[float], source_rows: Sequence[int], descending: bool) -> List[int]:
    if descending:
        return sorted(range(len(risks)), key=lambda index: (-float(risks[index]), int(source_rows[index])))
    return sorted(range(len(risks)), key=lambda index: (float(risks[index]), int(source_rows[index])))


def jaccard(risk_bf16: Sequence[float], risk_int4: Sequence[float], source_rows: Sequence[int], quantile: float) -> float:
    k = max(1, math.ceil(quantile * len(source_rows)))
    first = set(ranked_indices(risk_bf16, source_rows, True)[:k])
    second = set(ranked_indices(risk_int4, source_rows, True)[:k])
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def aurc(risk: Sequence[float], error: Sequence[int], source_rows: Sequence[int]) -> float:
    order = ranked_indices(risk, source_rows, False)
    ordered_error = np.asarray(error, dtype=float)[order]
    return float(np.mean(np.cumsum(ordered_error) / np.arange(1, len(order) + 1)))


def rank_metric(value: Any) -> float | None:
    return None if value is None or not finite(value) else float(value)


def population_metrics(rows_bf16: Sequence[Dict[str, Any]], rows_int4: Sequence[Dict[str, Any]], population: str) -> Dict[str, Any]:
    if population == "full_clean":
        selected = [True] * len(rows_bf16)
    elif population == "strict_unseen":
        selected = [row.get("provenance_status") == "strict_unseen" for row in rows_bf16]
    else:
        raise ValueError(population)
    left = [row for row, keep in zip(rows_bf16, selected) if keep]
    right = [row for row, keep in zip(rows_int4, selected) if keep]
    if not left or len(left) != len(right):
        raise RuntimeError(f"invalid population size: {population}")
    source_rows = [int(row["source_row"]) for row in left]
    predictions_bf16 = np.asarray([str(row["normalized_prediction"]) for row in left], dtype=object)
    predictions_int4 = np.asarray([str(row["normalized_prediction"]) for row in right], dtype=object)
    ground_truth = np.asarray([str(row["ground_truth"]) for row in left], dtype=object)
    risk_bf16 = np.asarray([float(row["risk_primary"]) for row in left], dtype=float)
    risk_int4 = np.asarray([float(row["risk_primary"]) for row in right], dtype=float)
    correct_bf16 = predictions_bf16 == ground_truth
    correct_int4 = predictions_int4 == ground_truth
    error_bf16 = (~correct_bf16).astype(int)
    error_int4 = (~correct_int4).astype(int)
    auroc_bf16 = auroc_from_scores(error_bf16.tolist(), risk_bf16.tolist())
    auroc_int4 = auroc_from_scores(error_int4.tolist(), risk_int4.tolist())
    spearman = rank_metric(spearman_correlation(risk_bf16.tolist(), risk_int4.tolist()))
    kendall = rank_metric(kendall_tau_b(risk_bf16.tolist(), risk_int4.tolist()))
    aurc_bf16 = aurc(risk_bf16, error_bf16, source_rows)
    aurc_int4 = aurc(risk_int4, error_int4, source_rows)
    return {
        "population": population,
        "n": len(left),
        "bf16_accuracy": float(correct_bf16.mean()),
        "int4_accuracy": float(correct_int4.mean()),
        "delta_accuracy": float(correct_int4.mean() - correct_bf16.mean()),
        "answer_flip_rate": float(np.mean(predictions_bf16 != predictions_int4)),
        "spearman": spearman,
        "kendall_tau_b": kendall,
        "j5": jaccard(risk_bf16, risk_int4, source_rows, 0.05),
        "j10": jaccard(risk_bf16, risk_int4, source_rows, 0.10),
        "j20": jaccard(risk_bf16, risk_int4, source_rows, 0.20),
        "j30": jaccard(risk_bf16, risk_int4, source_rows, 0.30),
        "bf16_auroc": auroc_bf16,
        "int4_auroc": auroc_int4,
        "delta_auroc": None if auroc_bf16 is None or auroc_int4 is None else auroc_int4 - auroc_bf16,
        "bf16_aurc": aurc_bf16,
        "int4_aurc": aurc_int4,
        "delta_aurc": aurc_int4 - aurc_bf16,
        "correct_samples_bf16": int(correct_bf16.sum()),
        "incorrect_samples_bf16": int(error_bf16.sum()),
        "correct_samples_int4": int(correct_int4.sum()),
        "incorrect_samples_int4": int(error_int4.sum()),
        "metric_validity": {
            "correct_samples_bf16_gt_0": bool(correct_bf16.sum() > 0),
            "incorrect_samples_bf16_gt_0": bool(error_bf16.sum() > 0),
            "correct_samples_int4_gt_0": bool(correct_int4.sum() > 0),
            "incorrect_samples_int4_gt_0": bool(error_int4.sum() > 0),
            "bf16_auroc_defined": auroc_bf16 is not None,
            "int4_auroc_defined": auroc_int4 is not None,
            "bf16_aurc_defined": True,
            "int4_aurc_defined": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    g4 = read_json(root / "artifacts/eswa_revision_v1/04_noise_floor/077_noise_floor_v1_3.json")
    if g4.get("status") != "PASS" or g4.get("protocol_version") != "v1.3":
        raise RuntimeError("077 v1.3 is not a valid unlock")
    cells: Dict[str, Any] = {}
    for cell_name, config in CELLS.items():
        bf16, bf16_audit = joined_cache(root, config, "bf16")
        int4, int4_audit = joined_cache(root, config, "int4_g128")
        assert_pair(bf16, int4, config)
        cells[cell_name] = {
            "model": config["model"],
            "dataset": config["dataset"],
            "n_full_clean": config["n"],
            "joined_audit_status": {"bf16": bf16_audit["status"], "int4_g128": int4_audit["status"]},
            "full_clean": population_metrics(bf16, int4, "full_clean"),
            "strict_unseen": population_metrics(bf16, int4, "strict_unseen"),
        }
    artifact = {
        "stage": "078_PRIMARY_METRICS_V1_3",
        "status": "PASS",
        "protocol_version": "v1.3",
        "risk_definition": "risk_max = 1 - max(candidate_probabilities)",
        "auroc_positive_class": "incorrect/error = 1",
        "auroc_score": "risk_max",
        "aurc_discrete_definition": "sort by (risk ascending, source_row ascending); coverage k/n for k=1..n; selective risk is prefix errors/k; AURC is the mean of all prefix selective risks",
        "delta_convention": {
            "delta_accuracy": "INT4 - BF16",
            "delta_auroc": "INT4 - BF16",
            "delta_aurc": "INT4 - BF16",
        },
        "rank_rules": {
            "spearman": "average-rank tie handling",
            "kendall": "tau-b",
            "rank_order": "risk descending; canonical source_row ascending for exact ties",
            "jaccard_budgets": "ceil(q*n), q=0.05/0.10/0.20/0.30",
            "aurc_order": "risk ascending; canonical source_row ascending for exact ties",
        },
        "populations": {
            "full_clean": "all joined canonical rows",
            "strict_unseen": "provenance_status == strict_unseen",
        },
        "GT_accessed": True,
        "metrics_computed": True,
        "bootstrap_computed": False,
        "bootstrap_reserved_for": "083",
        "cells": cells,
    }
    output = root / "artifacts/eswa_revision_v1/05_primary_metrics/078_primary_metrics_v1_3_corrected.json"
    write_json(output, artifact)
    previous = root / "artifacts/eswa_revision_v1/05_primary_metrics/078_primary_metrics_v1_3.json"
    if not previous.exists():
        raise RuntimeError("superseded 078 artifact is missing; refusing corrected comparison")
    old = read_json(previous)
    unchanged = True
    for cell_name in CELLS:
        for population in ("full_clean", "strict_unseen"):
            old_metrics = old["cells"][cell_name][population]
            new_metrics = artifact["cells"][cell_name][population]
            for key, value in new_metrics.items():
                if key == "delta_accuracy":
                    continue
                if old_metrics.get(key) != value:
                    unchanged = False
    audit = {
        "stage": "078_METRIC_DEFINITION_AUDIT",
        "status": "PASS" if unchanged else "FAIL",
        "protocol_version": "v1.3",
        "corrected_artifact": str(output),
        "superseded_artifact_preserved_read_only": True,
        "delta_accuracy": "INT4 - BF16",
        "delta_auroc": "INT4 - BF16",
        "delta_aurc": "INT4 - BF16",
        "auroc_positive_class": "incorrect/error = 1",
        "auroc_score": "risk_max",
        "aurc_sort": "(risk ascending, canonical source_row ascending)",
        "aurc_discrete_definition": artifact["aurc_discrete_definition"],
        "spearman_ties": "average ranks",
        "kendall": "tau-b",
        "jaccard_tie_break": "(-risk, canonical source_row)",
        "full_clean_counts": {"MMStar": 1500, "AI2D": 3088},
        "strict_unseen_counts": {"MMStar": 1200, "AI2D": 2788},
        "all_other_point_metrics_unchanged": unchanged,
        "GT_accessed": True,
        "bootstrap_computed": False,
    }
    write_json(root / "artifacts/eswa_revision_v1/05_primary_metrics/078_metric_definition_audit.json", audit)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
