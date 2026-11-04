#!/usr/bin/env python3
"""Compute the authorized 081 operational queue metrics.

This unit is GT-dependent by design: it uses the canonical 075 joined caches
to measure queue overlap and recall of INT4 errors.  It does not bootstrap,
mutate joined caches, or run any later unit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


CELLS = {
    "internvl3_2b_mmstar": {"model": "internvl3_2b", "dataset": "MMStar", "dataset_key": "mmstar", "n": 1500, "joined": "04_gt_join/internvl3_2b/mmstar"},
    "internvl3_2b_ai2d": {"model": "internvl3_2b", "dataset": "AI2D", "dataset_key": "ai2d", "n": 3088, "joined": "04_gt_join/internvl3_2b/ai2d"},
    "qwen2_5_vl_7b_mmstar": {"model": "qwen2_5_vl_7b", "dataset": "MMStar", "dataset_key": "mmstar", "n": 1500, "joined": "04_gt_join/qwen2_5_vl_7b/mmstar"},
    "qwen2_5_vl_7b_ai2d": {"model": "qwen2_5_vl_7b", "dataset": "AI2D", "dataset_key": "ai2d", "n": 3088, "joined": "04_gt_join/qwen2_5_vl_7b/ai2d"},
}
BUDGETS = (0.05, 0.10, 0.20, 0.30)


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


def load_pair(root: Path, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    base = root / "artifacts/eswa_revision_v1" / config["joined"]
    bf16 = read_jsonl(base / "bf16/predictions_with_gt.jsonl")
    int4 = read_jsonl(base / "int4_g128/predictions_with_gt.jsonl")
    if len(bf16) != config["n"] or len(int4) != config["n"]:
        raise RuntimeError(f"joined row count mismatch: {base}")
    for rows in (bf16, int4):
        for index, row in enumerate(rows):
            if row.get("ground_truth") not in {"A", "B", "C", "D"}:
                raise RuntimeError(f"invalid ground_truth at {base}:{index + 1}")
            if row.get("GT_accessed") is not True or row.get("scientific_metrics_computed") is not False:
                raise RuntimeError(f"joined provenance flags mismatch: {base}")
            if not finite(row.get("risk_primary")):
                raise RuntimeError(f"invalid risk_primary at {base}:{index + 1}")
    if [row["sample_id"] for row in bf16] != [row["sample_id"] for row in int4]:
        raise RuntimeError(f"sample_id order mismatch: {base}")
    if [row["source_row"] for row in bf16] != [row["source_row"] for row in int4]:
        raise RuntimeError(f"source_row order mismatch: {base}")
    if [row.get("provenance_status") for row in bf16] != [row.get("provenance_status") for row in int4]:
        raise RuntimeError(f"provenance membership mismatch: {base}")
    audit = read_json(base / "075_join_audit.json")
    if audit.get("status") != "PASS" or audit.get("metrics_computed") is not False:
        raise RuntimeError(f"075 join audit unlock failed: {base}")
    return bf16, int4, audit


def population_rows(rows_bf16: Sequence[Dict[str, Any]], rows_int4: Sequence[Dict[str, Any]], population: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    keep = [population == "full_clean" or row.get("provenance_status") == "strict_unseen" for row in rows_bf16]
    left = [row for row, include in zip(rows_bf16, keep) if include]
    right = [row for row, include in zip(rows_int4, keep) if include]
    if not left or len(left) != len(right):
        raise RuntimeError(f"invalid population size: {population}")
    return left, right


def rank_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    return sorted(range(len(rows)), key=lambda index: (-float(rows[index]["risk_primary"]), int(rows[index]["source_row"])))


def queue_result(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]], quantile: float) -> Dict[str, Any]:
    n = len(left)
    k = max(1, math.ceil(quantile * n))
    bf16_order = rank_indices(left)
    int4_order = rank_indices(right)
    bf16_queue = {int(left[index]["source_row"]) for index in bf16_order[:k]}
    int4_queue = {int(right[index]["source_row"]) for index in int4_order[:k]}
    shared = len(bf16_queue & int4_queue)
    replaced = k - shared
    raw_jaccard = shared / len(bf16_queue | int4_queue)
    chance_jaccard = quantile / (2.0 - quantile)
    adjusted_jaccard = (raw_jaccard - chance_jaccard) / (1.0 - chance_jaccard)
    int4_error_by_source = {
        int(row["source_row"]): str(row["normalized_prediction"]) != str(row["ground_truth"])
        for row in right
    }
    total_int4_errors = sum(int(value) for value in int4_error_by_source.values())
    selected_int4_errors = sum(int(int4_error_by_source[source_row]) for source_row in int4_queue)
    selected_bf16_errors = sum(int(int4_error_by_source[source_row]) for source_row in bf16_queue)
    return {
        "quantile": quantile,
        "k": k,
        "shared_slots": shared,
        "replaced_slots": replaced,
        "replacement_fraction": replaced / k,
        "raw_jaccard": raw_jaccard,
        "chance_jaccard": chance_jaccard,
        "adjusted_jaccard": adjusted_jaccard,
        "total_int4_errors": total_int4_errors,
        "int4_risk_recall": None if total_int4_errors == 0 else selected_int4_errors / total_int4_errors,
        "bf16_risk_recall": None if total_int4_errors == 0 else selected_bf16_errors / total_int4_errors,
        "random_recall_baseline": quantile,
        "int4_errors_selected_by_int4_queue": selected_int4_errors,
        "int4_errors_selected_by_bf16_queue": selected_bf16_errors,
    }


def population_result(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]], population: str) -> Dict[str, Any]:
    left, right = population_rows(left, right, population)
    budgets = {str(q): queue_result(left, right, q) for q in BUDGETS}
    return {
        "population": population,
        "n": len(left),
        "queue_budgets": budgets,
        "primary_budget": "0.2",
        "primary": budgets["0.2"],
        "metric_validity": {
            "int4_errors_gt_0": budgets["0.2"]["total_int4_errors"] > 0,
            "recall_defined_all_budgets": all(value["int4_risk_recall"] is not None and value["bf16_risk_recall"] is not None for value in budgets.values()),
        },
    }


def cross_artifact_gate(root: Path, cells: Dict[str, Any]) -> Dict[str, Any]:
    canonical_078 = read_json(root / "artifacts/eswa_revision_v1/05_primary_metrics/078_primary_metrics_v1_3_corrected.json")
    canonical_079 = read_json(root / "artifacts/eswa_revision_v1/06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json")
    if canonical_078.get("status") != "PASS" or canonical_079.get("status") != "PASS":
        raise RuntimeError("078/079 canonical unlock failed")
    jaccard_comparisons: Dict[str, Any] = {}
    recall_comparisons: Dict[str, Any] = {}
    all_jaccard_exact = True
    all_recall_exact = True
    for cell_name in CELLS:
        for population in ("full_clean", "strict_unseen"):
            ours = cells[cell_name][population]
            canonical = canonical_078["cells"][cell_name][population]
            comparison: Dict[str, Any] = {}
            for q, key in ((0.05, "j5"), (0.10, "j10"), (0.20, "j20"), (0.30, "j30")):
                left = float(ours["queue_budgets"][str(q)]["raw_jaccard"])
                right = float(canonical[key])
                equal = left == right
                all_jaccard_exact = all_jaccard_exact and equal
                comparison[key] = {"exact_equal": equal, "computed": left, "canonical_078": right, "abs_diff": abs(left - right)}
            jaccard_comparisons[f"{cell_name}|{population}"] = comparison
            ours_recall = float(ours["primary"]["int4_risk_recall"])
            risk_max = float(canonical_079["cells"][cell_name][population]["signals"]["risk_max"]["int4_recall_at_20"])
            equal = ours_recall == risk_max
            all_recall_exact = all_recall_exact and equal
            recall_comparisons[f"{cell_name}|{population}"] = {"exact_equal": equal, "081_int4_risk_recall_at_20": ours_recall, "079_risk_max_recall_at_20": risk_max, "abs_diff": abs(ours_recall - risk_max)}
    return {
        "all_078_raw_jaccard_exact": all_jaccard_exact,
        "078_raw_jaccard": jaccard_comparisons,
        "all_079_int4_risk_recall_at_20_exact": all_recall_exact,
        "079_int4_risk_recall_at_20": recall_comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cells: Dict[str, Any] = {}
    join_audits: Dict[str, Any] = {}
    for cell_name, config in CELLS.items():
        bf16, int4, audit_bf16 = load_pair(root, config)
        audit_int4 = audit_bf16
        if audit_bf16.get("status") != "PASS":
            raise RuntimeError(f"075 join audit failed: {cell_name}")
        cells[cell_name] = {
            "model": config["model"],
            "dataset": config["dataset"],
            "n_full_clean": config["n"],
            "full_clean": population_result(bf16, int4, "full_clean"),
            "strict_unseen": population_result(bf16, int4, "strict_unseen"),
        }
        join_audits[cell_name] = {"bf16": audit_bf16, "int4_g128": audit_int4}
    gate = cross_artifact_gate(root, cells)
    validity = all(
        cells[cell][population]["metric_validity"]["int4_errors_gt_0"]
        and cells[cell][population]["metric_validity"]["recall_defined_all_budgets"]
        for cell in CELLS for population in ("full_clean", "strict_unseen")
    )
    audit_pass = validity and gate["all_078_raw_jaccard_exact"] and gate["all_079_int4_risk_recall_at_20_exact"]
    artifact = {
        "stage": "081_OPERATIONAL_QUEUE_METRICS_V1_3",
        "status": "PASS" if audit_pass else "HOLD",
        "protocol_version": "v1.3",
        "risk_definition": "risk_max = joined_row.risk_primary = 1 - pmax",
        "rank_order": "(-risk_max, canonical source_row)",
        "budgets": list(BUDGETS),
        "primary_budget": 0.20,
        "jaccard_adjustment": "chance_jaccard = q/(2-q); adjusted_jaccard = (raw_jaccard-chance_jaccard)/(1-chance_jaccard)",
        "recall_definition": "INT4 errors selected by top-q queue / all INT4 errors; report both INT4-risk and BF16-risk ordering",
        "delta_convention": "not applicable",
        "GT_accessed": True,
        "scientific_metrics_computed": False,
        "bootstrap_computed": False,
        "cells": cells,
        "cross_artifact_gate": gate,
    }
    audit = {
        "stage": "081_OPERATIONAL_QUEUE_VALIDITY_AUDIT",
        "status": "PASS" if audit_pass else "HOLD",
        "protocol_version": "v1.3",
        "canonical_risk_source": "joined_row.risk_primary",
        "rank_tie_break": "canonical source_row ascending",
        "population_membership": "provenance_status == strict_unseen for sensitivity; all joined rows for full-clean",
        "jaccard_chance_correction": "chance=q/(2-q); no clamping of adjusted_jaccard",
        "queue_recall_reports": ["int4_risk_recall", "bf16_risk_recall"],
        "join_audits": join_audits,
        "cross_artifact_gate": gate,
        "GT_accessed": True,
        "scientific_metrics_computed": False,
        "bootstrap_computed": False,
        "metrics_computed": audit_pass,
    }
    output_dir = root / "artifacts/eswa_revision_v1/08_queue_utility"
    write_json(output_dir / "081_queue_metrics_v1_3_corrected.json", artifact)
    write_json(output_dir / "081_queue_validity_audit_v1_3_corrected.json", audit)
    print(json.dumps({"artifact": artifact, "audit": audit}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
