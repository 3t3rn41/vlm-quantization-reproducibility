#!/usr/bin/env python3
"""Produce the read-only 079 cross-artifact consistency audit.

The audit deliberately does not rewrite 078/079 artifacts or joined caches.
It records exact equality and absolute differences for the canonical
risk_primary field, the raw probability reconstruction, and the current 079
normalized-probability reconstruction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


LETTERS = ("A", "B", "C", "D")


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"non-object row: {path}")
    return rows


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("vlm_079_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probability_values(row: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    probabilities = row.get("candidate_probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(LETTERS):
        raise RuntimeError("candidate_probabilities must contain exactly A/B/C/D")
    values = [float(probabilities[label]) for label in LETTERS]
    total = sum(values)
    if not all(math.isfinite(value) and value >= 0.0 for value in values) or total <= 0.0:
        raise RuntimeError("invalid candidate probabilities")
    raw_risk = 1.0 - max(values)
    normalized = [value / total for value in values]
    p1, p2 = sorted(normalized, reverse=True)[:2]
    normalized_risk = 1.0 - p1
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0) / math.log(4.0)
    margin = 1.0 - (p1 - p2)
    return float(row["risk_primary"]), raw_risk, normalized_risk, entropy, margin, total


def scalar_audit(values_left: Sequence[float], values_right: Sequence[float]) -> Dict[str, Any]:
    diffs = [float(a) - float(b) for a, b in zip(values_left, values_right)]
    return {
        "n": len(diffs),
        "different_rows": sum(a != b for a, b in zip(values_left, values_right)),
        "exact_equal": all(a == b for a, b in zip(values_left, values_right)),
        "max_abs_diff": max((abs(value) for value in diffs), default=0.0),
    }


def metric_audit(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for key in left:
        if key not in right:
            continue
        difference = float(left[key]) - float(right[key])
        metrics[key] = {"exact_equal": float(left[key]) == float(right[key]), "abs_diff": abs(difference), "left": left[key], "right": right[key]}
    return metrics


def primary_metrics(module: Any, left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]], left_risk: Sequence[float], right_risk: Sequence[float]) -> Dict[str, float]:
    source_rows = [int(row["source_row"]) for row in left]
    errors = [int(str(row["normalized_prediction"]) != str(row["ground_truth"])) for row in right]
    left_order = module.ranked_indices(left_risk, source_rows)
    right_order = module.ranked_indices(right_risk, source_rows)
    return {
        "spearman": module.spearman(left_risk, right_risk),
        "kendall_tau_b": module.kendall_tau_b(left_risk, right_risk),
        "j20": module.top_jaccard(left_risk, right_risk, source_rows),
        "rbo_p090": module.rbo(left_order, right_order, 0.90),
        "int4_auroc": module.auroc(errors, right_risk),
        "int4_aurc": module.aurc(errors, right_risk, source_rows),
        "int4_recall_at_20": module.recall_at_20(errors, right_risk, source_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    module = load_module(root / "scripts/079_compute_risk_signal_ablation.py")
    artifact_079 = read_json(root / "artifacts/eswa_revision_v1/06_risk_ablation/079_risk_signal_ablation_v1_3.json")
    artifact_078 = read_json(root / "artifacts/eswa_revision_v1/05_primary_metrics/078_primary_metrics_v1_3_corrected.json")
    artifact_077 = read_json(root / "artifacts/eswa_revision_v1/04_noise_floor/077_noise_floor_v1_3.json")

    cells: Dict[str, Any] = {}
    risk_source_all_equal = True
    metrics_all_equal = True
    signal_validity_all_pass = True
    for cell_name, config in module.CELLS.items():
        base = root / "artifacts/eswa_revision_v1" / config["joined"]
        bf16 = read_jsonl(base / "bf16/predictions_with_gt.jsonl")
        int4 = read_jsonl(base / "int4_g128/predictions_with_gt.jsonl")
        row_sources: Dict[str, Any] = {}
        for side, rows in (("bf16", bf16), ("int4", int4)):
            stored, raw, current, entropy, margin, totals = [], [], [], [], [], []
            for row in rows:
                stored_value, raw_value, current_value, entropy_value, margin_value, total = probability_values(row)
                stored.append(stored_value)
                raw.append(raw_value)
                current.append(current_value)
                entropy.append(entropy_value)
                margin.append(margin_value)
                totals.append(total)
            source_audit = scalar_audit(stored, raw)
            current_audit = scalar_audit(stored, current)
            risk_source_all_equal = risk_source_all_equal and source_audit["exact_equal"]
            finite_and_bounded = all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in current + entropy + margin)
            # p1 >= p2 is guaranteed by the sorted construction; retain an explicit audit flag.
            p1_ge_p2 = True
            signal_validity_all_pass = signal_validity_all_pass and finite_and_bounded and p1_ge_p2
            source_rows = [int(row["source_row"]) for row in rows]
            row_sources[side] = {
                "n": len(rows),
                "candidate_probability_sum_min": min(totals),
                "candidate_probability_sum_max": max(totals),
                "risk_max_078_source": "joined row risk_primary",
                "risk_max_canonical_recomputed": "1 - max(raw candidate_probabilities)",
                "risk_max_079_current_source": "1 - max(candidate_probabilities / sum(candidate_probabilities))",
                "risk_primary_vs_canonical_recomputed": source_audit,
                "risk_primary_vs_079_current_recomputed": current_audit,
                "risk_max_078_source_order_equals_canonical_order": module.ranked_indices(stored, source_rows) == module.ranked_indices(raw, source_rows),
                "risk_max_079_current_order_equals_canonical_order": module.ranked_indices(current, source_rows) == module.ranked_indices(raw, source_rows),
                "signal_definition_validity": {
                    "all_finite": finite_and_bounded,
                    "all_within_0_1": finite_and_bounded,
                    "entropy_uses_normalized_4_way_probabilities": True,
                    "entropy_divisor": "log(4)",
                    "p1_ge_p2": p1_ge_p2,
                    "margin_definition": "1 - (p1 - p2)",
                },
            }

        population_audits: Dict[str, Any] = {}
        for population in ("full_clean", "strict_unseen"):
            keep = [population == "full_clean" or row.get("provenance_status") == "strict_unseen" for row in bf16]
            left = [row for row, include in zip(bf16, keep) if include]
            right = [row for row, include in zip(int4, keep) if include]
            stored_left = [probability_values(row)[0] for row in left]
            stored_right = [probability_values(row)[0] for row in right]
            canonical_metrics = primary_metrics(module, left, right, stored_left, stored_right)
            reported = artifact_079["cells"][cell_name][population]["signals"]["risk_max"]
            reported_metrics = {
                "spearman": reported["bf16_int4_spearman"],
                "kendall_tau_b": reported["bf16_int4_kendall_tau_b"],
                "j20": reported["j20"],
                "rbo_p090": reported["rbo_p090"],
                "int4_auroc": reported["int4_auroc"],
                "int4_aurc": reported["int4_aurc"],
                "int4_recall_at_20": reported["int4_recall_at_20"],
            }
            canonical_078 = artifact_078["cells"][cell_name][population]
            metrics_078 = {"spearman": canonical_078["spearman"], "kendall_tau_b": canonical_078["kendall_tau_b"], "j20": canonical_078["j20"], "int4_auroc": canonical_078["int4_auroc"], "int4_aurc": canonical_078["int4_aurc"]}
            risk_source_all_equal = risk_source_all_equal and all(value["risk_primary_vs_canonical_recomputed"]["exact_equal"] for value in row_sources.values())
            report_vs_canonical = metric_audit(reported_metrics, canonical_metrics)
            canonical_vs_078 = metric_audit({key: canonical_metrics[key] for key in metrics_078}, metrics_078)
            metrics_all_equal = metrics_all_equal and all(item["exact_equal"] for item in canonical_vs_078.values())
            rbo_077 = None
            rbo_077_audit = None
            if population == "full_clean":
                rbo_077 = artifact_077["cells"][cell_name]["quantization_contrast"]["RBO"]["p=0.90"]
                rbo_077_audit = {"exact_equal": reported_metrics["rbo_p090"] == rbo_077, "abs_diff": abs(reported_metrics["rbo_p090"] - rbo_077), "079": reported_metrics["rbo_p090"], "077": rbo_077}
                metrics_all_equal = metrics_all_equal and rbo_077_audit["exact_equal"]
            population_audits[population] = {
                "n": len(left),
                "079_risk_max_vs_canonical_078": report_vs_canonical,
                "canonical_risk_primary_metrics_vs_078": canonical_vs_078,
                "full_clean_rbo_079_vs_077": rbo_077_audit,
            }

        cells[cell_name] = {"model": config["model"], "dataset": config["dataset"], "row_source_audit": row_sources, "population_audit": population_audits}

    artifact = {
        "stage": "079_CROSS_ARTIFACT_CONSISTENCY_AUDIT",
        "status": "PASS" if risk_source_all_equal and metrics_all_equal and signal_validity_all_pass else "HOLD",
        "read_only": True,
        "joined_caches_modified": False,
        "inference_run": False,
        "gt_join_run": False,
        "risk_max_078_source": "joined row risk_primary, used by canonical 078/077",
        "risk_max_canonical_recomputed": "1 - max(candidate_probabilities) without re-normalizing serialized values",
        "risk_max_079_current_source": "1 - max(candidate_probabilities / sum(candidate_probabilities))",
        "risk_max_source_audit_pass": risk_source_all_equal,
        "canonical_risk_metrics_reproduce_078": metrics_all_equal,
        "signal_definition_validity_pass": signal_validity_all_pass,
        "auroc_definition": "positive=incorrect/error=1; score=corresponding risk signal",
        "aurc_definition": "sort=(risk ascending, source_row ascending); prefix selective risk=errors_prefix/k; AURC=mean(prefix selective risk)",
        "recall_at_20_definition": "k=ceil(0.20*n); numerator=INT4 errors in signal-specific INT4 top-k queue; denominator=all INT4 errors",
        "cells": cells,
    }
    output = root / "artifacts/eswa_revision_v1/06_risk_ablation/079_cross_artifact_consistency_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
