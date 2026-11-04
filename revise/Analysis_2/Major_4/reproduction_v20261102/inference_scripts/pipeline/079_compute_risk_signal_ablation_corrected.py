#!/usr/bin/env python3
"""CPU-only correction of 079 risk_max source, with implementation audit.

The original 079 artifact remains untouched.  This corrected run reuses the
joined row's canonical risk_primary for risk_max and leaves the entropy and
margin calculations byte-for-byte on the original implementation path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List


LETTERS = ("A", "B", "C", "D")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("vlm_079_original_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def corrected_risk_signals(row: Dict[str, Any]) -> Dict[str, float]:
    probabilities = row.get("candidate_probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(LETTERS):
        raise RuntimeError("candidate_probabilities must contain exactly A/B/C/D")
    values = [float(probabilities[label]) for label in LETTERS]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError("invalid candidate probabilities")
    total = sum(values)
    if total <= 0.0:
        raise RuntimeError("candidate probabilities sum must be positive")
    raw_risk = 1.0 - max(values)
    stored_risk = float(row["risk_primary"])
    if stored_risk != raw_risk:
        raise RuntimeError("risk_primary is not exactly 1-max(raw serialized candidate_probabilities)")
    normalized = [value / total for value in values]
    ordered = sorted(normalized, reverse=True)
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0) / math.log(4.0)
    margin = 1.0 - (ordered[0] - ordered[1])
    return {"risk_max": stored_risk, "risk_entropy": entropy, "risk_margin": margin}


def signal_metric_map(artifact: Dict[str, Any], signal: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for cell in artifact["cells"].values():
        for population in ("full_clean", "strict_unseen"):
            values[f"{cell['model']}|{cell['dataset']}|{population}"] = cell[population]["signals"][signal]
    return values


def exact_signal_unchanged(old: Dict[str, Any], new: Dict[str, Any], signal: str) -> Dict[str, Any]:
    old_values = signal_metric_map(old, signal)
    new_values = signal_metric_map(new, signal)
    comparisons = {key: old_values[key] == new_values[key] for key in old_values}
    return {"signal": signal, "all_exact_equal": all(comparisons.values()), "different_populations": [key for key, equal in comparisons.items() if not equal]}


def risk_row_source_audit(root: Path, module: Any) -> Dict[str, Any]:
    cell_audit: Dict[str, Any] = {}
    all_exact = True
    for cell_name, config in module.CELLS.items():
        base = root / "artifacts/eswa_revision_v1" / config["joined"]
        side_audit: Dict[str, Any] = {}
        for side, backend in (("bf16", "bf16"), ("int4", "int4_g128")):
            path = base / backend / "predictions_with_gt.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            differences: List[float] = []
            different_rows = 0
            for row in rows:
                probabilities = [float(row["candidate_probabilities"][label]) for label in LETTERS]
                stored = float(row["risk_primary"])
                raw = 1.0 - max(probabilities)
                difference = stored - raw
                differences.append(difference)
                if stored != raw:
                    different_rows += 1
            exact = different_rows == 0
            all_exact = all_exact and exact
            side_audit[side] = {"n": len(rows), "different_rows": different_rows, "max_abs_diff": max((abs(value) for value in differences), default=0.0), "exact_equal": exact}
        cell_audit[cell_name] = side_audit
    return {"all_rows_exact_equal": all_exact, "cells": cell_audit}


def metric_gate(root: Path, corrected: Dict[str, Any], module: Any) -> Dict[str, Any]:
    canonical = read_json(root / "artifacts/eswa_revision_v1/05_primary_metrics/078_primary_metrics_v1_3_corrected.json")
    noise = read_json(root / "artifacts/eswa_revision_v1/04_noise_floor/077_noise_floor_v1_3.json")
    metric_keys = {
        "spearman": ("bf16_int4_spearman", "spearman"),
        "kendall_tau_b": ("bf16_int4_kendall_tau_b", "kendall_tau_b"),
        "j20": ("j20", "j20"),
        "int4_auroc": ("int4_auroc", "int4_auroc"),
        "int4_aurc": ("int4_aurc", "int4_aurc"),
    }
    comparisons: Dict[str, Any] = {}
    all_exact = True
    for cell_name in module.CELLS:
        for population in ("full_clean", "strict_unseen"):
            corrected_values = corrected["cells"][cell_name][population]["signals"]["risk_max"]
            canonical_values = canonical["cells"][cell_name][population]
            population_comparison: Dict[str, Any] = {}
            for metric, (corrected_key, canonical_key) in metric_keys.items():
                left = float(corrected_values[corrected_key])
                right = float(canonical_values[canonical_key])
                equal = left == right
                all_exact = all_exact and equal
                population_comparison[metric] = {"exact_equal": equal, "abs_diff": abs(left - right), "corrected": left, "canonical_078": right}
            if population == "full_clean":
                corrected_rbo = float(corrected_values["rbo_p090"])
                canonical_rbo = float(noise["cells"][cell_name]["quantization_contrast"]["RBO"]["p=0.90"])
                equal = corrected_rbo == canonical_rbo
                all_exact = all_exact and equal
                population_comparison["rbo_p090_vs_077"] = {"exact_equal": equal, "abs_diff": abs(corrected_rbo - canonical_rbo), "corrected": corrected_rbo, "canonical_077": canonical_rbo}
            comparisons[f"{cell_name}|{population}"] = population_comparison
    return {"all_exact_equal": all_exact, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    module = load_module(root / "scripts/079_compute_risk_signal_ablation.py")
    old_path = root / "artifacts/eswa_revision_v1/06_risk_ablation/079_risk_signal_ablation_v1_3.json"
    old = read_json(old_path)
    if old.get("status") != "PASS":
        raise RuntimeError("original 079 artifact is not the expected read-only candidate")
    module.risk_signals = corrected_risk_signals
    cells: Dict[str, Any] = {}
    for cell_name, config in module.CELLS.items():
        bf16, int4 = module.load_pair(root, config)
        cells[cell_name] = {
            "model": config["model"],
            "dataset": config["dataset"],
            "n_full_clean": config["n"],
            "full_clean": module.population_result(bf16, int4, "full_clean"),
            "strict_unseen": module.population_result(bf16, int4, "strict_unseen"),
        }
    corrected = {
        "stage": "079_RISK_SIGNAL_ABLATION_V1_3_CORRECTED",
        "status": "PASS",
        "protocol_version": "v1.3",
        "correction": {
            "root_cause": "unintended renormalization of serialized candidate probabilities when recomputing risk_max",
            "canonical_risk_max_source": "joined_row.risk_primary",
            "risk_max_rule": "use joined_row.risk_primary directly; do not normalize candidate_probabilities for risk_max",
            "old_artifact": str(old_path),
            "old_artifact_status": "SUPERSEDED_READ_ONLY",
        },
        "signals": {
            "risk_max": "joined row risk_primary; canonical 1 - max(raw serialized candidate_probabilities)",
            "risk_entropy": "normalized 4-way entropy",
            "risk_margin": "1 - (p1 - p2)",
        },
        "populations": {"full_clean": "all joined canonical rows", "strict_unseen": "provenance_status == strict_unseen"},
        "rank_rules": {"spearman": "average ranks", "kendall": "tau-b", "rank_tie_break": "(-risk, canonical source_row)", "rbo": "RBO_EXT, p=0.90"},
        "recall_at_20_definition": "INT4 error recall among the top ceil(0.20*n) rows ranked by the signal-specific INT4 risk",
        "operational_queue_metrics_computed": False,
        "bootstrap_computed": False,
        "GT_accessed": True,
        "metrics_computed": True,
        "cells": cells,
    }
    source_audit = risk_row_source_audit(root, module)
    entropy_audit = exact_signal_unchanged(old, corrected, "risk_entropy")
    margin_audit = exact_signal_unchanged(old, corrected, "risk_margin")
    gate = metric_gate(root, corrected, module)
    audit = {
        "stage": "079_IMPLEMENTATION_CORRECTION_AUDIT",
        "status": "PASS" if source_audit["all_rows_exact_equal"] and gate["all_exact_equal"] and entropy_audit["all_exact_equal"] and margin_audit["all_exact_equal"] else "HOLD",
        "old_artifact": str(old_path),
        "old_artifact_action": "retained read-only and marked superseded",
        "corrected_artifact": str(root / "artifacts/eswa_revision_v1/06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json"),
        "root_cause": "unintended renormalization of serialized candidate probabilities when recomputing risk_max",
        "canonical_risk_max_source": "joined_row.risk_primary",
        "stored_risk_primary_equals_raw_1_minus_pmax": source_audit,
        "risk_max_cross_artifact_gate": gate,
        "entropy_unchanged_from_original_079": entropy_audit,
        "margin_unchanged_from_original_079": margin_audit,
        "GT_source_changed": False,
        "joined_cache_modified": False,
        "inference_rerun": False,
        "bootstrap_computed": False,
        "hash_computed": False,
    }
    corrected_path = root / "artifacts/eswa_revision_v1/06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json"
    audit_path = root / "artifacts/eswa_revision_v1/06_risk_ablation/079_implementation_correction_audit.json"
    module.write_json(corrected_path, corrected)
    module.write_json(audit_path, audit)
    print(json.dumps({"corrected_artifact": str(corrected_path), "audit_artifact": str(audit_path), "corrected": corrected, "audit": audit}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
