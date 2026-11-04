#!/usr/bin/env python3
"""Compute the authorized 080 candidate-mass point metrics and audit.

Candidate mass is read from the Amendment 004 recovery cache.  The script never
reconstructs it from the four candidate probabilities and never reads GT for
the computation.  It fails closed if stored candidate_mass disagrees with the
saved logsumexp fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


CELLS = {
    "internvl3_2b_mmstar": {"model": "internvl3_2b", "dataset": "MMStar", "n": 1500, "manifest": "01_manifests/mmstar_full_clean.jsonl", "joined": "04_gt_join/internvl3_2b/mmstar", "recovery": "08_candidate_mass_recovery/internvl3_2b/mmstar"},
    "internvl3_2b_ai2d": {"model": "internvl3_2b", "dataset": "AI2D", "n": 3088, "manifest": "01_manifests/ai2d_full_clean.jsonl", "joined": "04_gt_join/internvl3_2b/ai2d", "recovery": "08_candidate_mass_recovery/internvl3_2b/ai2d"},
    "qwen2_5_vl_7b_mmstar": {"model": "qwen2_5_vl_7b", "dataset": "MMStar", "n": 1500, "manifest": "01_manifests/mmstar_full_clean.jsonl", "joined": "04_gt_join/qwen2_5_vl_7b/mmstar", "recovery": "08_candidate_mass_recovery/qwen2_5_vl_7b/mmstar"},
    "qwen2_5_vl_7b_ai2d": {"model": "qwen2_5_vl_7b", "dataset": "AI2D", "n": 3088, "manifest": "01_manifests/ai2d_full_clean.jsonl", "joined": "04_gt_join/qwen2_5_vl_7b/ai2d", "recovery": "08_candidate_mass_recovery/qwen2_5_vl_7b/ai2d"},
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"non-object row: {path}")
    return rows


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        value = indexed[cursor][1]
        while end < len(indexed) and indexed[end][1] == value:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = rank
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_rank, right_rank))
    left_var = sum((a - left_mean) ** 2 for a in left_rank)
    right_var = sum((b - right_mean) ** 2 for b in right_rank)
    if left_var == 0.0 or right_var == 0.0:
        return 1.0 if left_rank == right_rank else 0.0
    return numerator / math.sqrt(left_var * right_var)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise RuntimeError("cannot quantile an empty vector")
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def distribution(values: Sequence[float]) -> Dict[str, float]:
    return {
        "median": quantile(values, 0.50),
        "q1": quantile(values, 0.25),
        "q3": quantile(values, 0.75),
        "iqr": quantile(values, 0.75) - quantile(values, 0.25),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def field_audit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stored: List[float] = []
    derived: List[float] = []
    log_deltas: List[float] = []
    finite = True
    bounded = True
    missing = 0
    for row in rows:
        try:
            mass = float(row["candidate_mass"])
            candidate_logsumexp = float(row["candidate_logsumexp"])
            vocab_logsumexp = float(row["vocab_logsumexp"])
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue
        finite = finite and all(math.isfinite(value) for value in (mass, candidate_logsumexp, vocab_logsumexp))
        bounded = bounded and 0.0 <= mass <= 1.0
        delta = candidate_logsumexp - vocab_logsumexp
        log_deltas.append(delta)
        try:
            reconstructed = math.exp(delta)
        except OverflowError:
            reconstructed = float("inf")
        stored.append(mass)
        derived.append(reconstructed)
    differences = [left - right for left, right in zip(stored, derived)]
    return {
        "n": len(rows),
        "missing_required_fields": missing,
        "all_required_fields_present": missing == 0 and len(stored) == len(rows),
        "all_finite": finite and all(math.isfinite(value) for value in derived),
        "all_candidate_mass_within_0_1": bounded,
        "logsumexp_delta_min": min(log_deltas) if log_deltas else None,
        "logsumexp_delta_max": max(log_deltas) if log_deltas else None,
        "stored_candidate_mass_vs_logsumexp_derived": {
            "different_rows": sum(left != right for left, right in zip(stored, derived)),
            "exact_equal": all(left == right for left, right in zip(stored, derived)) and len(stored) == len(rows),
            "max_abs_diff": max((abs(value) for value in differences), default=0.0),
        },
        "stored_values": stored,
    }


def load_pair(root: Path, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    base = root / "artifacts/eswa_revision_v1" / config["recovery"]
    joined_base = root / "artifacts/eswa_revision_v1" / config["joined"]
    manifest = read_jsonl(root / "artifacts/eswa_revision_v1" / config["manifest"])
    bf16 = read_jsonl(base / "bf16/predictions.jsonl")
    int4 = read_jsonl(base / "int4_g128/predictions.jsonl")
    joined_bf16 = read_jsonl(joined_base / "bf16/predictions_with_gt.jsonl")
    joined_int4 = read_jsonl(joined_base / "int4_g128/predictions_with_gt.jsonl")
    if len(bf16) != config["n"] or len(int4) != config["n"] or len(manifest) != config["n"]:
        raise RuntimeError(f"joined cache size mismatch: {base}")
    if [row["sample_id"] for row in bf16] != [row["sample_id"] for row in int4] or [row["sample_id"] for row in bf16] != [row["sample_id"] for row in joined_bf16] or [row["sample_id"] for row in int4] != [row["sample_id"] for row in joined_int4] or [row["sample_id"] for row in bf16] != [row["sample_id"] for row in manifest]:
        raise RuntimeError(f"sample_id alignment mismatch: {base}")
    if [row["source_row"] for row in bf16] != [row["source_row"] for row in int4] or [row["source_row"] for row in bf16] != [row["source_row"] for row in joined_bf16] or [row["source_row"] for row in int4] != [row["source_row"] for row in joined_int4] or [row["source_row"] for row in bf16] != [row["source_row"] for row in manifest]:
        raise RuntimeError(f"source_row alignment mismatch: {base}")
    # Carry only provenance membership from the GT-joined audit view; no GT field is read.
    bf16 = [dict(row, provenance_status=joined_row.get("provenance_status")) for row, joined_row in zip(bf16, joined_bf16)]
    int4 = [dict(row, provenance_status=joined_row.get("provenance_status")) for row, joined_row in zip(int4, joined_int4)]
    gate_audits: Dict[str, Any] = {}
    for backend in ("bf16", "int4_g128"):
        recovery_dir = base / backend
        gate = json.loads((recovery_dir / "recovery_equivalence_audit.json").read_text(encoding="utf-8"))
        metadata = json.loads((recovery_dir / "metadata.json").read_text(encoding="utf-8"))
        equivalence = gate.get("equivalence", {})
        internal = gate.get("candidate_mass_internal_validity", {})
        gate_pass = (
            gate.get("status") == "PASS"
            and equivalence.get("status") == "PASS"
            and equivalence.get("different_rows") == 0
            and equivalence.get("different_candidate_logits") == 0
            and equivalence.get("max_candidate_logits_abs_diff") == 0.0
            and equivalence.get("sample_id_exact") is True
            and equivalence.get("source_row_exact") is True
            and internal.get("all_finite") is True
            and internal.get("all_mass_in_0_1") is True
            and metadata.get("status") == "PASS"
            and metadata.get("GT_accessed") is False
        )
        gate_audits[backend] = {"status": "PASS" if gate_pass else "HOLD", "metadata": metadata, "equivalence_audit": gate}
    return bf16, int4, {"manifest_identity_exact": True, "recovery_gates": gate_audits}


def population_values(rows: Sequence[Dict[str, Any]], population: str) -> List[float]:
    return [float(row["candidate_mass"]) for row in rows if population == "full_clean" or row.get("provenance_status") == "strict_unseen"]


def population_result(bf16: Sequence[Dict[str, Any]], int4: Sequence[Dict[str, Any]], population: str) -> Dict[str, Any]:
    keep = [population == "full_clean" or row.get("provenance_status") == "strict_unseen" for row in bf16]
    left = [float(row["candidate_mass"]) for row, include in zip(bf16, keep) if include]
    right = [float(row["candidate_mass"]) for row, include in zip(int4, keep) if include]
    deltas = [int4_value - bf16_value for bf16_value, int4_value in zip(left, right)]
    return {
        "population": population,
        "n": len(left),
        "bf16": distribution(left),
        "int4": distribution(right),
        "paired_delta_int4_minus_bf16": distribution(deltas),
        "paired_median_delta": quantile(deltas, 0.50),
        "paired_mean_delta": float(np.mean(np.asarray(deltas, dtype=float))),
        "bf16_int4_spearman": spearman(left, right),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cell_rows: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    audit_cells: Dict[str, Any] = {}
    audit_pass = True
    for cell_name, config in CELLS.items():
        bf16, int4, load_audit = load_pair(root, config)
        cell_rows[cell_name] = (bf16, int4)
        side_audits = {"bf16": field_audit(bf16), "int4": field_audit(int4)}
        provenance_aligned = [left.get("provenance_status") for left in bf16] == [right.get("provenance_status") for right in int4]
        population_count = sum(row.get("provenance_status") == "strict_unseen" for row in bf16)
        cell_pass = all(audit["all_required_fields_present"] and audit["all_finite"] and audit["all_candidate_mass_within_0_1"] and audit["stored_candidate_mass_vs_logsumexp_derived"]["exact_equal"] for audit in side_audits.values()) and provenance_aligned and population_count > 0 and load_audit["manifest_identity_exact"] and all(gate["status"] == "PASS" for gate in load_audit["recovery_gates"].values())
        audit_pass = audit_pass and cell_pass
        audit_cells[cell_name] = {
            "model": config["model"],
            "dataset": config["dataset"],
            "row_counts_equal": len(bf16) == len(int4) == config["n"],
            "sample_id_alignment_exact": [row["sample_id"] for row in bf16] == [row["sample_id"] for row in int4],
            "source_row_alignment_exact": [row["source_row"] for row in bf16] == [row["source_row"] for row in int4],
            "provenance_alignment_exact": provenance_aligned,
            "manifest_identity_exact": load_audit["manifest_identity_exact"],
            "recovery_gates": load_audit["recovery_gates"],
            "strict_unseen_count": population_count,
            "bf16": side_audits["bf16"],
            "int4": side_audits["int4"],
            "GT_used_for_computation": False,
        }

    metrics_cells: Dict[str, Any] = {}
    if audit_pass:
        for cell_name, (bf16, int4) in cell_rows.items():
            metrics_cells[cell_name] = {
                "model": CELLS[cell_name]["model"],
                "dataset": CELLS[cell_name]["dataset"],
                "full_clean": population_result(bf16, int4, "full_clean"),
                "strict_unseen": population_result(bf16, int4, "strict_unseen"),
            }

    audit = {
        "stage": "080_CANDIDATE_MASS_IMPLEMENTATION_AUDIT",
        "status": "PASS" if audit_pass else "HOLD",
        "candidate_mass_source": "Amendment 004 08_candidate_mass_recovery candidate_mass field",
        "candidate_mass_definition": "exp(candidate_logsumexp - vocab_logsumexp)",
        "candidate_mass_reconstruction_audit": "stored candidate_mass == exp(candidate_logsumexp - vocab_logsumexp), exact float equality required",
        "populations": {"full_clean": "all joined canonical rows", "strict_unseen": "provenance_status == strict_unseen"},
        "GT_used_for_computation": False,
        "GT_accessed_for_metric": False,
        "outcome_conditioned_analysis": False,
        "bootstrap_computed": False,
        "cells": audit_cells,
    }
    artifact = {
        "stage": "080_CANDIDATE_MASS_METRICS_V1_3_RECOVERED",
        "status": "PASS" if audit_pass else "HOLD",
        "protocol_version": "v1.3",
        "candidate_mass_source": "Amendment 004 08_candidate_mass_recovery candidate_mass field",
        "candidate_mass_definition": "M_cand = sum_y exp(z_y) / sum_v exp(z_v) = exp(candidate_logsumexp - vocab_logsumexp)",
        "delta_convention": "candidate_mass_delta = INT4 - BF16",
        "iqr_definition": "Q3 - Q1 using deterministic linear quantiles",
        "populations": {"full_clean": "all joined canonical rows", "strict_unseen": "provenance_status == strict_unseen"},
        "GT_used_for_computation": False,
        "GT_accessed_for_metric": False,
        "outcome_conditioned_analysis": False,
        "bootstrap_computed": False,
        "metrics_computed": audit_pass,
        "cells": metrics_cells,
    }
    output_dir = root / "artifacts/eswa_revision_v1/07_candidate_mass"
    write_json(output_dir / "080_candidate_mass_metrics_v1_3_recovered.json", artifact)
    write_json(output_dir / "080_candidate_mass_validity_audit_v1_3.json", audit)
    print(json.dumps({"artifact": artifact, "audit": audit}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
