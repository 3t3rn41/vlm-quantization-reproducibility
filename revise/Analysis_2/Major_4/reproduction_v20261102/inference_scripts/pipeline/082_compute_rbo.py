#!/usr/bin/env python3
"""Compute the authorized 082 standalone RBO summary.

This stage is a GT-free summary/reproduction of the frozen RBO_EXT metric.
It reads only the canonical 075 joined rows needed for sample identity,
source-row order, provenance filtering, and the stored canonical
``risk_primary`` field.  Ground-truth and outcome fields are never accessed.

The implementation deliberately uses the Amendment 003 ranking rule
``(-risk_primary, source_row)`` and fails closed unless both cross-artifact
gates reproduce the canonical 077 full-clean values and the canonical 079
risk_max p=.90 values exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


CELLS = {
    "internvl3_2b_mmstar": {"model": "internvl3_2b", "dataset": "MMStar", "n": 1500, "joined": "04_gt_join/internvl3_2b/mmstar"},
    "internvl3_2b_ai2d": {"model": "internvl3_2b", "dataset": "AI2D", "n": 3088, "joined": "04_gt_join/internvl3_2b/ai2d"},
    "qwen2_5_vl_7b_mmstar": {"model": "qwen2_5_vl_7b", "dataset": "MMStar", "n": 1500, "joined": "04_gt_join/qwen2_5_vl_7b/mmstar"},
    "qwen2_5_vl_7b_ai2d": {"model": "qwen2_5_vl_7b", "dataset": "AI2D", "n": 3088, "joined": "04_gt_join/qwen2_5_vl_7b/ai2d"},
}
PERSISTENCE = (0.90, 0.80, 0.95)


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rbo_ext(ranking_a: Sequence[str], ranking_b: Sequence[str], persistence: float) -> float:
    """Finite-list RBO_EXT used by the canonical 077/079 implementations."""
    if not 0.0 < persistence < 1.0:
        raise ValueError("RBO persistence must be in (0, 1)")
    if len(ranking_a) != len(ranking_b) or not ranking_a:
        raise RuntimeError("RBO requires non-empty equal-length rankings")
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    weighted = 0.0
    for depth, (item_a, item_b) in enumerate(zip(ranking_a, ranking_b), 1):
        seen_a.add(item_a)
        seen_b.add(item_b)
        agreement = len(seen_a & seen_b) / depth
        weighted += agreement * (persistence ** (depth - 1))
    final_agreement = len(seen_a & seen_b) / len(ranking_a)
    return (1.0 - persistence) * weighted + (persistence ** len(ranking_a)) * final_agreement


def canonical_rows(root: Path, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    base = root / "artifacts/eswa_revision_v1" / config["joined"]
    audit = read_json(base / "075_join_audit.json")
    if audit.get("status") != "PASS":
        raise RuntimeError(f"075 join audit is not PASS: {base}")
    if audit.get("metrics_computed") is not False:
        raise RuntimeError(f"075 join audit metric flag is not false: {base}")
    sides: List[List[Dict[str, Any]]] = []
    for backend in ("bf16", "int4_g128"):
        rows = read_jsonl(base / backend / "predictions_with_gt.jsonl")
        if len(rows) != config["n"]:
            raise RuntimeError(f"unexpected row count for {base}/{backend}: {len(rows)}")
        sample_ids = [row.get("sample_id") for row in rows]
        source_rows = [row.get("source_row") for row in rows]
        if not all(isinstance(value, str) and value for value in sample_ids):
            raise RuntimeError(f"invalid sample_id in {base}/{backend}")
        if not all(isinstance(value, int) for value in source_rows):
            raise RuntimeError(f"invalid source_row in {base}/{backend}")
        if len(set(sample_ids)) != len(sample_ids) or len(set(source_rows)) != len(source_rows):
            raise RuntimeError(f"duplicate identity in {base}/{backend}")
        for row in rows:
            risk = row.get("risk_primary")
            if not isinstance(risk, (int, float)) or not math.isfinite(float(risk)):
                raise RuntimeError(f"invalid canonical risk_primary in {base}/{backend}")
        sides.append(rows)
    bf16, int4 = sides
    if [row["sample_id"] for row in bf16] != [row["sample_id"] for row in int4]:
        raise RuntimeError(f"BF16/INT4 sample_id alignment mismatch: {base}")
    if [row["source_row"] for row in bf16] != [row["source_row"] for row in int4]:
        raise RuntimeError(f"BF16/INT4 source_row alignment mismatch: {base}")
    if [row.get("provenance_status") for row in bf16] != [row.get("provenance_status") for row in int4]:
        raise RuntimeError(f"BF16/INT4 provenance alignment mismatch: {base}")
    return bf16, int4, audit


def population(rows: Sequence[Dict[str, Any]], population_name: str) -> List[Dict[str, Any]]:
    if population_name == "full_clean":
        return list(rows)
    if population_name == "strict_unseen":
        selected = [row for row in rows if row.get("provenance_status") == "strict_unseen"]
        if not selected:
            raise RuntimeError("strict_unseen population is empty")
        return selected
    raise ValueError(population_name)


def ranked_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        row["sample_id"]
        for row in sorted(rows, key=lambda row: (-float(row["risk_primary"]), int(row["source_row"])))
    ]


def population_metrics(bf16: Sequence[Dict[str, Any]], int4: Sequence[Dict[str, Any]], population_name: str) -> Dict[str, Any]:
    left = population(bf16, population_name)
    right = population(int4, population_name)
    if len(left) != len(right):
        raise RuntimeError(f"population size mismatch: {population_name}")
    ids_left = [row["sample_id"] for row in left]
    ids_right = [row["sample_id"] for row in right]
    source_left = [row["source_row"] for row in left]
    source_right = [row["source_row"] for row in right]
    if ids_left != ids_right or source_left != source_right:
        raise RuntimeError(f"population alignment mismatch: {population_name}")
    ranking_left = ranked_ids(left)
    ranking_right = ranked_ids(right)
    rbo_values = {f"p={p:.2f}": rbo_ext(ranking_left, ranking_right, p) for p in PERSISTENCE}
    return {
        "population": population_name,
        "n": len(left),
        "sample_id_unique": len(set(ids_left)) == len(ids_left),
        "source_row_unique": len(set(source_left)) == len(source_left),
        "bf16_int4_same_id_set": set(ids_left) == set(ids_right),
        "bf16_int4_same_source_row_alignment": source_left == source_right,
        "provenance_population": population_name,
        "risk_source": "canonical joined_row.risk_primary",
        "ranking_tie_break": "(-risk, canonical source_row)",
        "rbo_variant": "RBO_EXT",
        "RBO": rbo_values,
        "p090_finite": math.isfinite(rbo_values["p=0.90"]),
        "p080_finite": math.isfinite(rbo_values["p=0.80"]),
        "p095_finite": math.isfinite(rbo_values["p=0.95"]),
        "ground_truth_read": False,
        "outcome_conditioned_analysis": False,
    }


def exact_gate(expected: float, actual: float) -> Dict[str, Any]:
    return {"exact_equal": actual == expected, "expected": expected, "actual": actual, "abs_diff": abs(actual - expected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts/eswa_revision_v1"
    reference_077 = read_json(base / "04_noise_floor/077_noise_floor_v1_3.json")
    reference_079 = read_json(base / "06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json")
    if reference_077.get("status") != "PASS" or reference_079.get("status") != "PASS":
        raise RuntimeError("canonical 077/079 references are not PASS")

    cells: Dict[str, Any] = {}
    gate_a: Dict[str, Any] = {}
    gate_b: Dict[str, Any] = {}
    all_exact = True
    for cell_name, config in CELLS.items():
        bf16, int4, audit = canonical_rows(root, config)
        populations = {name: population_metrics(bf16, int4, name) for name in ("full_clean", "strict_unseen")}
        cells[cell_name] = {
            "model": config["model"],
            "dataset": config["dataset"],
            "n_full_clean": config["n"],
            "full_clean": populations["full_clean"],
            "strict_unseen": populations["strict_unseen"],
            "join_audit_status": audit.get("status"),
        }
        full_expected = reference_077["cells"][cell_name]["quantization_contrast"]["RBO"]
        gate_a[cell_name] = {}
        for key in ("p=0.90", "p=0.80", "p=0.95"):
            comparison = exact_gate(float(full_expected[key]), float(populations["full_clean"]["RBO"][key]))
            gate_a[cell_name][key] = comparison
            all_exact = all_exact and comparison["exact_equal"]
        gate_b[cell_name] = {}
        for population_name in ("full_clean", "strict_unseen"):
            expected = float(reference_079["cells"][cell_name][population_name]["signals"]["risk_max"]["rbo_p090"])
            actual = float(populations[population_name]["RBO"]["p=0.90"])
            comparison = exact_gate(expected, actual)
            gate_b[cell_name][population_name] = comparison
            all_exact = all_exact and comparison["exact_equal"]

    validity = {
        "stage": "082_RBO_VALIDITY_AUDIT_V1_3",
        "status": "PASS" if all_exact else "HOLD",
        "protocol_version": "v1.3",
        "rbo_variant": "RBO_EXT",
        "risk_source": "canonical joined_row.risk_primary = 1 - pmax",
        "ranking_rule": "(-risk, canonical source_row)",
        "persistence": {"primary": 0.90, "sensitivity": [0.80, 0.95]},
        "full_clean_exact_match_077": all(item["p=0.90"]["exact_equal"] and item["p=0.80"]["exact_equal"] and item["p=0.95"]["exact_equal"] for item in gate_a.values()),
        "risk_max_p090_exact_match_079": all(item[pop]["exact_equal"] for item in gate_b.values() for pop in ("full_clean", "strict_unseen")),
        "cross_artifact_gate_A_077_full_clean": gate_a,
        "cross_artifact_gate_B_079_risk_max_p090": gate_b,
        "GT_required_for_metric": False,
        "GT_accessed": False,
        "ground_truth_read": False,
        "outcome_conditioned_analysis": False,
        "scientific_metrics_computed": True,
        "bootstrap_computed": False,
        "cells": {
            cell: {
                population: {
                    "n": value[population]["n"],
                    "sample_id_unique": value[population]["sample_id_unique"],
                    "source_row_unique": value[population]["source_row_unique"],
                    "bf16_int4_same_id_set": value[population]["bf16_int4_same_id_set"],
                    "same_source_row_alignment": value[population]["bf16_int4_same_source_row_alignment"],
                    "provenance_population": value[population]["provenance_population"],
                    "risk_source": value[population]["risk_source"],
                    "ranking_tie_break": value[population]["ranking_tie_break"],
                    "rbo_variant": value[population]["rbo_variant"],
                    "p090_finite": value[population]["p090_finite"],
                    "p080_finite": value[population]["p080_finite"],
                    "p095_finite": value[population]["p095_finite"],
                }
                for population in ("full_clean", "strict_unseen")
            }
            for cell, value in cells.items()
        },
    }
    if validity["status"] != "PASS":
        raise RuntimeError(json.dumps(validity, ensure_ascii=False, sort_keys=True))

    artifact = {
        "stage": "082_RBO_V1_3",
        "status": "PASS",
        "protocol_version": "v1.3",
        "risk_definition": "risk_max = canonical joined_row.risk_primary = 1 - pmax",
        "ranking_rule": "(-risk, canonical source_row)",
        "rbo_variant": "RBO_EXT",
        "persistence": {"primary": 0.90, "sensitivity": [0.80, 0.95]},
        "GT_required_for_metric": False,
        "GT_accessed": False,
        "ground_truth_read": False,
        "outcome_conditioned_analysis": False,
        "scientific_metrics_computed": True,
        "bootstrap_computed": False,
        "canonical_references": {
            "077": str(base / "04_noise_floor/077_noise_floor_v1_3.json"),
            "079_corrected": str(base / "06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json"),
        },
        "cells": cells,
    }
    output_dir = base / "09_rbo"
    write_json(output_dir / "082_rbo_v1_3.json", artifact)
    write_json(output_dir / "082_rbo_validity_audit_v1_3.json", validity)
    print(json.dumps({"status": "PASS", "metrics_artifact": str(output_dir / "082_rbo_v1_3.json"), "audit_artifact": str(output_dir / "082_rbo_validity_audit_v1_3.json"), "full_clean_exact_match_077": validity["full_clean_exact_match_077"], "risk_max_p090_exact_match_079": validity["risk_max_p090_exact_match_079"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
