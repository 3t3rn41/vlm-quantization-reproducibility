#!/usr/bin/env python3
"""Compute the authorized GT-free BF16 repeat noise floor for VLM R1/R2.

This stage reads only canonical prediction caches and their metadata.  It does
not open manifests, labels, or any ground-truth file.  Pairing is guarded by
sample-id/source-row order, dataset revision, model revision, and manifest
identity checks before rank metrics are computed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EXPECTED = {
    "internvl3_2b_mmstar": {
        "model": "internvl3_2b",
        "dataset": "mmstar",
        "dataset_label": "MMStar",
        "n": 1500,
        "r1_bf16": "02_full_inference/internvl3_2b/mmstar/bf16",
        "r1_int4": "02_full_inference/internvl3_2b/mmstar/int4_g128",
        "r2_bf16": "03_r2_repeat/internvl3_2b/mmstar/bf16_r2",
        "r1_stage": "073_FULL_INFERENCE_R1_1",
        "r1_int4_stage": "073_FULL_INFERENCE_R1_2",
        "r2_stage": "076_R2_1_BF16_REPEAT",
    },
    "internvl3_2b_ai2d": {
        "model": "internvl3_2b",
        "dataset": "ai2d",
        "dataset_label": "AI2D",
        "n": 3088,
        "r1_bf16": "02_full_inference/internvl3_2b/ai2d/bf16",
        "r1_int4": "02_full_inference/internvl3_2b/ai2d/int4_g128",
        "r2_bf16": "03_r2_repeat/internvl3_2b/ai2d/bf16_r2",
        "r1_stage": "073_FULL_INFERENCE_R1_3",
        "r1_int4_stage": "073_FULL_INFERENCE_R1_4",
        "r2_stage": "076_R2_2_BF16_REPEAT",
    },
    "qwen2_5_vl_7b_mmstar": {
        "model": "qwen2_5_vl_7b",
        "dataset": "mmstar",
        "dataset_label": "MMStar",
        "n": 1500,
        "r1_bf16": "02_full_inference/qwen2_5_vl_7b/mmstar/bf16",
        "r1_int4": "02_full_inference/qwen2_5_vl_7b/mmstar/int4_g128",
        "r2_bf16": "03_r2_repeat/qwen2_5_vl_7b/mmstar/bf16_r2",
        "r1_stage": "073_FULL_INFERENCE_R1_5",
        "r1_int4_stage": "073_FULL_INFERENCE_R1_6",
        "r2_stage": "076_R2_3_BF16_REPEAT",
    },
    "qwen2_5_vl_7b_ai2d": {
        "model": "qwen2_5_vl_7b",
        "dataset": "ai2d",
        "dataset_label": "AI2D",
        "n": 3088,
        "r1_bf16": "02_full_inference/qwen2_5_vl_7b/ai2d/bf16",
        "r1_int4": "02_full_inference/qwen2_5_vl_7b/ai2d/int4_g128",
        "r2_bf16": "03_r2_repeat/qwen2_5_vl_7b/ai2d/bf16_r2",
        "r1_stage": "073_FULL_INFERENCE_R1_7",
        "r1_int4_stage": "073_FULL_INFERENCE_R1_8",
        "r2_stage": "076_R2_4_BF16_REPEAT",
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def canonical_manifest(root: Path, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"missing manifest identity: {value!r}")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return os.path.normcase(os.path.normpath(str(candidate)))


def assert_no_gt_metadata(meta: Dict[str, Any], path: Path) -> None:
    if meta.get("GT_accessed") is not False:
        raise RuntimeError(f"GT_accessed is not false: {path}")
    if meta.get("scientific_metrics_computed") is not False:
        raise RuntimeError(f"inference cache already has scientific metrics: {path}")


def validate_cache(
    root: Path,
    directory: Path,
    expected: Dict[str, Any],
    expected_backend: str,
    expected_stage: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta_path = directory / "metadata.json"
    prediction_path = directory / "predictions.jsonl"
    meta = load_json(meta_path)
    rows = load_jsonl(prediction_path)

    if meta.get("status") != "PASS":
        raise RuntimeError(f"cache status is not PASS: {meta_path}")
    if meta.get("stage") != expected_stage:
        raise RuntimeError(f"unexpected stage in {meta_path}: {meta.get('stage')!r}")
    if meta.get("model") != expected["model"] or meta.get("dataset") != expected["dataset"]:
        raise RuntimeError(f"model/dataset mismatch: {meta_path}")
    if meta.get("backend") != expected_backend:
        raise RuntimeError(f"backend mismatch in {meta_path}: {meta.get('backend')!r}")
    assert_no_gt_metadata(meta, meta_path)
    canonical_manifest(root, meta.get("manifest"))

    if len(rows) != expected["n"]:
        raise RuntimeError(f"unexpected row count in {prediction_path}: {len(rows)}")
    ids: List[str] = []
    source_rows: List[int] = []
    revisions: List[str] = []
    for index, row in enumerate(rows):
        if row.get("model") != expected["model"] or row.get("backend") != expected_backend:
            raise RuntimeError(f"row model/backend mismatch at {prediction_path}:{index + 1}")
        if row.get("stage") != expected_stage:
            raise RuntimeError(f"row stage mismatch at {prediction_path}:{index + 1}")
        if row.get("GT_accessed") is not False or row.get("gold_accessed") is not False:
            raise RuntimeError(f"GT access flag is not false at {prediction_path}:{index + 1}")
        if row.get("ground_truth") is not None or row.get("correct") is not None:
            raise RuntimeError(f"outcome field is populated at {prediction_path}:{index + 1}")
        sample_id = row.get("sample_id")
        source_row = row.get("source_row")
        revision = row.get("dataset_revision")
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(f"invalid sample_id at {prediction_path}:{index + 1}")
        if not isinstance(source_row, int):
            raise RuntimeError(f"invalid source_row at {prediction_path}:{index + 1}")
        if not isinstance(revision, str) or not revision:
            raise RuntimeError(f"invalid dataset_revision at {prediction_path}:{index + 1}")
        if row.get("model_revision") != meta.get("model_revision"):
            raise RuntimeError(f"model revision mismatch at {prediction_path}:{index + 1}")
        ids.append(sample_id)
        source_rows.append(source_row)
        revisions.append(revision)
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate sample_id in {prediction_path}")
    if len(set(source_rows)) != len(source_rows):
        raise RuntimeError(f"duplicate source_row in {prediction_path}")
    if len(set(revisions)) != 1:
        raise RuntimeError(f"dataset_revision is not constant in {prediction_path}")
    return meta, rows


def assert_pair_alignment(
    root: Path,
    left_meta: Dict[str, Any],
    left_rows: Sequence[Dict[str, Any]],
    right_meta: Dict[str, Any],
    right_rows: Sequence[Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:
    if len(left_rows) != len(right_rows):
        raise RuntimeError(f"row-count mismatch for {label}")
    left_ids = [row["sample_id"] for row in left_rows]
    right_ids = [row["sample_id"] for row in right_rows]
    left_sources = [row["source_row"] for row in left_rows]
    right_sources = [row["source_row"] for row in right_rows]
    if left_ids != right_ids:
        raise RuntimeError(f"canonical sample-id order mismatch for {label}")
    if left_sources != right_sources:
        raise RuntimeError(f"canonical source_row order mismatch for {label}")
    left_revision = {row["dataset_revision"] for row in left_rows}
    right_revision = {row["dataset_revision"] for row in right_rows}
    if left_revision != right_revision:
        raise RuntimeError(f"dataset_revision mismatch for {label}")
    left_manifest = canonical_manifest(root, left_meta["manifest"])
    right_manifest = canonical_manifest(root, right_meta["manifest"])
    if left_manifest != right_manifest:
        raise RuntimeError(f"manifest identity mismatch for {label}")
    if left_meta.get("model_revision") != right_meta.get("model_revision"):
        raise RuntimeError(f"model revision mismatch for {label}")
    return {
        "sample_count": len(left_rows),
        "canonical_sample_id_order_equal": True,
        "canonical_source_row_order_equal": True,
        "dataset_revision": next(iter(left_revision)),
        "manifest_identity_equal": True,
        "model_revision_equal": True,
        "GT_accessed": False,
        "ground_truth_read": False,
        "outcome_fields_read": False,
    }


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


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
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


def kendall_tau_a(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        left_tail = left[i + 1 :]
        right_tail = right[i + 1 :]
        for left_delta, right_delta in zip(
            (left[i] - value for value in left_tail),
            (right[i] - value for value in right_tail),
        ):
            product = left_delta * right_delta
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    return (concordant - discordant) / (0.5 * n * (n - 1))


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    """Tie-corrected Kendall tau-b, frozen by Amendment 003."""
    n = len(left)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    ties_left_only = 0
    ties_right_only = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta_left = left[i] - left[j]
            delta_right = right[i] - right[j]
            if delta_left == 0.0 and delta_right == 0.0:
                continue
            if delta_left == 0.0:
                ties_left_only += 1
            elif delta_right == 0.0:
                ties_right_only += 1
            elif delta_left * delta_right > 0.0:
                concordant += 1
            else:
                discordant += 1
    numerator = concordant - discordant
    denominator = math.sqrt(
        (concordant + discordant + ties_left_only)
        * (concordant + discordant + ties_right_only)
    )
    if denominator == 0.0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def risk_from_row(row: Dict[str, Any]) -> float:
    probabilities = row.get("candidate_probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise RuntimeError("candidate_probabilities is missing or not an object")
    values = [float(value) for value in probabilities.values()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("candidate_probabilities contains a non-finite value")
    risk = 1.0 - max(values)
    stored = row.get("risk_primary")
    if not isinstance(stored, (int, float)) or not math.isclose(float(stored), risk, rel_tol=0.0, abs_tol=2e-6):
        raise RuntimeError("risk_primary does not match 1-max(candidate_probabilities)")
    return risk


def risk_vector(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[float], List[str]]:
    ids = [row["sample_id"] for row in rows]
    risks = [risk_from_row(row) for row in rows]
    answers = []
    for row in rows:
        answer = row.get("normalized_prediction")
        if not isinstance(answer, str):
            raise RuntimeError("normalized_prediction is missing")
        answers.append(answer)
    return ids, risks, answers


def ranked_ids(ids: Sequence[str], risks: Sequence[float], source_rows: Sequence[int]) -> List[str]:
    return [
        sample_id
        for sample_id, _, source_row in sorted(
            zip(ids, risks, source_rows), key=lambda item: (-item[1], item[2])
        )
    ]


def tail_jaccard(
    ids_a: Sequence[str],
    risks_a: Sequence[float],
    source_rows_a: Sequence[int],
    ids_b: Sequence[str],
    risks_b: Sequence[float],
    source_rows_b: Sequence[int],
    quantile: float,
) -> float:
    size = max(1, math.ceil(quantile * len(ids_a)))
    top_a = set(ranked_ids(ids_a, risks_a, source_rows_a)[:size])
    top_b = set(ranked_ids(ids_b, risks_b, source_rows_b)[:size])
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 1.0


def rbo(ranking_a: Sequence[str], ranking_b: Sequence[str], persistence: float) -> float:
    if not 0.0 < persistence < 1.0:
        raise ValueError("RBO persistence must be in (0, 1)")
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


def risk_tie_summary(risks: Sequence[float]) -> Dict[str, Any]:
    counts = Counter(risks)
    tied_groups = [count for count in counts.values() if count > 1]
    return {
        "n_unique_risk": len(counts),
        "n_tied_samples": sum(tied_groups),
        "max_tie_group_size": max(tied_groups, default=1),
    }


def boundary_tie_size(risks: Sequence[float], quantile: float) -> int:
    size = max(1, math.ceil(quantile * len(risks)))
    boundary_risk = sorted(risks, reverse=True)[size - 1]
    return sum(risk == boundary_risk for risk in risks)


def pair_tie_audit(risks_left: Sequence[float], risks_right: Sequence[float]) -> Dict[str, Any]:
    boundaries: Dict[str, Dict[str, int]] = {}
    for name, quantile in (("top5", 0.05), ("top10", 0.10), ("top20", 0.20), ("top30", 0.30)):
        boundaries[f"{name}_boundary_tie_size"] = {
            "left": boundary_tie_size(risks_left, quantile),
            "right": boundary_tie_size(risks_right, quantile),
        }
    return {
        "left": risk_tie_summary(risks_left),
        "right": risk_tie_summary(risks_right),
        **boundaries,
    }


def pair_metrics(left_rows: Sequence[Dict[str, Any]], right_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids_left, risks_left, answers_left = risk_vector(left_rows)
    ids_right, risks_right, answers_right = risk_vector(right_rows)
    if ids_left != ids_right:
        raise RuntimeError("pair metrics received non-identical canonical IDs")
    source_rows_left = [row["source_row"] for row in left_rows]
    source_rows_right = [row["source_row"] for row in right_rows]
    if source_rows_left != source_rows_right:
        raise RuntimeError("pair metrics received non-identical source_row alignment")
    ranking_left = ranked_ids(ids_left, risks_left, source_rows_left)
    ranking_right = ranked_ids(ids_right, risks_right, source_rows_right)
    metrics: Dict[str, Any] = {
        "spearman": spearman(risks_left, risks_right),
        "kendall": kendall_tau_b(risks_left, risks_right),
        "kendall_tau_a_diagnostic_only": kendall_tau_a(risks_left, risks_right),
        "J5": tail_jaccard(ids_left, risks_left, source_rows_left, ids_right, risks_right, source_rows_right, 0.05),
        "J10": tail_jaccard(ids_left, risks_left, source_rows_left, ids_right, risks_right, source_rows_right, 0.10),
        "J20": tail_jaccard(ids_left, risks_left, source_rows_left, ids_right, risks_right, source_rows_right, 0.20),
        "J30": tail_jaccard(ids_left, risks_left, source_rows_left, ids_right, risks_right, source_rows_right, 0.30),
        "RBO": {
            "p=0.90": rbo(ranking_left, ranking_right, 0.90),
            "p=0.80": rbo(ranking_left, ranking_right, 0.80),
            "p=0.95": rbo(ranking_left, ranking_right, 0.95),
        },
        "answer_flip_rate": sum(a != b for a, b in zip(answers_left, answers_right)) / len(answers_left),
        "tie_audit": pair_tie_audit(risks_left, risks_right),
    }
    return metrics


def process_cell(root: Path, cell_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    base = root / "artifacts" / "eswa_revision_v1"
    r1_bf16_meta, r1_bf16_rows = validate_cache(
        root, base / config["r1_bf16"], config, "bf16", config["r1_stage"]
    )
    r1_int4_meta, r1_int4_rows = validate_cache(
        root, base / config["r1_int4"], config, "int4_g128", config["r1_int4_stage"]
    )
    r2_meta, r2_rows = validate_cache(
        root, base / config["r2_bf16"], config, "bf16", config["r2_stage"]
    )
    # R1 BF16 and R1 INT4 use the same R1 stage family but must have distinct
    # backends; the row-level stage is checked above using the matching R1_2/4/6/8 value.
    audit_repeat = assert_pair_alignment(root, r1_bf16_meta, r1_bf16_rows, r2_meta, r2_rows, f"{cell_name}: BF16-R1 vs BF16-R2")
    audit_quant = assert_pair_alignment(root, r1_bf16_meta, r1_bf16_rows, r1_int4_meta, r1_int4_rows, f"{cell_name}: BF16-R1 vs INT4")
    return {
        "model": config["model"],
        "dataset": config["dataset_label"],
        "n": config["n"],
        "pairing_audit": {
            "BF16_R1_vs_BF16_R2": audit_repeat,
            "BF16_R1_vs_INT4_R1": audit_quant,
        },
        "same_precision_noise_floor": pair_metrics(r1_bf16_rows, r2_rows),
        "quantization_contrast": pair_metrics(r1_bf16_rows, r1_int4_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to artifacts/eswa_revision_v1/04_noise_floor/077_noise_floor_v1_3.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")
    results = {cell: process_cell(root, cell, config) for cell, config in EXPECTED.items()}
    output = args.output or root / "artifacts" / "eswa_revision_v1" / "04_noise_floor" / "077_noise_floor_v1_3.json"
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "stage": "077_NOISE_FLOOR_V1_3",
        "status": "PASS",
        "protocol_version": "v1.3",
        "amendment_id": "VLM_ESWA_REVISION_AMENDMENT_003",
        "amendment_scope": "CPU-only clarification of previously underspecified rank-metric tie handling",
        "risk_definition": "risk_max = 1 - max(candidate_probabilities)",
        "tail_size_rule": "ceil(q*n)",
        "ranking_rule": "descending risk_max; canonical source_row ascending for exact ties",
        "kendall_definition": "Kendall tau-b (tie-corrected)",
        "kendall_tau_a": "diagnostic_only",
        "rbo_variant": "RBO_EXT",
        "metrics": ["Spearman", "Kendall_tau_b", "J5", "J10", "J20", "J30", "RBO_EXT", "answer_flip_rate"],
        "RBO": {"primary_p": 0.90, "sensitivity_p": [0.80, 0.95]},
        "GT_accessed": False,
        "GT_join_started": False,
        "ground_truth_read": False,
        "outcome_fields_read": False,
        "accuracy_computed": False,
        "noise_floor_metrics_computed": True,
        "scientific_metrics_computed": True,
        "cells": results,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    tie_audit_output = output.with_name("077_tie_audit_v1_3.json")
    tie_audit_artifact = {
        "stage": "077_TIE_AUDIT_V1_3",
        "status": "PASS",
        "protocol_version": "v1.3",
        "amendment_id": "VLM_ESWA_REVISION_AMENDMENT_003",
        "description": "Descriptive risk-tie and queue-boundary audit performed before canonical 077 v1.3 reporting.",
        "GT_accessed": False,
        "GT_join_started": False,
        "outcome_fields_read": False,
        "hashes_used": False,
        "cells": {
            cell: {
                "same_precision_BF16_R1_vs_BF16_R2": value["same_precision_noise_floor"]["tie_audit"],
                "quantization_BF16_R1_vs_INT4_R1": value["quantization_contrast"]["tie_audit"],
            }
            for cell, value in results.items()
        },
    }
    tie_audit_temporary = tie_audit_output.with_suffix(tie_audit_output.suffix + ".tmp")
    with tie_audit_temporary.open("w", encoding="utf-8") as handle:
        json.dump(tie_audit_artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tie_audit_temporary, tie_audit_output)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
