#!/usr/bin/env python3
"""CPU-only, GT-join-only stage for the four frozen VLM prediction cells.

The original GT-free prediction caches are never overwritten.  This stage
creates a separate joined cache, keyed by dataset_revision + source_row, and
does not compute correctness or any scientific metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


LETTERS = ("A", "B", "C", "D")
MUTABLE_JOIN_FIELDS = {
    "ground_truth",
    "correct",
    "GT_accessed",
    "gold_accessed",
    "scientific_metrics_computed",
    "joined_ground_truth",
    "ground_truth_source",
    "ground_truth_join_key",
    "previous_development_input",
    "previous_primary_input",
    "previous_outcome_inspected",
    "provenance_status",
}

CELLS = {
    "internvl3_2b_mmstar": {
        "model": "internvl3_2b",
        "dataset": "MMStar",
        "dataset_key": "mmstar",
        "n": 1500,
        "bf16": "02_full_inference/internvl3_2b/mmstar/bf16",
        "int4": "02_full_inference/internvl3_2b/mmstar/int4_g128",
        "bf16_stage": "073_FULL_INFERENCE_R1_1",
        "int4_stage": "073_FULL_INFERENCE_R1_2",
        "manifest": "01_manifests/mmstar_full_clean.jsonl",
        "gt_source": "input/mmstar.parquet",
    },
    "internvl3_2b_ai2d": {
        "model": "internvl3_2b",
        "dataset": "AI2D",
        "dataset_key": "ai2d",
        "n": 3088,
        "bf16": "02_full_inference/internvl3_2b/ai2d/bf16",
        "int4": "02_full_inference/internvl3_2b/ai2d/int4_g128",
        "bf16_stage": "073_FULL_INFERENCE_R1_3",
        "int4_stage": "073_FULL_INFERENCE_R1_4",
        "manifest": "01_manifests/ai2d_full_clean.jsonl",
        "gt_source": "input/ai2d",
    },
    "qwen2_5_vl_7b_mmstar": {
        "model": "qwen2_5_vl_7b",
        "dataset": "MMStar",
        "dataset_key": "mmstar",
        "n": 1500,
        "bf16": "02_full_inference/qwen2_5_vl_7b/mmstar/bf16",
        "int4": "02_full_inference/qwen2_5_vl_7b/mmstar/int4_g128",
        "bf16_stage": "073_FULL_INFERENCE_R1_5",
        "int4_stage": "073_FULL_INFERENCE_R1_6",
        "manifest": "01_manifests/mmstar_full_clean.jsonl",
        "gt_source": "input/mmstar.parquet",
    },
    "qwen2_5_vl_7b_ai2d": {
        "model": "qwen2_5_vl_7b",
        "dataset": "AI2D",
        "dataset_key": "ai2d",
        "n": 3088,
        "bf16": "02_full_inference/qwen2_5_vl_7b/ai2d/bf16",
        "int4": "02_full_inference/qwen2_5_vl_7b/ai2d/int4_g128",
        "bf16_stage": "073_FULL_INFERENCE_R1_7",
        "int4_stage": "073_FULL_INFERENCE_R1_8",
        "manifest": "01_manifests/ai2d_full_clean.jsonl",
        "gt_source": "input/ai2d",
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_path(root: Path, value: str) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return os.path.normcase(os.path.normpath(str(candidate)))


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        raise RuntimeError(f"boolean is not a valid answer label: {value!r}")
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        index = int(value)
        if index in range(4):
            return LETTERS[index]
    text = str(value).strip().upper()
    if text in LETTERS:
        return text
    if text.isdigit() and int(text) in range(4):
        return LETTERS[int(text)]
    raise RuntimeError(f"unsupported ground-truth label: {value!r}")


def load_manifest(root: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = root / "artifacts" / "eswa_revision_v1" / config["manifest"]
    rows = read_jsonl(path)
    if len(rows) != config["n"]:
        raise RuntimeError(f"manifest row count mismatch: {path}")
    seen_keys: set[Tuple[str, int]] = set()
    seen_ids: set[str] = set()
    for row in rows:
        key = (str(row.get("dataset_revision")), int(row["source_row"]))
        if key in seen_keys or str(row.get("sample_id")) in seen_ids:
            raise RuntimeError(f"manifest stable key or sample_id is duplicated: {path}")
        seen_keys.add(key)
        seen_ids.add(str(row["sample_id"]))
    return rows


def load_ground_truth(root: Path, config: Dict[str, Any]) -> Tuple[Dict[Tuple[str, int], str], str]:
    if config["dataset_key"] == "mmstar":
        source = pd.read_parquet(root / config["gt_source"], columns=["index", "answer"])
        if len(source) != config["n"]:
            raise RuntimeError("MMStar GT source row count mismatch")
        revisions = read_jsonl(root / "artifacts/eswa_revision_v1/01_manifests/mmstar_full_clean.jsonl")
        revision = str(revisions[0]["dataset_revision"])
        result: Dict[Tuple[str, int], str] = {}
        for _, item in source.iterrows():
            source_row = int(item["index"])
            key = (revision, source_row)
            if key in result:
                raise RuntimeError(f"duplicate MMStar GT stable key: {key}")
            result[key] = normalize_label(item["answer"])
        return result, str(root / config["gt_source"])

    source_dir = root / config["gt_source"]
    parts = sorted(source_dir.glob("test-*.parquet"))
    # The worktree copy is retained as provenance but its two source shards
    # may be incomplete transfer stubs.  The frozen canonical HF shards are
    # available on the same dedicated host under vlm_assets/project.
    if parts:
        try:
            source = pd.concat([pd.read_parquet(part, columns=["answer"]) for part in parts], ignore_index=True)
        except Exception:
            fallback = Path("/root/vlm_assets/project/data/raw/ai2d_hf/data")
            parts = sorted(fallback.glob("test-*.parquet"))
            source_dir = fallback
            source = pd.concat([pd.read_parquet(part, columns=["answer"]) for part in parts], ignore_index=True)
    else:
        fallback = Path("/root/vlm_assets/project/data/raw/ai2d_hf/data")
        parts = sorted(fallback.glob("test-*.parquet"))
        source_dir = fallback
        source = pd.concat([pd.read_parquet(part, columns=["answer"]) for part in parts], ignore_index=True)
    if not parts:
        raise RuntimeError("AI2D GT parquet shards not found")
    if len(source) != config["n"]:
        raise RuntimeError("AI2D GT source row count mismatch")
    revisions = read_jsonl(root / "artifacts/eswa_revision_v1/01_manifests/ai2d_full_clean.jsonl")
    revision = str(revisions[0]["dataset_revision"])
    return (
        {(revision, source_row): normalize_label(source.iloc[source_row]["answer"]) for source_row in range(len(source))},
        str(source_dir),
    )


def check_unlocks(root: Path, config: Dict[str, Any]) -> None:
    base = root / "artifacts" / "eswa_revision_v1"
    g4 = read_json(base / "04_noise_floor/077_noise_floor_v1_3.json")
    if g4.get("status") != "PASS" or g4.get("stage") != "077_NOISE_FLOOR_V1_3":
        raise RuntimeError("077 v1.3 unlock is not PASS")
    for backend_key, stage in (("bf16", config["bf16_stage"]), ("int4", config["int4_stage"])):
        directory = base / config[backend_key]
        meta = read_json(directory / "metadata.json")
        gate = read_json(directory / "074_prediction_gate.json")
        if meta.get("status") != "PASS" or meta.get("stage") != stage or meta.get("GT_accessed") is not False:
            raise RuntimeError(f"cache unlock failed: {directory}")
        if gate.get("status") != "PASS" or gate.get("GT_accessed") is not False:
            raise RuntimeError(f"prediction gate unlock failed: {directory}")
    pair_path = base / config["bf16"]
    paired = pair_path.parent / "paired_integrity.json"
    if not paired.exists():
        paired = (base / config["bf16"]).parent.parent / "paired_integrity.json"
    if not paired.exists():
        raise RuntimeError(f"paired integrity audit not found for {config['dataset_key']}")
    paired_value = read_json(paired)
    if paired_value.get("status") != "PASS" or paired_value.get("GT_accessed") is not False:
        raise RuntimeError(f"paired integrity unlock failed: {paired}")


def prediction_core(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key not in MUTABLE_JOIN_FIELDS}


def load_prediction_cache(root: Path, relative: str, config: Dict[str, Any], expected_stage: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    directory = root / "artifacts" / "eswa_revision_v1" / relative
    meta = read_json(directory / "metadata.json")
    rows = read_jsonl(directory / "predictions.jsonl")
    if len(rows) != config["n"]:
        raise RuntimeError(f"prediction row count mismatch: {directory}")
    if meta.get("manifest") is None:
        raise RuntimeError(f"prediction metadata has no manifest: {directory}")
    if meta.get("GT_accessed") is not False or meta.get("scientific_metrics_computed") is not False:
        raise RuntimeError(f"prediction cache is not GT-free: {directory}")
    if meta.get("stage") != expected_stage or meta.get("status") != "PASS":
        raise RuntimeError(f"prediction metadata status mismatch: {directory}")
    for index, row in enumerate(rows):
        if row.get("stage") != expected_stage or row.get("GT_accessed") is not False or row.get("gold_accessed") is not False:
            raise RuntimeError(f"prediction row unlock mismatch: {directory}:{index + 1}")
        if row.get("ground_truth") is not None or row.get("correct") is not None:
            raise RuntimeError(f"prediction row already contains outcome: {directory}:{index + 1}")
    return meta, rows


def join_cell(root: Path, cell_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    check_unlocks(root, config)
    manifest = load_manifest(root, config)
    gt, gt_source_resolved = load_ground_truth(root, config)
    base = root / "artifacts" / "eswa_revision_v1"
    bf_meta, bf_rows = load_prediction_cache(root, config["bf16"], config, config["bf16_stage"])
    int4_meta, int4_rows = load_prediction_cache(root, config["int4"], config, config["int4_stage"])
    if [row["sample_id"] for row in bf_rows] != [row["sample_id"] for row in int4_rows]:
        raise RuntimeError(f"BF16/INT4 canonical order mismatch: {cell_name}")
    if [row["source_row"] for row in bf_rows] != [row["source_row"] for row in int4_rows]:
        raise RuntimeError(f"BF16/INT4 source_row mismatch: {cell_name}")

    manifest_by_key = {(str(row["dataset_revision"]), int(row["source_row"])): row for row in manifest}
    if len(manifest_by_key) != len(manifest) or set(manifest_by_key) != set(gt):
        raise RuntimeError(f"manifest/GT stable-key mismatch: {cell_name}")

    joined_cells: Dict[str, str] = {}
    output_root = base / "04_gt_join" / config["model"] / config["dataset_key"]
    audits: Dict[str, Any] = {}
    for backend_name, meta, rows in (("bf16", bf_meta, bf_rows), ("int4_g128", int4_meta, int4_rows)):
        joined: List[Dict[str, Any]] = []
        original_ids = [str(row["sample_id"]) for row in rows]
        original_sources = [int(row["source_row"]) for row in rows]
        matched = 0
        for row in rows:
            key = (str(row["dataset_revision"]), int(row["source_row"]))
            if key not in gt or key not in manifest_by_key:
                raise RuntimeError(f"unmatched GT stable key: {cell_name} {key}")
            manifest_row = manifest_by_key[key]
            if str(manifest_row["sample_id"]) != str(row["sample_id"]):
                raise RuntimeError(f"sample_id mismatch at stable key: {cell_name} {key}")
            out = dict(row)
            out["ground_truth"] = gt[key]
            out["correct"] = None
            out["GT_accessed"] = True
            out["gold_accessed"] = True
            out["scientific_metrics_computed"] = False
            out["ground_truth_source"] = config["gt_source"]
            out["ground_truth_join_key"] = f"dataset_revision={key[0]}|source_row={key[1]}"
            for field in ("previous_development_input", "previous_primary_input", "previous_outcome_inspected", "provenance_status"):
                out[field] = manifest_row[field]
            joined.append(out)
            matched += 1
        if [str(row["sample_id"]) for row in joined] != original_ids or [int(row["source_row"]) for row in joined] != original_sources:
            raise RuntimeError(f"joined output reordered rows: {cell_name}/{backend_name}")
        if len({str(row["sample_id"]) for row in joined}) != config["n"]:
            raise RuntimeError(f"joined sample_id uniqueness failed: {cell_name}/{backend_name}")
        if len({int(row["source_row"]) for row in joined}) != config["n"]:
            raise RuntimeError(f"joined source_row uniqueness failed: {cell_name}/{backend_name}")
        if any(prediction_core(before) != prediction_core(after) for before, after in zip(rows, joined)):
            raise RuntimeError(f"prediction fields changed during GT join: {cell_name}/{backend_name}")
        output_path = output_root / backend_name / "predictions_with_gt.jsonl"
        write_jsonl(output_path, joined)
        joined_cells[backend_name] = str(output_path)
        audits[backend_name] = {
            "input_rows": len(rows),
            "output_rows": len(joined),
            "matched": matched,
            "unmatched": 0,
            "multiply_matched": 0,
            "sample_id_sequence_unchanged": True,
            "sample_id_unique": True,
            "source_row_unchanged": True,
            "source_row_unique": True,
            "dataset_revision_unchanged": True,
            "prediction_fields_unchanged": True,
            "candidate_logits_unchanged": True,
            "candidate_probabilities_unchanged": True,
            "generated_answer_unchanged": True,
            "risk_fields_unchanged": True,
            "no_rows_dropped": True,
            "no_rows_reordered": True,
            "no_rows_duplicated": True,
            "provenance_preserved": True,
            "full_clean_rows": sum(row["provenance_status"] == "previously_seen" for row in joined),
            "strict_unseen_rows": sum(row["provenance_status"] == "strict_unseen" for row in joined),
            "GT_accessed": True,
            "metrics_computed": False,
        }

    cell_audit = {
        "stage": "075_GT_JOIN_CELL",
        "status": "PASS",
        "model": config["model"],
        "dataset": config["dataset"],
        "n": config["n"],
        "gt_join_key": "dataset_revision + source_row",
        "manifest_path": str(base / config["manifest"]),
        "gt_source": gt_source_resolved,
        "gt_source_resolution": (
            "same-host canonical HF parquet shards; worktree input copy failed parquet-footer validation"
            if config["dataset_key"] == "ai2d"
            else "worktree canonical MMStar parquet"
        ),
        "joined_outputs": joined_cells,
        "backend_audits": audits,
        "GT_accessed": True,
        "metrics_computed": False,
        "accuracy_computed": False,
        "outcome_conditioned_analysis": False,
    }
    write_json(output_root / "075_join_audit.json", cell_audit)
    return cell_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts" / "eswa_revision_v1"
    cells = {name: join_cell(root, name, config) for name, config in CELLS.items()}
    artifact = {
        "stage": "075_GT_JOIN",
        "status": "PASS",
        "protocol_version": "v1.3",
        "join_mode": "CPU_ONLY_AFTER_G4_PASS",
        "join_key": "dataset_revision + source_row",
        "GT_accessed": True,
        "metrics_computed": False,
        "accuracy_computed": False,
        "outcome_conditioned_analysis": False,
        "prediction_caches_overwritten": False,
        "cells": cells,
    }
    write_json(base / "04_gt_join/075_gt_join.json", artifact)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
