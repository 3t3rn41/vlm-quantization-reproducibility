#!/usr/bin/env python3
"""Build the isolated VLM-ESWA R0 full-clean manifests.

This is deliberately a CPU-only, outcome-blind preparation step.  It excludes
answer/label columns at read time and never writes them to inference manifests.
Existing 300-row inputs are mapped back to canonical source rows by actual
question text and image bytes.  The mapping is kept as provenance evidence;
the old sample identifier is retained only when that actual input is uniquely
resolved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from io import BytesIO


FORBIDDEN = {
    "answer", "label", "gold", "ground_truth", "correct", "correct_answer",
}
MM_REV = "bc98d668301da7b14f648724866e57302778ab27"
AI_REV = "c83a9b9692933aff8349157c88a413df9d02c4e5"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def bytes_from_image(value: Any) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"unsupported image value: {type(value).__name__}")
    return bytes(value)


def image_size(data: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(data)) as im:
        return im.size


def source_identity(dataset: str, source_row: int, image_path: str) -> str:
    return f"{dataset}|source_row={source_row}|image_path={image_path}"


def normalize_options(options: Any) -> list[str]:
    if hasattr(options, "tolist"):
        options = options.tolist()
    return [str(x) for x in options]


def ai2d_question(question: str, options: list[str]) -> str:
    letters = "ABCD"
    rendered = "\n".join(f"{letters[i]}: {options[i]}" for i in range(4))
    return f"Question: {question}\nOptions:\n{rendered}"


def resolve_prior_image(path_value: str, prior_manifest: Path) -> Path:
    p = Path(path_value)
    normalized_name = Path(str(path_value).replace("\\", "/")).name
    candidates = [
        p,
        prior_manifest.parent / p,
        prior_manifest.parent.parent / p,
        Path.cwd() / p,
        # Prior manifests may have been produced on Windows and copied to the
        # Linux experiment host.  The basename fallback preserves the actual
        # image identity without trusting the stale local absolute prefix.
        prior_manifest.parent / normalized_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"prior image not found: {path_value}")


def assert_no_forbidden(row: dict[str, Any]) -> None:
    bad = FORBIDDEN.intersection(row)
    if bad:
        raise AssertionError(f"forbidden label fields in inference row: {sorted(bad)}")


def write_image_once(path: Path, data: bytes) -> None:
    """Write a derived image, but fail closed if a prior partial run disagrees."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"existing derived image disagrees with canonical bytes: {path}")
        return
    path.write_bytes(data)


def build_mmstar(args: argparse.Namespace, out_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = ["index", "question", "category", "l2_category", "image", "meta_info"]
    source = pd.read_parquet(args.mmstar_parquet, columns=required)
    if len(source) != 1500:
        raise RuntimeError(f"MMStar canonical row count is {len(source)}, expected 1500")
    canonical_indices = [int(x) for x in source["index"].tolist()]
    if set(canonical_indices) != set(range(1500)):
        raise RuntimeError("MMStar canonical index column is not exactly 0..1499")

    pilot = load_jsonl(args.mmstar_pilot_manifest)
    if len(pilot) != 300:
        raise RuntimeError(f"MMStar frozen pilot manifest has {len(pilot)} rows, expected 300")
    pilot_by_id: dict[str, int] = {}
    for pilot_row in pilot:
        sid = str(pilot_row.get("sample_id", ""))
        source_index = int(pilot_row.get("source_index"))
        if not sid or sid in pilot_by_id or not 0 <= source_index < 1500:
            raise RuntimeError("MMStar frozen pilot manifest has invalid or non-unique sample identity")
        pilot_by_id[sid] = source_index

    # The canonical source is indexed by actual input content.  If duplicated
    # content occurs in multiple official rows, the frozen pilot's canonical
    # source_index is used only as a tie-breaker; the old shuffled position is
    # never treated as the authoritative source row.
    canonical_by_key: dict[tuple[str, bytes], list[int]] = {}
    canonical_payload: dict[int, tuple[Any, bytes]] = {}
    for _, item in source.iterrows():
        source_row = int(item["index"])
        raw = bytes_from_image(item["image"])
        canonical_payload[source_row] = (item, raw)
        key = (str(item["question"]), raw)
        canonical_by_key.setdefault(key, []).append(source_row)

    prior = load_jsonl(args.mmstar_prior_manifest)
    prior_sample_ids: set[str] = set()
    old_by_source_row: dict[int, dict[str, Any]] = {}
    candidate_rows_by_id: dict[str, list[int]] = {}
    tie_break_ids: set[str] = set()
    for old in prior:
        if "question" not in old or "image_path" not in old:
            raise RuntimeError("MMStar prior manifest lacks actual question/image identity")
        old_sample_id = str(old.get("sample_id", ""))
        if not old_sample_id or old_sample_id in prior_sample_ids:
            raise RuntimeError("MMStar prior manifest sample_id is missing or not unique")
        prior_sample_ids.add(old_sample_id)
        image_path = resolve_prior_image(str(old["image_path"]), args.mmstar_prior_manifest)
        key = (str(old["question"]), image_path.read_bytes())
        candidates = canonical_by_key.get(key, [])
        candidate_rows_by_id[old_sample_id] = list(candidates)
        if not candidates:
            raise RuntimeError(f"MMStar historical input has no canonical source-row match: {old_sample_id}")
        if len(candidates) > 1:
            pilot_source_index = pilot_by_id.get(old_sample_id)
            if pilot_source_index not in candidates:
                raise RuntimeError(
                    f"MMStar duplicate-content mapping cannot be resolved for {old_sample_id}: "
                    f"candidates={candidates}, frozen_pilot_source_index={pilot_source_index}"
                )
            selected_source_row = pilot_source_index
            tie_break_ids.add(old_sample_id)
        else:
            selected_source_row = candidates[0]
        if selected_source_row in old_by_source_row:
            raise RuntimeError(f"MMStar historical rows collide at canonical source_row={selected_source_row}")
        old_by_source_row[selected_source_row] = {"old": old, "image_path": image_path}

    image_root = out_root / "images" / "mmstar"
    rows: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    resolved_prior_ids: set[str] = set()
    for pos, item in source.iterrows():
        source_row = int(item["index"])
        _, raw = canonical_payload[source_row]
        meta = item["meta_info"] if isinstance(item["meta_info"], dict) else {}
        canonical_path = str(meta.get("image_path", f"images/{source_row}.jpg"))
        historical = old_by_source_row.get(source_row)
        old = historical["old"] if historical is not None else None
        if old is not None:
            resolved_prior_ids.add(str(old["sample_id"]))
        sample_id = str(old["sample_id"]) if old is not None else f"mmstar-source-row-{source_row:04d}"
        out_image = image_root / f"{sample_id}.jpg"
        write_image_once(out_image, raw)
        width, height = image_size(raw)
        old_source_row = int(old["source_row"]) if old is not None and "source_row" in old else None
        old_id = str(old["sample_id"]) if old is not None else None
        row = {
            "sample_id": sample_id,
            "dataset": "MMStar",
            "dataset_revision": MM_REV,
            "config": "val",
            "split": "val",
            "manifest_source_index": pos,
            "source_row": source_row,
            "input_fingerprint_or_stable_key": source_identity("MMStar", source_row, canonical_path),
            "question": str(item["question"]),
            "category": str(item["category"]),
            "l2_category": str(item["l2_category"]),
            "image_path": str(out_image),
            "image_bytes": len(raw),
            "image_width": width,
            "image_height": height,
            "previous_development_input": old is not None,
            "previous_primary_input": old is not None,
            "previous_outcome_inspected": old is not None,
            "provenance_status": "previously_seen" if old is not None else "strict_unseen",
        }
        assert_no_forbidden(row)
        rows.append(row)
        mapping.append({
            "dataset": "MMStar",
            "source_row": source_row,
            "sample_id": sample_id,
            "old_sample_id": str(old["sample_id"]) if old is not None else None,
            "old_source_row": old_source_row,
            "pilot_source_index": pilot_by_id.get(old_id) if old_id is not None else None,
            "candidate_source_rows": candidate_rows_by_id.get(old_id, []) if old_id is not None else [],
            "matched_by": (
                "question_and_actual_image_bytes"
                + ("_plus_frozen_pilot_source_index_tiebreak" if old_id in tie_break_ids else "")
                if old is not None else None
            ),
            "mapping_status": "resolved" if old is not None else "new_strict_unseen",
        })
    unresolved_ids = sorted(prior_sample_ids - resolved_prior_ids)
    if unresolved_ids:
        raise RuntimeError(
            "MMStar historical mapping left prior rows unresolved: "
            + ", ".join(unresolved_ids[:10])
            + (" ..." if len(unresolved_ids) > 10 else "")
        )
    return rows, {
        "dataset": "MMStar", "canonical_rows": 1500, "prior_rows": len(prior),
        "resolved_prior_rows": len(resolved_prior_ids),
        "unresolved_prior_rows": len(unresolved_ids), "ambiguous_canonical_rows": 0,
        "ambiguous_input_groups": len(tie_break_ids),
        "mapping_strategy": "actual_question_and_image_bytes; frozen_pilot_source_index_only_for_duplicate-content_tiebreak",
        "mapping": mapping,
    }


def build_ai2d(args: argparse.Namespace, out_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parts = [pd.read_parquet(p, columns=["question", "options", "image"])
             for p in sorted(Path(args.ai2d_dir).glob("test-*.parquet"))]
    source = pd.concat(parts, ignore_index=True)
    if len(source) != 3088:
        raise RuntimeError(f"AI2D canonical row count is {len(source)}, expected 3088")

    prior = load_jsonl(args.ai2d_prior_manifest)
    prior_by_row = {int(x["source_row"]): x for x in prior}
    if len(prior_by_row) != len(prior):
        raise RuntimeError("AI2D prior source_row is not unique")
    if any(row < 0 or row >= 3088 for row in prior_by_row):
        raise RuntimeError("AI2D prior source_row is outside canonical 0..3087")
    if len({str(x.get("sample_id", "")) for x in prior}) != len(prior) or any(not x.get("sample_id") for x in prior):
        raise RuntimeError("AI2D prior manifest sample_id is missing or not unique")
    image_root = out_root / "images" / "ai2d"
    rows: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for source_row, item in source.iterrows():
        raw = bytes_from_image(item["image"])
        options = normalize_options(item["options"])
        if len(options) != 4:
            raise RuntimeError(f"AI2D source_row={source_row} does not have four options")
        prompt = ai2d_question(str(item["question"]), options)
        old = prior_by_row.get(int(source_row))
        if old is not None:
            old_image = resolve_prior_image(str(old["image_path"]), args.ai2d_prior_manifest)
            if old_image.read_bytes() != raw:
                raise RuntimeError(f"AI2D historical source_row/image mismatch at {source_row}")
        sample_id = str(old["sample_id"]) if old is not None else f"ai2d-source-row-{source_row:04d}"
        out_image = image_root / f"{sample_id}.png"
        write_image_once(out_image, raw)
        width, height = image_size(raw)
        row = {
            "sample_id": sample_id,
            "dataset": "AI2D",
            "dataset_revision": AI_REV,
            "config": "default",
            "split": "test",
            "manifest_source_index": source_row,
            "source_row": source_row,
            "input_fingerprint_or_stable_key": source_identity("AI2D", source_row, str(item["image"]["path"])),
            "question": prompt,
            "image_path": str(out_image),
            "image_bytes": len(raw),
            "image_width": width,
            "image_height": height,
            "number_of_options": 4,
            "previous_development_input": old is not None,
            "previous_primary_input": old is not None,
            "previous_outcome_inspected": old is not None,
            "provenance_status": "previously_seen" if old is not None else "strict_unseen",
        }
        assert_no_forbidden(row)
        rows.append(row)
        mapping.append({
            "dataset": "AI2D", "source_row": source_row, "sample_id": sample_id,
            "old_sample_id": str(old["sample_id"]) if old is not None else None,
            "matched_by": "canonical_source_row_and_actual_image_bytes" if old is not None else None,
            "mapping_status": "resolved" if old is not None else "new_strict_unseen",
        })
    return rows, {
        "dataset": "AI2D", "canonical_rows": 3088, "prior_rows": len(prior),
        "resolved_prior_rows": sum(x["mapping_status"] == "resolved" for x in mapping),
        "unresolved_prior_rows": 0, "ambiguous_canonical_rows": 0,
        "mapping": mapping,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmstar-parquet", type=Path, required=True)
    ap.add_argument("--ai2d-dir", type=Path, required=True)
    ap.add_argument("--mmstar-pilot-manifest", type=Path, required=True)
    ap.add_argument("--mmstar-prior-manifest", type=Path, required=True)
    ap.add_argument("--ai2d-prior-manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out = args.output_root
    manifests = out / "01_manifests"
    protocol = out / "00_protocol"
    mm_rows, mm_report = build_mmstar(args, out)
    ai_rows, ai_report = build_ai2d(args, out)
    all_rows = mm_rows + ai_rows
    for row in all_rows:
        assert_no_forbidden(row)
    dump_jsonl(manifests / "mmstar_full_clean.jsonl", mm_rows)
    dump_jsonl(manifests / "ai2d_full_clean.jsonl", ai_rows)
    registry = [{k: row[k] for k in [
        "sample_id", "dataset", "dataset_revision", "source_row",
        "input_fingerprint_or_stable_key", "previous_development_input",
        "previous_primary_input", "previous_outcome_inspected", "provenance_status",
    ]} for row in all_rows]
    dump_jsonl(manifests / "provenance_registry.jsonl", registry)
    seen = [r["sample_id"] for r in registry if r["provenance_status"] == "previously_seen"]
    unseen = [r["sample_id"] for r in registry if r["provenance_status"] == "strict_unseen"]
    dump_json(manifests / "previously_seen_ids.json", {"count": len(seen), "sample_ids": seen})
    dump_json(manifests / "strict_unseen_ids.json", {"count": len(unseen), "sample_ids": unseen})
    dump_json(protocol / "dataset_revisions.json", {
        "protocol": "VLM-ESWA-REVISION-R0",
        "outcome_blind": True,
        "labels_written_to_inference_manifest": False,
        "datasets": {
            "MMStar": {"source": "local canonical MMStar parquet", "revision": MM_REV, "split": "val", "rows": 1500},
            "AI2D": {"source": "local canonical HuggingFace parquet shards", "revision": AI_REV, "split": "test", "rows": 3088},
        },
    })
    dump_json(protocol / "r0_implementation_summary.json", {
        "protocol": "VLM-ESWA-REVISION-R0",
        "stage": "MANIFEST_IMPLEMENTATION",
        "scientific_inference_started": False,
        "gt_accessed": False,
        "scientific_metrics_computed": False,
        "datasets": {"MMStar": mm_report, "AI2D": ai_report},
        "counts": {"MMStar": len(mm_rows), "AI2D": len(ai_rows), "total": len(all_rows),
                   "previously_seen": len(seen), "strict_unseen": len(unseen)},
    })
    print(json.dumps({
        "status": "R0_MANIFESTS_BUILT",
        "MMStar": {"rows": len(mm_rows), "previously_seen": sum(r["provenance_status"] == "previously_seen" for r in mm_rows),
                   "strict_unseen": sum(r["provenance_status"] == "strict_unseen" for r in mm_rows),
                   "resolved_prior_rows": mm_report["resolved_prior_rows"], "ambiguous": mm_report["ambiguous_canonical_rows"]},
        "AI2D": {"rows": len(ai_rows), "previously_seen": sum(r["provenance_status"] == "previously_seen" for r in ai_rows),
                 "strict_unseen": sum(r["provenance_status"] == "strict_unseen" for r in ai_rows),
                 "resolved_prior_rows": ai_report["resolved_prior_rows"]},
        "GT_accessed": False, "scientific_metrics_computed": False,
        "output_root": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
