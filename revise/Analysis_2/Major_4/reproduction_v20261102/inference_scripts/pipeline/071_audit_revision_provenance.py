#!/usr/bin/env python3
"""Fail-closed R0 manifest/provenance audit for VLM-ESWA revision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN = {"answer", "label", "gold", "ground_truth", "correct", "correct_answer"}
EXPECTED = {"MMStar": 1500, "AI2D": 3088}
REQUIRED = {
    "sample_id", "dataset", "dataset_revision", "split", "manifest_source_index",
    "source_row", "input_fingerprint_or_stable_key", "question", "image_path",
    "previous_development_input", "previous_primary_input", "previous_outcome_inspected",
    "provenance_status",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fail(checks: dict[str, Any], name: str, detail: Any) -> None:
    checks[name] = {"pass": False, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.output_root
    man = root / "01_manifests"
    audit_dir = root / "02_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    errors: list[str] = []
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    try:
        mm = read_jsonl(man / "mmstar_full_clean.jsonl")
        ai = read_jsonl(man / "ai2d_full_clean.jsonl")
        registry = read_jsonl(man / "provenance_registry.jsonl")
        seen_doc = json.loads((man / "previously_seen_ids.json").read_text(encoding="utf-8"))
        unseen_doc = json.loads((man / "strict_unseen_ids.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"load_failure:{type(exc).__name__}:{exc}")
        mm, ai, registry, seen_doc, unseen_doc = [], [], [], {}, {}
    rows_by_dataset = {"MMStar": mm, "AI2D": ai}

    for dataset, rows in rows_by_dataset.items():
        expected = EXPECTED[dataset]
        if len(rows) != expected:
            errors.append(f"{dataset}:row_count={len(rows)} expected={expected}")
            fail(checks, f"{dataset}.row_count", {"actual": len(rows), "expected": expected})
        else:
            checks[f"{dataset}.row_count"] = {"pass": True, "actual": len(rows)}
        sample_ids = [r.get("sample_id") for r in rows]
        source_rows = [r.get("source_row") for r in rows]
        stable = [r.get("input_fingerprint_or_stable_key") for r in rows]
        if any(x in (None, "") for x in sample_ids) or len(set(sample_ids)) != len(rows):
            errors.append(f"{dataset}:sample_id_not_unique_or_null")
            fail(checks, f"{dataset}.sample_id_unique", False)
        else:
            checks[f"{dataset}.sample_id_unique"] = {"pass": True, "count": len(rows)}
        if any(x in (None, "") for x in stable) or len(set(stable)) != len(rows):
            errors.append(f"{dataset}:stable_key_not_unique_or_null")
            fail(checks, f"{dataset}.stable_key_unique", False)
        else:
            checks[f"{dataset}.stable_key_unique"] = {"pass": True, "count": len(rows)}
        if any(not isinstance(x, int) for x in source_rows) or len(set(source_rows)) != len(rows):
            errors.append(f"{dataset}:source_row_not_unique_or_null")
            fail(checks, f"{dataset}.source_row_unique", False)
        else:
            expected_rows = set(range(expected))
            if set(source_rows) != expected_rows:
                errors.append(f"{dataset}:source_row_domain_mismatch")
                fail(checks, f"{dataset}.source_row_domain", {"min": min(source_rows), "max": max(source_rows)})
            else:
                checks[f"{dataset}.source_row_unique"] = {"pass": True, "count": len(rows)}
                checks[f"{dataset}.source_row_domain"] = {"pass": True, "domain": f"0..{expected - 1}"}
        for i, row in enumerate(rows):
            missing = REQUIRED.difference(row)
            forbidden = FORBIDDEN.intersection(row)
            if missing:
                errors.append(f"{dataset}:row={i}:missing={sorted(missing)}")
            if forbidden:
                errors.append(f"{dataset}:row={i}:forbidden={sorted(forbidden)}")
            flags = [row.get(x) for x in [
                "previous_development_input", "previous_primary_input", "previous_outcome_inspected"]]
            if any(type(x) is not bool for x in flags):
                errors.append(f"{dataset}:row={i}:historical_flags_not_boolean")
            expected_status = "strict_unseen" if not any(flags) else "previously_seen"
            if row.get("provenance_status") != expected_status:
                errors.append(f"{dataset}:row={i}:provenance_status_mismatch")
            image_path = Path(str(row.get("image_path", "")))
            if not image_path.is_file() or image_path.stat().st_size == 0:
                errors.append(f"{dataset}:row={i}:image_missing_or_empty")

    all_rows = mm + ai
    if len(registry) != len(all_rows):
        errors.append(f"registry:row_count={len(registry)} expected={len(all_rows)}")
    manifest_by_id = {r.get("sample_id"): r for r in all_rows}
    registry_by_id = {r.get("sample_id"): r for r in registry}
    if len(registry_by_id) != len(registry):
        errors.append("registry:sample_id_not_unique")
    for sid, row in manifest_by_id.items():
        reg = registry_by_id.get(sid)
        if reg is None:
            errors.append(f"registry:missing_sample_id={sid}")
            continue
        for key in ["dataset", "dataset_revision", "source_row", "input_fingerprint_or_stable_key",
                    "previous_development_input", "previous_primary_input", "previous_outcome_inspected", "provenance_status"]:
            if reg.get(key) != row.get(key):
                errors.append(f"registry:mismatch:{sid}:{key}")
    checks["registry_manifest_alignment"] = {"pass": not any(x.startswith("registry:") for x in errors), "rows": len(registry)}

    expected_seen = sorted(r["sample_id"] for r in all_rows if r["provenance_status"] == "previously_seen")
    expected_unseen = sorted(r["sample_id"] for r in all_rows if r["provenance_status"] == "strict_unseen")
    if sorted(seen_doc.get("sample_ids", [])) != expected_seen or int(seen_doc.get("count", -1)) != len(expected_seen):
        errors.append("previously_seen_registry_mismatch")
        fail(checks, "previously_seen_registry", False)
    else:
        checks["previously_seen_registry"] = {"pass": True, "count": len(expected_seen)}
    if sorted(unseen_doc.get("sample_ids", [])) != expected_unseen or int(unseen_doc.get("count", -1)) != len(expected_unseen):
        errors.append("strict_unseen_registry_mismatch")
        fail(checks, "strict_unseen_registry", False)
    else:
        checks["strict_unseen_registry"] = {"pass": True, "count": len(expected_unseen)}
    checks["no_label_or_gold_fields"] = {"pass": not any("forbidden=" in x for x in errors)}
    mapping_summary_path = root / "00_protocol" / "r0_implementation_summary.json"
    mapping_ok = False
    mapping_detail: dict[str, Any] = {"path": str(mapping_summary_path)}
    if mapping_summary_path.exists():
        try:
            summary = json.loads(mapping_summary_path.read_text(encoding="utf-8"))
            mm_report = summary["datasets"]["MMStar"]
            mapping_detail.update({
                "prior_rows": mm_report["prior_rows"],
                "resolved_prior_rows": mm_report["resolved_prior_rows"],
                "unresolved_prior_rows": mm_report["unresolved_prior_rows"],
                "ambiguous_canonical_rows": mm_report["ambiguous_canonical_rows"],
            })
            mapping_ok = (
                mm_report["prior_rows"] == 300
                and mm_report["resolved_prior_rows"] == 300
                and mm_report["unresolved_prior_rows"] == 0
                and mm_report["ambiguous_canonical_rows"] == 0
            )
        except Exception as exc:
            mapping_detail["error"] = f"{type(exc).__name__}:{exc}"
    checks["mmstar_historical_mapping"] = {"pass": mapping_ok, "detail": mapping_detail}
    if not mapping_ok:
        errors.append("missing_mapping_evidence")

    result = {
        "status": "R0_PROVENANCE_AUDIT_PASS" if not errors else "R0_PROVENANCE_AUDIT_FAIL",
        "protocol": "VLM-ESWA-REVISION-R0",
        "stage": "G1_DATASET_PROVENANCE",
        "checks": checks,
        "errors": errors,
        "counts": {
            "MMStar": len(mm), "AI2D": len(ai), "total": len(all_rows),
            "previously_seen": len(expected_seen), "strict_unseen": len(expected_unseen),
        },
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "scientific_inference_started": False,
        "labels_in_inference_manifest": False,
    }
    out = audit_dir / "071_provenance_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
