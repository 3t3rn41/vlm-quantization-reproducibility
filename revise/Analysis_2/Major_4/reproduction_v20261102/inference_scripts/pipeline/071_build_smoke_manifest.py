#!/usr/bin/env python3
"""Build the frozen, outcome-blind 072 smoke composition.

The smoke set is fixed once from source-row order and is shared by every
model/backend.  This script reads only the already audited inference
manifests; it never reads answers or ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, default=None,
                    help="Optional POSIX image root used when the source manifest was built on Windows.")
    args = ap.parse_args()
    root = args.artifact_root
    out = root / "02_smoke"
    out.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[dict]] = {}
    for dataset, name in (("MMStar", "mmstar"), ("AI2D", "ai2d")):
        source = root / "01_manifests" / f"{name}_full_clean.jsonl"
        rows = sorted(read_jsonl(source), key=lambda row: int(row["source_row"]))[:50]
        if len(rows) != 50 or [int(row["source_row"]) for row in rows] != list(range(50)):
            raise SystemExit(f"{dataset} smoke composition is not source_row 0..49")
        if any(any(key in row for key in ("answer", "label", "gold", "ground_truth", "correct", "correct_answer")) for row in rows):
            raise SystemExit(f"{dataset} smoke manifest contains a forbidden outcome field")
        if args.image_root is not None:
            dataset_dir = "mmstar" if dataset == "MMStar" else "ai2d"
            for row in rows:
                row["image_path"] = str(args.image_root / dataset_dir / Path(row["image_path"]).name)
        selected[name] = rows
        write_jsonl(out / f"smoke_{name}_50.jsonl", rows)
    ids = {
        "schema": "vlm.eswa_revision_v1.smoke_ids.v1",
        "composition": "source_row_sorted_first_50_per_dataset",
        "outcome_blind": True,
        "shared_across_model_backends": True,
        "mmstar": [{"sample_id": r["sample_id"], "source_row": r["source_row"]} for r in selected["mmstar"]],
        "ai2d": [{"sample_id": r["sample_id"], "source_row": r["source_row"]} for r in selected["ai2d"]],
        "gt_accessed": False,
        "scientific_metrics_computed": False,
    }
    (out / "smoke_ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "072_SMOKE_MANIFESTS_BUILT", "MMStar": 50, "AI2D": 50,
                      "source_rows": "0..49", "outcome_blind": True,
                      "GT_accessed": False, "scientific_metrics_computed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
