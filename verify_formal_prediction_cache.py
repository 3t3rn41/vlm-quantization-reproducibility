"""Compare complete GPU prediction records with the released numeric caches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CELLS = {"mmstar": 1500, "ai2d": 3088}
MODELS = ("internvl3_2b", "qwen2_5_vl_7b")


def rows(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def predictions(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def compare(records: list[dict], cache: list[dict], prefix: str, cell: str) -> tuple[float, list[str]]:
    assert len(records) == len(cache), (cell, prefix, len(records), len(cache))
    difference = 0.0
    ids = []
    for index, (record, derived) in enumerate(zip(records, cache)):
        assert int(record["source_row"]) == int(derived["source_row"]) == index, (cell, prefix, index)
        assert record["normalized_prediction"] == derived[prefix + "_prediction"], (cell, prefix, index)
        assert record.get("parser_status", "PASS") == "PASS" and record["generation_score_agree"]
        ids.append(record["sample_id"])
        for label in "ABCD":
            difference = max(difference, abs(float(record["candidate_probabilities"][label]) - float(derived[prefix + "_prob_" + label])))
    assert len(set(ids)) == len(ids), (cell, prefix, "duplicate IDs")
    assert difference <= 1e-12, (cell, prefix, difference)
    return difference, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-inference-root", type=Path, required=True)
    parser.add_argument("--hqq-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"status": "PASS", "original_records": 0, "hqq_records": 0, "cells": {}}
    for model in MODELS:
        for dataset, expected in CELLS.items():
            cell = f"{model}__{dataset}"
            original = rows(args.cache_root / "manuscript/submission/numeric_audit_20261002" / f"{cell}_derived.csv")
            hqq = rows(args.cache_root / "revise/Analysis_2/Major_2/derived_audit" / f"{cell}_derived.csv")
            assert len(original) == len(hqq) == expected
            source = args.original_inference_root / model / dataset
            bf16 = predictions(source / "bf16/predictions.jsonl")
            torchao = predictions(source / "int4_g128/predictions.jsonl")
            hqq_predictions = predictions(args.hqq_root / model / f"{dataset}_predictions.jsonl")
            bf16_diff, bf16_ids = compare(bf16, original, "bf16", cell)
            torchao_diff, torchao_ids = compare(torchao, original, "int4", cell)
            hqq_diff, hqq_ids = compare(hqq_predictions, hqq, "int4", cell)
            assert bf16_ids == torchao_ids == hqq_ids, (cell, "backend sample IDs differ")
            result["original_records"] += len(bf16) + len(torchao)
            result["hqq_records"] += len(hqq_predictions)
            result["cells"][cell] = {"rows_per_backend": expected, "max_probability_difference": max(bf16_diff, torchao_diff, hqq_diff), "torchao_errors": sum(int(row["int4_error"]) for row in original), "hqq_errors": sum(int(row["int4_error"]) for row in hqq)}
    assert result["original_records"] == 18352 and result["hqq_records"] == 9176
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
