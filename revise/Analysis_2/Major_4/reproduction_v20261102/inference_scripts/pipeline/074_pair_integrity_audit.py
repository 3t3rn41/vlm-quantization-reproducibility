"""GT-free exact-pair integrity audit for the InternVL MMStar backends."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODEL_REVISION = "f6c7b60375759170fd49f5e9e298e2178485cba5"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_jsonl(path: Path) -> list[dict]:
    require(path.exists(), f"missing predictions file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 1500, f"{path}: expected 1500 rows, got {len(rows)}")
    return rows


def normalized_manifest_value(path_value: str) -> str:
    return Path(path_value).as_posix().rstrip("/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-output", type=Path, required=True)
    parser.add_argument("--int4-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = (Path(__file__).resolve().parents[2] / "artifacts" / "eswa_revision_v1" / "01_manifests" / "mmstar_full_clean.jsonl")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(manifest) == 1500, "frozen manifest row count mismatch")
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    manifest_sources = [row.get("source_row") for row in manifest]

    bf16_metadata = read_json(args.bf16_output / "metadata.json")
    int4_metadata = read_json(args.int4_output / "metadata.json")
    bf16_gate = read_json(args.bf16_output / "074_prediction_gate.json")
    int4_gate = read_json(args.int4_output / "074_prediction_gate.json")
    require(bf16_gate.get("status") == "PASS", "BF16 single-cell gate is not PASS")
    require(int4_gate.get("status") == "PASS", "INT4 single-cell gate is not PASS")
    require(bf16_metadata.get("stage") == "073_FULL_INFERENCE_R1_1", "BF16 metadata stage mismatch")
    require(int4_metadata.get("stage") == "073_FULL_INFERENCE_R1_2", "INT4 metadata stage mismatch")
    for metadata, backend in ((bf16_metadata, "bf16"), (int4_metadata, "int4_g128")):
        require(metadata.get("status") == "PASS", f"{backend} metadata is not PASS")
        require(metadata.get("model") == "internvl3_2b", f"{backend} model mismatch")
        require(metadata.get("dataset") == "mmstar", f"{backend} dataset mismatch")
        require(metadata.get("model_revision") == MODEL_REVISION, f"{backend} model revision mismatch")
        require(metadata.get("expected_samples") == 1500, f"{backend} expected sample count mismatch")
        require(metadata.get("completed_samples") == 1500, f"{backend} completed sample count mismatch")
        require(metadata.get("frozen_config_v1_1_match") is True, f"{backend} frozen config mismatch")
        require(metadata.get("GT_accessed") is False, f"{backend} GT flag is true")
        require(metadata.get("scientific_metrics_computed") is False, f"{backend} scientific flag is true")
    require(int4_metadata.get("quantization") == "TorchAO Int4WeightOnlyConfig group_size=128", "INT4 quantization mismatch")
    require(int4_metadata.get("group_size") == 128, "INT4 group size mismatch")

    bf16_rows = load_jsonl(args.bf16_output / "predictions.jsonl")
    int4_rows = load_jsonl(args.int4_output / "predictions.jsonl")
    bf16_ids = [str(row.get("sample_id")) for row in bf16_rows]
    int4_ids = [str(row.get("sample_id")) for row in int4_rows]
    require(bf16_ids == manifest_ids, "BF16 IDs do not match frozen manifest")
    require(int4_ids == manifest_ids, "INT4 IDs do not match frozen manifest")
    require(bf16_ids == int4_ids, "BF16/INT4 sample_id sequences differ")
    require(len(set(bf16_ids)) == 1500 and len(set(int4_ids)) == 1500, "duplicate backend sample_id")

    bf16_sources = [row.get("source_row") for row in bf16_rows]
    int4_sources = [row.get("source_row") for row in int4_rows]
    require(bf16_sources == manifest_sources, "BF16 source_row alignment mismatch")
    require(int4_sources == manifest_sources, "INT4 source_row alignment mismatch")
    require(bf16_sources == int4_sources, "BF16/INT4 source_row alignment differs")

    bf16_manifest = normalized_manifest_value(str(bf16_metadata.get("manifest", "")))
    int4_manifest = normalized_manifest_value(str(int4_metadata.get("manifest", "")))
    require(Path(bf16_manifest).name == Path(int4_manifest).name == manifest_path.name, "manifest identity mismatch")

    result = {
        "stage": "074_PAIRED_INTEGRITY_AUDIT_R1_1_R1_2",
        "status": "PASS",
        "model": "internvl3_2b",
        "dataset": "mmstar",
        "model_revision": MODEL_REVISION,
        "bf16_backend": "bf16",
        "int4_backend": "int4_g128",
        "bf16_row_count": len(bf16_rows),
        "int4_row_count": len(int4_rows),
        "bf16_sample_ids_unique": True,
        "int4_sample_ids_unique": True,
        "sample_id_sequence_equal": True,
        "sample_id_set_equal": True,
        "no_bf16_only_ids": True,
        "no_int4_only_ids": True,
        "no_duplicate_ids": True,
        "same_source_row_alignment": True,
        "same_manifest_identity": True,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "decision": "PASS",
        "next_action": "STOP_AND_REPORT_TO_WEB_GPT",
    }
    atomic_json(args.output, result)

    for directory in (args.bf16_output, args.int4_output):
        audit_path = directory / "audit.json"
        audit = read_json(audit_path) if audit_path.exists() else {}
        audit["paired_backend_id_check"] = "PASS"
        atomic_json(audit_path, audit)
        metadata_path = directory / "metadata.json"
        metadata = read_json(metadata_path)
        metadata["paired_backend_id_check"] = "PASS"
        atomic_json(metadata_path, metadata)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
