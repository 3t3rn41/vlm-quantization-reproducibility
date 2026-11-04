"""GT-free exact-pair integrity audit for one model/dataset backend pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODEL_REVISIONS = {
    "internvl3_2b": "f6c7b60375759170fd49f5e9e298e2178485cba5",
    "qwen2_5_vl_7b": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}
MODEL_ADAPTERS = {
    "internvl3_2b": "internvl-chat-v1",
    "qwen2_5_vl_7b": "vlm_qwen_candidate_adapter_v2",
}
AMENDMENT_001 = "VLM_ESWA_REVISION_AMENDMENT_001"
AMENDMENT_002 = "VLM_ESWA_REVISION_AMENDMENT_002"


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


def load_jsonl(path: Path, expected_rows: int) -> list[dict]:
    require(path.exists(), f"missing predictions file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == expected_rows, f"{path}: expected {expected_rows} rows, got {len(rows)}")
    return rows


def normalized_manifest_value(path_value: str) -> str:
    return Path(path_value).as_posix().rstrip("/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmstar", "ai2d"], required=True)
    parser.add_argument("--bf16-output", type=Path, required=True)
    parser.add_argument("--int4-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_REVISIONS), default=None)
    args = parser.parse_args()

    expected_rows = {"mmstar": 1500, "ai2d": 3088}[args.dataset]
    dataset_title = "MMStar" if args.dataset == "mmstar" else "AI2D"
    manifest_name = f"{args.dataset}_full_clean.jsonl"
    manifest_path = (Path(__file__).resolve().parents[2] / "artifacts" / "eswa_revision_v1" / "01_manifests") / manifest_name
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(manifest) == expected_rows, "frozen manifest row count mismatch")
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    manifest_sources = [row.get("source_row") for row in manifest]

    bf16_metadata = read_json(args.bf16_output / "metadata.json")
    int4_metadata = read_json(args.int4_output / "metadata.json")
    model = args.model or str(bf16_metadata.get("model", "")).lower()
    require(model in MODEL_REVISIONS, f"unsupported model: {model!r}")
    require(str(int4_metadata.get("model", "")).lower() == model, "backend model identities differ")
    model_revision = MODEL_REVISIONS[model]
    bf16_gate = read_json(args.bf16_output / "074_prediction_gate.json")
    int4_gate = read_json(args.int4_output / "074_prediction_gate.json")
    require(bf16_gate.get("status") == "PASS", "BF16 single-cell gate is not PASS")
    require(int4_gate.get("status") == "PASS", "INT4 single-cell gate is not PASS")
    expected_stages = {
        "bf16": ("073_FULL_INFERENCE_R1_1" if args.dataset == "mmstar" else "073_FULL_INFERENCE_R1_3")
        if model == "internvl3_2b" else ("073_FULL_INFERENCE_R1_5" if args.dataset == "mmstar" else "073_FULL_INFERENCE_R1_7"),
        "int4_g128": ("073_FULL_INFERENCE_R1_2" if args.dataset == "mmstar" else "073_FULL_INFERENCE_R1_4")
        if model == "internvl3_2b" else ("073_FULL_INFERENCE_R1_6" if args.dataset == "mmstar" else "073_FULL_INFERENCE_R1_8"),
    }
    for metadata, gate, backend in (
        (bf16_metadata, bf16_gate, "bf16"),
        (int4_metadata, int4_gate, "int4_g128"),
    ):
        require(metadata.get("stage") == expected_stages[backend], f"{backend} metadata stage mismatch")
        require(metadata.get("status") == "PASS", f"{backend} metadata is not PASS")
        require(metadata.get("model") == model, f"{backend} model mismatch")
        require(metadata.get("dataset") == args.dataset, f"{backend} dataset mismatch")
        require(metadata.get("model_revision") == model_revision, f"{backend} model revision mismatch")
        require(metadata.get("expected_samples") == expected_rows, f"{backend} expected sample count mismatch")
        require(metadata.get("completed_samples") == expected_rows, f"{backend} completed sample count mismatch")
        require(metadata.get("protocol_version") == "v1.2", f"{backend} protocol version mismatch")
        if model == "internvl3_2b":
            require(metadata.get("amendment_id") == AMENDMENT_002, f"{backend} amendment mismatch")
            require(gate.get("amendment_id") == AMENDMENT_002, f"{backend} gate amendment mismatch")
            require(metadata.get("torch_set_num_threads") == 8, f"{backend} InternVL thread binding mismatch")
            require(metadata.get("internvl_normalization") == "vectorized/batched normalization", f"{backend} InternVL normalization mismatch")
        else:
            require(metadata.get("amendment_id") == AMENDMENT_001, f"{backend} Qwen amendment mismatch")
            active = metadata.get("active_amendments") or []
            require(AMENDMENT_001 in active and AMENDMENT_002 in active, f"{backend} Qwen active amendments mismatch")
            require(metadata.get("qwen_processor_family") == "Qwen2_5_VLProcessor", f"{backend} Qwen processor mismatch")
            require(metadata.get("qwen_native_preprocessing") is True, f"{backend} Qwen preprocessing mismatch")
            require(metadata.get("qwen_candidate_adapter") == MODEL_ADAPTERS[model], f"{backend} Qwen adapter mismatch")
            require(gate.get("amendment_id") == AMENDMENT_001, f"{backend} Qwen gate amendment mismatch")
        require(metadata.get("frozen_config_v1_2_match") is True, f"{backend} frozen v1.2 config mismatch")
        require(gate.get("protocol_version") == "v1.2", f"{backend} gate protocol version mismatch")
        require(gate.get("frozen_config_v1_2_match") is True, f"{backend} gate frozen v1.2 mismatch")
        require(metadata.get("GT_accessed") is False, f"{backend} GT flag is true")
        require(metadata.get("scientific_metrics_computed") is False, f"{backend} scientific flag is true")
    require(int4_metadata.get("quantization") == "TorchAO Int4WeightOnlyConfig group_size=128", "INT4 quantization mismatch")
    require(int4_metadata.get("group_size") == 128, "INT4 group size mismatch")

    bf16_rows = load_jsonl(args.bf16_output / "predictions.jsonl", expected_rows)
    int4_rows = load_jsonl(args.int4_output / "predictions.jsonl", expected_rows)
    bf16_ids = [str(row.get("sample_id")) for row in bf16_rows]
    int4_ids = [str(row.get("sample_id")) for row in int4_rows]
    require(bf16_ids == manifest_ids, "BF16 IDs do not match frozen manifest")
    require(int4_ids == manifest_ids, "INT4 IDs do not match frozen manifest")
    require(bf16_ids == int4_ids, "BF16/INT4 sample_id sequences differ")
    require(len(set(bf16_ids)) == expected_rows and len(set(int4_ids)) == expected_rows, "duplicate backend sample_id")

    bf16_sources = [row.get("source_row") for row in bf16_rows]
    int4_sources = [row.get("source_row") for row in int4_rows]
    require(bf16_sources == manifest_sources, "BF16 source_row alignment mismatch")
    require(int4_sources == manifest_sources, "INT4 source_row alignment mismatch")
    require(bf16_sources == int4_sources, "BF16/INT4 source_row alignment differs")

    bf16_manifest = normalized_manifest_value(str(bf16_metadata.get("manifest", "")))
    int4_manifest = normalized_manifest_value(str(int4_metadata.get("manifest", "")))
    require(Path(bf16_manifest).name == Path(int4_manifest).name == manifest_path.name, "manifest identity mismatch")

    pair_stage = (
        ("074_PAIRED_INTEGRITY_AUDIT_R1_1_R1_2" if args.dataset == "mmstar" else "074_PAIRED_INTEGRITY_AUDIT_R1_3_R1_4")
        if model == "internvl3_2b"
        else ("074_PAIRED_INTEGRITY_AUDIT_R1_5_R1_6" if args.dataset == "mmstar" else "074_PAIRED_INTEGRITY_AUDIT_R1_7_R1_8")
    )
    result = {
        "stage": pair_stage,
        "status": "PASS",
        "model": model,
        "dataset": args.dataset,
        "dataset_title": dataset_title,
        "model_revision": model_revision,
        "protocol_version": "v1.2",
        "amendment_id": AMENDMENT_002 if model == "internvl3_2b" else AMENDMENT_001,
        "active_amendments": [AMENDMENT_001, AMENDMENT_002],
        "frozen_config_v1_2_match": True,
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
        audit.update({
            "paired_backend_id_check": "PASS",
            "protocol_version": "v1.2",
            "amendment_id": AMENDMENT_002 if model == "internvl3_2b" else AMENDMENT_001,
            "frozen_config_v1_2_match": True,
        })
        atomic_json(audit_path, audit)
        metadata_path = directory / "metadata.json"
        metadata = read_json(metadata_path)
        metadata["paired_backend_id_check"] = "PASS"
        atomic_json(metadata_path, metadata)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
