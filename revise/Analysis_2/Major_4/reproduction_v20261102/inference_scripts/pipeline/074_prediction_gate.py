"""GT-free Prediction Gate for the frozen ESWA revision R1.1 cell."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


LABELS = ["A", "B", "C", "D"]
MODEL_REVISIONS = {
    "internvl3_2b": "f6c7b60375759170fd49f5e9e298e2178485cba5",
    "qwen2_5_vl_7b": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}
MODEL_ADAPTERS = {
    "internvl3_2b": "internvl-chat-v1",
    "qwen2_5_vl_7b": "vlm_qwen_candidate_adapter_v2",
}


def gate_identity(model: str, dataset: str, backend: str) -> dict[str, str]:
    identities = {
        ("internvl3_2b", "mmstar", "bf16"): ("073_FULL_INFERENCE_R1_1", "074_PREDICTION_GATE_R1_1", "PENDING_INT4"),
        ("internvl3_2b", "mmstar", "int4_g128"): ("073_FULL_INFERENCE_R1_2", "074_PREDICTION_GATE_R1_2", "PENDING_BF16_INT4_PAIR_AUDIT"),
        ("internvl3_2b", "ai2d", "bf16"): ("073_FULL_INFERENCE_R1_3", "074_PREDICTION_GATE_R1_3", "PENDING_INT4"),
        ("internvl3_2b", "ai2d", "int4_g128"): ("073_FULL_INFERENCE_R1_4", "074_PREDICTION_GATE_R1_4", "PENDING_BF16_INT4_PAIR_AUDIT"),
        ("qwen2_5_vl_7b", "mmstar", "bf16"): ("073_FULL_INFERENCE_R1_5", "074_PREDICTION_GATE_R1_5", "PENDING_INT4"),
        ("qwen2_5_vl_7b", "mmstar", "int4_g128"): ("073_FULL_INFERENCE_R1_6", "074_PREDICTION_GATE_R1_6", "PENDING_BF16_INT4_PAIR_AUDIT"),
        ("qwen2_5_vl_7b", "ai2d", "bf16"): ("073_FULL_INFERENCE_R1_7", "074_PREDICTION_GATE_R1_7", "PENDING_INT4"),
        ("qwen2_5_vl_7b", "ai2d", "int4_g128"): ("073_FULL_INFERENCE_R1_8", "074_PREDICTION_GATE_R1_8", "PENDING_BF16_INT4_PAIR_AUDIT"),
    }
    try:
        inference_stage, gate_stage, paired = identities[(model, dataset, backend)]
    except KeyError as exc:
        raise RuntimeError(f"unsupported model/dataset/backend: {model}/{dataset}/{backend}") from exc
    return {"inference_stage": inference_stage, "gate_stage": gate_stage, "paired": paired}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_number(value, name: str) -> None:
    require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"non-finite {name}")


def load_manifest(path: Path, expected_rows: int, dataset: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == expected_rows, f"manifest row count is {len(rows)}, expected {expected_rows}")
    ids = [str(row.get("sample_id")) for row in rows]
    require(len(set(ids)) == len(ids), "manifest contains duplicate sample_id")
    for index, row in enumerate(rows):
        require(row.get("image_path"), f"manifest row {index} has no image_path")
    require({str(row.get("dataset", "")).lower() for row in rows} == {dataset}, "manifest dataset mismatch")
    return rows


def load_predictions(path: Path, expected_rows: int) -> list[dict]:
    require(path.exists(), f"missing predictions file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        require(line.strip(), f"empty prediction line {line_number}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON at prediction line {line_number}: {exc}") from exc
    require(len(rows) == expected_rows, f"prediction row count is {len(rows)}, expected {expected_rows}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["bf16", "int4_g128"], required=True)
    parser.add_argument("--dataset", choices=["mmstar", "ai2d"], default=None)
    parser.add_argument("--model", choices=list(MODEL_REVISIONS), default=None)
    args = parser.parse_args()
    output = args.output
    metadata_path = output / "metadata.json"
    metadata = read_json(metadata_path)
    dataset = args.dataset or str(metadata.get("dataset", "")).lower()
    model = args.model or str(metadata.get("model", "")).lower()
    require(model in MODEL_REVISIONS, f"unsupported model: {model!r}")
    expected_rows = {"mmstar": 1500, "ai2d": 3088}.get(dataset)
    require(expected_rows is not None, f"unsupported dataset: {dataset!r}")
    if args.dataset is not None:
        require(str(metadata.get("dataset", "")).lower() == args.dataset, "metadata dataset mismatch")
    identity = gate_identity(model, dataset, args.backend)
    expected_revision = MODEL_REVISIONS[model]
    expected_adapter = MODEL_ADAPTERS[model]
    paired_check = metadata.get("paired_backend_id_check")
    require(
        paired_check in {identity["paired"], "PASS"},
        f"metadata mismatch for paired_backend_id_check: {paired_check!r}",
    )
    manifest_name = "mmstar_full_clean.jsonl" if dataset == "mmstar" else "ai2d_full_clean.jsonl"
    manifest_path = (Path(__file__).resolve().parents[2] / "artifacts" / "eswa_revision_v1" / "01_manifests") / manifest_name
    manifest = load_manifest(manifest_path, expected_rows, dataset)
    predictions = load_predictions(output / "predictions.jsonl", expected_rows)

    require(metadata.get("stage") == identity["inference_stage"], "metadata stage mismatch")
    require(metadata.get("status") == "PASS", "073 metadata is not PASS")
    protocol_version = str(metadata.get("protocol_version", "v1.1"))
    protocol_binding = {
        "protocol_version": protocol_version,
    }
    if protocol_version == "v1.2":
        require(metadata.get("frozen_config_v1_2_match") is True, "frozen v1.2 config binding is not true")
        if model == "internvl3_2b":
            require(metadata.get("amendment_id") == "VLM_ESWA_REVISION_AMENDMENT_002", "InternVL Amendment 002 binding mismatch")
            require(metadata.get("torch_set_num_threads") == 8, "InternVL torch thread binding mismatch")
            require(metadata.get("internvl_normalization") == "vectorized/batched normalization", "InternVL normalization binding mismatch")
            protocol_binding.update({
                "amendment_id": "VLM_ESWA_REVISION_AMENDMENT_002",
                "torch_set_num_threads": 8,
                "internvl_normalization": "vectorized/batched normalization",
                "frozen_config_v1_2_match": True,
            })
        else:
            active = metadata.get("active_amendments")
            require(isinstance(active, list), "Qwen active_amendments is missing")
            require("VLM_ESWA_REVISION_AMENDMENT_001" in active, "Qwen Amendment 001 binding is missing")
            require("VLM_ESWA_REVISION_AMENDMENT_002" in active, "Qwen active amendment record is missing")
            require(metadata.get("qwen_processor_family") == "Qwen2_5_VLProcessor", "Qwen processor family mismatch")
            require(metadata.get("qwen_native_preprocessing") is True, "Qwen native preprocessing binding mismatch")
            require(metadata.get("qwen_candidate_adapter") == "vlm_qwen_candidate_adapter_v2", "Qwen adapter binding mismatch")
            protocol_binding.update({
                "amendment_id": "VLM_ESWA_REVISION_AMENDMENT_001",
                "active_amendments": active,
                "qwen_processor_family": "Qwen2_5_VLProcessor",
                "qwen_native_preprocessing": True,
                "qwen_candidate_adapter": "vlm_qwen_candidate_adapter_v2",
                "frozen_config_v1_2_match": True,
            })
    else:
        require(protocol_version == "v1.1", f"unsupported protocol version: {protocol_version!r}")
        protocol_binding["frozen_config_v1_1_match"] = True
    for key, expected in {
        "model": model,
        "dataset": dataset,
        "backend": args.backend,
        "model_revision": expected_revision,
        "expected_samples": expected_rows,
        "completed_samples": expected_rows,
        "batch_size": 1,
        "candidate_adapter": expected_adapter,
        "network_requests": 0,
        "parser_failures": 0,
        "probability_failures": 0,
        "oom": False,
        "crash_count": 0,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "paired_backend_id_check": paired_check,
        **protocol_binding,
    }.items():
        require(metadata.get(key) == expected, f"metadata mismatch for {key}: {metadata.get(key)!r}")
    if args.backend == "int4_g128":
        require(metadata.get("quantization") == "TorchAO Int4WeightOnlyConfig group_size=128", "INT4 quantization config mismatch")
        require(metadata.get("group_size") == 128, "INT4 group size mismatch")
        coverage = metadata.get("quantization_coverage") or {}
        require(coverage.get("coverage_valid") is True, "INT4 quantization coverage is invalid")

    expected_ids = [str(row["sample_id"]) for row in manifest]
    actual_ids = [str(record.get("sample_id")) for record in predictions]
    require(actual_ids == expected_ids, "prediction sample_id order does not match manifest")
    require(len(set(actual_ids)) == expected_rows, "prediction sample_id is not unique")

    row_dir = output / ".rows"
    row_files = sorted(row_dir.glob("*.json")) if row_dir.exists() else []
    require(len(row_files) == expected_rows, f"row file count is {len(row_files)}, expected {expected_rows}")

    missing_inputs = []
    for index, row in enumerate(manifest):
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            missing_inputs.append({"index": index, "path": str(image_path)})
    require(not missing_inputs, f"missing input images: {missing_inputs[:3]}")

    for index, record in enumerate(predictions):
        prefix = f"prediction row {index}"
        require(record.get("stage") == identity["inference_stage"], f"{prefix}: stage mismatch")
        require(record.get("sample_index") == index, f"{prefix}: sample_index mismatch")
        require(record.get("model") == model, f"{prefix}: model mismatch")
        require(record.get("backend") == args.backend, f"{prefix}: backend mismatch")
        require(record.get("model_revision") == expected_revision, f"{prefix}: model revision mismatch")
        require(record.get("candidate_adapter") == expected_adapter, f"{prefix}: adapter mismatch")
        if model == "qwen2_5_vl_7b":
            require(record.get("qwen_processor_family") == "Qwen2_5_VLProcessor", f"{prefix}: Qwen processor mismatch")
            require(record.get("qwen_native_preprocessing") is True, f"{prefix}: Qwen preprocessing mismatch")
            require(record.get("qwen_candidate_adapter") == "vlm_qwen_candidate_adapter_v2", f"{prefix}: Qwen adapter mismatch")
        require(record.get("candidate_token_count") == 4, f"{prefix}: candidate token count mismatch")
        require(record.get("parser_status") == "PASS", f"{prefix}: parser failure")
        require(record.get("generation_score_agree") is True, f"{prefix}: generation/score disagreement")
        require(record.get("gold_accessed") is False, f"{prefix}: gold_accessed is not false")
        require(record.get("GT_accessed") is False, f"{prefix}: GT_accessed is not false")
        require(record.get("ground_truth") is None, f"{prefix}: ground_truth is present")
        require(record.get("correct") is None, f"{prefix}: correct is present")
        require(record.get("scientific_metrics_computed") is False, f"{prefix}: scientific metric flag is true")

        token_ids = record.get("candidate_token_ids")
        require(isinstance(token_ids, dict) and set(token_ids) == set(LABELS), f"{prefix}: candidate token map mismatch")
        require(len(set(token_ids.values())) == 4, f"{prefix}: candidate token IDs are not distinct")

        logits = record.get("candidate_logits")
        require(isinstance(logits, list) and len(logits) == 4, f"{prefix}: candidate logits shape mismatch")
        for value in logits:
            finite_number(value, f"{prefix} candidate logit")

        probabilities = record.get("candidate_probabilities")
        require(isinstance(probabilities, dict) and set(probabilities) == set(LABELS), f"{prefix}: probability map mismatch")
        for label in LABELS:
            finite_number(probabilities[label], f"{prefix} probability {label}")
        probability_sum = sum(float(probabilities[label]) for label in LABELS)
        finite_number(record.get("probability_sum"), f"{prefix} stored probability sum")
        require(abs(probability_sum - 1.0) < 1e-5, f"{prefix}: probability sum invalid")
        require(abs(float(record["probability_sum"]) - 1.0) < 1e-5, f"{prefix}: stored probability sum invalid")

        generated = record.get("generated_answer")
        require(generated in LABELS, f"{prefix}: generated answer invalid")
        require(generated == record.get("normalized_prediction") == record.get("score_argmax_label"), f"{prefix}: argmax agreement failed")

    gate = {
        "stage": identity["gate_stage"],
        "status": "PASS",
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "backend": metadata["backend"],
        "model_revision": metadata["model_revision"],
        "expected_rows": expected_rows,
        "actual_rows": len(predictions),
        "unique_sample_ids": len(set(actual_ids)),
        "candidate_token_ids_distinct": True,
        "all_candidate_logits_finite": True,
        "all_candidate_probabilities_finite": True,
        "all_probability_sums_valid": True,
        "parser_success_percent": 100.0,
        "generated_answer_equals_candidate_argmax": True,
        "missing_image_or_input": False,
        "duplicate_output_row": False,
        "paired_backend_id_check": paired_check,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "decision": "PASS",
        "next_action": "STOP_AND_REPORT_TO_WEB_GPT",
    }
    gate.update(protocol_binding)
    atomic_json(output / "074_prediction_gate.json", gate)
    audit = read_json(output / "audit.json") if (output / "audit.json").exists() else {}
    audit.pop("frozen_config_v1_1_match", None)
    audit.update(gate)
    atomic_json(output / "audit.json", audit)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
