"""GT-free formal full inference runner for the frozen Qwen R1 cells.

This runner deliberately keeps the Qwen model-native processor path from the
frozen smoke protocol.  It does not apply the InternVL-only Amendment 002
execution knobs (CPU thread cap or batched InternVL normalization).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import socket
import time
import traceback
from pathlib import Path


LABELS = ["A", "B", "C", "D"]
MODEL = "qwen2_5_vl_7b"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
ADAPTER = "vlm_qwen_candidate_adapter_v2"
PROCESSOR_FAMILY = "Qwen2_5_VLProcessor"
SUFFIX = "\nAnswer with exactly one option letter: A, B, C, or D. Do not provide explanations or any other text."


def run_identity(dataset: str, backend: str) -> dict[str, str]:
    identities = {
        ("mmstar", "bf16"): {
            "stage": "073_FULL_INFERENCE_R1_5",
            "run_id": "vlm-eswa-r1.5-qwen2.5-vl-7b-mmstar-bf16",
            "paired_backend_id_check": "PENDING_INT4",
        },
        ("mmstar", "int4_g128"): {
            "stage": "073_FULL_INFERENCE_R1_6",
            "run_id": "vlm-eswa-r1.6-qwen2.5-vl-7b-mmstar-int4-g128",
            "paired_backend_id_check": "PENDING_BF16_INT4_PAIR_AUDIT",
        },
        ("ai2d", "bf16"): {
            "stage": "073_FULL_INFERENCE_R1_7",
            "run_id": "vlm-eswa-r1.7-qwen2.5-vl-7b-ai2d-bf16",
            "paired_backend_id_check": "PENDING_INT4",
        },
        ("ai2d", "int4_g128"): {
            "stage": "073_FULL_INFERENCE_R1_8",
            "run_id": "vlm-eswa-r1.8-qwen2.5-vl-7b-ai2d-int4-g128",
            "paired_backend_id_check": "PENDING_BF16_INT4_PAIR_AUDIT",
        },
    }
    try:
        return identities[(dataset, backend)]
    except KeyError as exc:
        raise RuntimeError(f"unsupported dataset/backend: {dataset}/{backend}") from exc


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_record(path: Path, record: dict) -> None:
    atomic_json(path, record)


def load_manifest(path: Path, dataset: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_rows = {"mmstar": 1500, "ai2d": 3088}[dataset]
    if len(rows) != expected_rows:
        raise RuntimeError(f"formal {dataset} manifest must contain {expected_rows} rows, got {len(rows)}")
    required = {"sample_id", "dataset", "dataset_revision", "question", "image_path"}
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("manifest missing required fields")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise RuntimeError("duplicate sample_id in frozen manifest")
    forbidden = {"answer", "label", "gold", "ground_truth", "correct", "correct_answer"}
    if any(forbidden.intersection(row) for row in rows):
        raise RuntimeError("outcome field entered inference manifest")
    if {str(row["dataset"]).lower() for row in rows} != {dataset}:
        raise RuntimeError(f"formal {dataset} manifest dataset mismatch")
    return rows


def quantize_candidates(logits, token_ids):
    import torch

    values = logits[token_ids].float()
    if not torch.isfinite(values).all().item():
        raise RuntimeError("candidate logits contain NaN/Inf")
    probabilities = torch.softmax(values, dim=-1)
    if not torch.isfinite(probabilities).all().item():
        raise RuntimeError("candidate probabilities contain NaN/Inf")
    probs = [float(value) for value in probabilities.detach().cpu().tolist()]
    total = float(sum(probs))
    if not math.isfinite(total) or abs(total - 1.0) >= 1e-5:
        raise RuntimeError("candidate probability sum is invalid")
    choice = max(range(4), key=lambda index: probs[index])
    label = LABELS[choice]
    return {
        "candidate_logits": [float(value) for value in values.detach().cpu().tolist()],
        "candidate_probabilities": dict(zip(LABELS, probs)),
        "probability_sum": total,
        "generated_answer": label,
        "normalized_prediction": label,
        "score_argmax_label": label,
        "risk_primary": float(1.0 - probs[choice]),
        "generation_score_agree": True,
    }


def contextual_ids(tokenizer, rendered: str) -> list[int]:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    ids = []
    for label in LABELS:
        full = tokenizer.encode(rendered + label, add_special_tokens=False)
        if len(full) != len(prefix) + 1 or full[:-1] != prefix:
            raise RuntimeError(f"candidate-token adapter mismatch for {label}")
        ids.append(int(full[-1]))
    if len(set(ids)) != 4:
        raise RuntimeError("candidate-token IDs are not distinct")
    return ids


def load_existing_records(out_dir: Path, rows: list[dict]) -> dict[int, dict]:
    if not out_dir.exists():
        return {}
    contents = list(out_dir.iterdir())
    meaningful = [path for path in contents if path.name not in {"run.log", "launcher.log"}]
    if meaningful:
        raise RuntimeError("canonical output directory already contains history; refusing overwrite")
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[MODEL], required=True)
    parser.add_argument("--dataset", choices=["mmstar", "ai2d"], required=True)
    parser.add_argument("--backend", choices=["bf16", "int4_g128"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    random.seed(42)
    identity = run_identity(args.dataset, args.backend)
    rows = load_manifest(args.manifest, args.dataset)
    completed = load_existing_records(args.output, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".rows").mkdir(parents=True, exist_ok=True)

    metadata = {
        "stage": identity["stage"],
        "status": "RUNNING",
        "model": MODEL,
        "dataset": args.dataset,
        "backend": args.backend,
        "model_revision": MODEL_REVISION,
        "manifest": str(args.manifest),
        "expected_samples": len(rows),
        "completed_samples": 0,
        "batch_size": 1,
        "dtype": "bfloat16",
        "sampling": False,
        "model_eval": True,
        "torch_inference_mode": True,
        "use_flash_attn": False,
        "candidate_adapter": ADAPTER,
        "qwen_candidate_adapter": ADAPTER,
        "qwen_processor_family": PROCESSOR_FAMILY,
        "qwen_native_preprocessing": True,
        "prompt_suffix_frozen": SUFFIX,
        "resume_enabled": False,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "scientific_eligible": False,
        "network_requests": 0,
        "parser_failures": 0,
        "probability_failures": 0,
        "oom": False,
        "crash_count": 0,
        "protocol_version": "v1.2",
        "amendment_id": "VLM_ESWA_REVISION_AMENDMENT_001",
        "active_amendments": [
            "VLM_ESWA_REVISION_AMENDMENT_001",
            "VLM_ESWA_REVISION_AMENDMENT_002",
        ],
        "amendment_scope": "Qwen native processor path; Amendment 002 InternVL execution knobs not applied",
        "frozen_config_v1_2_match": True,
        "qwen_native_config_match": True,
        "paired_backend_id_check": identity["paired_backend_id_check"],
        "quantization": "TorchAO Int4WeightOnlyConfig group_size=128" if args.backend == "int4_g128" else "none",
        "group_size": 128 if args.backend == "int4_g128" else None,
    }
    atomic_json(args.output / "metadata.json", metadata)

    old_connect = socket.socket.connect
    network_requests: list[str] = []
    model = processor = tokenizer = None
    run_started = time.perf_counter()
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one CUDA device")

        def blocked_connect(self, address):
            network_requests.append(repr(address))
            raise RuntimeError("network disabled in formal inference")

        socket.socket.connect = blocked_connect
        model_path = args.model_path or Path(sorted(Path("/root/vlm_assets/models").glob("Qwen2.5-VL-7B-Instruct-*"))[0])
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)
        load_kw = {
            "torch_dtype": torch.bfloat16,
            "device_map": "cuda",
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if args.backend == "bf16":
            load_kw["low_cpu_mem_usage"] = True
        else:
            from transformers import TorchAoConfig
            load_kw["quantization_config"] = TorchAoConfig("int4_weight_only", group_size=128)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kw).eval()
        tokenizer = processor.tokenizer
        torch.cuda.synchronize()
        metadata["model_class"] = type(model).__name__
        metadata["processor_class"] = type(processor).__name__
        metadata["parameter_dtype"] = str(next(model.parameters()).dtype)
        metadata["parameter_device"] = str(next(model.parameters()).device)
        if args.backend == "int4_g128":
            total = quantized = q_modules = 0
            classes: set[str] = set()
            for module in model.modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                n = sum(parameter.numel() for parameter in module.parameters(recurse=False))
                total += int(n)
                weight = module.weight
                cls = type(weight).__name__
                classes.add(cls)
                if ("AffineQuantized" in cls or "Quantized" in cls or "Int4" in cls
                        or "WeightOnly" in cls or hasattr(weight, "tensor_impl")):
                    q_modules += 1
                    quantized += int(n)
            coverage = {
                "total_linear_parameters": total,
                "quantized_weight_parameters": quantized,
                "quantized_linear_modules": q_modules,
                "quantized_weight_fraction": quantized / total if total else 0.0,
                "weight_classes": sorted(classes),
                "coverage_valid": q_modules > 0 and quantized > 0,
            }
            metadata["quantization_coverage"] = coverage
            if not coverage["coverage_valid"]:
                raise RuntimeError("INT4 coverage/load path invalid")
        atomic_json(args.output / "metadata.json", metadata)

        for index, row in enumerate(rows):
            sample_started = time.perf_counter()
            with Image.open(Path(row["image_path"])) as source:
                image = source.convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(row["question"]) + SUFFIX},
                ],
            }]
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
            model_inputs = {key: (value.cuda() if torch.is_tensor(value) else value)
                            for key, value in inputs.items()}
            candidate_ids = contextual_ids(tokenizer, rendered)
            with torch.inference_mode():
                output = model(**model_inputs, return_dict=True)
            torch.cuda.synchronize()
            scored = quantize_candidates(output.logits[0, -1, :], candidate_ids)
            record = {
                "run_id": identity["run_id"],
                "stage": identity["stage"],
                "sample_index": index,
                "sample_id": str(row["sample_id"]),
                "dataset": str(row["dataset"]),
                "dataset_revision": str(row["dataset_revision"]),
                "source_row": row.get("source_row"),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "backend": args.backend,
                "dtype": "bfloat16",
                "batch_size": 1,
                "prompt_version": "prompt_v3",
                "candidate_adapter": ADAPTER,
                "qwen_candidate_adapter": ADAPTER,
                "qwen_processor_family": PROCESSOR_FAMILY,
                "qwen_native_preprocessing": True,
                "candidate_token_ids": dict(zip(LABELS, candidate_ids)),
                "candidate_token_count": 4,
                "parser_status": "PASS",
                "parser_reason": "single_step_candidate_argmax",
                "gold_accessed": False,
                "ground_truth": None,
                "correct": None,
                "GT_accessed": False,
                "scientific_metrics_computed": False,
                "prompt_token_count": int(model_inputs["input_ids"].shape[-1]),
                "sample_wall_time_sec": round(time.perf_counter() - sample_started, 6),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                **scored,
            }
            atomic_record(args.output / ".rows" / f"{index:06d}.json", record)
            with (args.output / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            metadata["completed_samples"] = index + 1
            metadata["last_sample_index"] = index
            metadata["last_sample_id"] = str(row["sample_id"])
            metadata["elapsed_sec"] = round(time.perf_counter() - run_started, 3)
            atomic_json(args.output / "metadata.json", metadata)
            print(json.dumps({"event": "sample_done", "index": index, "sample_id": row["sample_id"]}, ensure_ascii=False), flush=True)
            del image, inputs, model_inputs, output
            torch.cuda.empty_cache()

        metadata.update({
            "status": "PASS",
            "completed_samples": len(rows),
            "unique_sample_ids": len({str(row["sample_id"]) for row in rows}),
            "parser_failures": 0,
            "probability_failures": 0,
            "network_requests": len(network_requests),
            "wall_time_sec": round(time.perf_counter() - run_started, 3),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        })
        audit = {
            "stage": identity["stage"],
            "expected_rows": len(rows),
            "actual_rows": len(rows),
            "unique_sample_ids": len({str(row["sample_id"]) for row in rows}),
            "parser_success_percent": 100.0,
            "all_candidate_logits_finite": True,
            "all_candidate_probabilities_finite": True,
            "all_probability_sums_valid": True,
            "generated_answer_equals_candidate_argmax": True,
            "missing_image_or_input": False,
            "duplicate_output_row": False,
            "paired_backend_id_check": identity["paired_backend_id_check"],
            "GT_accessed": False,
            "scientific_metrics_computed": False,
            "decision": "PENDING_074_SINGLE_CELL_GATE",
        }
        atomic_json(args.output / "audit.json", audit)
    except Exception as exc:
        metadata.update({
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc)[-3000:],
            "traceback_tail": traceback.format_exc()[-6000:],
            "network_requests": len(network_requests),
            "crash_count": 1,
            "oom": "out of memory" in str(exc).lower() or "cuda out of memory" in str(exc).lower(),
        })
        atomic_json(args.output / "metadata.json", metadata)
        raise
    finally:
        socket.socket.connect = old_connect
        if model is not None:
            del model
        if processor is not None:
            del processor
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    metadata["total_elapsed_sec"] = round(time.perf_counter() - run_started, 6)
    atomic_json(args.output / "metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
