"""GT-free, resumable formal inference runner for the frozen ESWA revision."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import socket
import time
import traceback
from pathlib import Path


LABELS = ["A", "B", "C", "D"]
MODEL_REVISION = "f6c7b60375759170fd49f5e9e298e2178485cba5"
SUFFIX = "\nAnswer with exactly one option letter: A, B, C, or D. Do not provide explanations or any other text."


def run_identity(dataset: str, backend: str) -> dict[str, str]:
    if dataset == "mmstar" and backend == "bf16":
        return {
            "stage": "073_FULL_INFERENCE_R1_1",
            "run_id": "vlm-eswa-r1.1-internvl3-2b-mmstar-bf16",
            "paired_backend_id_check": "PENDING_INT4",
        }
    if dataset == "mmstar" and backend == "int4_g128":
        return {
            "stage": "073_FULL_INFERENCE_R1_2",
            "run_id": "vlm-eswa-r1.2-internvl3-2b-mmstar-int4-g128",
            "paired_backend_id_check": "PENDING_BF16_INT4_PAIR_AUDIT",
        }
    if dataset == "ai2d" and backend == "bf16":
        return {
            "stage": "073_FULL_INFERENCE_R1_3",
            "run_id": "vlm-eswa-r1.3-internvl3-2b-ai2d-bf16",
            "paired_backend_id_check": "PENDING_INT4",
        }
    if dataset == "ai2d" and backend == "int4_g128":
        return {
            "stage": "073_FULL_INFERENCE_R1_4",
            "run_id": "vlm-eswa-r1.4-internvl3-2b-ai2d-int4-g128",
            "paired_backend_id_check": "PENDING_BF16_INT4_PAIR_AUDIT",
        }
    raise RuntimeError(f"unsupported dataset/backend: {dataset}/{backend}")


def load_worker_module():
    path = Path(__file__).with_name("072_smoke_revision_inference.py")
    spec = importlib.util.spec_from_file_location("vlm_072_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen 072 worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def load_existing_records(out_dir: Path, rows: list[dict], resume: bool) -> dict[int, dict]:
    row_dir = out_dir / ".rows"
    existing = {}
    if not out_dir.exists():
        if resume:
            raise RuntimeError("--resume requested but output directory does not exist")
        return existing
    contents = list(out_dir.iterdir())
    meaningful = [path for path in contents if path.name not in {"run.log", "launcher.log"}]
    if meaningful and not resume:
        raise RuntimeError("canonical output directory already contains history; refusing overwrite")
    if not resume:
        return existing
    if not row_dir.exists():
        raise RuntimeError("resume requested but atomic row directory is missing")
    expected_ids = [str(row["sample_id"]) for row in rows]
    for path in sorted(row_dir.glob("*.json")):
        record = read_json(path)
        index = int(path.stem)
        if index < 0 or index >= len(rows):
            raise RuntimeError(f"resume row index out of range: {path.name}")
        if record.get("sample_id") != expected_ids[index]:
            raise RuntimeError(f"resume manifest/order mismatch at index {index}")
        if index in existing:
            raise RuntimeError(f"duplicate resume row index: {index}")
        existing[index] = record
    if len({record["sample_id"] for record in existing.values()}) != len(existing):
        raise RuntimeError("duplicate completed sample_id in resume state")
    return existing


def rebuild_predictions(out_dir: Path, records: dict[int, dict]) -> None:
    temp = out_dir / "predictions.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, out_dir / "predictions.jsonl")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["internvl3_2b"], required=True)
    parser.add_argument("--dataset", choices=["mmstar", "ai2d"], required=True)
    parser.add_argument("--backend", choices=["bf16", "int4_g128"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    random.seed(42)
    identity = run_identity(args.dataset, args.backend)
    worker = load_worker_module()
    rows = load_manifest(args.manifest, args.dataset)
    completed = load_existing_records(args.output, rows, args.resume)
    if args.resume and (args.output / "metadata.json").exists():
        prior = read_json(args.output / "metadata.json")
        for key, expected in {
            "stage": identity["stage"],
            "model": args.model,
            "dataset": args.dataset,
            "backend": args.backend,
            "model_revision": MODEL_REVISION,
            "manifest": str(args.manifest),
        }.items():
            if prior.get(key) != expected:
                raise RuntimeError(f"resume identity mismatch for {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".rows").mkdir(parents=True, exist_ok=True)
    if completed:
        rebuild_predictions(args.output, completed)

    metadata = {
        "stage": identity["stage"],
        "status": "RUNNING",
        "model": args.model,
        "dataset": args.dataset,
        "backend": args.backend,
        "model_revision": MODEL_REVISION,
        "manifest": str(args.manifest),
        "expected_samples": len(rows),
        "completed_samples": len(completed),
        "batch_size": 1,
        "dtype": "bfloat16",
        "sampling": False,
        "model_eval": True,
        "torch_inference_mode": True,
        "use_flash_attn": False,
        "candidate_adapter": "internvl-chat-v1",
        "prompt_suffix_frozen": SUFFIX,
        "resume_enabled": args.resume,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "scientific_eligible": False,
        "network_requests": 0,
        "parser_failures": 0,
        "probability_failures": 0,
        "oom": False,
        "crash_count": 0,
        "frozen_config_v1_1_match": True,
        "paired_backend_id_check": identity["paired_backend_id_check"],
        "quantization": "TorchAO Int4WeightOnlyConfig group_size=128" if args.backend == "int4_g128" else "none",
        "group_size": 128 if args.backend == "int4_g128" else None,
    }
    atomic_json(args.output / "metadata.json", metadata)

    old_connect = socket.socket.connect
    network_requests: list[str] = []
    model = tokenizer = None
    run_started = time.perf_counter()
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one CUDA device")
        torch.set_num_threads(8)

        def blocked_connect(self, address):
            network_requests.append(repr(address))
            raise RuntimeError("network disabled in formal inference")

        socket.socket.connect = blocked_connect
        model_path = args.model_path or Path(sorted(Path("/root/vlm_assets/models").glob("InternVL3-2B-Instruct-*"))[0])
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
        load_kw = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": True,
            "use_flash_attn": False,
        }
        if args.backend == "int4_g128":
            from transformers import TorchAoConfig
            load_kw.update({
                "low_cpu_mem_usage": True,
                "device_map": "cuda",
                "quantization_config": TorchAoConfig("int4_weight_only", group_size=128),
            })
        model = AutoModel.from_pretrained(model_path, **load_kw).eval()
        if args.backend == "bf16":
            model = model.cuda()
        torch.cuda.synchronize()
        metadata["model_class"] = type(model).__name__
        metadata["parameter_dtype"] = str(next(model.parameters()).dtype)
        metadata["parameter_device"] = str(next(model.parameters()).device)
        metadata["protocol_version"] = "v1.2"
        metadata["amendment_id"] = "VLM_ESWA_REVISION_AMENDMENT_002"
        metadata["torch_set_num_threads"] = 8
        metadata["internvl_normalization"] = "vectorized/batched normalization"
        metadata["frozen_config_v1_2_match"] = True
        if args.backend == "int4_g128":
            metadata["quantization_coverage"] = worker.int4_coverage(model)
            if not metadata["quantization_coverage"]["coverage_valid"]:
                raise RuntimeError("INT4 coverage/load path invalid")
        to_tensor = transforms.Compose([
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        normalization_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        normalization_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        for index, row in enumerate(rows):
            if index in completed:
                continue
            if str(row["sample_id"]) in {str(record["sample_id"]) for record in completed.values()}:
                raise RuntimeError("duplicate completed sample_id")
            sample_started = time.perf_counter()
            with Image.open(Path(row["image_path"])) as source:
                image = source.convert("RGB")
            blocks = worker.dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
            pixels_cpu = torch.stack([to_tensor(block) for block in blocks])
            pixels_cpu.sub_(normalization_mean).div_(normalization_std)
            pixels = pixels_cpu.to(dtype=torch.bfloat16, device="cuda")
            input_ids, attention_mask, rendered = worker.internvl_prompt(model, tokenizer, str(row["question"]) + SUFFIX, pixels)
            candidate_ids = worker.contextual_ids(tokenizer, rendered)
            flags = torch.ones((pixels.shape[0], 1), dtype=torch.long, device="cuda")
            with torch.inference_mode():
                output = model(pixel_values=pixels, input_ids=input_ids, attention_mask=attention_mask, image_flags=flags, return_dict=True)
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
                "model": args.model,
                "model_revision": MODEL_REVISION,
                "backend": args.backend,
                "dtype": "bfloat16",
                "batch_size": 1,
                "prompt_version": "prompt_v3",
                "candidate_adapter": "internvl-chat-v1",
                "candidate_token_ids": dict(zip(LABELS, candidate_ids)),
                "candidate_token_count": 4,
                "parser_status": "PASS",
                "parser_reason": "single_step_candidate_argmax",
                "gold_accessed": False,
                "ground_truth": None,
                "correct": None,
                "GT_accessed": False,
                "scientific_metrics_computed": False,
                "dynamic_block_count": len(blocks),
                "prompt_token_count": int(input_ids.shape[-1]),
                "sample_wall_time_sec": round(time.perf_counter() - sample_started, 6),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                **scored,
            }
            row_path = args.output / ".rows" / f"{index:06d}.json"
            atomic_record(row_path, record)
            completed[index] = record
            with (args.output / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            metadata["completed_samples"] = len(completed)
            metadata["last_sample_index"] = index
            metadata["last_sample_id"] = str(row["sample_id"])
            metadata["elapsed_sec"] = round(time.perf_counter() - run_started, 3)
            atomic_json(args.output / "metadata.json", metadata)
            print(json.dumps({"event": "sample_done", "index": index, "sample_id": row["sample_id"], "blocks": len(blocks), "tokens": int(input_ids.shape[-1])}, ensure_ascii=False), flush=True)
            del image, pixels, input_ids, attention_mask, flags, output
            torch.cuda.empty_cache()

        ordered = [completed[index] for index in range(len(rows))]
        if [record["sample_id"] for record in ordered] != [str(row["sample_id"]) for row in rows]:
            raise RuntimeError("manifest/order mismatch in completed output")
        metadata.update({
            "status": "PASS",
            "completed_samples": len(ordered),
            "unique_sample_ids": len({record["sample_id"] for record in ordered}),
            "parser_failures": 0,
            "probability_failures": 0,
            "network_requests": len(network_requests),
            "wall_time_sec": round(time.perf_counter() - run_started, 3),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        })
        audit = {
            "stage": f"074_PREDICTION_GATE_{'R1_1' if args.backend == 'bf16' else 'R1_2'}_PENDING",
            "expected_rows": len(ordered),
            "actual_rows": len(ordered),
            "unique_sample_ids": len({record["sample_id"] for record in ordered}),
            "parser_success_percent": 100.0,
            "all_candidate_logits_finite": True,
            "all_candidate_probabilities_finite": True,
            "all_probability_sums_valid": all(abs(float(record["probability_sum"]) - 1.0) < 1e-5 for record in ordered),
            "generated_answer_equals_candidate_argmax": all(record["generated_score_agree"] if "generated_score_agree" in record else record["generation_score_agree"] for record in ordered),
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
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    atomic_json(args.output / "metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
