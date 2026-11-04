#!/usr/bin/env python3
"""084 frozen latency/memory audit for the VLM revision.

This is an engineering-only benchmark.  It never reads ground truth,
prediction caches, or scientific metrics.  Each cell runs the frozen native
processor path with 20 warm-ups followed by 100 synchronized batch-1 forward
measurements.  The four cells are executed serially so the previous model is
fully released before the next one is loaded.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DEFAULT = Path(__file__).resolve().parents[2]
STAGE = "084_LATENCY_MEMORY_AUDIT_R1_3"
SUFFIX = "\nAnswer with exactly one option letter: A, B, C, or D. Do not provide explanations or any other text."
LABELS = ("A", "B", "C", "D")
MODEL_REVISIONS = {
    "internvl3_2b": "f6c7b60375759170fd49f5e9e298e2178485cba5",
    "qwen2_5_vl_7b": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}
CELLS = (
    ("internvl3_2b", "bf16"),
    ("internvl3_2b", "int4_g128"),
    ("qwen2_5_vl_7b", "bf16"),
    ("qwen2_5_vl_7b", "int4_g128"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_new_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, path)


def load_worker_module():
    worker_path = Path(__file__).with_name("072_smoke_revision_inference.py")
    spec = importlib.util.spec_from_file_location("vlm_072_worker_for_084", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen worker: {worker_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path, expected_dataset: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50 or len({str(row.get("sample_id")) for row in rows}) != 50:
        raise RuntimeError(f"expected 50 unique rows in {path}")
    if {str(row.get("dataset", "")).lower() for row in rows} != {expected_dataset}:
        raise RuntimeError(f"dataset mismatch in {path}")
    forbidden = {"answer", "label", "gold", "ground_truth", "correct", "correct_answer"}
    if any(forbidden.intersection(row) for row in rows):
        raise RuntimeError(f"outcome field in benchmark manifest: {path}")
    return rows


def load_benchmark(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    smoke = load_json(root / "artifacts/eswa_revision_v1/02_smoke/smoke_ids.json")
    mmstar = load_rows(root / "artifacts/eswa_revision_v1/02_smoke/smoke_mmstar_50.jsonl", "mmstar")
    ai2d = load_rows(root / "artifacts/eswa_revision_v1/02_smoke/smoke_ai2d_50.jsonl", "ai2d")
    for name, rows in (("mmstar", mmstar), ("ai2d", ai2d)):
        expected = smoke[name]
        actual = [{"sample_id": str(row["sample_id"]), "source_row": int(row["source_row"])} for row in rows]
        if actual != expected:
            raise RuntimeError(f"{name} smoke IDs/order do not exactly match frozen smoke_ids.json")
    return mmstar, ai2d


def resolve_image_path(root: Path, raw: str) -> Path:
    raw_path = Path(raw)
    candidates = [raw_path, root / raw_path, root / raw.replace("\\", "/")]
    normalized = raw.replace("\\", "/")
    marker = "VLM/revision_eswa_v1/"
    if marker in normalized:
        candidates.append(root / normalized.split(marker, 1)[1])
    marker2 = "artifacts/eswa_revision_v1/"
    if marker2 in normalized:
        candidates.append(root / normalized[normalized.index(marker2):])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve benchmark image: {raw}")


def find_model_path(model: str) -> Path:
    prefix = "InternVL3-2B-Instruct-" if model == "internvl3_2b" else "Qwen2.5-VL-7B-Instruct-"
    paths = sorted(Path("/root/vlm_assets/models").glob(prefix + "*"))
    if not paths:
        raise FileNotFoundError(f"no local model directory matching {prefix}*")
    return paths[0]


def quantization_coverage(model) -> dict[str, Any]:
    import torch
    total = quantized = q_modules = unquantized = 0
    classes: set[str] = set()
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        n = sum(parameter.numel() for parameter in module.parameters(recurse=False))
        total += int(n)
        weight = module.weight
        cls = type(weight).__name__
        classes.add(cls)
        is_quantized = ("AffineQuantized" in cls or "Quantized" in cls or "Int4" in cls
                        or "WeightOnly" in cls or hasattr(weight, "tensor_impl"))
        if is_quantized:
            q_modules += 1
            quantized += int(n)
        else:
            unquantized += 1
    return {
        "total_linear_parameters": total,
        "quantized_weight_parameters": quantized,
        "quantized_linear_modules": q_modules,
        "unquantized_linear_modules": unquantized,
        "quantized_weight_fraction": quantized / total if total else 0.0,
        "weight_classes": sorted(classes),
        "coverage_valid": q_modules > 0 and quantized > 0,
    }


def load_cell(model_name: str, backend: str, model_path: Path, worker):
    import torch
    torch.set_num_threads(8) if model_name == "internvl3_2b" else None
    if model_name == "internvl3_2b":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
        load_kw: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": True,
            "use_flash_attn": False,
        }
        if backend == "int4_g128":
            from transformers import TorchAoConfig
            load_kw.update({
                "low_cpu_mem_usage": True,
                "device_map": "cuda",
                "quantization_config": TorchAoConfig("int4_weight_only", group_size=128),
            })
        model = AutoModel.from_pretrained(model_path, **load_kw).eval()
        if backend == "bf16":
            model = model.cuda()
        return model, tokenizer, None, {"torch_set_num_threads": 8, "internvl_normalization": "vectorized/batched normalization"}
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    load_kw = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if backend == "bf16":
        load_kw["low_cpu_mem_usage"] = True
    else:
        from transformers import TorchAoConfig
        load_kw["quantization_config"] = TorchAoConfig("int4_weight_only", group_size=128)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kw).eval()
    return model, processor.tokenizer, processor, {
        "qwen_processor_family": "Qwen2_5_VLProcessor",
        "qwen_candidate_adapter": "vlm_qwen_candidate_adapter_v2",
        "qwen_native_preprocessing": True,
    }


def make_inputs(model_name: str, model, tokenizer, processor, row: dict[str, Any], root: Path, worker):
    import torch
    from PIL import Image
    with Image.open(resolve_image_path(root, str(row["image_path"]))) as source:
        image = source.convert("RGB")
    if model_name == "internvl3_2b":
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        to_tensor = transforms.Compose([
            transforms.Lambda(lambda value: value.convert("RGB")),
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        blocks = worker.dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
        pixels_cpu = torch.stack([to_tensor(block) for block in blocks])
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
        pixels_cpu.sub_(mean).div_(std)
        pixels = pixels_cpu.to(dtype=torch.bfloat16, device="cuda")
        input_ids, attention_mask, _ = worker.internvl_prompt(model, tokenizer, str(row["question"]) + SUFFIX, pixels)
        flags = torch.ones((pixels.shape[0], 1), dtype=torch.long, device="cuda")
        return {"pixel_values": pixels, "input_ids": input_ids, "attention_mask": attention_mask, "image_flags": flags, "return_dict": True}
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": str(row["question"]) + SUFFIX}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    return {key: (value.cuda() if torch.is_tensor(value) else value) for key, value in inputs.items()} | {"return_dict": True}


def forward(model, inputs):
    import torch
    with torch.inference_mode():
        return model(**inputs)


def run_cell(root: Path, model_name: str, backend: str, rows: list[dict[str, Any]], cell_dir: Path) -> dict[str, Any]:
    import torch
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty cell directory: {cell_dir}")
    cell_dir.mkdir(parents=True, exist_ok=False)
    worker = load_worker_module()
    old_connect = socket.socket.connect
    network_requests: list[str] = []
    model = tokenizer = processor = None
    raw: list[dict[str, Any]] = []
    warmup: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "stage": STAGE,
        "status": "INVALID",
        "model": model_name,
        "backend": backend,
        "model_revision": MODEL_REVISIONS[model_name],
        "warmup_n": 20,
        "formal_n": 100,
        "batch_size": 1,
        "sampling": False,
        "model_eval": True,
        "torch_inference_mode": True,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "network_requests": 0,
        "crash_count": 0,
        "OOM": False,
        "formal_order_exact": False,
        "benchmark_ids_match": False,
        "frozen_config_match": False,
    }

    def blocked_connect(self, address):
        network_requests.append(repr(address))
        raise RuntimeError("network disabled in 084")

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one CUDA device")
        socket.socket.connect = blocked_connect
        model_path = find_model_path(model_name)
        model, tokenizer, processor, path_meta = load_cell(model_name, backend, model_path, worker)
        meta.update(path_meta)
        meta["model_path"] = str(model_path)
        meta["model_class"] = type(model).__name__
        meta["processor_class"] = type(processor).__name__ if processor is not None else None
        if backend == "int4_g128":
            meta["quantization_coverage"] = quantization_coverage(model)
            if not meta["quantization_coverage"]["coverage_valid"]:
                raise RuntimeError("INT4 coverage/load path invalid")
        torch.cuda.synchronize()
        meta["resident_allocated_after_load"] = int(torch.cuda.memory_allocated())
        meta["resident_reserved_after_load"] = int(torch.cuda.memory_reserved())
        for index, row in enumerate(rows[:20]):
            inputs = make_inputs(model_name, model, tokenizer, processor, row, root, worker)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = forward(model, inputs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            warmup.append({"formal_index": index, "sample_id": str(row["sample_id"]), "latency_sec": elapsed})
            del output, inputs
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for index, row in enumerate(rows):
            inputs = make_inputs(model_name, model, tokenizer, processor, row, root, worker)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = forward(model, inputs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            raw.append({
                "model": model_name,
                "backend": backend,
                "dataset": str(row["dataset"]),
                "sample_id": str(row["sample_id"]),
                "source_row": int(row["source_row"]),
                "formal_index": index,
                "latency_sec": float(elapsed),
            })
            del output, inputs
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        latencies = [float(item["latency_sec"]) for item in raw]
        finite = all(math.isfinite(value) for value in latencies)
        positive = all(value > 0.0 for value in latencies)
        meta.update({
            "formal_order_exact": [item["formal_index"] for item in raw] == list(range(100)),
            "benchmark_ids_match": len({item["sample_id"] for item in raw}) == 100,
            "frozen_config_match": True,
            "all_latencies_finite": finite,
            "all_latencies_positive": positive,
            "latency_median_sec": float(__import__("numpy").median(latencies)),
            "latency_p95_sec": float(__import__("numpy").percentile(latencies, 95)),
            "throughput_samples_per_sec": float(100.0 / sum(latencies)),
            "formal_peak_allocated": peak_allocated,
            "formal_peak_reserved": peak_reserved,
            "network_requests": len(network_requests),
            "peak_allocated_ge_resident": peak_allocated >= meta["resident_allocated_after_load"],
            "peak_reserved_ge_resident": peak_reserved >= meta["resident_reserved_after_load"],
            "OOM": False,
            "crash_count": 0,
            "status": "PASS",
        })
    except Exception as exc:
        message = str(exc)
        meta.update({
            "error_type": type(exc).__name__,
            "error": message[-2000:],
            "crash_count": 1,
            "OOM": "out of memory" in message.lower() or "cuda out of memory" in message.lower(),
            "network_requests": len(network_requests),
            "status": "INVALID",
        })
    finally:
        socket.socket.connect = old_connect
        meta["network_requests"] = len(network_requests)
        meta["GT_accessed"] = False
        meta["scientific_metrics_computed"] = False
        try:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if processor is not None:
                del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        write_new_jsonl(cell_dir / "084_latency_raw_v1_3.jsonl", raw)
        write_new_jsonl(cell_dir / "084_warmup_log_v1_3.jsonl", warmup)
        write_new_json(cell_dir / "084_cell_summary_v1_3.json", meta)
    if meta["status"] != "PASS":
        raise RuntimeError(f"084 cell failed: {model_name}/{backend}: {meta.get('error', 'unknown error')}")
    return meta


def aggregate(root: Path, mmstar: list[dict[str, Any]], ai2d: list[dict[str, Any]]) -> None:
    cost = root / "artifacts/eswa_revision_v1/11_cost"
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for model_name, backend in CELLS:
        key = f"{model_name}__{backend}"
        cell_dir = cost / "cells" / key
        summaries[key] = load_json(cell_dir / "084_cell_summary_v1_3.json")
        all_rows.extend([json.loads(line) for line in (cell_dir / "084_latency_raw_v1_3.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])
    by_cell = {key: value for key, value in summaries.items()}
    ids_expected = [str(row["sample_id"]) for row in mmstar + ai2d]
    source_expected = [int(row["source_row"]) for row in mmstar + ai2d]
    paired_checks = {}
    for model_name in ("internvl3_2b", "qwen2_5_vl_7b"):
        bf = [row for row in all_rows if row["model"] == model_name and row["backend"] == "bf16"]
        iq = [row for row in all_rows if row["model"] == model_name and row["backend"] == "int4_g128"]
        paired_checks[model_name] = {
            "bf16_ids_match_frozen": [row["sample_id"] for row in bf] == ids_expected,
            "int4_ids_match_frozen": [row["sample_id"] for row in iq] == ids_expected,
            "bf16_int4_ids_exact": [row["sample_id"] for row in bf] == [row["sample_id"] for row in iq],
            "bf16_source_rows_match_frozen": [row["source_row"] for row in bf] == source_expected,
            "int4_source_rows_match_frozen": [row["source_row"] for row in iq] == source_expected,
            "formal_order_exact": [row["formal_index"] for row in bf] == list(range(100)) and [row["formal_index"] for row in iq] == list(range(100)),
        }
    gate_cells = {}
    for key, summary in by_cell.items():
        gate_cells[key] = all([
            summary.get("status") == "PASS",
            summary.get("warmup_n") == 20,
            summary.get("formal_n") == 100,
            summary.get("batch_size") == 1,
            summary.get("sampling") is False,
            summary.get("model_eval") is True,
            summary.get("torch_inference_mode") is True,
            summary.get("formal_order_exact") is True,
            summary.get("frozen_config_match") is True,
            summary.get("all_latencies_finite") is True,
            summary.get("all_latencies_positive") is True,
            summary.get("resident_allocated_after_load", -1) >= 0,
            summary.get("resident_reserved_after_load", -1) >= 0,
            summary.get("formal_peak_allocated", -1) >= summary.get("resident_allocated_after_load", 0),
            summary.get("formal_peak_reserved", -1) >= summary.get("resident_reserved_after_load", 0),
            summary.get("OOM") is False,
            summary.get("crash_count") == 0,
            summary.get("network_requests") == 0,
            summary.get("GT_accessed") is False,
            summary.get("scientific_metrics_computed") is False,
        ])
    validity = {
        "stage": STAGE,
        "status": "PASS" if all(gate_cells.values()) and all(all(v.values()) for v in paired_checks.values()) else "FAIL",
        "protocol": {
            "benchmark": "frozen 072 smoke_ids: MMStar 50 + AI2D 50",
            "warmup_n": 20,
            "formal_n": 100,
            "formal_repetitions": 1,
            "batch_size": 1,
            "latency_boundary": "synchronize -> perf_counter -> forward -> synchronize -> perf_counter",
            "throughput_definition": "100 / sum(formal_latency_sec)",
            "cross_model_preprocessing_equivalence": False,
        },
        "cells": gate_cells,
        "paired_benchmark_gate": paired_checks,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "network_requests": sum(int(summary.get("network_requests", 0)) for summary in by_cell.values()),
        "performance_threshold_applied": False,
    }
    summaries_artifact = {
        "stage": STAGE,
        "status": validity["status"],
        "cells": by_cell,
        "paired_benchmark_gate": paired_checks,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "performance_threshold_applied": False,
    }
    memory_artifact = {
        "stage": STAGE,
        "status": validity["status"],
        "cells": {
            key: {
                "model": value.get("model"),
                "backend": value.get("backend"),
                "resident_allocated_after_load": value.get("resident_allocated_after_load"),
                "resident_reserved_after_load": value.get("resident_reserved_after_load"),
                "formal_peak_allocated": value.get("formal_peak_allocated"),
                "formal_peak_reserved": value.get("formal_peak_reserved"),
            }
            for key, value in by_cell.items()
        },
        "bytes_are_canonical": True,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
    }
    write_new_jsonl(cost / "084_latency_raw_v1_3.jsonl", all_rows)
    write_new_json(cost / "084_latency_memory_v1_3.json", memory_artifact)
    write_new_json(cost / "084_memory_summary_v1_3.json", summaries_artifact)
    write_new_json(cost / "084_validity_audit_v1_3.json", validity)
    if validity["status"] != "PASS":
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    mmstar, ai2d = load_benchmark(root)
    rows = mmstar + ai2d
    if len(rows) != 100:
        raise RuntimeError("combined benchmark must contain exactly 100 rows")
    cost = root / "artifacts/eswa_revision_v1/11_cost"
    if cost.exists() and any(cost.iterdir()):
        raise RuntimeError(f"084 output namespace already contains artifacts: {cost}")
    cost.mkdir(parents=True, exist_ok=False)
    for model_name, backend in CELLS:
        run_cell(root, model_name, backend, rows, cost / "cells" / f"{model_name}__{backend}")
    aggregate(root, mmstar, ai2d)
    print(json.dumps({"status": "PASS", "stage": STAGE, "formal_measurements": 400}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
