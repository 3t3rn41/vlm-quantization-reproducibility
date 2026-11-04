#!/usr/bin/env python3
"""Study-authorized 072 engineering smoke for one model/backend/dataset.

This worker deliberately records only engineering stability.  It does not
load outcome labels, compute probabilities/risk, calculate scientific
metrics, or write correctness fields.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

LABELS = ["A", "B", "C", "D"]
SUFFIX = "\nAnswer with exactly one option letter: A, B, C, or D. Do not provide explanations or any other text."
M1_REVISION = "f6c7b60375759170fd49f5e9e298e2178485cba5"
M2_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def build_transform(input_size: int = 448):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def closest_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
            best_diff, best = diff, ratio
    return best


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    width, height = image.size
    aspect = width / height
    ratios = set((i, j) for n in range(min_num, max_num + 1)
                 for i in range(1, n + 1) for j in range(1, n + 1)
                 if min_num <= i * j <= max_num)
    target = closest_ratio(aspect, sorted(ratios, key=lambda x: x[0] * x[1]), width, height, image_size)
    resized = image.resize((image_size * target[0], image_size * target[1]))
    blocks = []
    for index in range(target[0] * target[1]):
        columns = target[0]
        x = index % columns
        y = index // columns
        blocks.append(resized.crop((x * image_size, y * image_size,
                                    (x + 1) * image_size, (y + 1) * image_size)))
    if use_thumbnail and len(blocks) != 1:
        blocks.append(image.resize((image_size, image_size)))
    return blocks


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50 or len({row.get("sample_id") for row in rows}) != 50:
        raise RuntimeError("smoke manifest must contain exactly 50 unique sample IDs")
    for row in rows:
        if any(key in row for key in ("answer", "label", "gold", "ground_truth", "correct", "correct_answer")):
            raise RuntimeError("forbidden outcome field in smoke manifest")
    return rows


def int4_coverage(model) -> dict:
    import torch
    total = quantized = 0
    q_modules = 0
    unquantized = 0
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


def internvl_prompt(model, tokenizer, question: str, pixel_values):
    sys.path.insert(0, str(Path(model.config._name_or_path)))
    from conversation import get_conv_template
    if "<image>" not in question:
        question = "<image>\n" + question
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * pixel_values.shape[0] + "</img>"
    query = query.replace("<image>", image_tokens, 1)
    encoded = tokenizer(query, return_tensors="pt")
    return encoded["input_ids"].to(pixel_values.device), encoded["attention_mask"].to(pixel_values.device), query


def write_outputs(out_dir: Path, metadata: dict, records: list[dict], audit: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "engineering_records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args) -> int:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    random.seed(42)
    out_dir = Path(args.output)
    rows = load_rows(Path(args.manifest))
    metadata = {
        "stage": "072_THROUGHPUT_SMOKE",
        "status": "072_SMOKE_FAIL",
        "result_tier": "INFRA_ONLY",
        "scientific": False,
        "model": args.model,
        "dataset": args.dataset,
        "backend": args.backend,
        "expected_samples": 50,
        "completed_samples": 0,
        "batch_size": 1,
        "dtype": "bfloat16",
        "sampling": False,
        "model_eval": True,
        "torch_inference_mode": True,
        "use_flash_attn": False,
        "prompt_suffix_frozen": SUFFIX,
        "candidate_adapter": "internvl-chat-v1" if args.model == "internvl3_2b" else "vlm_qwen_candidate_adapter_v2",
        "quantization": "TorchAO Int4WeightOnlyConfig group_size=128" if args.backend == "int4_g128" else "none",
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "network_requests": 0,
        "parser_failures": 0,
        "nan_count": 0,
        "inf_count": 0,
        "oom": False,
        "crash_count": 0,
        "candidate_adapter_ok": False,
        "frozen_config_match": True,
        "samples_per_sec": None,
        "wall_time_sec": None,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }
    records: list[dict] = []
    audit = {"expected_samples": 50, "actual_samples": 0, "unique_sample_ids": 0,
             "parser_failures": 0, "nan_count": 0, "inf_count": 0,
             "GT_accessed": False, "scientific_metrics_computed": False,
             "scientific_eligible": False}
    net: list[str] = []
    old_connect = socket.socket.connect
    model = processor = tokenizer = None
    started = time.perf_counter()

    def blocked_connect(self, address):
        net.append(repr(address))
        raise RuntimeError("network disabled in 072 smoke")

    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one CUDA device")
        socket.socket.connect = blocked_connect
        model_path = args.model_path
        print(json.dumps({"event": "model_loading_start", "model": args.model, "backend": args.backend}, ensure_ascii=False), flush=True)
        if args.model == "internvl3_2b":
            from transformers import AutoModel, AutoTokenizer, TorchAoConfig
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
            print(json.dumps({"event": "tokenizer_loaded"}, ensure_ascii=False), flush=True)
            load_kw = {"torch_dtype": torch.bfloat16, "trust_remote_code": True,
                       "local_files_only": True, "use_flash_attn": False}
            if args.backend == "int4_g128":
                load_kw.update({"low_cpu_mem_usage": True, "device_map": "cuda",
                                "quantization_config": TorchAoConfig("int4_weight_only", group_size=128)})
            model = AutoModel.from_pretrained(model_path, **load_kw).eval()
            if args.backend == "bf16":
                model = model.cuda()
            print(json.dumps({"event": "model_loaded"}, ensure_ascii=False), flush=True)
            model_revision = M1_REVISION
        else:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TorchAoConfig
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            load_kw = {"torch_dtype": torch.bfloat16, "device_map": "cuda",
                       "trust_remote_code": True, "local_files_only": True}
            if args.backend == "bf16":
                load_kw["low_cpu_mem_usage"] = True
            else:
                load_kw["quantization_config"] = TorchAoConfig("int4_weight_only", group_size=128)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kw).eval()
            tokenizer = processor.tokenizer
            print(json.dumps({"event": "model_loaded"}, ensure_ascii=False), flush=True)
            model_revision = M2_REVISION
        torch.cuda.synchronize()
        metadata["model_revision"] = model_revision
        if args.backend == "int4_g128":
            metadata["quantization_coverage"] = int4_coverage(model)
            if not metadata["quantization_coverage"]["coverage_valid"]:
                raise RuntimeError("INT4 coverage/load path invalid")
        if net:
            raise RuntimeError("unexpected network access")
        metadata["candidate_adapter_ok"] = True
        sample_start = time.perf_counter()
        transform = build_transform(448) if args.model == "internvl3_2b" else None
        for index, row in enumerate(rows):
            print(json.dumps({"event": "sample_start", "index": index, "sample_id": row["sample_id"]}, ensure_ascii=False), flush=True)
            from PIL import Image
            sample_start_one = time.perf_counter()
            image_path = Path(row["image_path"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            if args.model == "internvl3_2b":
                blocks = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
                print(json.dumps({"event": "image_blocks_ready", "index": index, "blocks": len(blocks)}, ensure_ascii=False), flush=True)
                pixels = torch.stack([transform(block) for block in blocks]).to(dtype=torch.bfloat16, device="cuda")
                print(json.dumps({"event": "pixels_ready", "index": index, "shape": list(pixels.shape)}, ensure_ascii=False), flush=True)
                input_ids, attention_mask, rendered = internvl_prompt(model, tokenizer, row["question"] + SUFFIX, pixels)
                print(json.dumps({"event": "prompt_ready", "index": index, "tokens": int(input_ids.shape[-1])}, ensure_ascii=False), flush=True)
                candidate_ids = contextual_ids(tokenizer, rendered)
                print(json.dumps({"event": "candidate_ids_ready", "index": index}, ensure_ascii=False), flush=True)
                flags = torch.ones((pixels.shape[0], 1), dtype=torch.long, device="cuda")
                print(json.dumps({"event": "forward_start", "index": index}, ensure_ascii=False), flush=True)
                with torch.inference_mode():
                    output = model(pixel_values=pixels, input_ids=input_ids,
                                   attention_mask=attention_mask, image_flags=flags, return_dict=True)
                print(json.dumps({"event": "forward_done", "index": index}, ensure_ascii=False), flush=True)
            else:
                messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                               {"type": "text", "text": row["question"] + SUFFIX}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
                model_inputs = {key: (value.cuda() if torch.is_tensor(value) else value)
                                for key, value in inputs.items()}
                candidate_ids = contextual_ids(tokenizer, rendered)
                with torch.inference_mode():
                    output = model(**model_inputs, return_dict=True)
            torch.cuda.synchronize()
            logits = output.logits[0, -1, :]
            vals = [float(logits[token_id].float().item()) for token_id in candidate_ids]
            finite = all(math.isfinite(value) for value in vals)
            if not finite:
                metadata["nan_count"] += sum(math.isnan(value) for value in vals)
                metadata["inf_count"] += sum(math.isinf(value) for value in vals)
            chosen_index = max(range(4), key=lambda i: vals[i]) if finite else None
            parser_status = "PASS" if chosen_index is not None else "FAIL"
            if parser_status != "PASS":
                metadata["parser_failures"] += 1
            records.append({
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "source_row": row["source_row"],
                "candidate_token_count": 4,
                "candidate_adapter_ok": True,
                "parser_status": parser_status,
                "generated_token_count": 1 if chosen_index is not None else 0,
                "eos_observed": True if chosen_index is not None else False,
                "logits_finite": finite,
                "sample_wall_time_sec": round(time.perf_counter() - sample_start_one, 6),
            })
            metadata["completed_samples"] = index + 1
            print(json.dumps({"event": "sample_done", "index": index}, ensure_ascii=False), flush=True)
        elapsed = time.perf_counter() - sample_start
        metadata["wall_time_sec"] = round(elapsed, 6)
        metadata["samples_per_sec"] = round(len(records) / elapsed, 6) if elapsed > 0 else None
        metadata["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        metadata["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        audit.update({"actual_samples": len(records), "unique_sample_ids": len({r["sample_id"] for r in records}),
                      "parser_failures": metadata["parser_failures"], "nan_count": metadata["nan_count"],
                      "inf_count": metadata["inf_count"], "network_requests": len(net),
                      "candidate_adapter_ok": metadata["candidate_adapter_ok"],
                      "frozen_config_match": metadata["frozen_config_match"]})
        passed = (len(records) == 50 and audit["unique_sample_ids"] == 50
                  and metadata["parser_failures"] == 0 and metadata["nan_count"] == 0
                  and metadata["inf_count"] == 0 and not net)
        metadata["status"] = "072_SMOKE_PASS" if passed else "072_SMOKE_FAIL"
        return_code = 0 if passed else 2
    except Exception as exc:
        message = str(exc)
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = message[-2000:]
        metadata["crash_count"] = 1
        metadata["oom"] = "out of memory" in message.lower() or "cuda out of memory" in message.lower()
        metadata["network_requests"] = len(net)
        return_code = 2
    finally:
        socket.socket.connect = old_connect
        metadata["network_requests"] = len(net)
        metadata["GT_accessed"] = False
        metadata["scientific_metrics_computed"] = False
        metadata["scientific_eligible"] = False
        try:
            if model is not None:
                del model
            if processor is not None:
                del processor
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        audit.update({"actual_samples": len(records), "unique_sample_ids": len({r["sample_id"] for r in records}),
                      "parser_failures": metadata["parser_failures"], "nan_count": metadata["nan_count"],
                      "inf_count": metadata["inf_count"], "network_requests": len(net),
                      "candidate_adapter_ok": metadata["candidate_adapter_ok"],
                      "frozen_config_match": metadata["frozen_config_match"]})
        metadata["total_elapsed_sec"] = round(time.perf_counter() - started, 6)
        write_outputs(out_dir, metadata, records, audit)
    return return_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["internvl3_2b", "qwen2_5_vl_7b"], required=True)
    ap.add_argument("--dataset", choices=["mmstar", "ai2d"], required=True)
    ap.add_argument("--backend", choices=["bf16", "int4_g128"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
