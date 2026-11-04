#!/usr/bin/env python3
"""GT-blind Amendment 004 candidate-mass field recovery for one backend cell."""

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
from typing import Any, Dict, List


LABELS = ["A", "B", "C", "D"]
AMENDMENT = "VLM_ESWA_REVISION_AMENDMENT_004"
INTERNVL_REVISION = "f6c7b60375759170fd49f5e9e298e2178485cba5"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SUFFIX = "\nAnswer with exactly one option letter: A, B, C, or D. Do not provide explanations or any other text."


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"non-object row: {path}")
    return rows


def load_manifest(path: Path, dataset: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    expected = {"mmstar": 1500, "ai2d": 3088}[dataset]
    if len(rows) != expected:
        raise RuntimeError(f"manifest size mismatch: {len(rows)} != {expected}")
    required = {"sample_id", "dataset", "dataset_revision", "question", "image_path"}
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("manifest missing required fields")
    forbidden = {"answer", "label", "gold", "ground_truth", "correct", "correct_answer"}
    if any(forbidden.intersection(row) for row in rows):
        raise RuntimeError("outcome field entered recovery manifest")
    if {str(row["dataset"]).lower() for row in rows} != {dataset}:
        raise RuntimeError("manifest dataset mismatch")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise RuntimeError("duplicate manifest sample_id")
    return rows


def load_completed(out_dir: Path, rows: List[Dict[str, Any]], resume: bool) -> Dict[int, Dict[str, Any]]:
    if not out_dir.exists():
        if resume:
            raise RuntimeError("resume requested but recovery output does not exist")
        return {}
    contents = list(out_dir.iterdir())
    meaningful = [path for path in contents if path.name not in {"run.log", "launcher.log"}]
    if meaningful and not resume:
        raise RuntimeError("recovery output exists; refusing overwrite")
    if not resume:
        return {}
    row_dir = out_dir / ".rows"
    if not row_dir.exists():
        raise RuntimeError("resume requested without .rows")
    expected = [str(row["sample_id"]) for row in rows]
    completed: Dict[int, Dict[str, Any]] = {}
    for path in sorted(row_dir.glob("*.json")):
        index = int(path.stem)
        record = read_json(path)
        if index < 0 or index >= len(rows) or record.get("sample_id") != expected[index]:
            raise RuntimeError(f"resume row mismatch: {path}")
        completed[index] = record
    return completed


def rebuild_predictions(out_dir: Path, records: Dict[int, Dict[str, Any]]) -> None:
    temporary = out_dir / "predictions.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    os.replace(temporary, out_dir / "predictions.jsonl")


def score_logits(logits, candidate_ids):
    import torch

    full = logits.float()
    candidate = full[candidate_ids]
    if not torch.isfinite(full).all().item() or not torch.isfinite(candidate).all().item():
        raise RuntimeError("recovery logits contain NaN/Inf")
    probabilities = torch.softmax(candidate, dim=-1)
    if not torch.isfinite(probabilities).all().item():
        raise RuntimeError("recovery candidate probabilities contain NaN/Inf")
    probs = [float(value) for value in probabilities.detach().cpu().tolist()]
    choice = max(range(4), key=lambda index: probs[index])
    candidate_logsumexp = float(torch.logsumexp(candidate, dim=0).detach().cpu().item())
    vocab_logsumexp = float(torch.logsumexp(full, dim=0).detach().cpu().item())
    log_mass = candidate_logsumexp - vocab_logsumexp
    candidate_mass = float(math.exp(log_mass))
    if not math.isfinite(candidate_mass) or candidate_mass < 0.0 or candidate_mass > 1.0:
        raise RuntimeError("candidate_mass outside [0,1]")
    return {
        "candidate_logits": [float(value) for value in candidate.detach().cpu().tolist()],
        "candidate_probabilities": dict(zip(LABELS, probs)),
        "probability_sum": float(sum(probs)),
        "candidate_logsumexp": candidate_logsumexp,
        "vocab_logsumexp": vocab_logsumexp,
        "candidate_mass": candidate_mass,
        "argmax_candidate_answer": LABELS[choice],
    }


def canonical_path(root: Path, model: str, dataset: str, backend: str) -> Path:
    return root / "artifacts/eswa_revision_v1/02_full_inference" / model / dataset / backend / "predictions.jsonl"


def audit_equivalence(root: Path, output: Path, model: str, dataset: str, backend: str, n: int) -> Dict[str, Any]:
    import torch

    recovery = read_jsonl(output / "predictions.jsonl")
    original_path = canonical_path(root, model, dataset, backend)
    original = read_jsonl(original_path)
    if len(recovery) != n or len(original) != n:
        return {"status": "HOLD", "reason": "row_count_mismatch", "recovery_rows": len(recovery), "original_rows": len(original)}
    different_logits = 0
    max_abs_diff = 0.0
    different_probabilities = 0
    different_rows = 0
    different_argmax = 0
    different_tokens = 0
    internal_mass_different = 0
    max_mass_abs_diff = 0.0
    internal_candidate_logsumexp_different = 0
    max_candidate_logsumexp_abs_diff = 0.0
    audit_device = "cuda" if torch.cuda.is_available() else "cpu"
    for old, new in zip(original, recovery):
        row_diff = False
        if old.get("sample_id") != new.get("sample_id") or old.get("source_row") != new.get("source_row"):
            row_diff = True
        old_logits = [float(value) for value in old.get("candidate_logits", [])]
        new_logits = [float(value) for value in new.get("candidate_logits", [])]
        if old_logits != new_logits:
            different_logits += 1
            row_diff = True
            if len(old_logits) == len(new_logits):
                max_abs_diff = max(max_abs_diff, max((abs(a - b) for a, b in zip(old_logits, new_logits)), default=0.0))
            else:
                max_abs_diff = float("inf")
        old_probs = old.get("candidate_probabilities")
        new_probs = new.get("candidate_probabilities")
        if isinstance(old_probs, dict) and old_probs != new_probs:
            different_probabilities += 1
            row_diff = True
        old_argmax = old.get("score_argmax_label", old.get("generated_answer"))
        if old_argmax != new.get("argmax_candidate_answer"):
            different_argmax += 1
            row_diff = True
        if old.get("candidate_token_ids") != new.get("candidate_token_ids"):
            different_tokens += 1
            row_diff = True
        if len(new_logits) == 4:
            tensor = torch.tensor(new_logits, dtype=torch.float32, device=audit_device)
            derived_logsumexp = float(torch.logsumexp(tensor, dim=0).item())
            max_candidate_logsumexp_abs_diff = max(max_candidate_logsumexp_abs_diff, abs(float(new["candidate_logsumexp"]) - derived_logsumexp))
            if derived_logsumexp != float(new["candidate_logsumexp"]):
                internal_candidate_logsumexp_different += 1
        derived_mass = float(math.exp(float(new["candidate_logsumexp"]) - float(new["vocab_logsumexp"])))
        mass_diff = float(new["candidate_mass"]) - derived_mass
        max_mass_abs_diff = max(max_mass_abs_diff, abs(mass_diff))
        if float(new["candidate_mass"]) != derived_mass:
            internal_mass_different += 1
            row_diff = True
        if row_diff:
            different_rows += 1
    passed = not any((different_rows, different_logits, different_probabilities, different_argmax, different_tokens, internal_mass_different)) and max_mass_abs_diff == 0.0
    return {
        "status": "PASS" if passed else "HOLD",
        "original_cache": str(original_path),
        "recovery_cache": str(output / "predictions.jsonl"),
        "expected_rows": n,
        "actual_rows": len(recovery),
        "sample_id_exact": all(old.get("sample_id") == new.get("sample_id") for old, new in zip(original, recovery)),
        "source_row_exact": all(old.get("source_row") == new.get("source_row") for old, new in zip(original, recovery)),
        "different_rows": different_rows,
        "different_candidate_logits": different_logits,
        "max_candidate_logits_abs_diff": max_abs_diff,
        "different_candidate_probabilities": different_probabilities,
        "different_argmax_candidate": different_argmax,
        "different_candidate_token_ids": different_tokens,
        "candidate_logsumexp_internal_different_rows": internal_candidate_logsumexp_different,
        "candidate_logsumexp_internal_max_abs_diff": max_candidate_logsumexp_abs_diff,
        "candidate_logsumexp_audit_device": audit_device,
        "candidate_mass_internal_different_rows": internal_mass_different,
        "candidate_mass_internal_max_abs_diff": max_mass_abs_diff,
        "GT_accessed": False,
        "scientific_metrics_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=["internvl3_2b", "qwen2_5_vl_7b"], required=True)
    parser.add_argument("--dataset", choices=["mmstar", "ai2d"], required=True)
    parser.add_argument("--backend", choices=["bf16", "int4_g128"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    args.output = args.output.resolve()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    random.seed(42)
    rows = load_manifest(args.manifest, args.dataset)
    completed = load_completed(args.output, rows, args.resume)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".rows").mkdir(parents=True, exist_ok=True)
    if completed:
        rebuild_predictions(args.output, completed)
    metadata: Dict[str, Any] = {
        "stage": "080_CANDIDATE_MASS_RECOVERY",
        "status": "RUNNING",
        "amendment_id": AMENDMENT,
        "model": args.model,
        "dataset": args.dataset,
        "backend": args.backend,
        "manifest": str(args.manifest),
        "model_revision": INTERNVL_REVISION if args.model == "internvl3_2b" else QWEN_REVISION,
        "expected_samples": len(rows),
        "completed_samples": len(completed),
        "batch_size": 1,
        "dtype": "bfloat16",
        "GT_accessed": False,
        "scientific_metrics_computed": False,
        "bootstrap_computed": False,
        "output_namespace": "08_candidate_mass_recovery",
        "original_namespace_read_only": "02_full_inference",
        "resume_enabled": args.resume,
    }
    atomic_json(args.output / "metadata.json", metadata)
    old_connect = socket.socket.connect
    network_requests: List[str] = []
    model = processor = tokenizer = None
    run_started = time.perf_counter()
    try:
        import torch
        from PIL import Image

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one CUDA device")

        def blocked_connect(self, address):
            network_requests.append(repr(address))
            raise RuntimeError("network disabled in recovery inference")

        socket.socket.connect = blocked_connect
        if args.model == "internvl3_2b":
            original = load_script(root / "scripts/073_run_full_inference.py", "vlm_073_internvl")
            worker = original.load_worker_module()
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
            from transformers import AutoModel, AutoTokenizer

            torch.set_num_threads(8)
            model_path = args.model_path or Path(sorted(Path("/root/vlm_assets/models").glob("InternVL3-2B-Instruct-*"))[0])
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
            load_kw = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "local_files_only": True, "use_flash_attn": False}
            if args.backend == "int4_g128":
                from transformers import TorchAoConfig
                load_kw.update({"low_cpu_mem_usage": True, "device_map": "cuda", "quantization_config": TorchAoConfig("int4_weight_only", group_size=128)})
            model = AutoModel.from_pretrained(model_path, **load_kw).eval()
            if args.backend == "bf16":
                model = model.cuda()
            if args.backend == "int4_g128" and not worker.int4_coverage(model)["coverage_valid"]:
                raise RuntimeError("INT4 coverage/load path invalid")
            to_tensor = transforms.Compose([transforms.Lambda(lambda image: image.convert("RGB")), transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor()])
            normalization_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
            normalization_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        else:
            original = load_script(root / "scripts/073_run_qwen_full_inference.py", "vlm_073_qwen")
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            model_path = args.model_path or Path(sorted(Path("/root/vlm_assets/models").glob("Qwen2.5-VL-7B-Instruct-*"))[0])
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            load_kw = {"torch_dtype": torch.bfloat16, "device_map": "cuda", "trust_remote_code": True, "local_files_only": True}
            if args.backend == "bf16":
                load_kw["low_cpu_mem_usage"] = True
            else:
                from transformers import TorchAoConfig
                load_kw["quantization_config"] = TorchAoConfig("int4_weight_only", group_size=128)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kw).eval()
            tokenizer = processor.tokenizer

        for index, row in enumerate(rows):
            if index in completed:
                continue
            sample_started = time.perf_counter()
            with Image.open(Path(row["image_path"])) as source:
                image = source.convert("RGB")
            if args.model == "internvl3_2b":
                blocks = worker.dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
                pixels_cpu = torch.stack([to_tensor(block) for block in blocks])
                pixels_cpu.sub_(normalization_mean).div_(normalization_std)
                pixels = pixels_cpu.to(dtype=torch.bfloat16, device="cuda")
                input_ids, attention_mask, rendered = worker.internvl_prompt(model, tokenizer, str(row["question"]) + SUFFIX, pixels)
                candidate_ids = worker.contextual_ids(tokenizer, rendered)
                flags = torch.ones((pixels.shape[0], 1), dtype=torch.long, device="cuda")
                with torch.inference_mode():
                    output = model(pixel_values=pixels, input_ids=input_ids, attention_mask=attention_mask, image_flags=flags, return_dict=True)
                prompt_token_count = int(input_ids.shape[-1])
            else:
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": str(row["question"]) + SUFFIX}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
                model_inputs = {key: (value.cuda() if torch.is_tensor(value) else value) for key, value in inputs.items()}
                candidate_ids = original.contextual_ids(tokenizer, rendered)
                with torch.inference_mode():
                    output = model(**model_inputs, return_dict=True)
                prompt_token_count = int(model_inputs["input_ids"].shape[-1])
            scored = score_logits(output.logits[0, -1, :], candidate_ids)
            record = {
                "stage": "080_CANDIDATE_MASS_RECOVERY",
                "amendment_id": AMENDMENT,
                "sample_index": index,
                "sample_id": str(row["sample_id"]),
                "dataset": str(row["dataset"]),
                "dataset_revision": str(row["dataset_revision"]),
                "source_row": row.get("source_row"),
                "model": args.model,
                "model_revision": INTERNVL_REVISION if args.model == "internvl3_2b" else QWEN_REVISION,
                "backend": args.backend,
                "dtype": "bfloat16",
                "batch_size": 1,
                "prompt_version": "prompt_v3",
                "candidate_token_ids": dict(zip(LABELS, [int(value) for value in candidate_ids])),
                "candidate_token_count": 4,
                "argmax_candidate_answer": scored["argmax_candidate_answer"],
                "prompt_token_count": prompt_token_count,
                "dynamic_block_count": len(blocks) if args.model == "internvl3_2b" else None,
                "sample_wall_time_sec": round(time.perf_counter() - sample_started, 6),
                "GT_accessed": False,
                "scientific_metrics_computed": False,
                **scored,
            }
            atomic_json(args.output / ".rows" / f"{index:06d}.json", record)
            completed[index] = record
            with (args.output / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            metadata.update({"completed_samples": len(completed), "last_sample_index": index, "last_sample_id": str(row["sample_id"]), "elapsed_sec": round(time.perf_counter() - run_started, 3)})
            atomic_json(args.output / "metadata.json", metadata)
            print(json.dumps({"event": "sample_done", "index": index, "sample_id": row["sample_id"]}, ensure_ascii=False), flush=True)
            del image, output
            if args.model == "internvl3_2b":
                del pixels, input_ids, attention_mask, flags
            else:
                del inputs, model_inputs
            torch.cuda.empty_cache()

        ordered = [completed[index] for index in range(len(rows))]
        rebuild_predictions(args.output, completed)
        equivalence = audit_equivalence(root, args.output, args.model, args.dataset, args.backend, len(rows))
        audit = {
            "stage": "080_CANDIDATE_MASS_RECOVERY_EQUIVALENCE_GATE",
            "amendment_id": AMENDMENT,
            "status": equivalence["status"],
            "equivalence": equivalence,
            "candidate_mass_internal_validity": {
                "all_finite": all(math.isfinite(float(row["candidate_mass"])) and math.isfinite(float(row["candidate_logsumexp"])) and math.isfinite(float(row["vocab_logsumexp"])) for row in ordered),
                "all_mass_in_0_1": all(0.0 <= float(row["candidate_mass"]) <= 1.0 for row in ordered),
            },
            "GT_accessed": False,
            "scientific_metrics_computed": False,
            "bootstrap_computed": False,
        }
        atomic_json(args.output / "recovery_equivalence_audit.json", audit)
        metadata.update({"status": "PASS" if equivalence["status"] == "PASS" else "HOLD", "completed_samples": len(ordered), "network_requests": len(network_requests), "wall_time_sec": round(time.perf_counter() - run_started, 3), "recovery_equivalence_status": equivalence["status"]})
        atomic_json(args.output / "metadata.json", metadata)
        if equivalence["status"] != "PASS":
            raise RuntimeError("recovery equivalence gate HOLD")
    except Exception as exc:
        metadata.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)[-3000:], "traceback_tail": traceback.format_exc()[-6000:], "network_requests": len(network_requests)})
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
