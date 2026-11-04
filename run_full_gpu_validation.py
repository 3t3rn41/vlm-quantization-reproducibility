"""Rerun every frozen GPU inference cell in an isolated output directory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


CELLS = {"mmstar": 1500, "ai2d": 3088}
MODELS = {"internvl3_2b": "InternVL3-2B-Instruct-*", "qwen2_5_vl_7b": "Qwen2.5-VL-7B-Instruct-*"}


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-original", type=Path, required=True)
    parser.add_argument("--python-hqq", type=Path, required=True)
    parser.add_argument("--hqq-script", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible CUDA device required"
    assert shutil.disk_usage(args.output_root.parent).free > 2_000_000_000, "insufficient output space"
    models = {}
    for model, pattern in MODELS.items():
        matches = list(args.models_root.glob(pattern))
        assert len(matches) == 1, (model, matches)
        models[model] = matches[0]
    for dataset in CELLS:
        assert (args.workspace / "artifacts/eswa_revision_v1/01_manifests" / f"{dataset}_full_clean.jsonl").is_file()
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "10", "MKL_NUM_THREADS": "10"})
    sys.path.insert(0, str(args.cache_root))
    from verify_formal_prediction_cache import compare, predictions, rows
    state = {"status": "RUNNING", "started": time.time(), "jobs": []}
    atomic(args.output_root / "progress.json", state)
    for model in MODELS:
        for dataset, expected in CELLS.items():
            for backend in ("bf16", "int4_g128"):
                out = args.output_root / "original" / model / dataset / backend
                meta = out / "metadata.json"
                if meta.exists() and json.loads(meta.read_text())["status"] == "PASS":
                    assert json.loads(meta.read_text())["completed_samples"] == expected
                else:
                    assert not out.exists(), f"partial output requires manual review: {out}"
                    script = "073_run_full_inference.py" if model == "internvl3_2b" else "073_run_qwen_full_inference.py"
                    command = [str(args.python_original), str(args.workspace / "scripts" / script), "--model", model, "--dataset", dataset, "--backend", backend,
                               "--manifest", str(args.workspace / "artifacts/eswa_revision_v1/01_manifests" / f"{dataset}_full_clean.jsonl"),
                               "--output", str(out), "--model-path", str(models[model])]
                    run(command, args.output_root / "logs" / f"{model}_{dataset}_{backend}.log", env)
                    produced = json.loads(meta.read_text())
                    assert produced["status"] == "PASS" and produced["completed_samples"] == expected, (model, dataset, backend)
                cell = f"{model}__{dataset}"
                cache = rows(args.cache_root / "manuscript/submission/numeric_audit_20261002" / f"{cell}_derived.csv")
                difference, _ = compare(predictions(out / "predictions.jsonl"), cache, "bf16" if backend == "bf16" else "int4", cell)
                state["jobs"].append({"model": model, "dataset": dataset, "backend": backend, "rows": expected, "status": "PASS"})
                atomic(args.output_root / "progress.json", state)
    for model in MODELS:
        out = args.output_root / "hqq/formal" / model
        completion = out / "completion.json"
        if completion.exists() and json.loads(completion.read_text())["status"] == "PASS":
            assert json.loads(completion.read_text())["counts"] == CELLS
        else:
            assert not out.exists(), f"partial output requires manual review: {out}"
            command = [str(args.python_hqq), str(args.hqq_script), "--model", model, "--phase", "formal", "--base", str(args.workspace), "--output-root", str(args.output_root / "hqq"), "--model-path", str(models[model])]
            run(command, args.output_root / "logs" / f"{model}_hqq.log", env)
            assert json.loads(completion.read_text())["status"] == "PASS"
        for dataset, expected in CELLS.items():
            cell = f"{model}__{dataset}"
            cache = rows(args.cache_root / "revise/Analysis_2/Major_2/derived_audit" / f"{cell}_derived.csv")
            difference, _ = compare(predictions(out / f"{dataset}_predictions.jsonl"), cache, "int4", cell)
        state["jobs"].append({"model": model, "backend": "hqq_int4_g128", "rows": sum(CELLS.values()), "status": "PASS"})
        atomic(args.output_root / "progress.json", state)
    verifier = args.cache_root / "verify_formal_prediction_cache.py"
    completed = subprocess.run([str(args.python_original), str(verifier), "--original-inference-root", str(args.output_root / "original"), "--hqq-root", str(args.output_root / "hqq/formal"), "--cache-root", str(args.cache_root)], capture_output=True, text=True, check=True, env=env)
    state.update({"status": "PASS", "completed": time.time(), "comparison": json.loads(completed.stdout)})
    atomic(args.output_root / "progress.json", state)
    print(json.dumps({"status": "PASS", "jobs": len(state["jobs"]), "comparison": state["comparison"]}))


if __name__ == "__main__":
    main()
