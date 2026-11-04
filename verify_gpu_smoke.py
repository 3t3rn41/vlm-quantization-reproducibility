"""Check independent GPU smoke outputs against archived numeric references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = {"internvl3_2b": 294, "qwen2_5_vl_7b": 326}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hqq-root", type=Path, required=True, help="Directory containing smoke/<model>/ outputs")
    parser.add_argument("--queue-smoke", type=Path, action="append", default=[], help="One 50-input BF16 or TorchAO smoke output directory; repeat as needed")
    args = parser.parse_args()
    reference = load(ROOT / "validation/bf16_parity_reference.json")
    result = {"hqq_models": {}, "queue_smokes": []}
    for model, expected_modules in MODELS.items():
        folder = args.hqq_root / "smoke" / model
        completion = load(folder / "completion.json")
        metadata = load(folder / "metadata.json")
        rows = load(folder / "bf16_parity_rows.json")
        expected = reference[model]
        assert completion["status"] == "PASS" and completion["counts"] == {"mmstar": 3, "ai2d": 3}
        assert metadata["quantized_modules"] == expected_modules
        assert len(rows) == len(expected) == 6
        difference = 0.0
        for got, archived in zip(rows, expected):
            for field in ("sample_id", "prompt_token_count", "normalized_prediction"):
                assert got[field] == archived[field], (model, field)
            difference = max(difference, *(abs(got["candidate_probabilities"][label] - archived["candidate_probabilities"][label]) for label in "ABCD"))
        assert difference <= 1e-6, (model, difference)
        for dataset in ("mmstar", "ai2d"):
            assert len((folder / f"{dataset}_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
        result["hqq_models"][model] = {"rows": 6, "quantized_modules": expected_modules, "max_probability_difference": difference}
    for folder in args.queue_smoke:
        metadata = load(folder / "metadata.json")
        audit = load(folder / "audit.json")
        assert metadata["status"] == "072_SMOKE_PASS" and metadata["completed_samples"] == 50
        assert audit["actual_samples"] == 50 and audit["unique_sample_ids"] == 50
        for field in ("parser_failures", "nan_count", "inf_count", "network_requests"):
            assert metadata[field] == audit[field] == 0, (folder, field)
        result["queue_smokes"].append({"model": metadata["model"], "backend": metadata["backend"], "dataset": metadata["dataset"], "rows": 50})
    result["status"] = "PASS"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
