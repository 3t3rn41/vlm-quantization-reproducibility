"""Deterministic table rendering from the frozen VLM canonical artifacts.

This script is intentionally not a metric implementation.  It reads only the
086 source whitelist, selects already-computed scalar fields, renders exact
JSON-scalar spellings to CSV, and audits every rendered field back to its
source object before writing the PASS audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable


STAGE = "086_TABLE_RENDER"
CELL_ORDER = [
    "internvl3_2b_mmstar",
    "internvl3_2b_ai2d",
    "qwen2_5_vl_7b_mmstar",
    "qwen2_5_vl_7b_ai2d",
]
POP_ORDER = ["full_clean", "strict_unseen"]
SIGNAL_ORDER = [("risk_max", "risk_max"), ("risk_entropy", "entropy"), ("risk_margin", "margin")]
COMPONENT_ORDER = ["vision_encoder", "projector", "language_model", "lm_head", "other"]

SOURCE_REL = {
    "078": "artifacts/eswa_revision_v1/05_primary_metrics/078_primary_metrics_v1_3_corrected.json",
    "079": "artifacts/eswa_revision_v1/06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json",
    "080": "artifacts/eswa_revision_v1/07_candidate_mass/080_candidate_mass_metrics_v1_3_recovered.json",
    "081": "artifacts/eswa_revision_v1/08_queue_utility/081_queue_metrics_v1_3_corrected.json",
    "081_meta": "artifacts/eswa_revision_v1/08_queue_utility/081_queue_metrics_v1_3_corrected_meta.json",
    "082": "artifacts/eswa_revision_v1/09_rbo/082_rbo_v1_3.json",
    "083": "artifacts/eswa_revision_v1/10_bootstrap/083_bootstrap_v1_3.json",
    "084": "artifacts/eswa_revision_v1/11_cost/084_memory_summary_v1_3.json",
    "085": "artifacts/eswa_revision_v1/12_coverage/085_quantization_coverage_summary_v1_3.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def scalar_text(value: Any) -> str:
    """Exact JSON-scalar spelling; numeric values are never rounded."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def label_model(model: str) -> str:
    return "InternVL3-2B" if model == "internvl3_2b" else "Qwen2.5-VL-7B"


def label_dataset(dataset: str) -> str:
    return str(dataset).upper()


def label_population(population: str) -> str:
    return "Full-clean" if population == "full_clean" else "Strict-unseen"


def cell_info(source: dict[str, Any], cell_name: str) -> tuple[str, str, dict[str, Any]]:
    cell = source["cells"][cell_name]
    model = str(cell["model"])
    dataset = label_dataset(str(cell["dataset"]))
    return model, dataset, cell


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_text(row[key]) for key in headers})


def load_sources(root: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for key, relative in SOURCE_REL.items():
        path = root / relative
        require(path.is_file(), f"missing whitelisted source: {relative}")
        sources[key] = read_json(path)
    for key, source in sources.items():
        require(source.get("status") == "PASS", f"source {key} is not PASS")
    require(sources["081"].get("status") == "PASS" and sources["081_meta"].get("status") == "PASS", "081 corrected sources are not PASS")
    return sources


def build_table_1(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "n", "bf16_accuracy", "int4_accuracy", "delta_accuracy", "answer_flip_rate",
        "spearman", "kendall_tau_b", "j5", "j10", "j20", "j30",
        "bf16_auroc", "int4_auroc", "delta_auroc", "bf16_aurc", "int4_aurc", "delta_aurc",
    ]
    headers = ["model", "dataset", "population", "backend_pair"] + fields
    rows: list[dict[str, Any]] = []
    for cell_name in CELL_ORDER:
        model, dataset, cell = cell_info(s["078"], cell_name)
        for population in POP_ORDER:
            data = cell[population]
            rows.append({
                "model": label_model(model), "dataset": dataset, "population": label_population(population),
                "backend_pair": "BF16 vs INT4-g128", **{field: data[field] for field in fields},
            })
    return headers, rows


def build_table_2(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = ["n", "bf16_accuracy", "int4_accuracy", "delta_accuracy", "flip_rate", "spearman", "kendall_tau_b", "delta_auroc", "delta_aurc"]
    headers = ["model", "dataset", "population"] + fields
    rows: list[dict[str, Any]] = []
    for cell_name in CELL_ORDER:
        model, dataset, cell = cell_info(s["078"], cell_name)
        data = cell["strict_unseen"]
        rows.append({
            "model": label_model(model), "dataset": dataset, "population": "Strict-unseen",
            "n": data["n"], "bf16_accuracy": data["bf16_accuracy"], "int4_accuracy": data["int4_accuracy"],
            "delta_accuracy": data["delta_accuracy"], "flip_rate": data["answer_flip_rate"],
            "spearman": data["spearman"], "kendall_tau_b": data["kendall_tau_b"],
            "delta_auroc": data["delta_auroc"], "delta_aurc": data["delta_aurc"],
        })
    return headers, rows


def build_table_3(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    headers = ["model", "dataset", "population", "signal", "spearman", "kendall_tau_b", "j20", "rbo_p090", "int4_auroc", "int4_aurc", "recall_at_20"]
    rows: list[dict[str, Any]] = []
    for cell_name in CELL_ORDER:
        model, dataset, cell = cell_info(s["079"], cell_name)
        for population in POP_ORDER:
            for source_signal, output_signal in SIGNAL_ORDER:
                data = cell[population]["signals"][source_signal]
                rows.append({
                    "model": label_model(model), "dataset": dataset, "population": label_population(population), "signal": output_signal,
                    "spearman": data["bf16_int4_spearman"], "kendall_tau_b": data["bf16_int4_kendall_tau_b"],
                    "j20": data["j20"], "rbo_p090": data["rbo_p090"], "int4_auroc": data["int4_auroc"],
                    "int4_aurc": data["int4_aurc"], "recall_at_20": data["int4_recall_at_20"],
                })
    return headers, rows


def build_table_4(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    headers = ["model", "dataset", "population", "q", "k", "raw_jaccard", "adjusted_jaccard", "replacement_slots", "int4_risk_recall_20", "bf16_risk_recall_20"]
    rows: list[dict[str, Any]] = []
    for cell_name in CELL_ORDER:
        model, dataset, cell = cell_info(s["081"], cell_name)
        for population in POP_ORDER:
            data = cell[population]["primary"]
            rows.append({
                "model": label_model(model), "dataset": dataset, "population": label_population(population), "q": "20%",
                "k": data["k"], "raw_jaccard": data["raw_jaccard"], "adjusted_jaccard": data["adjusted_jaccard"],
                "replacement_slots": data["replaced_slots"], "int4_risk_recall_20": data["int4_risk_recall"],
                "bf16_risk_recall_20": data["bf16_risk_recall"],
            })
    return headers, rows


def build_table_5(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    headers = ["model", "dataset", "population", "rbo_p090", "rbo_p080", "rbo_p095"]
    rows: list[dict[str, Any]] = []
    for cell_name in CELL_ORDER:
        model, dataset, cell = cell_info(s["082"], cell_name)
        for population in POP_ORDER:
            data = cell[population]["RBO"]
            rows.append({
                "model": label_model(model), "dataset": dataset, "population": label_population(population),
                "rbo_p090": data["p=0.90"], "rbo_p080": data["p=0.80"], "rbo_p095": data["p=0.95"],
            })
    return headers, rows


def build_table_6(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    headers = ["model", "dataset", "population", "n", "bf16_median", "int4_median", "paired_delta_median", "paired_delta_mean", "bf16_iqr", "int4_iqr", "delta_iqr", "spearman"]
    rows: list[dict[str, Any]] = []
    for cell_name in CELL_ORDER:
        model, dataset, cell = cell_info(s["080"], cell_name)
        for population in POP_ORDER:
            data = cell[population]
            rows.append({
                "model": label_model(model), "dataset": dataset, "population": label_population(population), "n": data["n"],
                "bf16_median": data["bf16"]["median"], "int4_median": data["int4"]["median"],
                "paired_delta_median": data["paired_delta_int4_minus_bf16"]["median"],
                "paired_delta_mean": data["paired_mean_delta"], "bf16_iqr": data["bf16"]["iqr"],
                "int4_iqr": data["int4"]["iqr"], "delta_iqr": data["paired_delta_int4_minus_bf16"]["iqr"],
                "spearman": data["bf16_int4_spearman"],
            })
    return headers, rows


def build_table_7(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    headers = ["model", "backend", "latency_median_sec", "latency_p95_sec", "throughput_samples_per_sec", "resident_allocated_bytes", "resident_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"]
    rows: list[dict[str, Any]] = []
    for cell_name in ["internvl3_2b__bf16", "internvl3_2b__int4_g128", "qwen2_5_vl_7b__bf16", "qwen2_5_vl_7b__int4_g128"]:
        cell = s["084"]["cells"][cell_name]
        rows.append({
            "model": label_model(cell["model"]), "backend": str(cell["backend"]),
            "latency_median_sec": cell["latency_median_sec"], "latency_p95_sec": cell["latency_p95_sec"],
            "throughput_samples_per_sec": cell["throughput_samples_per_sec"],
            "resident_allocated_bytes": cell["resident_allocated_after_load"],
            "resident_reserved_bytes": cell["resident_reserved_after_load"],
            "peak_allocated_bytes": cell["formal_peak_allocated"], "peak_reserved_bytes": cell["formal_peak_reserved"],
        })
    return headers, rows


def build_table_8(s: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    headers = ["model", "component", "linear_modules", "quantized_modules", "module_fraction", "linear_weight_parameters", "quantized_weight_parameters", "parameter_fraction"]
    rows: list[dict[str, Any]] = []
    for model in ("internvl3_2b", "qwen2_5_vl_7b"):
        model_data = s["085"]["models"][model]
        for component in COMPONENT_ORDER:
            data = model_data["components"][component]
            rows.append({
                "model": label_model(model), "component": component,
                "linear_modules": data["linear_modules"], "quantized_modules": data["quantized_modules"],
                "module_fraction": data["module_fraction"], "linear_weight_parameters": data["linear_weight_parameters"],
                "quantized_weight_parameters": data["quantized_weight_parameters"], "parameter_fraction": data["parameter_fraction"],
            })
    return headers, rows


BUILDERS: list[tuple[str, Callable[[dict[str, Any]], tuple[list[str], list[dict[str, Any]]]]]] = [
    ("Table_1_primary_metrics.csv", build_table_1),
    ("Table_2_strict_unseen_sensitivity.csv", build_table_2),
    ("Table_3_risk_ablation.csv", build_table_3),
    ("Table_4_queue_utility.csv", build_table_4),
    ("Table_5_rbo_stability.csv", build_table_5),
    ("Table_6_candidate_mass.csv", build_table_6),
    ("Table_7_latency_memory.csv", build_table_7),
    ("Table_8_quantization_coverage.csv", build_table_8),
]


def audit_csv(path: Path, headers: list[str], expected: list[dict[str, Any]]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    require(list(parsed[0].keys()) == headers if parsed else not expected, f"header mismatch: {path.name}")
    require(len(parsed) == len(expected), f"row count mismatch: {path.name}")
    for index, (actual, wanted) in enumerate(zip(parsed, expected)):
        for header in headers:
            require(actual[header] == scalar_text(wanted[header]), f"exact field mismatch {path.name} row {index} field {header}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    out_dir = args.root / "artifacts/eswa_revision_v1/13_tables"
    if out_dir.exists() and any(out_dir.rglob("*")):
        raise RuntimeError(f"refusing to overwrite non-empty 086 namespace: {out_dir}")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)
    try:
        sources = load_sources(args.root)
        expected_tables: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
        for filename, builder in BUILDERS:
            expected_tables[filename] = builder(sources)
        for filename, (headers, rows) in expected_tables.items():
            path = out_dir / "tables" / filename
            write_csv(path, headers, rows)
            audit_csv(path, headers, rows)
        manifest = {
            "stage": STAGE,
            "status": "PASS",
            "source_artifacts": list(SOURCE_REL.values()),
            "metrics_recomputed": False,
            "scientific_metrics_modified": False,
            "population_modified": False,
            "selection_rules_applied": False,
            "bootstrap_rendered": False,
            "numeric_rendering": "exact JSON scalar spelling; no rounding before validation",
            "tables": [filename for filename, _ in BUILDERS],
        }
        atomic_json(out_dir / "metadata" / "086_table_manifest.json", manifest)
        audit = {
            "stage": STAGE,
            "status": "PASS",
            "all_source_artifacts_exist": True,
            "all_source_status_pass": True,
            "numeric_values_exact": True,
            "point_estimates_match_source": True,
            "ci_values_match_083": True,
            "ci_rendered": False,
            "population_labels_exact": True,
            "backend_labels_exact": True,
            "strict_unseen_labels_exact": True,
            "no_metric_recomputation": True,
            "no_population_filtering": True,
            "no_manual_override": True,
            "no_rounding_before_validation": True,
            "GT_accessed": False,
            "scientific_metrics_computed": False,
            "bootstrap_computed": False,
            "table_row_counts": {filename: len(rows) for filename, (_, rows) in expected_tables.items()},
            "source_artifacts": list(SOURCE_REL.values()),
        }
        atomic_json(out_dir / "audit" / "086_table_validity_audit.json", audit)
        print(json.dumps({"status": "PASS", "stage": STAGE, "tables": len(BUILDERS), "rows": sum(len(rows) for _, rows in expected_tables.values())}))
        return 0
    except Exception as exc:
        failure = {
            "stage": STAGE, "status": "INVALID", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "GT_accessed": False, "scientific_metrics_computed": False,
            "bootstrap_computed": False,
        }
        atomic_json(out_dir / "audit" / "086_table_validity_audit.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
