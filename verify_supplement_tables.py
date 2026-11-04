"""Check numerical supplement tables against released analysis records."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from pathlib import Path

from verify_manuscript_tables import ROOT, DATA, table_rows, check_cell, cell_id, numbers


CELLS = ("internvl3_2b__mmstar", "internvl3_2b__ai2d", "qwen2_5_vl_7b__mmstar", "qwen2_5_vl_7b__ai2d")
MODEL_LABEL = {"InternVL": "internvl3_2b", "Qwen": "qwen2_5_vl_7b"}
BACKENDS = ("TorchAO", "HQQ")
ACCURACY = ("bf16_accuracy", "int4_accuracy", "delta_accuracy", "flip", "spearman", "kendall_tau_b")
RANKING = tuple(f"{source}_risk_{target}_errors_{metric}" for source, target, metric in (
    ("bf16", "bf16", "auroc"), ("bf16", "bf16", "aurc"),
    ("bf16", "int4", "auroc"), ("bf16", "int4", "aurc"),
    ("int4", "bf16", "auroc"), ("int4", "bf16", "aurc"),
    ("int4", "int4", "auroc"), ("int4", "int4", "aurc"),
)) + ("delta_bf16_errors_auroc", "delta_bf16_errors_aurc", "delta_int4_errors_auroc", "delta_int4_errors_aurc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, required=True)
    args = parser.parse_args()
    tables = table_rows(args.supplement, expected=29)
    endpoint = {(r["backend"], r["cell"], r["metric"]): r for r in read_csv(DATA / "complete_image_group_endpoints.csv")}
    matrix = {(r["cell"], r["signal"], r["q"], r["risk_source"], r["error_target"]): r for r in read_csv(DATA / "cross_matrices_complete.csv")}
    interaction = {(r["cell"], r["contrast"], r["unit"]): r for r in read_csv(DATA / "interaction_results.csv")}
    count = 0
    rows = 0
    for row in tables[0][1:]:
        cell = cell_id(row[0]);contrast = "policy_interaction" if row[1] == "Policy" else "target_interaction"
        record = interaction[(cell, contrast, "errors")]
        count += check_cell(row[2], [float(record[k]) for k in ("point", "ci95_low", "ci95_high")], cell + contrast)
        count += check_cell(row[3], [float(record[k]) for k in ("point", "bonferroni_low", "bonferroni_high")], cell + contrast + " adjusted")
        rows += 1
    for cell_index, cell in enumerate(CELLS):
        for offset, metrics in enumerate((ACCURACY, RANKING, tuple(f"{kind}_{q}" for q in ("0.05", "0.10", "0.20", "0.30") for kind in ("jaccard", "net", "yield")))):
            table = tables[1 + 3 * cell_index + offset]
            assert len(table) - 1 == len(metrics), (cell, offset)
            for row, metric in zip(table[1:], metrics):
                for column, backend in ((1, "TorchAO"), (2, "HQQ")):
                    record = endpoint[(backend, cell, metric)]
                    count += check_cell(row[column], [float(record[k]) for k in ("point", "ci95_low", "ci95_high")], cell + backend + metric)
                rows += 1
    for cell_index, cell in enumerate(CELLS):
        table = tables[13 + cell_index]
        assert len(table) == 4
        for row, source in zip(table[1:], ("BF16", "TorchAO", "HQQ")):
            for column, target in enumerate(("BF16", "TorchAO", "HQQ"), 1):
                record = matrix[(cell, "max", "0.2", source, target)]
                count += check_cell(row[column], [float(record[k]) for k in ("captured_errors_point", "captured_errors_ci95_low", "captured_errors_ci95_high")], cell + source + target)
            rows += 1
    assert len(tables[17]) == 6 and all(all(cell.strip() for cell in row) for row in tables[17][1:])
    rows += 5
    primary = json.loads((ROOT / "revise/Analysis_2/Major_4/reproduction_v20261102/artifacts/eswa_revision_v1/10_bootstrap/083_bootstrap_v1_3.json").read_text(encoding="utf-8"))["cells"]
    primary_keys = ("078.delta_accuracy", "079.risk_max.spearman", "079.risk_max.j20")
    for row in tables[18][1:]:
        cell = cell_id(row[0]);metrics = primary[cell.replace("__", "_")]["populations"]["full_clean"]["metrics"]
        for display, metric in zip(row[1:4], primary_keys):
            record = metrics[metric]
            count += check_cell(display, [float(record[k]) for k in ("point_estimate", "low_2.5", "high_97.5")], cell + metric)
        for display, metric in ((row[4], "kendall_tau_b"), (row[5], "flip")):
            record = endpoint[("TorchAO", cell, metric)]
            count += check_cell(display, [float(record[k]) for k in ("point", "ci95_low", "ci95_high")], cell + metric)
        rows += 1
    archival_interval_endpoints = 0
    for table_index, metric_suffix, archive_name in ((19, "auroc", "manuscript_Table_5.csv"), (20, "aurc", "manuscript_Table_6.csv")):
        archived = read_csv(ROOT / "manuscript/submission/numeric_audit_20261002" / archive_name)
        assert len(archived) == 4
        metric_names = tuple(f"{source}_risk_{target}_errors_{metric_suffix}" for source, target in (("bf16", "bf16"), ("bf16", "int4"), ("int4", "bf16"), ("int4", "int4")))
        for row, saved in zip(tables[table_index][1:], archived):
            cell = cell_id(row[0]);assert cell_id(saved["Cell"].replace("\n", "")) == cell
            for column, metric in enumerate(metric_names, 1):
                display = numbers(row[column]);reference = numbers(list(saved.values())[column])
                assert display == reference, (table_index, cell, metric)
                count += check_cell(display[0], [float(endpoint[("TorchAO", cell, metric)]["point"])], cell + metric)
                archival_interval_endpoints += 2
            rows += 1
    archived_delta = read_csv(ROOT / "manuscript/submission/numeric_audit_20261002/manuscript_Table_7.csv")
    assert len(archived_delta) == 4
    for row, saved in zip(tables[21][1:], archived_delta):
        cell = cell_id(row[0]);assert cell_id(saved["Cell"].replace("\n", "")) == cell
        for column, metric in ((1, "delta_int4_errors_auroc"), (2, "delta_int4_errors_aurc")):
            display = numbers(row[column]);reference = numbers(list(saved.values())[column])
            assert display == reference and len(display) == 5, (cell, metric)
            record = endpoint[("TorchAO", cell, metric)]
            count += check_cell(" ".join((display[0], display[3], display[4])), [float(record[k]) for k in ("point", "ci95_low", "ci95_high")], cell + metric)
            archival_interval_endpoints += 2
        count += check_cell(row[3], [float(endpoint[("TorchAO", cell, "net_0.20")]["point"])], cell + " net")
        rows += 1
    tie = {(r["cell"], r["backend"]): r for r in read_csv(ROOT / "manuscript/submission/numeric_audit_20261002/risk_ties.csv")}
    tie_cutoff = {(r["cell"], r["q"]): ast.literal_eval(r["tie_groups_at_cutoff"]) for r in read_csv(ROOT / "manuscript/submission/numeric_audit_20261002/archived_random_tie_sensitivity.csv") if r["population"] == "full_clean"}
    for row in tables[22][1:]:
        cell = cell_id(row[0])
        for column, backend in ((1, "bf16"), (2, "int4")):
            record = tie[(cell, backend)]
            expected = f'{int(record["unique_risks"]):,}{int(record["tied_rows"])} ({float(record["tied_percent"]):.1f}%)'
            assert row[column].replace(" ", "") == expected.replace(" ", ""), (cell, backend, row[column], expected)
            count += 3
        for column, q in enumerate(("0.05", "0.1", "0.2", "0.3"), 3):
            group = tie_cutoff[(cell, q)]
            expected = f'{group["bf16"]["group_size"]}/{group["int4_g128"]["group_size"]}'
            assert row[column] == expected, (cell, q)
            count += 2
        rows += 1
    cost = json.loads((ROOT / "revise/Analysis_2/Major_4/reproduction_v20261102/artifacts/eswa_revision_v1/11_cost/084_memory_summary_v1_3.json").read_text(encoding="utf-8"))["cells"]
    for row in tables[23][1:]:
        model = MODEL_LABEL[row[0]]
        backend = "bf16" if row[1] == "BF16" else "int4_g128"
        record = cost[f"{model}__{backend}"]
        values = [record["latency_median_sec"], record["latency_p95_sec"], record["throughput_samples_per_sec"], record["resident_allocated_after_load"] / 2**30, record["formal_peak_allocated"] / 2**30]
        for display, value in zip(row[2:], values):
            count += check_cell(display, [value], row[0] + backend)
        rows += 1
    coverage = json.loads((ROOT / "revise/Analysis_2/Major_4/reproduction_v20261102/artifacts/eswa_revision_v1/12_coverage/085_quantization_coverage_summary_v1_3.json").read_text(encoding="utf-8"))["models"]
    for row in tables[24][1:]:
        record = coverage[MODEL_LABEL[row[0]]]
        assert row[1].replace(" ", "") == f'{record["quantized_linear_modules"]}/{record["total_linear_modules"]}'
        count += check_cell(row[2], [record["quantized_module_fraction"]], row[0] + " module fraction")
        assert row[3].replace(" ", "") == f'{record["quantized_linear_weight_parameters"]:,}/{record["total_linear_weight_parameters"]:,}'
        count += check_cell(row[4], [record["quantized_linear_parameter_fraction"]], row[0] + " parameter fraction")
        count += 4;rows += 1
    for row in tables[25][1:]:
        cell = cell_id(row[0]);bf = matrix[(cell, "max", "0.2", "BF16", "TorchAO")];q = matrix[(cell, "max", "0.2", "TorchAO", "TorchAO")]
        count += check_cell(row[1], [float(q["total_errors"])], cell + " errors")
        for display, record in ((row[2], bf), (row[3], q)):
            count += check_cell(display, [float(record["captured_errors_point"]), 100 * float(record["error_recall_point"])], cell + " capture")
        count += check_cell(row[4], [100 * float(q["random_recall"])], cell + " random")
        count += check_cell(row[5], [100 * float(q["oracle_recall"])], cell + " oracle")
        rows += 1
    composition = {(r["cell"], r["backend"]): r for r in read_csv(DATA / "error_composition_complete.csv")}
    for row in tables[26][1:]:
        cell = cell_id(row[0]);record = composition[(cell, row[1])]
        for display, key in zip(row[2:], ("new_errors_vs_BF16_point", "corrected_errors_vs_BF16_point", "shared_errors_point", "shared_correct_point", "total_errors_point")):
            count += check_cell(display, [float(record[key])], cell + row[1] + key)
        rows += 1
    semantic = {r["model"]: r["semantic_categories"] for r in json.loads((ROOT / "revise/Analysis_1/Minor_1/runtime_component_summary.json").read_text(encoding="utf-8"))}
    category = {"Vision encoder": "vision_encoder", "Projector or merger": "projector_or_merger", "Language model": "language_model", "LM head": "lm_head"}
    for row in tables[27][1:]:
        values = semantic[MODEL_LABEL[row[0]]][category[row[1]]]
        reference_modules, quantized_modules, reference_parameters, quantized_parameters = values
        assert row[2].replace(" ", "") == f"{quantized_modules}/{reference_modules}"
        count += check_cell(row[3], [100 * quantized_modules / reference_modules], row[0] + row[1] + " modules")
        assert row[4].replace(" ", "") == f"{quantized_parameters:,}/{reference_parameters:,}"
        count += check_cell(row[5], [100 * quantized_parameters / reference_parameters], row[0] + row[1] + " parameters")
        count += 4;rows += 1
    internvl = json.loads((ROOT / "revise/Analysis_2/Minor_3/internvl_recovered_token_split.json").read_text(encoding="utf-8"))
    qwen = json.loads((ROOT / "revise/Analysis_2/Minor_3/qwen_token_summary.json").read_text(encoding="utf-8"))
    for row in tables[28][1:]:
        if row[0] == "InternVL":
            groups = ([r[k] for r in internvl] for k in ("prompt_token_count", "visual_tokens", "nonvisual_sequence_tokens"))
            values = [[min(g), statistics.median(g), max(g)] for g in groups]
        else:
            values = [[qwen[k][field] for field in ("min", "median", "max")] for k in ("prompt_token_count", "visual_token_count_from_pinned_resize", "nonvisual_sequence_tokens")]
        for display, triple in zip(row[1:], values):
            count += check_cell(display, triple, row[0] + " tokens")
        rows += 1
    print(json.dumps({"status": "PASS", "tables": 29, "data_rows": rows, "numeric_values_checked": count, "archived_interval_endpoints_preserved": archival_interval_endpoints, "qualitative_table": 18}, ensure_ascii=False))


if __name__ == "__main__":
    main()
