"""Compare the current or archived manuscript tables with released numeric evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "revise/CPU_strengthening_20261103"
ARCHIVE = ROOT / "manuscript/submission/numeric_audit_20261002"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MODEL = {"InternVL": "internvl3_2b", "Qwen": "qwen2_5_vl_7b"}
NUMBER = re.compile(r"[+\-−]?\d+(?:\.\d+)?")
ARCHIVED_SHAPES = ((4, 8), (4, 8), (16, 5), (12, 4), (4, 6), (4, 5), (4, 5))
CURRENT_SHAPES = (
    (5, 4), (4, 8), (4, 8), (16, 5), (12, 4), (4, 6),
    (4, 6), (4, 6), (4, 5), (8, 4), (4, 5),
)


def cell_text(cell: ET.Element) -> str:
    parts = []
    for paragraph in cell.iter(W + "p"):
        for node in paragraph.iter():
            if node.tag == W + "t":
                parts.append(node.text or "")
            elif node.tag in (W + "br", W + "cr", W + "tab"):
                parts.append("\n")
        parts.append("\n")
    return "".join(parts).strip()


def table_rows(docx: Path) -> list[list[list[str]]]:
    with ZipFile(docx) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    tables = []
    for table in root.iter(W + "tbl"):
        rows = []
        for row in table.findall(W + "tr"):
            rows.append([cell_text(cell) for cell in row.findall(W + "tc")])
        tables.append(rows)
    return tables


def check_table_shape(table: list[list[str]], data_rows: int, columns: int, context: str) -> None:
    if len(table) != data_rows + 1 or any(len(row) != columns for row in table):
        raise ValueError(f"{context}: expected {data_rows} data rows and {columns} columns")


def compact(value: str) -> str:
    return "".join(value.split())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numbers(value: str) -> list[str]:
    return NUMBER.findall(value.replace("−", "-"))


def check(display: str, actual: float, context: str) -> None:
    token = display.replace("−", "-").replace("+", "")
    places = len(token.partition(".")[2]) if "." in token else 0
    tolerance = 0.5 * 10 ** (-places) + 1e-7
    within_rounding = abs(float(token) - actual) <= tolerance
    if not within_rounding:
        raise AssertionError((context, display, actual))


def check_cell(display: str, actual: list[float], context: str) -> int:
    found = numbers(display)
    if len(found) != len(actual):
        raise AssertionError((context, display, actual))
    for index, (text, value) in enumerate(zip(found, actual)):
        check(text, value, f"{context}[{index}]")
    return len(found)


def cell_id(label: str) -> str:
    model = next(value for key, value in MODEL.items() if label.startswith(key))
    dataset = "mmstar" if "MMStar" in label else "ai2d"
    return f"{model}__{dataset}"


def indexed(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in read_csv(path)}


def interval(record: dict[str, str], scale: int = 1) -> list[float]:
    return [scale * float(record[key]) for key in ("point", "ci95_low", "ci95_high")]


def queue_exchange(path: Path) -> tuple[int, int, int, int]:
    cache = read_csv(path)
    k = math.ceil(0.2 * len(cache))

    def queue(prefix: str) -> set[int]:
        key = prefix + "_risk_max"
        ordered = sorted(range(len(cache)), key=lambda i: (-float(cache[i][key]), int(cache[i]["source_row"])))
        return set(ordered[:k])

    bf16, quantized = queue("bf16"), queue("int4")
    removed, added = bf16 - quantized, quantized - bf16
    entering = sum(int(cache[i]["int4_error"]) for i in added)
    exiting = sum(int(cache[i]["int4_error"]) for i in removed)
    return k, len(removed), entering, exiting


def check_recall_rows(table: list[list[str]], matrices: dict) -> int:
    checked = 0
    for row in table[1:]:
        cell = cell_id(row[0])
        bf16 = matrices[(cell, "max", "0.2", "BF16", "TorchAO")]
        torchao = matrices[(cell, "max", "0.2", "TorchAO", "TorchAO")]
        checked += check_cell(row[1], [float(torchao["total_errors"])], cell + " errors")
        for display, record in ((row[2], bf16), (row[3], torchao)):
            values = [float(record["captured_errors_point"]), 100 * float(record["error_recall_point"])]
            checked += check_cell(display, values, cell + " capture")
        checked += check_cell(row[4], [100 * float(torchao["random_recall"])], cell + " random")
        checked += check_cell(row[5], [100 * float(torchao["oracle_recall"])], cell + " oracle")
    return checked


def check_count_interactions(table: list[list[str]], interactions: dict) -> int:
    checked = 0
    for row in table[1:]:
        cell = cell_id(row[0])
        contrast = {"Policy": "policy_interaction", "Target": "target_interaction"}[row[1]]
        record = interactions[(cell, contrast, "errors")]
        checked += check_cell(row[2], interval(record), cell + contrast)
        adjusted = [float(record[key]) for key in ("point", "bonferroni_low", "bonferroni_high")]
        checked += check_cell(row[3], adjusted, cell + contrast + " adjusted")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, required=True)
    args = parser.parse_args()
    tables = table_rows(args.manuscript)
    current = len(tables) == len(CURRENT_SHAPES)
    shapes = CURRENT_SHAPES if current else ARCHIVED_SHAPES
    if len(tables) != len(shapes):
        raise ValueError(f"Unsupported manuscript layout: found {len(tables)} tables, expected 11 or 7")
    for index, (table, (rows, columns)) in enumerate(zip(tables, shapes), 1):
        check_table_shape(table, rows, columns, f"Main Table {index}")
    if current:
        study, accuracy, exchange, budgets, signals, recall, hqq, hqq_exchange, exposure, counts, rates = tables
        if any(not cell for row in study[1:] for cell in row):
            raise ValueError("Main Table 1: incomplete qualitative comparison")
    else:
        accuracy, exchange, budgets, signals, hqq, exposure, rates = tables

    endpoints = indexed(DATA / "complete_image_group_endpoints.csv", ("backend", "cell", "metric"))
    sources = read_csv(DATA / "risk_source_contrasts_complete.csv")
    matrices = indexed(DATA / "cross_matrices_complete.csv", ("cell", "signal", "q", "risk_source", "error_target"))
    backend = indexed(DATA / "backend_contrasts_complete.csv", ("cell", "signal", "q", "policy"))
    interactions = indexed(DATA / "interaction_results.csv", ("cell", "contrast", "unit"))
    historical_rows = read_csv(ARCHIVE / "manuscript_Table_4.csv")
    assert len(historical_rows) == 4
    historical_intervals = {cell_id(row["Cell"].replace("\n", "")): numbers(list(row.values())[6])[1:3] for row in historical_rows}
    checked = 0
    issues = []

    def endpoint(kind: str, cell: str, metric: str) -> dict[str, str]:
        return endpoints[(kind, cell, metric)]

    def source(cell: str, signal: str, q: str, target: str, a: str, b: str) -> dict[str, str]:
        found = [r for r in sources if r["cell"] == cell and r["signal"] == signal and r["q"] == q and r["error_target"] == target and r["source_a"] == a and r["source_b"] == b]
        assert len(found) == 1, (cell, signal, q, target, a, b)
        return found[0]

    for row in accuracy[1:]:
        cell = cell_id(row[0])
        checked += check_cell(row[1], [float(endpoint("TorchAO", cell, "bf16_accuracy")["n"])], cell)
        metrics = ("bf16_accuracy", "int4_accuracy", "delta_accuracy", "flip", "spearman", "jaccard_0.20")
        for display, metric in zip(row[2:], metrics):
            record = endpoint("TorchAO", cell, metric)
            values = [float(record["point"])]
            if current and metric in ("delta_accuracy", "spearman", "jaccard_0.20"):
                values = interval(record)
            checked += check_cell(display, values, cell + metric)

    for row in exchange[1:]:
        cell = cell_id(row[0])
        k, m, entry_errors, exit_errors = queue_exchange(ARCHIVE / f"{cell}_derived.csv")
        checked += check_cell(row[1], [float(endpoint("TorchAO", cell, "jaccard_0.20")["point"])], cell + " J20")
        if current:
            checked += check_cell(row[2], [m, k, m / k], cell + " exchange")
            checked += check_cell(row[3], [exit_errors, m, exit_errors / m], cell + " exit")
            checked += check_cell(row[4], [entry_errors, m, entry_errors / m], cell + " entry")
        else:
            assert compact(row[2]) == f"{m}/{k}{m/k:.4f}", (cell, "exchange")
            assert compact(row[3]) == f"{exit_errors}/{m}{exit_errors/m:.4f}", (cell, "exit")
            assert compact(row[4]) == f"{entry_errors}/{m}{entry_errors/m:.4f}", (cell, "entry")
        checked += check_cell(row[5], [entry_errors/m - exit_errors/m], cell + " Q")
        image = endpoint("TorchAO", cell, "yield_0.20")
        row_ci = historical_intervals[cell]
        displayed_yield = numbers(row[6])
        assert len(displayed_yield) == 5
        checked += check_cell(displayed_yield[0], [float(image["point"])], cell + " yield point")
        checked += check_cell(" ".join(displayed_yield[3:]), [float(image["ci95_low"]), float(image["ci95_high"])], cell + " image interval")
        try:
            checked += check_cell(" ".join(displayed_yield[1:3]), [float(v) for v in row_ci], cell + " archived row interval")
        except AssertionError:
            issues.append({"cell": cell, "table": 3 if current else 2, "field": "row interval", "manuscript": displayed_yield[1:3], "cached_reconstruction": row_ci})
        checked += check_cell(row[7], [entry_errors - exit_errors], cell + " net")

    for row in budgets[1:]:
        cell = cell_id(row[0])
        q = float(row[1]) / 100
        contrast = source(cell, "max", str(q), "TorchAO", "TorchAO", "BF16")
        matrix = matrices[(cell, "max", str(q), "BF16", "TorchAO")]
        checked += check_cell(row[2], [float(matrix["k"])], cell + " k")
        checked += check_cell(row[3], [float(contrast[k]) for k in ("delta_captured_errors_point", "delta_captured_errors_ci95_low", "delta_captured_errors_ci95_high")], cell + " net")
        checked += check_cell(row[4], [float(contrast[k]) for k in ("delta_yield_per100_point", "delta_yield_per100_ci95_low", "delta_yield_per100_ci95_high")], cell + " yield")

    for row in signals[1:]:
        cell = cell_id(row[0])
        signal = row[1]
        contrast = source(cell, signal, "0.2", "TorchAO", "TorchAO", "BF16")
        checked += check_cell(row[2], [float(contrast["delta_captured_errors_point"])], cell + signal + " net")
        checked += check_cell(row[3], [float(contrast[k]) for k in ("delta_yield_per100_point", "delta_yield_per100_ci95_low", "delta_yield_per100_ci95_high")], cell + signal + " yield")

    accuracy_scale = 1 if current else 100
    for row in hqq[1:]:
        cell = cell_id(row[0])
        for display, metric, scale in ((row[1], "delta_accuracy", accuracy_scale), (row[2], "spearman", 1), (row[3], "jaccard_0.20", 1), (row[5], "yield_0.20", 1)):
            record = endpoint("HQQ", cell, metric)
            checked += check_cell(display, interval(record, scale), cell + metric)
        checked += check_cell(row[4], [float(endpoint("HQQ", cell, "net_0.20")["point"])], cell + " HQQ net")

    if current:
        checked += check_recall_rows(recall, matrices)
        hqq_cache = ROOT / "revise/Analysis_2/Major_2/derived_audit"
        for row in hqq_exchange[1:]:
            cell = cell_id(row[0])
            k, m, entering, exiting = queue_exchange(hqq_cache / f"{cell}_derived.csv")
            expected = (k, m, entering, exiting, entering - exiting)
            for column, value in enumerate(expected, 1):
                checked += check_cell(row[column], [value], f"Main Table 8 {cell} column {column + 1}")
        checked += check_count_interactions(counts, interactions)

    for row in exposure[1:]:
        cell = cell_id(row[0])
        for display, target in ((row[1], "TorchAO"), (row[2], "HQQ")):
            record = matrices[(cell, "max", "0.2", target, target)]
            checked += check_cell(display, [float(record[k]) for k in ("total_errors", "captured_errors_point", "unreviewed_errors_point")], cell + target)
        for display, policy in ((row[3], "native risk per backend"), (row[4], "BF16 risk")):
            record = backend[(cell, "max", "0.2", policy)]
            checked += check_cell(display, [float(record[k]) for k in ("delta_unreviewed_errors_point", "delta_unreviewed_errors_ci95_low", "delta_unreviewed_errors_ci95_high")], cell + policy)

    for row in rates[1:]:
        cell = cell_id(row[0])
        for display, contrast, adjusted in ((row[1], "policy_interaction", False), (row[2], "policy_interaction", True), (row[3], "target_interaction", False), (row[4], "target_interaction", True)):
            record = interactions[(cell, contrast, "per1000")]
            bounds = ("bonferroni_low", "bonferroni_high") if adjusted else ("ci95_low", "ci95_high")
            checked += check_cell(display, [float(record["point"]), *(float(record[k]) for k in bounds)], cell + contrast)

    result = {"status": "PASS" if not issues else "MISMATCH", "tables": len(tables), "data_rows": sum(len(table) - 1 for table in tables), "numeric_values_checked": checked, "issues": issues}
    if current:
        result["qualitative_table"] = 1
    print(json.dumps(result, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
