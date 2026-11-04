"""Compare the seven revised manuscript tables with released numeric evidence."""
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
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MODEL = {"InternVL": "internvl3_2b", "Qwen": "qwen2_5_vl_7b"}
NUMBER = re.compile(r"[+\-−]?\d+(?:\.\d+)?")


def table_rows(docx: Path, expected: int = 7) -> list[list[list[str]]]:
    with ZipFile(docx) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    tables = []
    for table in root.iter(W + "tbl"):
        rows = []
        for row in table.findall(W + "tr"):
            cells = []
            for cell in row.findall(W + "tc"):
                cells.append("".join(node.text or "" for node in cell.iter(W + "t")).strip())
            rows.append(cells)
        tables.append(rows)
    assert len(tables) == expected, len(tables)
    return tables


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numbers(value: str) -> list[str]:
    return NUMBER.findall(value.replace("−", "-"))


def check(display: str, actual: float, context: str) -> None:
    token = display.replace("−", "-").replace("+", "")
    places = len(token.partition(".")[2]) if "." in token else 0
    tolerance = 0.5 * 10 ** (-places) + 1e-7
    assert abs(float(token) - actual) <= tolerance, (context, display, actual)


def check_cell(display: str, actual: list[float], context: str) -> int:
    found = numbers(display)
    assert len(found) == len(actual), (context, display, actual)
    for index, (text, value) in enumerate(zip(found, actual)):
        check(text, value, f"{context}[{index}]")
    return len(found)


def cell_id(label: str) -> str:
    model = next(value for key, value in MODEL.items() if label.startswith(key))
    dataset = "mmstar" if "MMStar" in label else "ai2d"
    return f"{model}__{dataset}"


def indexed(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in read_csv(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, required=True)
    args = parser.parse_args()
    tables = table_rows(args.manuscript)
    endpoints = indexed(DATA / "complete_image_group_endpoints.csv", ("backend", "cell", "metric"))
    sources = read_csv(DATA / "risk_source_contrasts_complete.csv")
    matrices = indexed(DATA / "cross_matrices_complete.csv", ("cell", "signal", "q", "risk_source", "error_target"))
    backend = indexed(DATA / "backend_contrasts_complete.csv", ("cell", "signal", "q", "policy"))
    interactions = indexed(DATA / "interaction_results.csv", ("cell", "contrast", "unit"))
    historical_rows = read_csv(ROOT / "manuscript/submission/numeric_audit_20261002/manuscript_Table_4.csv")
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

    for row in tables[0][1:]:
        cell = cell_id(row[0])
        checked += check_cell(row[1], [float(endpoint("TorchAO", cell, "bf16_accuracy")["n"])], cell)
        for display, metric in zip(row[2:], ("bf16_accuracy", "int4_accuracy", "delta_accuracy", "flip", "spearman", "jaccard_0.20")):
            checked += check_cell(display, [float(endpoint("TorchAO", cell, metric)["point"])], cell + metric)

    for row in tables[1][1:]:
        cell = cell_id(row[0])
        cache = read_csv(ROOT / "manuscript/submission/numeric_audit_20261002" / f"{cell}_derived.csv")
        n = len(cache)
        k = math.ceil(0.2 * n)
        ordered = [set(sorted(range(n), key=lambda i: (-float(cache[i][prefix + "_risk_max"]), int(cache[i]["source_row"])))[:k]) for prefix in ("bf16", "int4")]
        removed, added = ordered[0] - ordered[1], ordered[1] - ordered[0]
        m = len(removed)
        exit_errors = sum(int(cache[i]["int4_error"]) for i in removed)
        entry_errors = sum(int(cache[i]["int4_error"]) for i in added)
        checked += check_cell(row[1], [float(endpoint("TorchAO", cell, "jaccard_0.20")["point"])], cell + " J20")
        assert row[2].replace(" ", "") == f"{m}/{k}{m/k:.4f}", (cell, "exchange")
        assert row[3].replace(" ", "") == f"{exit_errors}/{m}{exit_errors/m:.4f}", (cell, "exit")
        assert row[4].replace(" ", "") == f"{entry_errors}/{m}{entry_errors/m:.4f}", (cell, "entry")
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
            issues.append({"cell": cell, "table": 2, "field": "row interval", "manuscript": displayed_yield[1:3], "cached_reconstruction": row_ci})
        checked += check_cell(row[7], [entry_errors - exit_errors], cell + " net")

    for row in tables[2][1:]:
        cell = cell_id(row[0]);q = float(row[1]) / 100
        contrast = source(cell, "max", str(q), "TorchAO", "TorchAO", "BF16")
        matrix = matrices[(cell, "max", str(q), "BF16", "TorchAO")]
        checked += check_cell(row[2], [float(matrix["k"])], cell + " k")
        checked += check_cell(row[3], [float(contrast[k]) for k in ("delta_captured_errors_point", "delta_captured_errors_ci95_low", "delta_captured_errors_ci95_high")], cell + " net")
        checked += check_cell(row[4], [float(contrast[k]) for k in ("delta_yield_per100_point", "delta_yield_per100_ci95_low", "delta_yield_per100_ci95_high")], cell + " yield")

    for row in tables[3][1:]:
        cell = cell_id(row[0]);signal = row[1]
        contrast = source(cell, signal, "0.2", "TorchAO", "TorchAO", "BF16")
        checked += check_cell(row[2], [float(contrast["delta_captured_errors_point"])], cell + signal + " net")
        checked += check_cell(row[3], [float(contrast[k]) for k in ("delta_yield_per100_point", "delta_yield_per100_ci95_low", "delta_yield_per100_ci95_high")], cell + signal + " yield")

    for row in tables[4][1:]:
        cell = cell_id(row[0])
        for display, metric, scale in ((row[1], "delta_accuracy", 100), (row[2], "spearman", 1), (row[3], "jaccard_0.20", 1), (row[5], "yield_0.20", 1)):
            record = endpoint("HQQ", cell, metric)
            checked += check_cell(display, [scale * float(record[k]) for k in ("point", "ci95_low", "ci95_high")], cell + metric)
        checked += check_cell(row[4], [float(endpoint("HQQ", cell, "net_0.20")["point"])], cell + " HQQ net")

    for row in tables[5][1:]:
        cell = cell_id(row[0])
        for display, target in ((row[1], "TorchAO"), (row[2], "HQQ")):
            record = matrices[(cell, "max", "0.2", target, target)]
            checked += check_cell(display, [float(record[k]) for k in ("total_errors", "captured_errors_point", "unreviewed_errors_point")], cell + target)
        for display, policy in ((row[3], "native risk per backend"), (row[4], "BF16 risk")):
            record = backend[(cell, "max", "0.2", policy)]
            checked += check_cell(display, [float(record[k]) for k in ("delta_unreviewed_errors_point", "delta_unreviewed_errors_ci95_low", "delta_unreviewed_errors_ci95_high")], cell + policy)

    for row in tables[6][1:]:
        cell = cell_id(row[0])
        for display, contrast, adjusted in ((row[1], "policy_interaction", False), (row[2], "policy_interaction", True), (row[3], "target_interaction", False), (row[4], "target_interaction", True)):
            record = interactions[(cell, contrast, "per1000")]
            bounds = ("bonferroni_low", "bonferroni_high") if adjusted else ("ci95_low", "ci95_high")
            checked += check_cell(display, [float(record["point"]), *(float(record[k]) for k in bounds)], cell + contrast)

    result = {"status": "PASS" if not issues else "MISMATCH", "tables": 7, "data_rows": sum(len(table) - 1 for table in tables), "numeric_values_checked": checked, "issues": issues}
    print(json.dumps(result, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
