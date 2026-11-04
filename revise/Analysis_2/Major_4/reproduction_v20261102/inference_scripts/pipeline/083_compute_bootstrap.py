#!/usr/bin/env python3
"""Compute the authorized paired bootstrap for the canonical VLM metrics.

The implementation generates one set of sample-ID resample indices per
cell/population and reuses it for every metric, both backends, and every risk
signal.  It is deliberately CPU-only and reads the canonical 075 joined cache
for outcome-dependent metrics plus the Amendment 004 recovery namespace for
candidate mass.  It never reruns inference or modifies a joined cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


SEED = 20260913
B = 10000
CI_PERCENTILES = (2.5, 97.5)
LETTERS = ("A", "B", "C", "D")
POPULATIONS = ("full_clean", "strict_unseen")
BUDGET = 0.20
SIGNALS = ("risk_max", "risk_entropy", "risk_margin")

CELLS = {
    "internvl3_2b_mmstar": {"model": "internvl3_2b", "dataset": "MMStar", "n": 1500, "joined": "04_gt_join/internvl3_2b/mmstar", "recovery": "08_candidate_mass_recovery/internvl3_2b/mmstar"},
    "internvl3_2b_ai2d": {"model": "internvl3_2b", "dataset": "AI2D", "n": 3088, "joined": "04_gt_join/internvl3_2b/ai2d", "recovery": "08_candidate_mass_recovery/internvl3_2b/ai2d"},
    "qwen2_5_vl_7b_mmstar": {"model": "qwen2_5_vl_7b", "dataset": "MMStar", "n": 1500, "joined": "04_gt_join/qwen2_5_vl_7b/mmstar", "recovery": "08_candidate_mass_recovery/qwen2_5_vl_7b/mmstar"},
    "qwen2_5_vl_7b_ai2d": {"model": "qwen2_5_vl_7b", "dataset": "AI2D", "n": 3088, "joined": "04_gt_join/qwen2_5_vl_7b/ai2d", "recovery": "08_candidate_mass_recovery/qwen2_5_vl_7b/ai2d"},
}


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        value = indexed[cursor][1]
        while end < len(indexed) and indexed[end][1] == value:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = rank
        cursor = end
    return ranks


def average_ranks_np(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    if len(values) == 0:
        return ranks
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts.tolist(), ends.tolist()):
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    mean_left = sum(left_rank) / len(left_rank)
    mean_right = sum(right_rank) / len(right_rank)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left_rank, right_rank))
    denom_left = sum((a - mean_left) ** 2 for a in left_rank)
    denom_right = sum((b - mean_right) ** 2 for b in right_rank)
    if denom_left == 0.0 or denom_right == 0.0:
        return 1.0 if left_rank == right_rank else 0.0
    return numerator / math.sqrt(denom_left * denom_right)


def spearman_np(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks_np(left)
    right_rank = average_ranks_np(right)
    mean_left = float(left_rank.mean())
    mean_right = float(right_rank.mean())
    numerator = float(np.sum((left_rank - mean_left) * (right_rank - mean_right)))
    denom_left = float(np.sum((left_rank - mean_left) ** 2))
    denom_right = float(np.sum((right_rank - mean_right) ** 2))
    if denom_left == 0.0 or denom_right == 0.0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return numerator / math.sqrt(denom_left * denom_right)


class Fenwick:
    def __init__(self, size: int) -> None:
        self.values = [0] * (size + 1)

    def add(self, index: int) -> None:
        while index < len(self.values):
            self.values[index] += 1
            index += index & -index

    def prefix(self, index: int) -> int:
        total = 0
        while index:
            total += self.values[index]
            index -= index & -index
        return total


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    if n < 2:
        return 1.0
    n0 = n * (n - 1) // 2
    tie_left = sum(count * (count - 1) // 2 for count in Counter(left).values())
    tie_right = sum(count * (count - 1) // 2 for count in Counter(right).values())
    tie_both = sum(count * (count - 1) // 2 for count in Counter(zip(left, right)).values())
    right_ranks = {value: index for index, value in enumerate(sorted(set(right)), 1)}
    ordered = sorted(zip(left, right), key=lambda pair: (pair[0], pair[1]))
    fenwick = Fenwick(len(right_ranks))
    inversions = 0
    processed = 0
    cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and ordered[end][0] == ordered[cursor][0]:
            end += 1
        for _, value in ordered[cursor:end]:
            rank = right_ranks[value]
            inversions += processed - fenwick.prefix(rank)
        for _, value in ordered[cursor:end]:
            fenwick.add(right_ranks[value])
            processed += 1
        cursor = end
    comparable = n0 - tie_left - tie_right + tie_both
    numerator = comparable - 2 * inversions
    denominator = math.sqrt((n0 - tie_left) * (n0 - tie_right))
    if denominator == 0.0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def ranked_indices(risk: Sequence[float], source: Sequence[int], descending: bool = True) -> List[int]:
    if descending:
        return sorted(range(len(risk)), key=lambda i: (-float(risk[i]), int(source[i]), i))
    return sorted(range(len(risk)), key=lambda i: (float(risk[i]), int(source[i]), i))


def ranked_indices_np(risk: np.ndarray, source: np.ndarray, descending: bool) -> np.ndarray:
    occurrence = np.arange(len(risk), dtype=np.int64)
    if descending:
        return np.lexsort((occurrence, source, -risk))
    return np.lexsort((occurrence, source, risk))


def jaccard_np(left: np.ndarray, right: np.ndarray, source: np.ndarray, q: float) -> float:
    k = max(1, math.ceil(q * len(source)))
    first = ranked_indices_np(left, source, True)[:k]
    second = ranked_indices_np(right, source, True)[:k]
    return float(np.intersect1d(first, second).size / np.union1d(first, second).size)


def rbo_loop(left: Sequence[int], right: Sequence[int], p: float) -> float:
    seen_left: set[int] = set()
    seen_right: set[int] = set()
    weighted = 0.0
    for depth, (a, b) in enumerate(zip(left, right), 1):
        seen_left.add(a)
        seen_right.add(b)
        weighted += (len(seen_left & seen_right) / depth) * (p ** (depth - 1))
    final_agreement = len(seen_left & seen_right) / len(left)
    return (1.0 - p) * weighted + (p ** len(left)) * final_agreement


def rbo_np(left: np.ndarray, right: np.ndarray, p: float) -> float:
    n = len(left)
    position_right = np.empty(n, dtype=np.int64)
    position_right[right] = np.arange(n, dtype=np.int64)
    starts = np.maximum(np.arange(n, dtype=np.int64), position_right[left])
    overlap = np.cumsum(np.bincount(starts, minlength=n))
    depths = np.arange(1, n + 1, dtype=float)
    weighted = float(np.sum((overlap / depths) * (p ** np.arange(n, dtype=float))))
    final_agreement = float(overlap[-1] / n)
    return (1.0 - p) * weighted + (p ** n) * final_agreement


def auroc(error: Sequence[int], scores: Sequence[float]) -> float | None:
    positive = sum(int(value) for value in error)
    negative = len(error) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, error) if label)
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def auroc_np(error: np.ndarray, scores: np.ndarray) -> float | None:
    positive = int(error.sum())
    negative = len(error) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = average_ranks_np(scores)
    rank_sum = float(ranks[error.astype(bool)].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def aurc_np(error: np.ndarray, risk: np.ndarray, source: np.ndarray) -> float:
    order = ranked_indices_np(risk, source, False)
    ordered_error = error[order].astype(float)
    return float(np.mean(np.cumsum(ordered_error) / np.arange(1, len(order) + 1, dtype=float)))


def signal_arrays(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, List[float]] = {name: [] for name in SIGNALS}
    for row in rows:
        risk_max = float(row["risk_primary"])
        probs = [float(row["candidate_probabilities"][letter]) for letter in LETTERS]
        if not all(math.isfinite(value) and value >= 0.0 for value in probs):
            raise RuntimeError("invalid candidate probabilities")
        total = sum(probs)
        if total <= 0.0:
            raise RuntimeError("candidate probability sum is not positive")
        normalized = [value / total for value in probs]
        ordered = sorted(normalized, reverse=True)
        entropy = -sum(value * math.log(value) for value in normalized if value > 0.0) / math.log(4.0)
        margin = 1.0 - (ordered[0] - ordered[1])
        arrays["risk_max"].append(risk_max)
        arrays["risk_entropy"].append(entropy)
        arrays["risk_margin"].append(margin)
    return {name: np.asarray(values, dtype=float) for name, values in arrays.items()}


def load_cell(root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    joined = root / "artifacts/eswa_revision_v1" / config["joined"]
    recovery = root / "artifacts/eswa_revision_v1" / config["recovery"]
    bf16 = read_jsonl(joined / "bf16/predictions_with_gt.jsonl")
    int4 = read_jsonl(joined / "int4_g128/predictions_with_gt.jsonl")
    rec_bf16 = read_jsonl(recovery / "bf16/predictions.jsonl")
    rec_int4 = read_jsonl(recovery / "int4_g128/predictions.jsonl")
    if not (len(bf16) == len(int4) == len(rec_bf16) == len(rec_int4) == config["n"]):
        raise RuntimeError(f"row count mismatch: {joined}")
    for left, right in ((bf16, int4), (rec_bf16, rec_int4)):
        if [row["sample_id"] for row in left] != [row["sample_id"] for row in right] or [row["source_row"] for row in left] != [row["source_row"] for row in right]:
            raise RuntimeError(f"pair alignment mismatch: {joined}")
    if [row["sample_id"] for row in bf16] != [row["sample_id"] for row in rec_bf16] or [row["source_row"] for row in bf16] != [row["source_row"] for row in rec_bf16]:
        raise RuntimeError(f"joined/recovery alignment mismatch: {joined}")
    audit = read_json(joined / "075_join_audit.json")
    if audit.get("status") != "PASS" or audit.get("metrics_computed") is not False:
        raise RuntimeError(f"075 audit is not a valid canonical input: {joined}")
    provenance = np.asarray([row.get("provenance_status") for row in bf16], dtype=object)
    source = np.asarray([int(row["source_row"]) for row in bf16], dtype=np.int64)
    pred_bf16 = np.asarray([str(row["normalized_prediction"]) for row in bf16], dtype=object)
    pred_int4 = np.asarray([str(row["normalized_prediction"]) for row in int4], dtype=object)
    gt = np.asarray([str(row["ground_truth"]) for row in bf16], dtype=object)
    errors_bf16 = (pred_bf16 != gt).astype(np.int8)
    errors_int4 = (pred_int4 != gt).astype(np.int8)
    mass_bf16 = np.asarray([float(row["candidate_mass"]) for row in rec_bf16], dtype=float)
    mass_int4 = np.asarray([float(row["candidate_mass"]) for row in rec_int4], dtype=float)
    return {
        "bf16_rows": bf16,
        "int4_rows": int4,
        "recovery_bf16": rec_bf16,
        "recovery_int4": rec_int4,
        "provenance": provenance,
        "source": source,
        "pred_bf16": pred_bf16,
        "pred_int4": pred_int4,
        "gt": gt,
        "errors_bf16": errors_bf16,
        "errors_int4": errors_int4,
        "mass_bf16": mass_bf16,
        "mass_int4": mass_int4,
        "signals_bf16": signal_arrays(bf16),
        "signals_int4": signal_arrays(int4),
    }


def selected(data: Dict[str, Any], population: str) -> Dict[str, Any]:
    if population == "full_clean":
        idx = np.arange(len(data["source"]), dtype=np.int64)
    else:
        idx = np.flatnonzero(data["provenance"] == "strict_unseen").astype(np.int64)
    if len(idx) == 0:
        raise RuntimeError(f"empty population: {population}")
    result = {
        key: (value[idx] if isinstance(value, np.ndarray) and len(value) == len(data["source"]) else value)
        for key, value in data.items()
    }
    result["signals_bf16"] = {signal: data["signals_bf16"][signal][idx] for signal in SIGNALS}
    result["signals_int4"] = {signal: data["signals_int4"][signal][idx] for signal in SIGNALS}
    result["idx"] = idx
    return result


def point_metrics(data: Dict[str, Any], population: str) -> Dict[str, Any]:
    d = selected(data, population)
    n = len(d["idx"])
    source = d["source"]
    rb = d["signals_bf16"]["risk_max"]
    ri = d["signals_int4"]["risk_max"]
    err_b = d["errors_bf16"]
    err_i = d["errors_int4"]
    point: Dict[str, Any] = {
        "n": n,
        "bf16_accuracy": float(np.mean(1 - err_b)),
        "int4_accuracy": float(np.mean(1 - err_i)),
        "delta_accuracy": float(np.mean(1 - err_i) - np.mean(1 - err_b)),
        "answer_flip_rate": float(np.mean(d["pred_bf16"] != d["pred_int4"])),
        "spearman": spearman(rb.tolist(), ri.tolist()),
        "kendall_tau_b": kendall_tau_b(rb.tolist(), ri.tolist()),
        "j5": jaccard_np(rb, ri, source, 0.05),
        "j10": jaccard_np(rb, ri, source, 0.10),
        "j20": jaccard_np(rb, ri, source, 0.20),
        "j30": jaccard_np(rb, ri, source, 0.30),
        "bf16_auroc": auroc(err_b.tolist(), rb.tolist()),
        "int4_auroc": auroc(err_i.tolist(), ri.tolist()),
        "bf16_aurc": aurc_np(err_b, rb, source),
        "int4_aurc": aurc_np(err_i, ri, source),
    }
    point["delta_auroc"] = None if point["bf16_auroc"] is None or point["int4_auroc"] is None else point["int4_auroc"] - point["bf16_auroc"]
    point["delta_aurc"] = point["int4_aurc"] - point["bf16_aurc"]
    signal_points: Dict[str, Any] = {}
    for signal in SIGNALS:
        left = d["signals_bf16"][signal]
        right = d["signals_int4"][signal]
        signal_points[signal] = {
            "bf16_int4_spearman": spearman(left.tolist(), right.tolist()),
            "bf16_int4_kendall_tau_b": kendall_tau_b(left.tolist(), right.tolist()),
            "j5": jaccard_np(left, right, source, 0.05),
            "j10": jaccard_np(left, right, source, 0.10),
            "j20": jaccard_np(left, right, source, 0.20),
            "j30": jaccard_np(left, right, source, 0.30),
            "rbo_p090": rbo_loop(ranked_indices(left.tolist(), source.tolist()), ranked_indices(right.tolist(), source.tolist()), 0.90),
            "int4_auroc": auroc(err_i.tolist(), right.tolist()),
            "int4_aurc": aurc_np(err_i, right, source),
            "int4_recall_at_20": None if int(err_i.sum()) == 0 else float(err_i[ranked_indices_np(right, source, True)[:max(1, math.ceil(0.2 * n))]].sum() / err_i.sum()),
        }
    queue_order_b = ranked_indices_np(rb, source, True)
    queue_order_i = ranked_indices_np(ri, source, True)
    k = max(1, math.ceil(0.2 * n))
    q_b = set(queue_order_b[:k].tolist())
    q_i = set(queue_order_i[:k].tolist())
    total_i = int(err_i.sum())
    point["queue_replacement_fraction_20"] = (k - len(q_b & q_i)) / k
    point["int4_risk_recall_20"] = None if total_i == 0 else float(err_i[queue_order_i[:k]].sum() / total_i)
    point["bf16_risk_recall_20"] = None if total_i == 0 else float(err_i[queue_order_b[:k]].sum() / total_i)
    point["candidate_mass_mean_delta"] = float(np.mean(d["mass_int4"] - d["mass_bf16"]))
    point["candidate_mass_median_delta"] = float(np.quantile(d["mass_int4"] - d["mass_bf16"], 0.50, method="linear"))
    point["rbo_p090"] = rbo_loop(queue_order_b.tolist(), queue_order_i.tolist(), 0.90)
    point["rbo_p080"] = rbo_loop(queue_order_b.tolist(), queue_order_i.tolist(), 0.80)
    point["rbo_p095"] = rbo_loop(queue_order_b.tolist(), queue_order_i.tolist(), 0.95)
    point["risk_signal_points"] = signal_points
    return point


def metric_record(point: float | None, values: List[float], invalid: int) -> Dict[str, Any]:
    if not values:
        return {"point_estimate": point, "low_2.5": None, "high_97.5": None, "valid_B": 0, "invalid_B": invalid}
    interval = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975], method="linear")
    return {"point_estimate": point, "low_2.5": float(interval[0]), "high_97.5": float(interval[1]), "valid_B": len(values), "invalid_B": invalid}


def add_metric(store: Dict[str, List[float]], invalids: Dict[str, int], name: str, value: float | None) -> None:
    if value is None or not math.isfinite(float(value)):
        invalids[name] = invalids.get(name, 0) + 1
    else:
        store.setdefault(name, []).append(float(value))


def bootstrap_population(data: Dict[str, Any], population: str, point: Dict[str, Any]) -> Dict[str, Any]:
    d = selected(data, population)
    n = len(d["idx"])
    source_base = d["source"]
    rng = np.random.default_rng(SEED)
    series: Dict[str, List[float]] = {}
    invalids: Dict[str, int] = {}
    for replicate in range(B):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        source = np.arange(n, dtype=np.int64)
        err_b = d["errors_bf16"][idx]
        err_i = d["errors_int4"][idx]
        pred_b = d["pred_bf16"][idx]
        pred_i = d["pred_int4"][idx]
        risk_b = {signal: d["signals_bf16"][signal][idx] for signal in SIGNALS}
        risk_i = {signal: d["signals_int4"][signal][idx] for signal in SIGNALS}
        rb = risk_b["risk_max"]
        ri = risk_i["risk_max"]
        add_metric(series, invalids, "078.delta_accuracy", float(np.mean(1 - err_i) - np.mean(1 - err_b)))
        add_metric(series, invalids, "078.answer_flip_rate", float(np.mean(pred_b != pred_i)))
        add_metric(series, invalids, "078.spearman", spearman_np(rb, ri))
        add_metric(series, invalids, "078.kendall_tau_b", kendall_tau_b(rb.tolist(), ri.tolist()))
        for q, label in ((0.05, "j5"), (0.10, "j10"), (0.20, "j20"), (0.30, "j30")):
            add_metric(series, invalids, f"078.{label}", jaccard_np(rb, ri, source, q))
        j20 = series["078.j20"][-1]
        chance = BUDGET / (2.0 - BUDGET)
        add_metric(series, invalids, "078.adjusted_j20", (j20 - chance) / (1.0 - chance))
        auc_b = auroc_np(err_b, rb)
        auc_i = auroc_np(err_i, ri)
        add_metric(series, invalids, "078.delta_auroc", None if auc_b is None or auc_i is None else auc_i - auc_b)
        add_metric(series, invalids, "078.delta_aurc", aurc_np(err_i, ri, source) - aurc_np(err_b, rb, source))

        order_b = ranked_indices_np(rb, source, True)
        order_i = ranked_indices_np(ri, source, True)
        k = max(1, math.ceil(BUDGET * n))
        queue_b = set(order_b[:k].tolist())
        queue_i = set(order_i[:k].tolist())
        add_metric(series, invalids, "081.replacement_fraction_20", (k - len(queue_b & queue_i)) / k)
        total_errors = int(err_i.sum())
        add_metric(series, invalids, "081.int4_risk_recall_20", None if total_errors == 0 else float(err_i[order_i[:k]].sum() / total_errors))
        add_metric(series, invalids, "081.bf16_risk_recall_20", None if total_errors == 0 else float(err_i[order_b[:k]].sum() / total_errors))

        delta_mass = d["mass_int4"][idx] - d["mass_bf16"][idx]
        add_metric(series, invalids, "080.paired_mean_delta", float(np.mean(delta_mass)))
        add_metric(series, invalids, "080.paired_median_delta", float(np.quantile(delta_mass, 0.50, method="linear")))
        for p, label in ((0.90, "p090"), (0.80, "p080"), (0.95, "p095")):
            add_metric(series, invalids, f"082.rbo_{label}", rbo_np(order_b, order_i, p))

        for signal in SIGNALS:
            sb = risk_b[signal]
            si = risk_i[signal]
            prefix = f"079.{signal}"
            add_metric(series, invalids, f"{prefix}.spearman", spearman_np(sb, si))
            if signal == "risk_max":
                add_metric(series, invalids, f"{prefix}.kendall_tau_b", series["078.kendall_tau_b"][-1])
            for q, label in ((0.05, "j5"), (0.10, "j10"), (0.20, "j20"), (0.30, "j30")):
                add_metric(series, invalids, f"{prefix}.{label}", jaccard_np(sb, si, source, q))
            add_metric(series, invalids, f"{prefix}.rbo_p090", rbo_np(ranked_indices_np(sb, source, True), ranked_indices_np(si, source, True), 0.90))
            auc_signal = auroc_np(err_i, si)
            add_metric(series, invalids, f"{prefix}.int4_auroc", auc_signal)
            add_metric(series, invalids, f"{prefix}.int4_aurc", aurc_np(err_i, si, source))
            order_signal = ranked_indices_np(si, source, True)
            add_metric(series, invalids, f"{prefix}.int4_recall_at_20", None if total_errors == 0 else float(err_i[order_signal[:k]].sum() / total_errors))

    # Store point estimates under the same metric names as the bootstrap series.
    point_map = {
        "078.delta_accuracy": point["delta_accuracy"],
        "078.answer_flip_rate": point["answer_flip_rate"],
        "078.spearman": point["spearman"],
        "078.kendall_tau_b": point["kendall_tau_b"],
        "078.j5": point["j5"], "078.j10": point["j10"], "078.j20": point["j20"], "078.j30": point["j30"],
        "078.adjusted_j20": (point["j20"] - BUDGET / (2.0 - BUDGET)) / (1.0 - BUDGET / (2.0 - BUDGET)),
        "078.delta_auroc": point["delta_auroc"], "078.delta_aurc": point["delta_aurc"],
        "081.replacement_fraction_20": point["queue_replacement_fraction_20"],
        "081.int4_risk_recall_20": point["int4_risk_recall_20"], "081.bf16_risk_recall_20": point["bf16_risk_recall_20"],
        "080.paired_mean_delta": point["candidate_mass_mean_delta"], "080.paired_median_delta": point["candidate_mass_median_delta"],
        "082.rbo_p090": point["rbo_p090"], "082.rbo_p080": point["rbo_p080"], "082.rbo_p095": point["rbo_p095"],
    }
    for signal in SIGNALS:
        p = point["risk_signal_points"][signal]
        prefix = f"079.{signal}"
        point_map.update({
            f"{prefix}.spearman": p["bf16_int4_spearman"],
            f"{prefix}.j5": p["j5"], f"{prefix}.j10": p["j10"], f"{prefix}.j20": p["j20"], f"{prefix}.j30": p["j30"],
            f"{prefix}.rbo_p090": p["rbo_p090"], f"{prefix}.int4_auroc": p["int4_auroc"],
            f"{prefix}.int4_aurc": p["int4_aurc"], f"{prefix}.int4_recall_at_20": p["int4_recall_at_20"],
        })
        if signal == "risk_max":
            point_map[f"{prefix}.kendall_tau_b"] = p["bf16_int4_kendall_tau_b"]
    result: Dict[str, Any] = {}
    for name in sorted(point_map):
        result[name] = metric_record(point_map[name], series.get(name, []), invalids.get(name, 0))
    return {
        "n": n,
        "seed": SEED,
        "replicates_requested": B,
        "resample_unit": "sample_id",
        "paired_resampling": True,
        "rng": "numpy.default_rng / PCG64",
        "metrics": result,
        "invalid_replicates": {name: count for name, count in sorted(invalids.items()) if count},
    }


def compare_exact(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return float(actual) == float(expected)


def point_estimate_gate(root: Path, cell_name: str, population: str, point: Dict[str, Any], refs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    c078 = refs["078"]["cells"][cell_name][population]
    c079 = refs["079"]["cells"][cell_name][population]["signals"]
    c080 = refs["080"]["cells"][cell_name][population]
    c081 = refs["081"]["cells"][cell_name][population]
    c082 = refs["082"]["cells"][cell_name][population]
    comparisons: Dict[str, bool] = {}
    for key in ("delta_accuracy", "answer_flip_rate", "spearman", "kendall_tau_b", "j5", "j10", "j20", "j30", "delta_auroc", "delta_aurc"):
        comparisons[f"078.{key}"] = compare_exact(point[key], c078[key])
    comparisons["080.paired_mean_delta"] = compare_exact(point["candidate_mass_mean_delta"], c080["paired_mean_delta"])
    comparisons["080.paired_median_delta"] = compare_exact(point["candidate_mass_median_delta"], c080["paired_median_delta"])
    qprimary = c081["primary"]
    comparisons["081.replacement_fraction_20"] = compare_exact(point["queue_replacement_fraction_20"], qprimary["replacement_fraction"])
    comparisons["081.int4_risk_recall_20"] = compare_exact(point["int4_risk_recall_20"], qprimary["int4_risk_recall"])
    comparisons["081.bf16_risk_recall_20"] = compare_exact(point["bf16_risk_recall_20"], qprimary["bf16_risk_recall"])
    comparisons["082.rbo_p090"] = compare_exact(point["rbo_p090"], c082["RBO"]["p=0.90"])
    comparisons["082.rbo_p080"] = compare_exact(point["rbo_p080"], c082["RBO"]["p=0.80"])
    comparisons["082.rbo_p095"] = compare_exact(point["rbo_p095"], c082["RBO"]["p=0.95"])
    for signal in SIGNALS:
        p = point["risk_signal_points"][signal]
        c = c079[signal]
        for metric, key in (("bf16_int4_spearman", "bf16_int4_spearman"), ("bf16_int4_kendall_tau_b", "bf16_int4_kendall_tau_b"), ("j20", "j20"), ("rbo_p090", "rbo_p090"), ("int4_auroc", "int4_auroc"), ("int4_aurc", "int4_aurc"), ("int4_recall_at_20", "int4_recall_at_20")):
            comparisons[f"079.{signal}.{metric}"] = compare_exact(p[metric], c[key])
    return {"all_exact": all(comparisons.values()), "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts/eswa_revision_v1"
    refs = {
        "078": read_json(base / "05_primary_metrics/078_primary_metrics_v1_3_corrected.json"),
        "079": read_json(base / "06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json"),
        "080": read_json(base / "07_candidate_mass/080_candidate_mass_metrics_v1_3_recovered.json"),
        "081": read_json(base / "08_queue_utility/081_queue_metrics_v1_3_corrected.json"),
        "082": read_json(base / "09_rbo/082_rbo_v1_3.json"),
    }
    if any(value.get("status") != "PASS" for value in refs.values()):
        raise RuntimeError("one or more canonical references is not PASS")

    all_cells: Dict[str, Any] = {}
    point_gates: Dict[str, Any] = {}
    all_point_exact = True
    for cell_name, config in CELLS.items():
        data = load_cell(root, config)
        all_cells[cell_name] = {"model": config["model"], "dataset": config["dataset"], "populations": {}}
        point_gates[cell_name] = {}
        for population in POPULATIONS:
            print(f"bootstrap_start {cell_name} {population}", flush=True)
            point = point_metrics(data, population)
            gate = point_estimate_gate(root, cell_name, population, point, refs)
            point_gates[cell_name][population] = gate
            all_point_exact = all_point_exact and gate["all_exact"]
            if not gate["all_exact"]:
                raise RuntimeError(f"point-estimate cross-artifact gate failed: {cell_name} {population}: {gate}")
            all_cells[cell_name]["populations"][population] = bootstrap_population(data, population, point)
            print(f"bootstrap_done {cell_name} {population}", flush=True)

    audit = {
        "stage": "083_BOOTSTRAP_VALIDITY_AUDIT_V1_3",
        "status": "PASS" if all_point_exact else "HOLD",
        "protocol_version": "v1.3",
        "seed": SEED,
        "replicates_requested": B,
        "resample_unit": "sample_id",
        "paired_resampling": True,
        "same_resample_indices_across_metrics": True,
        "same_indices_BF16_INT4": True,
        "percentile_CI": [2.5, 97.5],
        "all_population_counts_correct": all(value["populations"][population]["n"] == ({"full_clean": CELLS[cell]["n"], "strict_unseen": 1200 if CELLS[cell]["dataset"] == "MMStar" else 2788}[population]) for cell, value in all_cells.items() for population in POPULATIONS),
        "point_estimate_exact_match_078": all(any(key.startswith("078.") and not equal for key, equal in gate["comparisons"].items()) is False for value in point_gates.values() for gate in value.values()),
        "point_estimate_exact_match_079": all(all(equal for key, equal in gate["comparisons"].items() if key.startswith("079.")) for value in point_gates.values() for gate in value.values()),
        "point_estimate_exact_match_080": all(all(equal for key, equal in gate["comparisons"].items() if key.startswith("080.")) for value in point_gates.values() for gate in value.values()),
        "point_estimate_exact_match_081": all(all(equal for key, equal in gate["comparisons"].items() if key.startswith("081.")) for value in point_gates.values() for gate in value.values()),
        "point_estimate_exact_match_082": all(all(equal for key, equal in gate["comparisons"].items() if key.startswith("082.")) for value in point_gates.values() for gate in value.values()),
        "point_estimate_gates": point_gates,
        "GT_source": "canonical 075 joined cache where required",
        "candidate_mass_source": "Amendment 004 08_candidate_mass_recovery",
        "no_inference_rerun": True,
        "no_joined_cache_modification": True,
        "GT_accessed": True,
        "scientific_metrics_computed": True,
        "bootstrap_computed": True,
    }
    audit["status"] = "PASS" if audit["all_population_counts_correct"] and all_point_exact and all(audit[key] for key in ("point_estimate_exact_match_078", "point_estimate_exact_match_079", "point_estimate_exact_match_080", "point_estimate_exact_match_081", "point_estimate_exact_match_082")) else "HOLD"
    artifact = {
        "stage": "083_PAIRED_BOOTSTRAP_V1_3",
        "status": audit["status"],
        "protocol_version": "v1.3",
        "seed": SEED,
        "replicates_requested": B,
        "resample_unit": "sample_id",
        "paired_resampling": True,
        "percentile_CI": [2.5, 97.5],
        "metric_scope": {
            "078_primary": "delta_accuracy, answer_flip_rate, Spearman, Kendall tau-b, J5/J10/J20/J30, adjusted J20, delta AUROC, delta AURC",
            "079_risk_ablation": "all three signals: Spearman, J5/J10/J20/J30, RBO p=.90, INT4 AUROC/AURC/Recall@20; Kendall tau-b for risk_max",
            "080_candidate_mass": "paired mean delta and paired median delta",
            "081_queue": "replacement_fraction@20, INT4-risk Recall@20, BF16-risk Recall@20",
            "082_rbo": "p=.90 primary and p=.80/.95 sensitivity",
        },
        "GT_accessed": True,
        "candidate_mass_source": "Amendment 004 08_candidate_mass_recovery",
        "bootstrap_computed": True,
        "cells": all_cells,
        "validity_audit": str(base / "10_bootstrap/083_bootstrap_validity_audit_v1_3.json"),
    }
    output_dir = base / "10_bootstrap"
    write_json(output_dir / "083_bootstrap_v1_3.json", artifact)
    write_json(output_dir / "083_bootstrap_validity_audit_v1_3.json", audit)
    print(json.dumps({"status": audit["status"], "metrics_artifact": str(output_dir / "083_bootstrap_v1_3.json"), "audit_artifact": str(output_dir / "083_bootstrap_validity_audit_v1_3.json"), "all_population_counts_correct": audit["all_population_counts_correct"], "all_point_estimate_gates": all_point_exact}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
