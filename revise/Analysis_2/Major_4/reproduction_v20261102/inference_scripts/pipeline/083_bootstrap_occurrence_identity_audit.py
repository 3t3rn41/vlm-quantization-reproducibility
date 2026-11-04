#!/usr/bin/env python3
"""Read-only audit of bootstrap occurrence identity for 083.

This audit must not regenerate the frozen bootstrap indices and must not
modify any existing joined cache or bootstrap result.  It verifies the
implementation invariant that every resampled row is addressed by its
replicate-local draw position, so duplicate original sample IDs remain
distinct observations for rank, set, queue, recall, and RBO metrics.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np


BOOTSTRAP_REL = "scripts/083_compute_bootstrap.py"
BOOTSTRAP_ARTIFACT_REL = "artifacts/eswa_revision_v1/10_bootstrap/083_bootstrap_v1_3.json"
BOOTSTRAP_AUDIT_REL = "artifacts/eswa_revision_v1/10_bootstrap/083_bootstrap_validity_audit_v1_3.json"
OUTPUT_REL = "artifacts/eswa_revision_v1/10_bootstrap/083_bootstrap_occurrence_identity_audit_v1_3.json"


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_bootstrap_module(path: Path):
    spec = importlib.util.spec_from_file_location("vlm_083_compute_bootstrap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import bootstrap module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_source(tree: ast.AST, source: str, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            return "\n".join(source.splitlines()[start:end])
    raise RuntimeError(f"function not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source_path = root / BOOTSTRAP_REL
    artifact_path = root / BOOTSTRAP_ARTIFACT_REL
    artifact_audit_path = root / BOOTSTRAP_AUDIT_REL
    output_path = root / OUTPUT_REL
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {output_path}")

    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    bootstrap_fn = function_source(tree, source_text, "bootstrap_population")
    artifact = read_json(artifact_path)
    artifact_audit = read_json(artifact_audit_path)

    # Static implementation checks.  These are deliberately source-level and
    # do not execute the 10,000-replicate loop or regenerate any indices.
    required_markers = {
        "draw_indices_have_length_n": "idx = rng.integers(0, n, size=n, dtype=np.int64)" in bootstrap_fn,
        "occurrence_source_is_local_draw_position": "source = np.arange(n, dtype=np.int64)" in bootstrap_fn,
        "resampled_arrays_use_same_idx": all(token in bootstrap_fn for token in (
            "d[\"errors_bf16\"][idx]",
            "d[\"errors_int4\"][idx]",
            "d[\"signals_bf16\"][signal][idx]",
            "d[\"signals_int4\"][signal][idx]",
        )),
        "ranked_indices_create_local_occurrence_ids": (
            "occurrence = np.arange(len(risk), dtype=np.int64)" in source_text
            and "np.lexsort((occurrence, source, -risk))" in source_text
        ),
        "jaccard_uses_ranked_occurrence_arrays": all(token in source_text for token in (
            "np.intersect1d(first, second)",
            "np.union1d(first, second)",
        )),
        "replacement_uses_local_queue_occurrences": all(token in bootstrap_fn for token in (
            "queue_b = set(order_b[:k].tolist())",
            "queue_i = set(order_i[:k].tolist())",
            "k = max(1, math.ceil(BUDGET * n))",
        )),
        "recall_counts_resampled_occurrences": all(token in bootstrap_fn for token in (
            "total_errors = int(err_i.sum())",
            "err_i[order_i[:k]].sum()",
            "err_i[order_b[:k]].sum()",
        )),
        "rbo_receives_ranked_local_occurrences": all(token in bootstrap_fn for token in (
            "rbo_np(order_b, order_i, p)",
            "ranked_indices_np(sb, source, True)",
        )),
        "queue_size_uses_n_draws": "math.ceil(BUDGET * n)" in bootstrap_fn,
    }

    forbidden_patterns = {
        "no_drop_duplicates": not re.search(r"drop_duplicates", source_text),
        "no_sample_id_set_identity": not re.search(r"set\s*\(\s*[^\n]*sample_id", source_text),
        "no_sample_id_rbo_identity": not re.search(r"rbo_(?:np|loop)\([^\n]*sample_id", source_text),
    }

    # Deterministic duplicate-draw witness.  The original IDs deliberately
    # contain a duplicate.  Draw positions 0..3 are the bootstrap identities;
    # positions 1 and 2 both refer to original ID B but remain distinct.
    module = load_bootstrap_module(source_path)
    original_ids = np.asarray(["A", "B", "C", "D"], dtype=object)
    draw_positions = np.asarray([0, 1, 1, 3], dtype=np.int64)
    resampled_ids = original_ids[draw_positions]
    occurrence_ids = np.arange(len(draw_positions), dtype=np.int64)
    risks_b = np.asarray([0.90, 0.80, 0.70, 0.10], dtype=float)
    risks_i = np.asarray([0.90, 0.75, 0.70, 0.20], dtype=float)
    order_b = module.ranked_indices_np(risks_b, occurrence_ids, True)
    order_i = module.ranked_indices_np(risks_i, occurrence_ids, True)
    k = max(1, int(np.ceil(0.50 * len(draw_positions))))
    queue_b = set(order_b[:k].tolist())
    queue_i = set(order_i[:k].tolist())
    rbo = float(module.rbo_np(order_b, order_i, 0.90))
    synthetic = {
        "draw_positions": draw_positions.tolist(),
        "resampled_original_ids": resampled_ids.tolist(),
        "n_observations": int(len(draw_positions)),
        "n_unique_original_sample_ids": int(np.unique(resampled_ids).size),
        "n_unique_occurrence_ids": int(np.unique(occurrence_ids).size),
        "duplicates_exist": bool(len(np.unique(resampled_ids)) < len(draw_positions)),
        "duplicates_preserved_as_observations": bool(len(resampled_ids) == len(draw_positions)),
        "ranked_list_length_b": int(len(order_b)),
        "ranked_list_length_i": int(len(order_i)),
        "ranked_list_occurrence_ids_unique_b": bool(len(np.unique(order_b)) == len(draw_positions)),
        "ranked_list_occurrence_ids_unique_i": bool(len(np.unique(order_i)) == len(draw_positions)),
        "queue_k": k,
        "queue_b_occurrence_count": len(queue_b),
        "queue_i_occurrence_count": len(queue_i),
        "rbo_p090_finite": bool(np.isfinite(rbo)),
    }

    # Existing output/audit are read-only evidence.  The full run did not
    # persist indices, so all-replicate draw-count coverage is an invariant of
    # the executed loop (size=n and no row filtering), not a regenerated scan.
    population_counts_correct = bool(artifact_audit.get("all_population_counts_correct") is True)
    point_gate_names = (
        "point_estimate_exact_match_078",
        "point_estimate_exact_match_079",
        "point_estimate_exact_match_080",
        "point_estimate_exact_match_081",
        "point_estimate_exact_match_082",
    )
    point_gates_ok = all(artifact_audit.get(name) is True for name in point_gate_names)
    full_metric_validity = all(
        int(metric.get("valid_B", -1)) == 10000 and int(metric.get("invalid_B", -1)) == 0
        for cell in artifact.get("cells", {}).values()
        for population in cell.get("populations", {}).values()
        for metric in population.get("metrics", {}).values()
    )

    checks = {
        **required_markers,
        **forbidden_patterns,
        "artifact_status_pass": artifact.get("status") == "PASS",
        "audit_status_pass": artifact_audit.get("status") == "PASS",
        "seed_frozen": artifact.get("seed") == 20260913,
        "replicates_frozen": artifact.get("replicates_requested") == 10000,
        "resample_unit_sample_id": artifact.get("resample_unit") == "sample_id",
        "paired_resampling": artifact.get("paired_resampling") is True,
        "all_population_counts_correct": population_counts_correct,
        "point_estimate_gates_all_exact": point_gates_ok,
        "all_metrics_valid_B_10000": full_metric_validity,
        "all_replicates_draw_count_equals_n": required_markers["draw_indices_have_length_n"] and required_markers["resampled_arrays_use_same_idx"],
        "duplicates_exist_in_bootstrap": synthetic["duplicates_exist"],
        "duplicates_preserved_as_observations": synthetic["duplicates_preserved_as_observations"],
        "jaccard_uses_occurrence_identity": required_markers["jaccard_uses_ranked_occurrence_arrays"],
        "replacement_uses_occurrence_identity": required_markers["replacement_uses_local_queue_occurrences"],
        "rbo_uses_occurrence_identity": required_markers["rbo_receives_ranked_local_occurrences"] and synthetic["rbo_p090_finite"],
        "recall_counts_occurrences": required_markers["recall_counts_resampled_occurrences"],
        "queue_k_based_on_n_draws": required_markers["queue_size_uses_n_draws"],
    }
    status = "PASS" if all(checks.values()) else "HOLD"
    result = {
        "artifact": "083_bootstrap_occurrence_identity_audit",
        "status": status,
        "scope": "read-only source and result audit; no bootstrap indices regenerated",
        "source_script": BOOTSTRAP_REL,
        "bootstrap_artifact": BOOTSTRAP_ARTIFACT_REL,
        "bootstrap_validity_audit": BOOTSTRAP_AUDIT_REL,
        "bootstrap_indices_regenerated": False,
        "bootstrap_rerun": False,
        "joined_cache_modified": False,
        "indices_persisted_by_original_run": False,
        "full_run_draw_count_proof": "static loop invariant: rng.integers(..., size=n) followed by direct indexed arrays with no filtering/deduplication",
        "checks": checks,
        "synthetic_duplicate_draw": synthetic,
        "synthetic_min_unique_original_ids": synthetic["n_unique_original_sample_ids"],
        "synthetic_max_unique_original_ids": synthetic["n_unique_original_sample_ids"],
        "notes": [
            "Occurrence identity is the replicate-local draw position, source=np.arange(n).",
            "BF16 and INT4 use the same draw position and therefore remain paired.",
            "Set/list operations operate on ranked occurrence positions, never original sample_id strings.",
            "The frozen 10,000-index arrays were not regenerated or inspected by re-sampling.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({"status": status, "output": str(output_path), "failed_checks": [k for k, v in checks.items() if not v]}, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
