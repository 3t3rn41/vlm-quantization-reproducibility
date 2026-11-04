# Script index

The scripts are grouped by role. They are source artifacts, not an unattended orchestration wrapper. Check each script's `--help`, the frozen protocol, and the applicable gate before running it.

| Stage | Scripts | Purpose |
|---|---|---|
| Provenance and manifests | `070_build_revision_manifests.py`, `071_audit_revision_provenance.py`, `071_build_smoke_manifest.py` | Prepare frozen full-clean manifests and audit provenance. |
| Smoke and full inference | `072_smoke_revision_inference.py`, `073_run_full_inference.py`, `073_run_qwen_full_inference.py` | Run gated smoke checks and the model-specific inference cells. |
| Prediction/pair gates | `074_prediction_gate.py`, `074_pair_integrity_audit.py`, `074_pair_integrity_audit_dataset.py` | Check prediction-only outputs and BF16/INT4 paired integrity. |
| Outcome join and metrics | `075_join_ground_truth.py`, `077_compute_noise_floor.py`, `078_compute_primary_metrics.py`, `079_compute_risk_signal_ablation_corrected.py`, `080_compute_candidate_mass_metrics.py`, `081_compute_queue_metrics.py`, `082_compute_rbo.py`, `083_compute_bootstrap.py` | Gated outcome analysis and primary/sensitivity metrics. `075` and downstream metrics must not be run until the protocol authorizes the outcome-linked stage. |
| Supporting analyses | `079_cross_artifact_consistency_audit.py`, `080_candidate_mass_recovery.py`, `083_bootstrap_occurrence_identity_audit.py`, `084_latency_memory.py`, `085_quantization_coverage.py`, `089_flip_queue_net_analysis.py` | Consistency, candidate-mass, bootstrap identity, engineering, coverage, and queue-yield audits. |
| Reporting | `086_render_tables.py` | Regenerate numeric tables from the required artifacts. |

The repository root is discovered relative to each packaged pipeline script. Scripts that need external inputs require explicit paths; the README deliberately avoids embedding host-specific locations.
