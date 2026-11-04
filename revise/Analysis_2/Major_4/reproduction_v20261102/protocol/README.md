# Protocol record guide

These files are the frozen protocol and audit context for the packaged VLM experiment. Read the original records before launching a reproduction; this README is only an index.

- `analysis_plan.json`: populations, metric families, planned execution sequence, and stage gates.
- `frozen_config_v1_3.json`: active analysis choices and invariants after amendment 003.
- `protocol_amendment_*.json`: recorded changes and their activation rationale. Preserve the amendments with the protocol; do not treat the final config as if it had always been in force.
- `model_revisions.json` and `dataset_revisions.json`: declared source revisions needed to identify the inputs.
- `environment.json`: sanitized, read-only snapshot taken at the freeze gate. It is not the final run environment attestation.
- `frozen_config_candidate_mass_recovery_v1.json`: separate frozen configuration for the candidate-mass recovery extension.

Machine identifiers and host-specific paths are intentionally omitted from this public-preparation snapshot. Aggregate gate and execution records are under `../artifacts/eswa_revision_v1/02_full_inference/`.
