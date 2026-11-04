# Vision-language model quantization result reproduction

Versioned inference and numeric analysis code, pinned model and dataset revisions, random streams, derived risk scores, error indicators, and cached numeric outputs for the BF16, TorchAO INT4, and HQQ INT4 experiment. The archive covers 9,176 paired benchmark inputs across two models and two datasets.

## CPU reproduction

Use Python 3.12 and install `requirements_cpu.txt`. Run from the repository root:

```sh
python reproduce_cpu.py --mode verify
python reproduce_cpu.py --mode interactions
python reproduce_cpu.py --mode full
python reproduce_cpu.py --mode data
python reproduce_cpu.py --mode archived-tables
```

`verify` checks 240 endpoint values and reconstructs 32 interaction intervals. `interactions` rebuilds eight paired interactions. `full` regenerates original and HQQ intervals, the backend audit, paired interactions, and consolidated numeric data. `data` regenerates the consolidated endpoint and backend audit tables. These analyses use cached numeric predictions.

`archived-tables` uses nine released aggregate stage inputs to regenerate eight archived numeric tables in a temporary directory and checks the table audit. The source inputs are under `revise/Analysis_2/Major_4/reproduction_v20261102/artifacts/eswa_revision_v1/`.

## Contents

- `protocol_current/`: model and processor revisions.
- `revise/Analysis_2/Major_4/reproduction_v20261102/`: inference and analysis scripts, protocol records, random streams, and derived numeric data.
- `revise/Analysis_2/Major_2/`: HQQ inference code, configuration, derived scores, and errors.
- `revise/Analysis_2/Major_3/`: paired image-group intervals.
- `revise/CPU_backend_audit_20261103/`: cross-source and backend numeric audits.
- `revise/CPU_strengthening_20261103/`: interaction analysis, paired replicates, and numeric tables.
- `manuscript/submission/numeric_audit_20261002/`: original numeric caches.

Obtain original benchmark images, question text, external answer keys, and model weights from their providers under their terms. Inference scripts use these inputs and the recorded inference environment. The CPU commands regenerate numeric analyses from saved predictions and aggregate stage data.

## Validation commands

With original full prediction outputs and the external input workspace available, run:

```sh
python verify_formal_prediction_cache.py --original-inference-root PATH_TO_BF16_TORCHAO_OUTPUTS --hqq-root PATH_TO_HQQ_FORMAL_OUTPUTS
python verify_gpu_smoke.py --hqq-root PATH_TO_NEW_HQQ_OUTPUTS --queue-smoke PATH_TO_50_INPUT_SMOKE_OUTPUT
python verify_manuscript_tables.py --manuscript PATH_TO_REVISED_DOCX
python verify_supplement_tables.py --supplement PATH_TO_SUPPLEMENT_DOCX
```

`verify_formal_prediction_cache.py` checks every saved prediction against the released per-input cache. `verify_gpu_smoke.py` checks HQQ BF16 parity against `validation/bf16_parity_reference.json`. The manuscript and supplement verifiers compare displayed table values with released numeric evidence. The latest validation receipt is `validation/validation_20261104.json`.

To rerun the complete GPU forward pass and compare all predictions with the released per-input caches, use:

```sh
python run_full_gpu_validation.py --workspace PATH_TO_EVALUATION_WORKSPACE --models-root PATH_TO_MODEL_WEIGHTS --output-root PATH_TO_NEW_OUTPUT --python-original PATH_TO_ORIGINAL_ENV_PYTHON --python-hqq PATH_TO_HQQ_ENV_PYTHON --hqq-script revise/Analysis_2/Major_2/run_hqq_full.py --cache-root .
```

The completed ten-job run checked 27,528 new predictions with zero candidate-probability difference. Its receipt is `validation/full_forward_20261104.json`.

No reuse license has been assigned to this repository yet.
