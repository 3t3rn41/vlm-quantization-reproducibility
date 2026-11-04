# Reproduction status (2026-11-04)

## Verified from released numeric inputs

- `python reproduce_cpu.py --mode full`: original and HQQ image-group intervals, backend audit, paired interactions, 240 endpoint point checks and 32 interaction interval reconstructions pass.
- `python reproduce_cpu.py --mode data`: 240 consolidated endpoint rows and four numeric backend audit tables regenerate from released inputs.
- `python reproduce_cpu.py --mode archived-tables`: eight aggregate tables (74 data rows) regenerate from nine released stage inputs; the table audit passes.

## GPU validation

The experimental container now exposes one NVIDIA GeForce RTX 4090 D. Four BF16/TorchAO evaluation smoke paths completed 50 inputs each with zero parser failures, nonfinite logits, or network requests. The public HQQ script completed six inputs per model; 294 and 326 Linear modules were quantized, respectively. All 12 BF16 parity rows matched the archived candidate probabilities exactly.

`run_full_gpu_validation.py` completed a fresh full forward pass in 3.532 hours: eight BF16/TorchAO cells and two HQQ model jobs. `verify_formal_prediction_cache.py` then compared all 18,352 fresh BF16/TorchAO records and 9,176 fresh HQQ records with the released numeric caches. Sample order, predictions and four candidate probabilities matched for every record, with maximum probability difference zero. The ten-job receipt is `validation/full_forward_20261104.json`.

## Manuscript comparison

`verify_manuscript_tables.py --manuscript PATH_TO_REVISED_DOCX` checked seven main tables, 48 data rows and 368 displayed numeric values against released numeric evidence. Eight endpoints of the original row-resampling intervals in Table 2 are preserved from the historical table archive; the other checked values use the released per-input caches or regenerated aggregate analysis. The fresh full forward pass reproduces the per-input predictions used by these analyses.

`verify_supplement_tables.py --supplement PATH_TO_SUPPLEMENT_DOCX` checked all 29 supplementary tables, 193 data rows and 1,218 numeric values. It also matched 80 preserved historical interval endpoints to their archived displays. Supplementary Table 18 is qualitative and has five complete data rows.
