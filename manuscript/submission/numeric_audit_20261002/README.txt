ARCHIVED HISTORICAL NOTE — 2026-10-02

Superseded by the repository root README for current access, commands, seeds and validation:
https://github.com/3t3rn41/vlm-quantization-reproducibility/blob/main/README.md

The historical text below records the disclosure status when this package was prepared. Current public materials include the exact random streams, derived caches, inference code and completed validation receipts. Run CPU reconstruction from the repository root with reproduce_cpu.py and the documented modes. Current model and processor revisions are in protocol_current/model_revisions.json; the root README explains the historical InternVL metadata correction.

BEGIN HISTORICAL TEXT

VLM supplementary audit package version 2026-10-02

This is locally prepared submission material, not a published repository. No reuse license has been assigned. It supports verification from paired derived records, not independent rerunning of model inference.

Run the point-result audit with Python and NumPy:
    python reconstruct.py
Expected net captured INT4 errors at 20% are +16, +15, +5 and -13 in the stated cell order. The script verifies all three signals at four budgets, fixed-INT4-label AUROC/AURC differences and the equal-budget accounting identity. It also checks a clearly identified constructed example with identical overlap and positive/zero/negative yield.

To rerun post-hoc paired intervals:
    python reconstruct.py --bootstrap --seed AUTHOR_PROVIDED_SEED --replicates 10000
The placeholder is a command argument, not an embedded seed. Exact seeds remain available from the authors upon request. Numeric seed fields in the archived primary summary have been replaced by this disclosure convention; its metric values and intervals are preserved. Monte Carlo endpoints need not match exactly with a different seed. Original primary row intervals are preserved separately and have not been silently replaced by the new post-hoc row draws. Each new random stream is initialized with NumPy SeedSequence([base_seed, cell_index, unit_index]); unit index is 0 for row and 1 for image. Cell order is InternVL/MMStar, InternVL/AI2D, Qwen/MMStar, Qwen/AI2D. Draws and score sources share weights within every replicate.

Derived CSV fields
source_row is the canonical dataset row, not a claim of statistical independence. image_group groups byte-identical materialized images; no cryptographic hashes were calculated. BF16/INT4 error indicators are evaluation-derived correctness outcomes. Scores increase with uncertainty. Candidate probabilities retain cached numeric precision. Files contain no raw images, question/option text, gold answer key, model weights, private host names or credentials.

Resampling scope
Row resampling draws n paired question rows with replacement. Image resampling draws G groups with replacement and retains all questions in each sampled group. Repeated occurrences are counted with multiplicity, queues are recomputed at ceil(q*n_replicate), and question-weighted metrics remain the endpoint. The 95% percentile intervals are pointwise and unadjusted for multiplicity. Image groups being exchangeable is an assumption, not an audited fact of population sampling. Bit-identical groups do not detect every semantic duplicate.

Supplementary tables
S1: full-manifest RBO at persistence 0.80/0.90/0.95.
S2: primary/entropy/margin 20% queue utility, row and image paired intervals.
S3: sample-size ranges for 10,000 row and image resamples.
The archived tie file includes historical strict-unseen sensitivity as well as full-clean rows; the present manuscript uses only full-clean. These randomization quantiles concern tie-breaking variation, not sampling uncertainty. The full-clean rows comprise the complete fixed manifests; their name does not imply absence of prior author-side exposure.

Access and limitations
Acquire the original MMStar and AI2D splits and pinned model snapshots under their applicable terms. Their raw data are omitted. Source scripts and pinned revision information remain in the existing inference reproduction package. This supplement does not assign a public repository URL, satisfy author/funding/declaration fields, or certify journal submission policy compliance. The official Neurocomputing Guide for Authors was inaccessible (403) during this editorial check.
