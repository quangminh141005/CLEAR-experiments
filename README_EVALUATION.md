# CLEAR evaluation on a remote SSH server

This adds a zero-shot research pipeline alongside the original code. Existing
`src/clear`, legacy scripts, results and the Indiana Word report are preserved.
No model is fitted or fine-tuned. The default is **full CLEAR**, not the direct
image/text baseline. No experiments, dependency installation, dataset downloads,
model downloads or synthetic tests were executed while implementing this code.

## Repository inspection and implementation choices

The reusable model loader is `src/clear/hub.py`: checkpoint inspection constructs
the paired text transformer and DINOv2 ViT-B/14 **with registers**, loads weights
strictly, and returns the official 448×448 bicubic RGB transform. Normalization
uses mean 101.48761 and std 83.43944 on 0–255 pixels (divided by 255 after
`ToTensor`). The loader already provides ordinary attention for PyTorch 1.12.

`src/clear/concepts.py` supports the released tensor and indexed SFR pickle.
`src/clear/pipeline.py::ZeroShotPipeline` implements CPU-resident concept banks,
chunked device transfer, normalized image/observation cosine scores and the full
projection. `examples/concepts/exp_zeroshot.py` documents the original experiment;
it contains machine-specific paths and random fallback embeddings, so the new
adapter reuses the library and **fails on missing/misaligned embeddings** instead.
The new observation reader explicitly selects `concept` and checks `concept_idx`
rather than assuming the first CSV column contains text. It never sorts bank rows.

Legacy `scripts/test_indiana.py` evaluates the backbone on a small selected cohort.
The new entry point `evaluate.py` is separate and never imports its downloading
code. Existing metric helpers assume dense labels or a different sampling scheme;
`evaluation/metrics.py` adds explicit masks and patient-cluster bootstrap rather
than changing old experiment behavior. All new modules have no import-time
network or model loading.

The paper's bank was curated by extracting radiological observations from
227,835 reports with an LLM (Ministral-8B), followed by deduplication and lowercase
normalization, yielding 368,294 observations. CLEAR's paired text tower and SFR
encode that same ordered vocabulary in different spaces. This pipeline reuses
the released bank; it does not regenerate concepts from evaluation reports,
retrain encoders, or replace missing concepts. See the
[original methods and Fig. 1](https://www.nature.com/articles/s41551-026-01741-4).

## 1. Set paths and create isolated environments

Run all following commands **yourself on the SSH server**, from this repository:

```bash
cd /your/path/to/CLEAR-experiments
export CLEAR_REPO="$PWD"
export SERVER_DATA=/your/storage/clear-data
export SERVER_ARCHIVES=/your/storage/clear-archives
export SERVER_ASSETS=/your/storage/clear-assets
export SERVER_RESULTS="$CLEAR_REPO/results/reproduction"
```

Install Python 3.10 with `venv`, git, curl and unzip manually if unavailable.
`setup.sh` never changes system packages, CUDA, the NVIDIA driver or the active
environment. It refuses an existing environment path. Requirements are pinned;
Torch is installed separately for the selected hardware. Keep the resulting
`requirements-resolved.txt` for the research record.

**Tesla K80 (Kepler, compute capability 3.7):**

```bash
bash setup.sh --profile k80 --python python3.10 --env-dir "$CLEAR_REPO/.venv-clear-k80"
source "$CLEAR_REPO/.venv-clear-k80/bin/activate"
source "$CLEAR_REPO/.venv-clear-k80/clear-env.sh"
```

This retains the repository's Python 3.10 / torch 1.12.1+cu102 /
torchvision 0.13.1+cu102 stack. Do **not** run `pip install -e .` in this legacy
environment: `pyproject.toml` requires torch >=2.0 and would conflict. The scripts
use `src` via PYTHONPATH. The setup prepares a separate DINOv2 architecture source
checkout pinned to `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, without fetching
backbone weights. The CLEAR checkpoint supplies all model weights.

Modern CUDA or CPU alternatives:

```bash
# CUDA 11.8-compatible NVIDIA GPU/driver; not K80.
bash setup.sh --profile cuda --env-dir "$CLEAR_REPO/.venv-clear-cuda"
source "$CLEAR_REPO/.venv-clear-cuda/bin/activate"
source "$CLEAR_REPO/.venv-clear-cuda/clear-env.sh"

# Or CPU only, in a different fresh environment.
bash setup.sh --profile cpu --env-dir "$CLEAR_REPO/.venv-clear-cpu"
```

Modern profiles pin torch 2.5.1 / torchvision 0.20.1. `--profile auto` detects
K80 by GPU name, otherwise selects CUDA or CPU; explicit selection is preferable
for a reproducible run. Always inspect setup's GPU names, wheel CUDA runtime and
compiled architectures. A K80 requires sm_37 support and a Kepler-capable driver
that supports CUDA 10.2. A recent system CUDA toolkit does not establish wheel or
driver compatibility. No BF16, FlashAttention, xFormers or mixed precision is used
in CLEAR evaluation. Legacy checkpoint loading uses pickle when old Torch cannot
use weights-only loading; use trusted official files only.

Use one device per process (`--device cuda:0` or `cpu`). A dual-GPU K80 board has
separate device memories, not pooled VRAM. Start with **batch size 1**, concept
chunk 256 or 512. The full SFR bank is about 6 GiB in FP32, plus about 1.1 GiB for
the CLEAR text bank; indexed-pickle deserialization/stacking has substantial peak
RAM overhead. Plan **32–64 GiB host RAM**, sufficient disk, and ample time for
CPU-to-GPU streaming. Input resolution remains 448; reducing it would change the
protocol. CPU inference is supported but can be very slow.

Official compatibility sources:
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/),
[NVIDIA legacy compute capabilities](https://developer.nvidia.com/cuda-legacy-gpus).
The exact server stack has not been tested by this implementation session.

## 2. Download each Kaggle dataset

```bash
bash download_datasets.sh --dataset indiana --data-root "$SERVER_DATA" --archive-dir "$SERVER_ARCHIVES"
bash download_datasets.sh --dataset padchest --data-root "$SERVER_DATA" --archive-dir "$SERVER_ARCHIVES"
bash download_datasets.sh --dataset chexpert --data-root "$SERVER_DATA" --archive-dir "$SERVER_ARCHIVES"
```

Use `--dataset all` to request all three. The commands use `curl -fL -C -`, preserve
`.zip.part`, check ZIP integrity, and extract once per dataset. Complete valid ZIPs
are reused; an extraction marker skips already CRC-verified image trees.
Extraction is in place (no duplicate staging tree). Archives remain available
for repair; after verifying the data you may move/archive them yourself. Do not
delete partial files blindly if a Kaggle/CDN endpoint rejects range requests;
the script explains how to validate or explicitly restart them. Extraction
interrupted inside a file is detected by CRC; remove only the named incomplete
file and rerun. Removing an extraction marker forces verification again.

For HTTP 401/403 or HTML login/terms responses, sign into the Kaggle dataset page,
accept access terms and configure either `~/.kaggle/kaggle.json`,
`KAGGLE_USERNAME`/`KAGGLE_KEY`, or `KAGGLE_API_TOKEN`. Secrets are read through a
private temporary curl config, not printed or passed as process arguments. Do
not commit credentials. The script does not bypass access restrictions. Free
disk must accommodate the compressed archive plus the extracted images.

Expected logical content (additional nested archive prefixes are supported):

```text
SERVER_DATA/
  indiana/ .../indiana_reports.csv, .../indiana_projections.csv, .../*.png
  padchest/ .../*labels*.csv[.gz], .../*.png
  chexpert/ .../valid.csv, .../train.csv, .../patient.../study.../*.jpg
```

If the PadChest mirror uses another annotation filename, pass its exact path
with `--annotations`; auto-discovery intentionally rejects ambiguous matches.
Images in any nested ZIPs must first be extracted into the corresponding dataset
directory by the server operator; loaders read ordinary image files, not ZIP
members or DICOM. Image paths use unique suffix matches, preserving CheXpert
patient/study paths; ambiguous filename matches raise an error.

## 3. Obtain and register required pretrained assets

Use the activated CLEAR environment (network access is needed for this explicit
step). Choose a published HF commit or resolve `main` once and record the exact
commit returned. You must have permission under the asset terms.

```bash
python scripts/prepare_clear_assets.py --revision main --output-dir "$SERVER_ASSETS"
```

This writes `assets.json` with resolved revision, SHA256s and ordered concept-text
hash. Existing different files/manifests are never overwritten. For an offline
server, copy the following **from the same official snapshot** into SERVER_ASSETS
and register them with the known revision:

```bash
python scripts/prepare_clear_assets.py --register-local \
  --revision YOUR_CLEAR_SNAPSHOT_COMMIT --output-dir "$SERVER_ASSETS"
```

| Required file | Meaning |
| --- | --- |
| `best_model.pt` | Released CLEAR image and paired text encoder, 768 dimensions |
| `mimic_concepts.csv` | Ordered 368,294 observations, `concept` and optionally `concept_idx` |
| `concept_embeddings_368294.pt` | Aligned 368,294 × 768 CLEAR observation embeddings |
| `concepts_embeddings_sfr_mistral.pickle` | Aligned 368,294 × 4,096 SFR observation embeddings |
| `assets.json` | Registration provenance and SHA256s of all four files |
| `prompts/DATASET.npz` | Precomputed SFR diagnostic prompt embeddings |

`inspect_assets` verifies content hashes before each run, including resume. Full
CLEAR never silently downloads missing assets. The checkpoint path defaults to
SERVER_ASSETS/best_model.pt; `--checkpoint` can point to another location of that
same registered official checkpoint. `--allow-unsafe-pickle` explicitly opts into
the official SFR pickle. No missing embeddings are replaced by random vectors.
Checksums detect changed files, not semantic alignment of an arbitrary third-party
matrix. Provenance from a single official snapshot remains essential.

Asset acquisition uses HF cache symlinks when possible to avoid a second copy.
For transfer to another server, dereference those symlinks (`cp -L` or `rsync -L`)
or transfer the linked cache too. `--register-local` computes local integrity
hashes; it cannot independently attest that supplied weights are authentic.

### Prepare SFR prompts separately

K80's torch 1.12 stack is unsuitable for modern SFR/Mistral code. Prepare the
small prompt cache on another compatible GPU or on a CPU with enough RAM in a
**separate** modern environment, then copy only the `.npz` files back. A 7B model
in FP32 needs roughly 28 GB for weights alone; allow at least 40–64 GB host RAM.
The prompt profile installs CPU Torch. For GPU prompt preparation, use the modern
CUDA environment and install the same pinned transformers dependencies there,
never in the K80 environment.

```bash
bash setup.sh --profile prompts --env-dir "$CLEAR_REPO/.venv-prompts"
source "$CLEAR_REPO/.venv-prompts/bin/activate"
source "$CLEAR_REPO/.venv-prompts/clear-env.sh"
SFR_REV=$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Salesforce/SFR-Embedding-Mistral").sha)')
printf '%s\n' "$SFR_REV"

python scripts/prepare_clear_prompts.py --dataset indiana --revision "$SFR_REV" \
  --device cpu --output "$SERVER_ASSETS/prompts/indiana.npz"
python scripts/prepare_clear_prompts.py --dataset padchest --revision "$SFR_REV" \
  --device cpu --output "$SERVER_ASSETS/prompts/padchest.npz"
python scripts/prepare_clear_prompts.py --dataset chexpert --revision "$SFR_REV" \
  --device cpu --output "$SERVER_ASSETS/prompts/chexpert.npz"

# Return to the K80 environment before image inference.
source "$CLEAR_REPO/.venv-clear-k80/bin/activate"
source "$CLEAR_REPO/.venv-clear-k80/clear-env.sh"
```

The prompt model is [Salesforce/SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral),
with last-token pooling and L2 normalization, 4096 dimensions. Default positive
and negative strings are the paper finding and `no {finding}`; canonical Indiana
`atelectasis` uses `pulmonary atelectasis` in the prompt. `--format raw` reproduces
the released `get_embed.py` SFR code branch. The paper describes task instructions,
but the released branch only adds instructions to names containing `instruct`.
This discrepancy is recorded, not concealed. `--format instruction --instruction
'...'` is an explicit alternative; confirm the bank's original recipe before
claiming equivalence. The cache records SFR revision, input strings, order hash,
dtype, max length, instruction and pooling. It never uses CLEAR's 768-dimensional
text encoder to encode prompts for the SFR-space classifier.

For offline SFR preparation, supply `--model /path/to/local/sfr-snapshot
--local-files-only --revision KNOWN_40_CHARACTER_COMMIT`. The model is not needed
during CLEAR image evaluation once caches exist. For custom `--labels` or
`--all-findings`, prepare the cohort first and pass `--cohort /path/to/cohort.json`
instead of `--dataset` to the prompt preparation script.

## 4. Check dataset structure and reference label support

```bash
python evaluate.py --config configs/indiana.yaml --data-dir "$SERVER_DATA/indiana" \
  --prepare-only --output-dir "$SERVER_RESULTS/check-indiana"
python evaluate.py --config configs/padchest.yaml --data-dir "$SERVER_DATA/padchest" \
  --prepare-only --output-dir "$SERVER_RESULTS/check-padchest"
python evaluate.py --config configs/chexpert.yaml --data-dir "$SERVER_DATA/chexpert" \
  --prepare-only --output-dir "$SERVER_RESULTS/check-chexpert"
```

These commands do not load Torch, weights or images into a model. They check
annotation structures and image existence, and write `cohort.json`,
`data_check.json` (positive/negative/missing support), and `required_prompts.json`.
Use fresh output directories for preparation and inference. To inspect high-bit
depth pixels, use a small actual run after reviewing preprocessing below;
prepare-only checks file paths and annotations, not decoding every image.

Reference-label policies are explicit:

| Dataset | Labels | Default missing/uncertain behavior |
| --- | --- | --- |
| Indiana | Exact semicolon-separated `Problems`; case-normalized `Pulmonary Atelectasis` → `atelectasis`; copied from study to each image | Unmentioned findings unknown; missing report/tags remain unknown under every policy |
| PadChest | Python-list `Labels`; `ImageID` and `PatientID`; full annotation CSV may describe images outside sample | Unmentioned findings unknown; missing label cell unknown; absent images counted and skipped only in PadChest config |
| CheXpert | Numeric finding columns; `Path` contains patient/study; default **valid.csv only**, never train+valid concatenation | Blank → unknown, -1 → unknown, 0 → negative, 1 → positive |

`--unmentioned-policy negative` enables the **closed-world weak-label assumption**
for a present Indiana/PadChest annotation list. This is often used in legacy
multi-hot benchmarks, but absence of a tag does not establish clinical absence.
`--normal-policy negative` instead infers other findings negative only from an
explicit `normal`/`no finding` tag. An absent annotation is never converted into a
negative by either tag policy. A literal empty `[]` counts as a supplied list;
under the explicit closed-world policy its findings become negative.

With conservative defaults, some or all tag-based classes have no negative
labels and therefore **null AUROC**. Inspect support and choose/document your
label assumption rather than interpreting null as a model failure. To get
exploratory tag-based AUROCs, the commands below explicitly choose `negative`.
For CheXpert, `--uncertain-policy ignore|negative|positive` and `--missing-policy
ignore|negative` are separately configurable and recorded.

Use `--label-map /path/mappings.json` for additional exact aliases, e.g.
`{"pulmonary atelectasis": "atelectasis"}`; `--labels atelectasis,cardiomegaly`
selects canonical targets. `--all-findings` uses all native/observed annotations,
not a disease set chosen by model performance. The default paper set is 30
Indiana, 55 PadChest and 13 CheXpert findings. No parent/child hierarchy merging,
NLP negation inference or report diagnosis extraction is performed. Native
misspellings should be explicitly mapped, not fuzzily guessed.

`--annotations PATH` chooses a particular label file; `--projections PATH`
overrides Indiana image/view metadata. `--view all|frontal` controls projections.
The paper's cohorts are not automatically reconstructed from these Kaggle mirrors.

PadChest native PNGs can be 16-bit. Default `--image-policy 8bit` rejects those
instead of silently clipping to white with PIL RGB conversion. For an explicit
linear unsigned 16-bit-to-8-bit conversion, use `--image-policy uint16-linear`
(divide by 257 and round). This may differ from the paper's image construction;
record that limitation. Already 8-bit files keep their values. No per-image
min/max normalization, histogram equalization or DICOM windowing is applied.

## 5. Evaluate Indiana

Start with a small patient/group budget; each group retains all its selected
views, so the image count may exceed subset-size:

```bash
python evaluate.py --config configs/indiana.yaml \
  --data-dir "$SERVER_DATA/indiana" --assets-dir "$SERVER_ASSETS" \
  --inference-mode clear --allow-unsafe-pickle --unmentioned-policy negative \
  --device cuda:0 --batch-size 1 --concept-chunk-size 256 \
  --subset-size 10 --seed 42 --output-dir "$SERVER_RESULTS/indiana-debug"

# Full available cohort; use a fresh output directory.
python evaluate.py --config configs/indiana.yaml \
  --data-dir "$SERVER_DATA/indiana" --assets-dir "$SERVER_ASSETS" \
  --inference-mode clear --allow-unsafe-pickle --unmentioned-policy negative \
  --device cuda:0 --batch-size 1 --concept-chunk-size 512 \
  --seed 42 --output-dir "$SERVER_RESULTS/indiana"
```

No dataset size is hardcoded. Indiana `uid` is a study/report identifier, not a
verified longitudinal patient identifier. Its default grouping prevents a study's
views separating, but cannot guarantee separation of different studies from one
patient. New `--split validation|test` requires a verified `--patient-map CSV`
containing `study_id,patient_id` for every included uid. Without it the CLI refuses
to construct these splits. Do not invent patient IDs from image filenames.

## 6. Evaluate PadChest sample

```bash
python evaluate.py --config configs/padchest.yaml \
  --data-dir "$SERVER_DATA/padchest" --assets-dir "$SERVER_ASSETS" \
  --allow-unsafe-pickle --unmentioned-policy negative --image-policy uint16-linear \
  --device cuda:0 --batch-size 1 --subset-size 10 \
  --output-dir "$SERVER_RESULTS/padchest-debug"

python evaluate.py --config configs/padchest.yaml \
  --data-dir "$SERVER_DATA/padchest" --assets-dir "$SERVER_ASSETS" \
  --allow-unsafe-pickle --unmentioned-policy negative --image-policy uint16-linear \
  --device cuda:0 --batch-size 1 --output-dir "$SERVER_RESULTS/padchest"
```

If all images are already 8-bit, use the default `--image-policy 8bit` instead.
PadChest's config explicitly allows absent image files because the full CSV may
describe images outside the Kaggle sample. Their count is saved; it is never
silently ignored. Review label provenance (`MethodLabel` etc. when supplied by
your mirror) before constructing a physician-only cohort externally. The loader
does not assert that every Kaggle sample label was manually annotated.

## 7. Evaluate CheXpert zero shot

```bash
python evaluate.py --config configs/chexpert.yaml \
  --data-dir "$SERVER_DATA/chexpert" --assets-dir "$SERVER_ASSETS" \
  --allow-unsafe-pickle --device cuda:0 --batch-size 1 --subset-size 10 \
  --output-dir "$SERVER_RESULTS/chexpert-debug"

python evaluate.py --config configs/chexpert.yaml \
  --data-dir "$SERVER_DATA/chexpert" --assets-dir "$SERVER_ASSETS" \
  --allow-unsafe-pickle --device cuda:0 --batch-size 1 \
  --output-dir "$SERVER_RESULTS/chexpert"
```

Default uses the mirror's `valid.csv`. Use `--annotations /path/to/train.csv`
only intentionally. CLEAR pretraining included CheXpert-Plus, so CheXpert
training-data evaluation may overlap pretraining and is not a clean external
test. The paper's expert-labelled 500-patient test cohort requires separate
access and is not automatically supplied by a Kaggle mirror. Do not treat the
validation set used for SFR selection in the paper as an untouched external test.

### Resume, baseline and independent validation thresholds

To resume, repeat the **same inference command** and add `--resume`. Completed
batches are atomic `prediction_parts/part-*.json`; a partial temporary file is
ignored. The cohort, source code, input size/mtime, asset hashes, prompt metadata,
mode, seed, device and batch/chunk sizes must match. Choose a new directory if
changing them. A failure before loading a model may leave identity metadata;
the same resume command can continue. Only one writer may use a run directory.
Input size/mtime checks do not prove image content identity; archive CRC and an
immutable prepared data tree are still required. Batch and chunk settings may
affect floating-point accumulation slightly across hardware.

The fingerprint also records installed package versions, the visible driver/GPU
inventory and hashes of the local DINOv2 Python source when CLEAR_DINOV2_REPO is
set. Keep that source checkout and environment unchanged throughout a resumed run.

For the optional direct baseline, use a separate directory and
`--inference-mode direct`. It uses paired CLEAR text-tower prompts, no concept
bank or SFR prompt cache, and is always labelled as a baseline. The registered
checkpoint and DINOv2 code are still required.

For a checkpoint-only baseline installation, register a separate asset directory:

```bash
python scripts/prepare_clear_assets.py --revision main --inference-mode direct \
  --output-dir /your/storage/clear-direct-assets
python evaluate.py --dataset indiana --data-dir "$SERVER_DATA/indiana" \
  --assets-dir /your/storage/clear-direct-assets --inference-mode direct \
  --unmentioned-policy negative --device cuda:0 --batch-size 1 \
  --output-dir "$SERVER_RESULTS/indiana-direct"
```

This does not fetch either concept embedding matrix. Use the full asset directory
for full CLEAR; a direct-only manifest is insufficient for concept projection.

Patient-level validation/test splits use deterministic patient-ID hashing and
`--validation-fraction 0.2 --seed 42`; they do not stratify by test labels. Keep
the same annotations/policies/seed in both commands. Example on CheXpert:

```bash
python evaluate.py --config configs/chexpert.yaml --data-dir "$SERVER_DATA/chexpert" \
  --assets-dir "$SERVER_ASSETS" --allow-unsafe-pickle --device cuda:0 --batch-size 1 \
  --split validation --validation-fraction 0.2 --seed 42 \
  --output-dir "$SERVER_RESULTS/chexpert-validation"

python -m evaluation.calibration --validation-run "$SERVER_RESULTS/chexpert-validation" \
  --output "$SERVER_RESULTS/chexpert-thresholds.json"

python evaluate.py --config configs/chexpert.yaml --data-dir "$SERVER_DATA/chexpert" \
  --assets-dir "$SERVER_ASSETS" --allow-unsafe-pickle --device cuda:0 --batch-size 1 \
  --split test --validation-fraction 0.2 --seed 42 \
  --threshold-file "$SERVER_RESULTS/chexpert-thresholds.json" \
  --output-dir "$SERVER_RESULTS/chexpert-heldout"
```

A small validation set may lack both outcomes for rare classes; threshold
selection then fails explicitly. Choose a prespecified supported finding set or
retain a fixed threshold. Do not repair this by using test labels for calibration.
Threshold files record the validation prediction hash, selection rule and patient
IDs. Application checks inference signatures and rejects patient overlap.

## Metrics and saved outputs

Each completed run includes:

```text
identity.json                  Resume fingerprint and source/asset/input identity
run.json                       Status, command configuration, cohort and environment
cohort.json, data_check.json    Selected records, labels, grouping, missing-image/support counts
prediction_parts/part-*.json    Atomic incremental predictions and reference labels
predictions.json               Full records, scores and positive/negative cosine logits
predictions.csv                Long form: sample/patient/study, disease, y_true, score, logits
metrics.json, metrics.csv      Per-class metrics, support, confusion matrix, bootstrap CIs
summary.csv                    Dataset-level AUROC summary
comparison.json/csv            Published references, CIs, differences and comparability
errors.csv                     Up to 20 FP and 20 FN per finding for manual review
figures/class-NNN.png           ROC, PR curve and confusion matrix, in cohort label order
figures/auroc.png, ap.png, f1.png
run.log                        Progress/errors; one appended log per resumed run
```

AUROC is null when no positive **or** no negative label exists, never zero-filled.
AP is null with no positives; an all-positive class may have AP=1 but still has
undefined AUROC. Undefined precision/recall/specificity/F1 denominators are null.
Macro AUROC uses valid classes only and records the class list. Missing labels
are excluded independently for each finding. Thresholded predictions use `>=`.
Confusion matrices are rows=true 0/1, columns=predicted 0/1. The default fixed
threshold is 0.5; paired cosine softmax scores are not calibrated probabilities.

Bootstrap samples patient/groups with replacement, preserving all group images
and cross-class dependence. Each per-class scalar metric and confusion count has
a 95% percentile CI. Degenerate resamples do not produce an AUROC; their count is
reported. Macro bootstrap keeps the original valid-class set fixed and excludes
resamples lacking support for any of those classes. Fewer than two valid
replicates yields null bounds. Label-support counts are observed counts, not
estimates; their optional bootstrap intervals describe sampling variation.

Recalculate from saved predictions with no model, checkpoint or image access:

```bash
python evaluate.py --metrics-only --output-dir "$SERVER_RESULTS/indiana" \
  --threshold 0.5 --bootstrap 1000 --seed 42
```

This replaces only derived metrics, comparisons, errors and figures in that run;
predictions are unchanged. Supply `--threshold-file` again if you want independent
validation thresholds; otherwise this command uses a fixed threshold. `--no-plots`
omits rendering; `--bootstrap 0` omits CI sampling for a quick debug check. If you
change labels/policies themselves, prepare a new experiment rather than relabel
saved outputs silently.

## 8. Compare with the original publication

```bash
python -m evaluation.comparison --run-dir "$SERVER_RESULTS/indiana"
python -m evaluation.comparison --run-dir "$SERVER_RESULTS/padchest"
python -m evaluation.comparison --run-dir "$SERVER_RESULTS/chexpert"
```

These comparisons are also generated by evaluation. Exact per-class AUROCs,
published CI endpoints and positive support were transcribed from the
[publisher's Supplementary Information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41551-026-01741-4/MediaObjects/41551_2026_1741_MOESM1_ESM.pdf):
Tables S9–S10 (55 PadChest findings), S11 (30 Indiana), S13 (13 supervised
CheXpert findings). See `evaluation/references/README.md` and `paper_auroc.csv`
for provenance and published rounding. CheXpert positive support is blank where
not supplied by that table; it is not inferred from another cohort.

The module reports paper AUROC, reproduced AUROC, absolute and signed difference,
local positive/negative/missing support, both CIs when available, table/experiment,
source URL and comparability status. Macro reference is computed from rounded
per-class values, rather than inventing precision for the approximately 0.700
PadChest and 0.870 CheXpert headlines. Matched-class macros are reported separately
when a local subset lacks valid classes.

**PadChest sample is not the full physician-labelled benchmark. CheXpert zero
shot is not supervised linear probing. Indiana comparison is conditional on
matching cohort, labels, inference and image preprocessing.** The pipeline never
automatically marks exact reproduction verified. No statistical significance is
claimed for differences between unrelated cohorts or different CI procedures.

## 9. Generate the final Vietnamese Word report

After completing the desired runs on your server:

```bash
python generate_report.py \
  --run "indiana=$SERVER_RESULTS/indiana" \
  --run "padchest=$SERVER_RESULTS/padchest" \
  --run "chexpert=$SERVER_RESULTS/chexpert" \
  --output-dir "$CLEAR_REPO/reports"
```

Outputs are **`reports/CLEAR_evaluation.md`** and
**`reports/CLEAR_evaluation.docx`**, with the 14 requested Vietnamese sections,
tables, recorded environment/protocol, plots if present, FP/FN counts/examples,
limitations and references. Runs must be supplied explicitly, so existing legacy
results cannot be mistaken for new full-CLEAR experiments. Omit any `--run` that
was not performed; its section says **Chưa thực hiện**. Failed/interrupted runs
are described as incomplete and their partial metrics are not promoted to final
results. With no `--run`, the generator produces a status report with quoted
paper references and no invented experimental metrics.

The previous CXR-LLaVA report is not present in this repository. The generator
follows the specified 14-section outline and offers `--template-docx
/path/to/previous-report.docx` for its Word styles/page template; it does not claim
an exact match to an unavailable report. A legacy `.doc` must first be converted
to `.docx` by the operator. Existing reports are protected; use another output
directory or explicitly `--overwrite` for the two named generated files only.

Review Word pagination after generation, especially with many finding rows. If
LibreOffice is installed, a manual server-side PDF preview is possible:

```bash
mkdir -p "$CLEAR_REPO/reports/preview"
libreoffice -env:UserInstallation=file:///tmp/clear-report-lo-profile --headless \
  --convert-to pdf --outdir "$CLEAR_REPO/reports/preview" \
  "$CLEAR_REPO/reports/CLEAR_evaluation.docx"
```

No Word/PDF/report artifact has been generated or visually verified in this coding
session. The generator embeds charts into DOCX; Markdown links depend on retaining
the referenced result directories when moving the report.

## Synthetic tests and static checks

Only run these on the server when ready; they do not download data/checkpoints,
load pretrained weights, or run a real model:

```bash
python -m pytest tests/test_evaluation_pipeline.py -q
```

Tests cover all three loaders, Indiana mapping, missing/uncertain labels,
patient-group splits, known metric examples, bootstrap determinism, matrix
dimensions/order, cache alignment, atomic prediction reload and resume,
validation leakage rejection, mocked CLI interruption/resume, and missing/result
report generation. The tiny projection test uses a mock torch module and synthetic
arrays to check the official chunked formula. The existing full test suite also
contains legacy tests; inspect those separately before invoking them, since this
task's offline synthetic tests are deliberately isolated in one file.

## Expected limitations and completion checklist

The server operator must supply Python/system tools, a compatible GPU driver,
authorized datasets, the official bank/weights, DINOv2 source, and a verified
SFR prompt recipe. No official cohort manifest or original prompt-cache snapshot
is included in the release inspected here, so row dimensions and snapshot hashes
do not by themselves prove exact reproduction. The indexed SFR pickle has high
host-memory peak use. Full streaming projection prioritizes limited VRAM and can
be slow on K80 PCIe/CPU. Datasets, assets, credentials, environments and generated
results are ignored by Git. Preserve data-check files, environment lock output,
run metadata and the recorded immutable revisions together with each result.
