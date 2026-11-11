# FRAMO-AMP

**FRAMework for Overall Antimicrobial Peptide Discovery**

FRAMO-AMP is a research software framework for antimicrobial peptide (AMP) discovery. It combines a three-model ensemble for AMP prediction (**AMP-P**), a ProGen-based sequence generator (**AMP-G**), and an interpretable workflow for prioritizing candidates for experimental validation (**ExpFrame**).

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [AMP prediction](#amp-prediction)
- [Sequence generation and fine-tuning](#sequence-generation-and-fine-tuning)
- [ExpFrame candidate prioritization](#expframe-candidate-prioritization)
- [Inputs and outputs](#inputs-and-outputs)
- [Models and reproducibility](#models-and-reproducibility)
- [Repository structure](#repository-structure)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## Overview

| Component | Purpose | Entry point |
|---|---|---|
| AMP-P | Extract ESM2, ProtBERT, and SaProt–ProstT5 features and classify sequences with three trained MLPs | `scripts/run_end_to_end.sh` |
| Ensemble inference | Classify aligned, precomputed feature matrices | `amp-ensemble` |
| AMP-G | Generate candidate peptide sequences with the bundled ProGen model | `scripts/generate_amp.py` |
| LoRA fine-tuning | Adapt the generator to a user-supplied FASTA dataset | `scripts/finetune_generation_lora.py` |
| ExpFrame | Prioritize predicted AMPs using sequence, structure, and embedding evidence | `scripts/expframe_filter.py` |

```text
Input FASTA
  ├── ESM2 mean embeddings ────────────────────── ESM2-MLP ──┐
  ├── ProtBERT mean embeddings ────────────── ProtBERT-MLP ──┼── Majority vote
  └── ProstT5 3Di tokens + SaProt mean embeddings ─ SaProt-MLP ┘       │
                                                              predictions.tsv
                                                                    │
                                    Sequence / structure / embedding evidence
                                                                    │
                                                                 ExpFrame
                                                                    │
                                                            retained_amp.fasta

AMP-G generates candidate FASTA files that can enter the same prediction workflow.
```

Each classifier uses its own training-set scaler and cross-validation decision threshold. By default, a sequence is classified as an AMP when at least two of the three classifiers predict AMP.

## Requirements

- Python 3.10 or later and PyTorch 2.1 or later.
- Linux for the end-to-end shell workflow, which uses GNU `realpath`.
- Git LFS to retrieve the bundled generator weights.

| Workflow | Additional requirements |
|---|---|
| Ensemble inference from NPY features | CPU or CUDA GPU; no feature encoders required |
| End-to-end feature extraction | NVIDIA GPU, CUDA-enabled PyTorch, and Foldseek 10 or later |
| Sequence generation | NVIDIA GPU recommended; CPU execution is also supported |
| LoRA fine-tuning | A user-supplied training FASTA; GPU resources appropriate to the training configuration |
| ExpFrame sequence evidence | MMseqs2, or precomputed MMseqs2 results |
| ExpFrame structure evidence | Candidate and reference structures, plus Foldseek; alternatively, precomputed Foldseek results |

The generator weights occupy approximately **3.06 GB**. The feature encoders require a further **8.7 GiB**. Allow additional disk space for caches, intermediate files, and outputs. Memory requirements depend on sequence length, batch size, and the selected workflow.

## Installation

Run the following commands in a terminal:

```bash
git lfs install
git clone https://github.com/huhubio/FRAMO-AMP.git
cd FRAMO-AMP

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[extract,generation]' 'peft==0.11.1'
```

The explicit PEFT pin selects a version compatible with this project's pinned Transformers 4.42.3 environment. For ensemble inference from existing NPY files only, install the base package with `python -m pip install -e .` instead.

All commands below assume that the virtual environment is active and the working directory is the **repository root**. Install Foldseek and MMseqs2 separately when required; they are not installed by `pip`.

Retrieve and verify the generator weights:

```bash
git lfs pull
ls -lh AMP_generation/model-*.safetensors
(cd AMP_generation && sha256sum -c SHA256SUMS)
(cd models && sha256sum -c SHA256SUMS)
```

The two generator shards should be approximately 1.78 GB and 1.28 GB, respectively. Files of only a few hundred bytes are Git LFS pointers, not model weights. On macOS, use `shasum -a 256 -c SHA256SUMS` in place of `sha256sum -c SHA256SUMS`.

## Quick start

### 1. Prepare an input FASTA

Save your peptide sequences as `input.fasta`, using a unique ID for each record:

```fasta
>peptide_1
KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK
>peptide_2
GIGKFLHSAKKFGKAFVGEIMNS
```

### 2. Predict AMP candidates

With the GPU environment and Foldseek available, run:

```bash
bash scripts/run_end_to_end.sh input.fasta results/example
```

On the first run, missing feature encoders are downloaded and verified. Predictions are written to `results/example/predictions.tsv`. An `ensemble_prediction` of `1` and an `ensemble_label` of `AMP` indicate a positive ensemble prediction.

### 3. Optionally generate sequences for screening

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_amp.py \
  --num-return-sequences 100 \
  --min-seq-length 8 \
  --max-seq-length 25 \
  --batch-size 16 \
  --out results/generated_amp.fasta

bash scripts/run_end_to_end.sh \
  results/generated_amp.fasta \
  results/generated_amp_screening
```

This writes generated sequences to `results/generated_amp.fasta`, generation settings and sequences to `results/generated_amp.json`, and screening predictions to `results/generated_amp_screening/predictions.tsv`.

## AMP prediction

### End-to-end prediction from FASTA

```bash
bash scripts/run_end_to_end.sh INPUT.fasta OUTPUT_DIR
```

The pipeline extracts residue-mean embeddings from ESM2-650M and ProtBERT, predicts 3Di tokens with Foldseek's ProstT5 model, and extracts mean embeddings from SaProt-650M using the paired amino acid/3Di tokens. It then applies the matching scalers and MLP classifiers before combining their predictions.

Configure resources through environment variables:

```bash
FRAMO_GPUS=0 \
ESM2_BATCH_SIZE=32 \
PROTBERT_BATCH_SIZE=16 \
SAPROT_BATCH_SIZE=8 \
PROSTT5_GPU=1 \
bash scripts/run_end_to_end.sh input.fasta results/run_01
```

| Variable | Default | Purpose |
|---|---|---|
| `FRAMO_GPUS` | `0` | GPU IDs used by the embedding extractors |
| `FRAMO_BACKBONE_DIR` | `models/backbones` under the repository root | Feature encoder directory |
| `ESM2_BATCH_SIZE` | `32` | ESM2 extraction batch size |
| `PROTBERT_BATCH_SIZE` | `16` | ProtBERT extraction batch size |
| `SAPROT_BATCH_SIZE` | `8` | SaProt extraction batch size |
| `MODEL_DTYPE` | `float16` | Encoder computation dtype |
| `FOLDSEEK_BIN` | `foldseek` | Foldseek executable |
| `FOLDSEEK_THREADS` | `16` | Foldseek thread count |
| `PROSTT5_GPU` | Unset | Any nonempty value enables the ProstT5 GPU flag; unset it for CPU execution |
| `ENSEMBLE_DEVICE` | `auto` | Device for the final MLP inference step |

To download and verify encoders before running the pipeline:

```bash
python scripts/download_backbones.py
```

For individual extraction steps, see the commands in [scripts/run_end_to_end.sh](scripts/run_end_to_end.sh) and the `--help` output of `extract_esm2_embeddings.py`, `extract_protein_lm_embeddings.py`, and `build_saprot_structure_tokens_with_prostt5.py` in `scripts/`.

### Inference from precomputed features

Edit [examples/manifest.json](examples/manifest.json) to point to your feature matrices, metadata, checkpoints, model configurations, and scalers. The example is a **template**: the referenced `data/query/` feature files are not included. Relative paths are resolved from the manifest's directory.

```bash
amp-ensemble \
  --manifest examples/manifest.json \
  --device cpu \
  --strategy majority \
  --output predictions.tsv
```

The three metadata files must contain the same `seq_id` values in the same order, with `feature_row` numbered consecutively from zero. Feature dimensions must be 1280 for ESM2, 1280 for SaProt–ProstT5, and 1024 for ProtBERT.

Optional soft voting averages the three classifier scores:

```bash
amp-ensemble \
  --manifest examples/manifest.json \
  --strategy soft \
  --soft-threshold 0.5 \
  --output predictions.tsv
```

With majority voting, `ensemble_score` is the fraction of positive votes: `0`, `1/3`, `2/3`, or `1`. With soft voting, it is the mean classifier score. These quantities should not be interpreted interchangeably.

## Sequence generation and fine-tuning

The bundled generator is stored in [AMP_generation/](AMP_generation/README.md). Default sampling and fine-tuning settings are defined in [configs/default.yaml](configs/default.yaml); command-line arguments override those settings.

For a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_amp.py \
  --model-dir AMP_generation \
  --num-return-sequences 1000 \
  --min-seq-length 8 \
  --max-seq-length 25 \
  --batch-size 16 \
  --seed 42 \
  --out results/generated_amp.fasta
```

For CPU execution, add `--device cpu` and use a small batch size. Multi-GPU generation, sampling parameters, LoRA training, and adapter loading are documented in the [generator guide](AMP_generation/README.md).

Generation produces computational candidates. The requested sequence count is a target: length filtering and the retry limit may yield fewer sequences. Generated sequences are not automatically deduplicated.

## ExpFrame candidate prioritization

ExpFrame combines three types of evidence after AMP-P prediction. Categories are assigned in the order shown below:

| Category | Evidence |
|---|---|
| `sequence_similarity` | MMseqs2 E-value `< 1e-3` |
| `structure_similarity` | No sequence-category match and Foldseek TM-score `> 0.5` |
| `divergent_generation` | Neither similarity category applies, but ESM2 CLS embeddings satisfy both reference-derived distance thresholds |
| `not_retained` | None of the above criteria are met |

![ExpFrame workflow](Expframe.png)

See [Expframe.md](Expframe.md) for the accompanying method description.

### Prepare reference data and structures

Supply a positive AMP reference FASTA and, optionally, a separate sequence-search reference FASTA. For structure-based screening, prepare candidate and positive-reference structures with ESMFold and place their PDB/mmCIF files in separate directories. Structure identifiers must match the corresponding sequence IDs. This repository does not automatically install or run ESMFold.

The paths in the following examples refer to user-supplied data or previously generated outputs.

### Run all evidence branches

```bash
python scripts/expframe_filter.py \
  --predictions-tsv results/generated_amp_screening/predictions.tsv \
  --positive-reference-fasta data/positive_amp.fasta \
  --sequence-reference-fasta data/high_confidence_predicted_amp.fasta \
  --run-mmseqs \
  --mmseqs-bin mmseqs \
  --run-foldseek \
  --foldseek-bin foldseek \
  --candidate-structure-dir results/esmfold/candidates \
  --reference-structure-dir results/esmfold/positive_reference \
  --esm2-model-dir models/backbones/esm2_t33_650M_UR50D \
  --gpus 0 \
  --output-dir results/expframe
```

If `--sequence-reference-fasta` is omitted, MMseqs2 uses the positive reference FASTA. Providing `--esm2-model-dir` triggers extraction of the leading `[CLS]` embeddings required by ExpFrame. The mean-pooled ESM2 embeddings used for AMP-P are not interchangeable with these features.

Embedding thresholds are estimated only from the positive reference set. By default, the method uses the 95th percentiles of reference-to-centroid distances and leave-one-out nearest-neighbor distances. A candidate must satisfy both thresholds. Adjust the quantile with `--embedding-quantile`.

### Reuse existing evidence

```bash
python scripts/expframe_filter.py \
  --predictions-tsv results/generated_amp_screening/predictions.tsv \
  --positive-reference-fasta data/positive_amp.fasta \
  --mmseqs-results results/mmseqs_hits.tsv \
  --foldseek-results results/foldseek_hits.tsv \
  --candidate-embeddings results/esm2_cls/candidates.npy \
  --candidate-metadata results/esm2_cls/candidates.metadata.tsv \
  --reference-embeddings results/esm2_cls/positive_reference.npy \
  --reference-metadata results/esm2_cls/positive_reference.metadata.tsv \
  --output-dir results/expframe
```

Use `--allow-partial` only when intentionally evaluating an incomplete set of evidence branches. Retained candidates are written to `results/expframe/retained_amp.fasta`; per-candidate evidence and decisions are recorded in `expframe_results.tsv`.

## Inputs and outputs

### FASTA and feature metadata

Use unique, whitespace-free sequence IDs in the first field of each FASTA header. The feature extractors map certain nonstandard residues to `X`. ExpFrame retains only sequences composed of the 20 canonical amino acids, removes duplicate sequences, and records preprocessing decisions in `input_audit.tsv`.

Each embedding extractor writes an NPY feature matrix and a matching `.metadata.tsv` file containing row indices, sequence IDs, cleaned sequences, truncation information, and source FASTA paths.

### Prediction outputs

```text
OUTPUT_DIR/
├── features/
│   ├── esm2/                 # NPY features and metadata
│   ├── protbert/             # NPY features and metadata
│   └── saprot_prostt5/        # NPY features and metadata
├── structure_tokens/         # Predicted 3Di tokens and audit table
├── foldseek_tmp/             # Intermediate Foldseek files
├── manifest.json
└── predictions.tsv
```

`predictions.tsv` includes sequence identifiers, sequences, each classifier's score, threshold and prediction, and the `ensemble_score`, `ensemble_prediction`, and `ensemble_label` columns.

### ExpFrame outputs

```text
results/expframe/
├── expframe_results.tsv
├── retained_amp.fasta
├── sequence_similarity.fasta
├── structure_similarity.fasta
├── divergent_generation.fasta
├── input_audit.tsv
├── summary.tsv
└── work/
```

## Models and reproducibility

The repository includes trained MLP checkpoints, their matching scalers and configurations, and the ProGen generator weights tracked through Git LFS.

| Classifier | Architecture | Dropout | Parameters | Decision threshold |
|---|---|---:|---:|---:|
| ESM2-MLP | 1280 → 512 → 1 | 0.20 | 656,385 | 0.500 |
| SaProt–ProstT5-MLP | 1280 → 512 → 256 → 128 → 1 | 0.35 | 820,225 | 0.795 |
| ProtBERT-MLP | 1024 → 512 → 256 → 128 → 1 | 0.50 | 689,153 | 0.850 |

Architecture details, feature encoder revisions, provenance, and checksums are recorded in:

- [docs/MODELS.md](docs/MODELS.md)
- [models/model_manifest.json](models/model_manifest.json)
- [configs/backbones.json](configs/backbones.json)
- [AMP_generation/model_manifest.json](AMP_generation/model_manifest.json)

For consistent inference, use mean pooling for all three classifier inputs, pair SaProt amino acids with the corresponding ProstT5/Foldseek 3Di tokens, and retain each classifier's own scaler. ExpFrame instead requires leading `[CLS]` embeddings from ESM2.

Record the repository commit, dependency versions, hardware, command-line arguments, manifests, feature metadata, and generation JSON files for each analysis. A fixed seed alone does not guarantee identical samples across hardware, library versions, batch sizes, or GPU counts. Reproducing a specific fine-tuning experiment also requires its dataset and adapter; the bundled base generator alone does not reproduce every adapted model.

## Repository structure

```text
FRAMO-AMP/
├── AMP_generation/       # ProGen configuration, tokenizer, implementation, and LFS weights
├── configs/              # Default settings and feature encoder specifications
├── docs/                 # Model documentation
├── examples/             # Inference manifest template
├── models/               # Trained MLPs, scalers, configurations, and checksums
├── scripts/              # Generation, extraction, prediction, and prioritization entry points
├── src/amp_ensemble/     # Ensemble inference package
├── src/framo_amp/        # Generation and ExpFrame implementations
├── tests/                # Unit tests
├── Expframe.md
└── Expframe.png
```

## Tests

Install the optional dependencies used by the tests and run the suite:

```bash
python -m pip install -e '.[test,extract,generation]' 'peft==0.11.1'
python -m unittest discover -s tests -v
```

The tests include mocked generation checks. Passing the unit tests does not establish successful GPU feature extraction, external-tool execution, or reproduction of manuscript results.

## Troubleshooting

| Problem | Suggested action |
|---|---|
| Generator shards contain only a few hundred bytes | Run `git lfs install` and `git lfs pull`, then verify the checksums. |
| CUDA out of memory | Reduce `--batch-size` for generation or the relevant extraction batch-size environment variable. |
| Foldseek or MMseqs2 cannot be found | Add the executable to `PATH`, or set `FOLDSEEK_BIN` for the shell pipeline and `--foldseek-bin` / `--mmseqs-bin` for ExpFrame. |
| Ensemble feature alignment fails | Check sequence IDs, row order, `feature_row`, feature dimensions, pooling, and scaler assignments. |
| The example manifest reports missing files | Replace the template's feature and metadata paths with your own files. |
| PEFT fails to import with Transformers | Use the documented PEFT 0.11.1 / Transformers 4.42.3 combination in a clean environment. |
| ExpFrame retains no divergent candidates | Confirm that the inputs are ESM2 CLS embeddings and inspect the centroid and nearest-neighbor distances and thresholds. |

## Limitations

Predicted and generated sequences are computational candidates, not experimentally validated AMPs. Results may reflect biases in training-set composition, sequence homology, and taxonomic coverage. ExpFrame prioritizes candidates; it does not assess toxicity, hemolysis, stability, or other properties required for experimental or downstream use.

## Citation

When using this software, cite this repository and the exact commit or release used. Also cite the original publications for the upstream models and tools used in your analysis, such as ESM2, ProtBERT, SaProt, ProstT5, ProGen, ESMFold, MMseqs2, and Foldseek. A manuscript citation or archived software DOI will be added when available.

## License

Repository source code is distributed under the [MIT License](LICENSE). Third-party model weights, datasets, and external tools remain subject to their respective licenses.
