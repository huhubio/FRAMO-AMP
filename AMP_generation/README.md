# AMP-G: ProGen-based peptide generation

This directory contains the local ProGen-based generator used by FRAMO-AMP to produce candidate peptide sequences. It provides the model configuration, tokenizer, custom model implementation, and pretrained weight shards used by `scripts/generate_amp.py` and `scripts/finetune_generation_lora.py`.

For the full prediction and candidate-prioritization workflow, see the [project README](../README.md).

## Model summary

| Property | Value |
|---|---|
| Model family | ProGen |
| Architecture | `ProGenForCausalLM` |
| Parameters | 764,803,616 |
| Transformer layers | 27 |
| Hidden dimension | 1,536 |
| Attention heads | 16 |
| Context positions | 2,048 |
| Vocabulary size | 32 |
| Stored weight dtype | `float32` |
| Weight format | Two `safetensors` shards, approximately 3.06 GB in total |

The model architecture and source record are documented in [model_manifest.json](model_manifest.json).

## Directory contents

| File | Purpose |
|---|---|
| `config.json` | Hugging Face model configuration, including the local `auto_map` |
| `configuration_progen.py` | Custom configuration class |
| `modeling_progen.py` | Custom ProGen causal language model implementation |
| `model-00001-of-00002.safetensors` | First weight shard, tracked with Git LFS |
| `model-00002-of-00002.safetensors` | Second weight shard, tracked with Git LFS |
| `model.safetensors.index.json` | Mapping from model tensors to weight shards |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | Tokenizer configuration and vocabulary |
| `added_tokens.json`, `special_tokens_map.json` | Additional and special token definitions |
| `generation_config.json` | Model-level generation configuration |
| `model_manifest.json` | Architecture, weight sizes, checksums, and provenance |
| `SHA256SUMS` | Checksums for the listed model assets |

The loader uses `trust_remote_code=True` to load the custom implementation from this directory. Model loading is local-only by default.

## Setup

Run all commands in this guide from the **repository root**, not from `AMP_generation/`.

After cloning the repository and activating a Python environment, install the generation dependencies and retrieve the weights:

```bash
python -m pip install -e '.[generation]' 'peft==0.11.1'
git lfs install
git lfs pull
```

PEFT 0.11.1 is specified for compatibility with the project's pinned Transformers 4.42.3 environment. See the [installation instructions](../README.md#installation) for the complete setup.

Verify the model assets before use:

```bash
(cd AMP_generation && sha256sum -c SHA256SUMS)
```

On macOS, replace `sha256sum` with `shasum -a 256`. The weight shards should be approximately 1.78 GB and 1.28 GB; files of only a few hundred bytes are unresolved Git LFS pointers.

An NVIDIA GPU is recommended for generation. CPU execution is available with `--device cpu`; runtime and memory use depend on batch size and sequence length.

## Generate sequences

### Single GPU

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

### Multiple GPUs

On Linux, use `torchrun` to distribute generation across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/generate_amp.py \
  --num-return-sequences 5000 \
  --batch-size 16 \
  --out results/generated_amp.fasta
```

`--num-return-sequences` is the total target across all processes, while `--batch-size` applies to each process. Each rank handles a separate set of sample indices; rank 0 gathers and writes the results. This does not guarantee unique sequence content.

### CPU

```bash
python scripts/generate_amp.py \
  --device cpu \
  --num-return-sequences 10 \
  --batch-size 1 \
  --seed 42 \
  --out results/generated_amp_cpu.fasta
```

## Sampling options

Defaults are defined in [configs/default.yaml](../configs/default.yaml). Command-line arguments take precedence. Use `python scripts/generate_amp.py --help` for the complete interface.

| Option | Default | Description |
|---|---|---|
| `--num-return-sequences` | `1000` | Target number of accepted sequences |
| `--min-seq-length` / `--max-seq-length` | `8` / `25` | Accepted sequence length range |
| `--batch-size` | `32` | Number of samples per generation batch per process |
| `--seed` | `42` | Base random seed |
| `--temperature-min` / `--temperature-max` | `0.8` / `1.0` | Range used to sample temperatures; generation uses their batch average |
| `--top-k` | `30` | Top-k sampling cutoff |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--repetition-penalty` | `1.1` | Repetition penalty |
| `--max-attempts-factor` | `5` | Retry-budget multiplier for length-filtered samples |
| `--device` | `cuda` | Execution device; set explicitly to `cpu` for CPU runs |
| `--adapter-dir` | None | Optional LoRA adapter directory |
| `--out` | Required | Output FASTA path |
| `--out-json` | FASTA path with `.json` suffix | Settings and sequence output |

The script filters decoded sequences by the requested length range. If the retry budget is exhausted, it may return fewer sequences than requested. It does not automatically remove duplicate sequences. A fixed seed does not guarantee identical outputs across batch sizes, GPU counts, hardware, or library versions.

## Output files

A run with `--out results/generated_amp.fasta` writes:

- `results/generated_amp.fasta`: accepted sequences with identifiers such as `Generated_sequence_1`.
- `results/generated_amp.json`: the resolved generation settings and accepted sequences.

The JSON file records sampling settings but is not a complete environment archive. Retain the repository revision, dependency versions, model checksums, and command alongside these files for reproducibility.

To screen the generated sequences with AMP-P, first install the extraction dependencies and external tools described in the project README, then run:

```bash
bash scripts/run_end_to_end.sh \
  results/generated_amp.fasta \
  results/generated_amp_screening
```

## LoRA fine-tuning

Supply your own training FASTA and choose an output directory. For example, on a four-GPU Linux system:

```bash
torchrun --nproc_per_node=4 scripts/finetune_generation_lora.py \
  --fasta data/my_amp_dataset.fasta \
  --dataset-name my_amp_dataset \
  --save-dir lora_runs/my_amp_dataset \
  --num-train-epochs 5 \
  --per-device-train-batch-size 8
```

`data/my_amp_dataset.fasta` is a placeholder for your dataset. For single-process training, replace `torchrun --nproc_per_node=4` with `python`. Use `--bf16` or `--fp16` only with compatible hardware.

The final adapter is saved to `lora_runs/my_amp_dataset/final/`. Load it together with the base model for generation:

```bash
python scripts/generate_amp.py \
  --model-dir AMP_generation \
  --adapter-dir lora_runs/my_amp_dataset/final \
  --num-return-sequences 1000 \
  --batch-size 16 \
  --out results/generated_amp_lora.fasta
```

Adapters are merged into the base model in memory by default during generation; use `--no-merge-adapter` to keep the adapter unmerged. Fine-tuning options are listed by `python scripts/finetune_generation_lora.py --help`.

The bundled generator and a fine-tuned adapter are distinct artifacts. Reproducing a particular adaptation experiment requires its training data, training settings, and resulting adapter in addition to the base model.

## Provenance and verification

The model manifest records source repository `huhubio/FRAMO-AMP` at commit `cf61c8cbeb9b915907b24aa71a63c027089b8c8a`. It lists 764,803,616 float32 parameters and a total shard size of 3,059,237,849 bytes. Use [SHA256SUMS](SHA256SUMS) to verify the listed assets after download or transfer.

## Troubleshooting

| Problem | Suggested action |
|---|---|
| Weight files are small text files | Run `git lfs pull` from the repository root and repeat checksum verification. |
| CUDA is unavailable | Select `--device cpu`, or install a CUDA-enabled PyTorch environment for GPU use. |
| GPU memory is exhausted | Reduce `--batch-size`, for example to `8`, `4`, or `1`. |
| PEFT fails to import | Use PEFT 0.11.1 with the project's Transformers 4.42.3 dependency. |
| Fewer sequences are returned than requested | Inspect the length bounds and retry budget; review the generation log. |
| Adapter loading fails | Confirm that `--adapter-dir` points to the saved adapter directory and that it matches the base model. |

## License and scope

Repository source code is covered by the [MIT License](../LICENSE). Upstream model weights, datasets, and tools retain their respective license terms. Generated sequences are computational candidates and require downstream assessment and experimental validation.
