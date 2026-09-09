# AMP Generator / AMP 生成模型

This directory stores the local ProGen-based AMP generator used by
`scripts/generate_amp.py` and `scripts/finetune_generation_lora.py`.

本目录保存 FRAMO-AMP 的本地 ProGen 系列 AMP 生成模型，由
`scripts/generate_amp.py` 与 `scripts/finetune_generation_lora.py` 调用。

## Contents / 文件内容

| File / 文件 | Purpose / 用途 |
|---|---|
| `config.json` | HuggingFace model config with local `auto_map`. |
| `configuration_progen.py` | Local custom config class loaded with `trust_remote_code=True`. |
| `modeling_progen.py` | Local custom ProGen causal-LM implementation. |
| `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors` | Sharded generator weights, tracked by Git LFS. |
| `model.safetensors.index.json` | HuggingFace shard index mapping tensors to weight shards. |
| tokenizer files | GPT2-style tokenizer vocabulary and special-token metadata. |

The generator weights are split into shards because GitHub LFS rejects single
files larger than 2 GB. After cloning, run `git lfs pull` if the shard files
are only small pointer files.

生成模型权重已按 HuggingFace 标准拆成多个 safetensors 分片，因为 GitHub
LFS 不接受超过 2 GB 的单个文件。如果 clone 后分片文件只是很小的 pointer，
请在仓库根目录运行 `git lfs pull`。

## Direct generation / 直接生成

```bash
python scripts/generate_amp.py \
    --num-return-sequences 1000 \
    --min-seq-length 8 \
    --max-seq-length 25 \
    --out results/generated_amp.fasta
```

For multi-GPU inference on Linux:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/generate_amp.py \
    --num-return-sequences 5000 \
    --out results/generated_amp.fasta
```

## LoRA fine-tuning / LoRA 微调

```bash
torchrun --nproc_per_node=4 scripts/finetune_generation_lora.py \
    --fasta data/my_amp_dataset.fasta \
    --dataset-name my_amp_dataset \
    --save-dir lora_runs/my_amp_dataset \
    --num-train-epochs 5 \
    --per-device-train-batch-size 8
```

The final LoRA weights are saved under `save-dir/final` for downstream use.

最终 LoRA 权重保存在 `save-dir/final`，供后续分析或部署使用。

## Provenance and verification / 来源与校验

This module was copied from `huhubio/FRAMO-AMP` commit `cf61c8c`. The bundled
model contains 764,803,616 float32 parameters. See `model_manifest.json` for
the machine-readable architecture and source record. Verify all model assets
from this directory with `sha256sum -c SHA256SUMS`.

本模块复制自 `huhubio/FRAMO-AMP` 的 `cf61c8c` 提交。内置生成模型包含
764,803,616 个 float32 参数。架构记录见 `model_manifest.json`，可在本目录运行
`sha256sum -c SHA256SUMS` 校验模型文件。
