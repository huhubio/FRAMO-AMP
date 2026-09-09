# FRAMO-AMP: FRAMework for Overall AMP Discovery

## ESM2 / SaProt-ProstT5 / ProtBERT 3-MLP Ensemble

这是一个包含候选肽生成和三模型筛选的完整 AMP 发现流程：

1. 使用仓库内置的 ProGen 模型生成候选 AMP 序列；
2. 从 FASTA 生成 ESM2、ProtBERT、SaProt-ProstT5 三套特征；
3. 使用三个 MLP 进行预测并执行 2/3 多数投票。

```text
FASTA
 ├─ ESM2 mean embedding ──────────────── ESM2-MLP ───────┐
 ├─ ProtBERT mean embedding ──────────── ProtBERT-MLP ───┼─ 2/3 majority vote
 └─ ProstT5 3Di → SaProt mean embedding ─ SaProt-MLP ─────┘
```

仓库支持三种用法：

- 使用本地 ProGen 权重直接生成候选 AMP，或通过 LoRA 微调生成模型；
- 从 FASTA 开始，自动生成三套 `.npy` 特征并完成集成预测；
- 输入已经提取好的三套 `.npy`，直接运行三个 MLP 和多数投票。

ProGen 生成模型以两个 Git LFS `safetensors` 分片保存在 `AMP_generation/`。三个训练完成的 MLP checkpoint、训练集 scaler、模型配置、阈值和 SHA-256 校验值均包含在 `models/` 中。

## 安装

需要 Python 3.10+、CUDA 版 PyTorch 和 NVIDIA GPU。SaProt 的 3Di 生成还需要支持 ProstT5 的 Foldseek 10 或更高版本。

```bash
git lfs install
git clone https://github.com/huhubio/FRAMO-AMP-FRAMework-for-overall-AMP-discovery-.git
cd FRAMO-AMP-FRAMework-for-overall-AMP-discovery-
python -m venv .venv
source .venv/bin/activate
pip install -e '.[extract,generation]'
```

如果仓库已克隆但 `AMP_generation/*.safetensors` 只是很小的 LFS 指针文件，请执行 `git lfs pull`。

Foldseek 请参考其官方安装方式；安装后确认：

```bash
foldseek version
```

## AMP 候选序列生成

### 使用哪个代码

生成阶段各文件的作用如下：

| 文件 | 作用 |
|---|---|
| `scripts/generate_amp.py` | 生成候选 AMP 的命令行入口，通常直接运行这个文件 |
| `src/framo_amp/generation/progen.py` | 加载 ProGen、执行采样、长度过滤并写出 FASTA/JSON |
| `AMP_generation/` | 本地 ProGen 模型代码、Tokenizer、配置和两个权重分片 |
| `configs/default.yaml` | 默认生成参数；命令行参数会覆盖这里的设置 |
| `scripts/finetune_generation_lora.py` | 使用自己的 AMP FASTA 对生成模型进行 LoRA 微调 |

克隆仓库后，先确认 Git LFS 权重已经下载。两个权重分片应分别约为
1.78 GB 和 1.28 GB，而不是几百字节的 LFS 指针：

```bash
git lfs install
git lfs pull
ls -lh AMP_generation/model-*.safetensors

# 可选：校验权重和模型相关文件
(cd AMP_generation && sha256sum -c SHA256SUMS)
```

安装生成模块所需依赖：

```bash
pip install -e '.[generation]'
```

### 单 GPU 生成

下面的命令使用仓库中的 `AMP_generation/` 权重，生成 100 条长度为
8–25 aa 的候选肽：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_amp.py \
  --model-dir AMP_generation \
  --num-return-sequences 100 \
  --min-seq-length 8 \
  --max-seq-length 25 \
  --batch-size 16 \
  --seed 42 \
  --out results/generated_amp.fasta
```

`--model-dir` 可以省略，因为默认值就是 `AMP_generation/`。如果显存不足，
将 `--batch-size` 调小，例如改为 `8`、`4` 或 `1`。也可以使用 CPU，
但生成会明显变慢：

```bash
python scripts/generate_amp.py \
  --device cpu \
  --num-return-sequences 10 \
  --batch-size 1 \
  --out results/generated_amp_cpu.fasta
```

生成完成后会得到两个文件：

```text
results/
├── generated_amp.fasta  # 后续特征提取和筛选直接使用这个文件
└── generated_amp.json   # 完整采样参数和生成序列，便于复现
```

FASTA 中的序列 ID 为 `Generated_sequence_1`、`Generated_sequence_2` 等。
程序会清理非氨基酸字符，并只保留位于 `--min-seq-length` 和
`--max-seq-length` 之间的序列。`--num-return-sequences` 表示希望最终写出的
有效序列总数；如果在有限尝试次数内未得到足够多的有效序列，日志会报告实际数量。

### 采样参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--num-return-sequences` | 1000 | 最终希望生成的候选序列总数 |
| `--min-seq-length` / `--max-seq-length` | 8 / 25 | 保留的肽长度范围 |
| `--batch-size` | 32 | 每个生成进程的批大小 |
| `--seed` | 42 | 随机种子；相同环境和参数下用于复现 |
| `--temperature-min` / `--temperature-max` | 0.8 / 1.0 | 每批采样温度范围；越高通常越多样 |
| `--top-k` | 30 | 只从概率最高的 K 个 token 中采样 |
| `--top-p` | 0.9 | nucleus sampling 的累计概率阈值 |
| `--repetition-penalty` | 1.1 | 重复惩罚，较高可减少重复模式 |
| `--max-attempts-factor` | 5 | 为获得合格长度序列允许的最大尝试倍数 |

例如，提高采样多样性：

```bash
python scripts/generate_amp.py \
  --num-return-sequences 1000 \
  --temperature-min 0.9 \
  --temperature-max 1.2 \
  --top-k 50 \
  --top-p 0.95 \
  --out results/generated_amp_diverse.fasta
```

### 多 GPU 生成

使用 `torchrun` 启动多进程，每张 GPU 对应一个进程。下面命令在 GPU 0 和
GPU 1 上合计生成 1000 条序列，而不是每张卡各生成 1000 条：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/generate_amp.py \
  --num-return-sequences 1000 \
  --batch-size 16 \
  --out results/generated_amp.fasta
```

只有 rank 0 会汇总各 GPU 的结果并写出最终 FASTA 和 JSON。

### 生成后直接运行三模型筛选

生成得到的 FASTA 可以直接作为端到端筛选脚本的输入：

```bash
scripts/run_end_to_end.sh \
  results/generated_amp.fasta \
  results/generated_amp_screening
```

最终集成预测位于：

```text
results/generated_amp_screening/predictions.tsv
```

该步骤会依次生成 ESM2、ProtBERT、SaProt-ProstT5 特征，再用三个 MLP
执行预测和 2/3 多数投票。

### 使用 LoRA 微调后的模型生成

使用自己的 AMP FASTA 进行 LoRA 微调：

```bash
torchrun --nproc_per_node=4 scripts/finetune_generation_lora.py \
  --fasta data/my_amp_dataset.fasta \
  --dataset-name my_amp_dataset \
  --save-dir lora_runs/my_amp_dataset \
  --num-train-epochs 5 \
  --per-device-train-batch-size 8 \
  --bf16
```

最终 LoRA adapter 默认保存在 `lora_runs/my_amp_dataset/final/`。加载该
adapter 生成候选序列：

```bash
python scripts/generate_amp.py \
  --model-dir AMP_generation \
  --adapter-dir lora_runs/my_amp_dataset/final \
  --num-return-sequences 1000 \
  --out results/generated_amp_lora.fasta
```

也可以在微调结束后立即生成：

```bash
torchrun --nproc_per_node=4 scripts/finetune_generation_lora.py \
  --fasta data/my_amp_dataset.fasta \
  --dataset-name my_amp_dataset \
  --save-dir lora_runs/my_amp_dataset \
  --num-train-epochs 5 \
  --per-device-train-batch-size 8 \
  --bf16 \
  --generate-after-train \
  --generation-out results/generated_after_lora.fasta \
  --num-return-sequences 1000
```

生成模型内部文件、参数量、来源和校验方式见 `AMP_generation/README.md`。

## 一条命令：FASTA 到集成结果

```bash
scripts/run_end_to_end.sh input.fasta output
```

第一次执行会下载并校验固定版本的 ESM2、ProtBERT、SaProt 和 Foldseek ProstT5 权重，共约 8.7 GiB。默认使用逻辑 GPU 0，可通过环境变量调整：

```bash
FRAMO_GPUS=0 \
ESM2_BATCH_SIZE=32 \
PROTBERT_BATCH_SIZE=16 \
SAPROT_BATCH_SIZE=8 \
PROSTT5_GPU=1 \
scripts/run_end_to_end.sh input.fasta output
```

主要输出：

```text
output/
├── features/
│   ├── esm2/*.npy
│   ├── protbert/*.npy
│   └── saprot_prostt5/*.npy
├── structure_tokens/*.prostt5_3di.tsv
├── manifest.json
└── predictions.tsv
```

## 分步生成 NPY

先下载固定版本的基础模型：

```bash
python scripts/download_backbones.py
```

下载目录默认为 `models/backbones/`，已被 `.gitignore` 排除。模型 ID、固定 revision、字节数及原始权重 SHA-256 位于 `configs/backbones.json`。

### 1. ESM2

```bash
python scripts/extract_esm2_embeddings.py input.fasta \
  --model-dir models/backbones/esm2_t33_650M_UR50D \
  --model-name esm2_t33_650M_UR50D \
  --output-dir output/features/esm2 \
  --gpus 0 --batch-size 32 --pooling mean \
  --model-dtype float16 --output-dtype float32
```

### 2. ProtBERT

```bash
python scripts/extract_protein_lm_embeddings.py input.fasta \
  --model-dir models/backbones/prot_bert \
  --model-name prot_bert --model-type protbert \
  --output-dir output/features/protbert \
  --gpus 0 --batch-size 16 --pooling mean \
  --model-dtype float16 --output-dtype float32
```

ProtBERT 输入会转换为空格分隔的氨基酸 token；非标准氨基酸会替换为 `X`。

### 3. ProstT5 3Di 与 SaProt

```bash
python scripts/build_saprot_structure_tokens_with_prostt5.py \
  --fasta input.fasta \
  --prostt5_model models/backbones/ProstT5_foldseek \
  --out_tsv output/structure_tokens/input.prostt5_3di.tsv \
  --audit_tsv output/structure_tokens/input.prostt5_3di.audit.tsv \
  --tmp_dir output/foldseek_tmp \
  --foldseek_bin foldseek --threads 16 --gpu

python scripts/extract_protein_lm_embeddings.py input.fasta \
  --model-dir models/backbones/SaProt_650M_AF2 \
  --model-name SaProt_650M_AF2 --model-type saprot \
  --structure-tsv output/structure_tokens/input.prostt5_3di.tsv \
  --output-dir output/features/saprot_prostt5 \
  --gpus 0 --batch-size 8 --pooling mean \
  --model-dtype float16 --output-dtype float32
```

三个提取脚本都会写出 `.npy` 与相匹配的 `.metadata.tsv`。metadata 保存 `feature_row`、`seq_id`、清洗后的序列、截断数量等审计信息。

## 从已有 NPY 推理

修改 `examples/manifest.json`，使三个成员分别指向 `.npy` 和对应的 `.metadata.tsv`，然后运行：

```bash
amp-ensemble --manifest examples/manifest.json --output predictions.tsv
```

三个 metadata 必须包含完全相同且顺序一致的 `seq_id`。默认策略使用各成员独立的交叉验证阈值，再执行 2/3 多数投票。也支持简单软投票：

```bash
amp-ensemble --manifest examples/manifest.json \
  --strategy soft --soft-threshold 0.5 --output predictions.tsv
```

## 模型参数

| 成员 | MLP 架构 | Dropout | 参数量 | CV 阈值 |
|---|---|---:|---:|---:|
| ESM2-MLP | 1280 → 512 → 1 | 0.20 | 656,385 | 0.500 |
| SaProt-ProstT5-MLP | 1280 → 512 → 256 → 128 → 1 | 0.35 | 820,225 | 0.795 |
| ProtBERT-MLP | 1024 → 512 → 256 → 128 → 1 | 0.50 | 689,153 | 0.850 |

完整参数清单见 `docs/MODELS.md` 和 `models/model_manifest.json`。验证已上传 MLP 参数：

```bash
cd models
sha256sum -c SHA256SUMS
```

基础语言模型权重合计约 8.7 GiB，超过 GitHub 单文件 100 MiB 限制，因此不会放入普通 Git 历史；下载脚本使用固定 revision 并校验原始权重哈希，确保输入模型一致。

## 测试

```bash
pip install -e '.[test]'
python -m unittest discover -s tests -v
```

## 注意事项

- 特征提取必须使用 `mean` pooling；更换 pooling 会使 MLP 输入分布不一致。
- SaProt 必须使用与序列逐位对齐的 ProstT5/Foldseek 3Di token。
- checkpoint 只应从可信来源加载；推理代码使用 `weights_only=True`。
- 本模型用于科研预测，最终候选仍需湿实验验证。
