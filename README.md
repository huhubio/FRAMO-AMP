# FRAMO-AMP: FRAMework for Overall AMP Discovery

## ESM2 / SaProt-ProstT5 / ProtBERT 3-MLP Ensemble

这是一个从蛋白质 FASTA 到抗菌肽（AMP）预测结果的完整流程：

```text
FASTA
 ├─ ESM2 mean embedding ──────────────── ESM2-MLP ───────┐
 ├─ ProtBERT mean embedding ──────────── ProtBERT-MLP ───┼─ 2/3 majority vote
 └─ ProstT5 3Di → SaProt mean embedding ─ SaProt-MLP ─────┘
```

仓库同时支持两种用法：

- 从 FASTA 开始，自动生成三套 `.npy` 特征并完成集成预测；
- 输入已经提取好的三套 `.npy`，直接运行三个 MLP 和多数投票。

三个训练完成的 MLP checkpoint、训练集 scaler、模型配置、阈值和 SHA-256 校验值均已包含在 `models/` 中。

## 安装

需要 Python 3.10+、CUDA 版 PyTorch 和 NVIDIA GPU。SaProt 的 3Di 生成还需要支持 ProstT5 的 Foldseek 10 或更高版本。

```bash
git clone https://github.com/huhubio/FRAMO-AMP-FRAMework-for-overall-AMP-discovery-.git
cd FRAMO-AMP-FRAMework-for-overall-AMP-discovery-
python -m venv .venv
source .venv/bin/activate
pip install -e '.[extract]'
```

Foldseek 请参考其官方安装方式；安装后确认：

```bash
foldseek version
```

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
