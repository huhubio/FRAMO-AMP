# AMP 3-MLP Ensemble

基于三种蛋白语言模型均值特征的抗菌肽（AMP）二分类集成：

- ESM2 `esm2_t33_650M_UR50D`（1280 维）+ MLP
- SaProt `SaProt_650M_AF2` / ProstT5 结构 token（1280 维）+ MLP
- ProtBERT（1024 维）+ MLP

默认决策规则是 **2-of-3 多数投票**。项目包含三个已训练 MLP 的 checkpoint、训练集拟合的标准化参数以及各模型独立的交叉验证阈值，因此推理时不再依赖原始训练矩阵，也不包含写死的本机路径。

## 安装

需要 Python 3.10 或更高版本：

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd amp_3mlp_ensemble
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

PyTorch 会自动使用可用的 CUDA；也可以用 `--device cpu` 或 `--device cuda:0` 显式指定。

## 输入

本项目从**已经提取好的均值 embedding** 开始推理。每个成员需要：

1. 一个形状为 `(N, D)` 的 float `.npy` 特征矩阵；
2. 一个 TSV metadata 文件，至少包含 `feature_row` 和 `seq_id`，可选 `sequence`；
3. 三个模型必须包含完全相同且顺序一致的 `seq_id`。

metadata 示例：

```text
feature_row\tseq_id\tsequence
0\tpeptide_001\tGIGKFLHSAKKFGKAFVGEIMNS
1\tpeptide_002\tKWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK
```

大体积 embedding 没有提交到 Git；请按 `examples/manifest.json` 中的路径放到 `data/query/`，或修改 manifest 指向自己的文件。模型配置、权重和 scaler 已位于 `models/`。

## 推理

```bash
amp-ensemble \
  --manifest examples/manifest.json \
  --output predictions.tsv
```

CPU 推理：

```bash
amp-ensemble --manifest examples/manifest.json --device cpu --output predictions.tsv
```

默认 `--strategy majority` 使用各成员自己的验证阈值，再进行 2/3 投票。也可进行未校准概率的简单软投票：

```bash
amp-ensemble --manifest examples/manifest.json --strategy soft --soft-threshold 0.5
```

输出 TSV 包含每个成员的概率、阈值、二分类结果，以及最终集成分数、类别和标签。多数投票模式下 `ensemble_score` 是阳性票比例（0、1/3、2/3 或 1）。

## Python API

```python
from pathlib import Path
from amp_ensemble.cli import load_manifest
from amp_ensemble import EnsemblePredictor

members = load_manifest(Path("examples/manifest.json"))
predictor = EnsemblePredictor(members, device="auto")
metadata, member_outputs, ensemble_score, prediction = predictor.predict()
```

## 模型阈值

| 成员 | 隐藏层 | Dropout | CV 阈值 |
|---|---:|---:|---:|
| ESM2-MLP | 512 | 0.20 | 0.500 |
| SaProt-ProstT5-MLP | 512, 256, 128 | 0.35 | 0.795 |
| ProtBERT-MLP | 512, 256, 128 | 0.50 | 0.850 |

这些阈值来自训练流程的五折交叉验证，不应使用待预测数据重新调参。若不同数据域需要概率校准，应单独保留不参与训练的验证集。

## 测试

```bash
pip install -e '.[test]'
pytest
```

## 注意事项

- SaProt 分支依赖 ProstT5/结构 token 的具体预处理方法；新数据必须使用与训练时相同的提取流程。
- checkpoint 只应从可信来源加载。本项目使用 PyTorch 的 `weights_only=True` 限制反序列化范围。
- 该模型用于科研预测，不替代湿实验验证。
