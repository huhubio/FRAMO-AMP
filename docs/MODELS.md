# Model and parameter inventory

## Frozen feature encoders

The three feature branches use mean pooling over residue tokens and exclude
special/padding tokens. Exact Hub revisions and original weight checksums are
recorded in [`configs/backbones.json`](../configs/backbones.json).

| Branch | Model | Revision | Dimensions | Local weight size |
|---|---|---|---:|---:|
| ESM2 | `facebook/esm2_t33_650M_UR50D` | `08e4846e...` | 1280 | 2.43 GiB |
| ProtBERT | `Rostlab/prot_bert` | `7a894481...` | 1024 | 1.57 GiB |
| SaProt | `westlake-repl/SaProt_650M_AF2` | `d9b9ad00...` | 1280 | 2.43 GiB |
| 3Di generation | Foldseek ProstT5 GGUF | downloaded by Foldseek | n/a | 2.25 GiB |

These frozen backbone files total about 8.7 GiB. They are intentionally not
stored in normal Git because GitHub rejects files larger than 100 MiB. Run
`scripts/download_backbones.py`; it downloads the pinned files and verifies
their byte sizes and SHA-256 hashes. This also avoids redistributing upstream
weights under this repository.

SaProt input is built by pairing every uppercase amino acid with the lowercase
ProstT5-predicted Foldseek 3Di token. The resulting paired tokens are passed to
SaProt, and the last hidden layer is mean pooled over non-special tokens.

## Trained MLP heads

All trainable FRAMO-AMP head parameters are committed under `models/`; no MLP
parameters are omitted. Each checkpoint contains a PyTorch `state_dict`.

| Head | Architecture | Dropout | Parameters | Decision threshold |
|---|---|---:|---:|---:|
| ESM2-MLP | 1280 → 512 → 1 | 0.20 | 656,385 | 0.500000 |
| SaProt-ProstT5-MLP | 1280 → 512 → 256 → 128 → 1 | 0.35 | 820,225 | 0.795000 |
| ProtBERT-MLP | 1024 → 512 → 256 → 128 → 1 | 0.50 | 689,153 | 0.850000 |

Each hidden layer is `Linear → ReLU → Dropout`; the output is one logit and is
converted to a probability with sigmoid. The per-feature standardization
arrays fitted on 30,687 training examples are committed as compressed NPZ
files. Full tuning configurations and cross-validation results are committed
as the corresponding `*_config.json` files.

Machine-readable architecture and checkpoint hashes are in
[`models/model_manifest.json`](../models/model_manifest.json). Verify every
committed head artifact from the `models` directory with:

```bash
cd models
sha256sum -c SHA256SUMS
```

## Ensemble

Each head first applies its own cross-validation threshold. The final label is
AMP when at least two of the three heads predict AMP. Thus the majority-vote
score is one of `0`, `1/3`, `2/3`, or `1`.
