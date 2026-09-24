# Model Weights for Chord Similarity (CS)

This directory contains pretrained model weights for the Chord Similarity (CS) metric.

## Files

### `polydis-v1-chd_encoder_only.pt` (29 MB, Recommended)

Lightweight chord encoder extracted from the PolyDisVAE model.

- **Size**: 28.90 MB (72.3% smaller than full model)
- **Purpose**: Encode 2-bar chord progressions to 256-dim latent vectors
- **Architecture**: Bidirectional GRU (36 → 1024) + Linear (2048 → 256)
- **Parameters**: 7,574,016 (only chord encoder)

### `polydis-v1.pt` (104 MB, Optional)

Full PolyDisVAE model (includes unused components).

- **Size**: 104.19 MB
- **Components**:
  - chd_encoder: 28.89 MB ✓ (used for CS)
  - txt_encoder: 36.14 MB ✗ (unused)
  - pnotree_decoder: 33.58 MB ✗ (unused)
  - chd_decoder: 5.57 MB ✗ (unused)

**Note**: The CS metric only uses `chd_encoder`, so the lightweight version is recommended.

## Download

- Lightweight (29 MB): `polydis-v1-chd_encoder_only.pt`
- Full model (104 MB): `polydis-v1.pt`

Place the downloaded file(s) in this directory (`smg_metrics/model_weights/`).

## Model Training

The chord encoder was trained as part of the EC2-VAE model:

**Reference**:
> Wang, Z., Wang, D., Zhang, Y., and Xia, G. "Learning interpretable representation 
> for controllable polyphonic music generation." Proceedings of 21st International 
> Society for Music Information Retrieval Conference (ISMIR), 2020.
> 
> https://arxiv.org/abs/2008.07122

**Training data**: Not publicly disclosed in the original paper.

**Model architecture**:
- Input: 36-dim chord vectors (Root + Chroma + Bass) over 8 beats (2 bars)
- Encoder: Bidirectional GRU with hidden size 1024
- Output: 256-dim Gaussian distribution (mean used for similarity)

## Usage

The CS metric automatically detects and loads the lightweight model:

```python
from smg_metrics import compute_cs, pair_eval

# Standalone CS metric
cs = compute_cs("generated.mid", "reference.mid")
print(f"Chord Similarity: {cs:.4f}")

# As part of pairwise evaluation
result = pair_eval("generated.mid", "reference.mid")
print(f"CS: {result.cs:.4f}")
```

## License

Model weights inherit the license from the original PolyDisVAE implementation.
Please cite the original paper if you use this metric:

```bibtex
@inproceedings{wang2020learning,
  title={Learning interpretable representation for controllable polyphonic music generation},
  author={Wang, Ziyu and Wang, Dingsu and Zhang, Yixiao and Xia, Gus},
  booktitle={Proceedings of the 21st International Society for Music Information Retrieval Conference},
  year={2020}
}
```
