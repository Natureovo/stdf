# Script Guide

This project now has three practical workflows.

## Core STDF Baseline

- `train.py`: train the original STDF/MFVQE model.
- `test.py`: evaluate the original STDF/MFVQE model.
- `test_one_video.py`: run STDF on one YUV video and save an enhanced YUV file.

## Detail-Guided Analysis

- `analyze_detail_loss.py`: analyze detail loss between two image files.
- `analyze_detail_loss_yuv.py`: extract paired raw/QP37 YUV frames and analyze compression detail loss.
- `analyze_stdf_detail_compare.py`: compare raw, compressed, and STDF outputs; this is the main analysis script for building diffusion guidance.

Default analysis outputs are written under:

```text
outputs/detail_loss/
outputs/detail_compare/
```

## Hybrid STDF + GRDR Diffusion

- `train_hybrid_grdr.py`: freeze a trained STDF checkpoint and train only the GRDR residual diffusion branch.
- `test_hybrid_grdr_one_video.py`: test STDF + guidance + GRDR on one YUV video.
- `check_grdr.py`: quick sanity check for the GRDR module.
- `check_hybrid_parts.py`: quick sanity check for the hybrid guidance and diffusion parts.

Default hybrid outputs are written under:

```text
outputs/hybrid_grdr/
```

## Dataset Utilities

- `create_lmdb_mfqev2.py`: create MFQEv2 LMDB files.
- `create_lmdb_vimeo90k.py`: create Vimeo90K LMDB files.

## Notes

- `net_stdf.py` is the original STDF/MFVQE model.
- `net_grdr.py` is the guided residual diffusion refinement module.
- `net_hybrid.py` connects STDF, detail guidance, and GRDR.
- `utils/detail_loss.py` is the offline gradient/frequency detail-loss analyzer.
- `utils/detail_guidance.py` is the PyTorch training-time guidance-map implementation.
