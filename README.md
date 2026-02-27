# Multi-level and Multi-modal Action Anticipation (m&m-Ant)

**Paper (arXiv):** https://arxiv.org/abs/2506.02382  
**Project Page & Demo:** https://alregib.ece.gatech.edu/multi-level-and-multi-modal-action-anticipation/

## TL;DR
Action anticipation predicts future actions from partially observed videos, which is difficult due to incomplete context and inherent uncertainty. This work introduces **m&m-Ant**, a **multi-modal** (video + text) and **multi-level (hierarchical)** framework that improves long-term action anticipation by:
- combining visual cues with textual information,
- explicitly modeling hierarchical semantics,
- and introducing a **fine-grained text generator** trained with a **temporal consistency loss** to mitigate errors from coarse/inaccurate action labels.

Experiments on **Breakfast**, **50Salads**, and **DARai** show consistent improvements over prior methods, reporting an average anticipation accuracy gain of **+3.08%**. :contentReference[oaicite:0]{index=0}

## Run
To run the code:
```bash
python3 main.py
```

## Citation
```
@inproceedings{kim2025multi,
  title={Multi-level and Multi-modal Action Anticipation},
  author={Kim, Seulgi and Kaviani, Ghazal and Prabhushankar, Mohit and AlRegib, Ghassan},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  pages={265--270},
  year={2025},
  organization={IEEE}
}
```
