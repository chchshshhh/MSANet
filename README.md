# MSANet

Code release for **A Novel Multi-view Perception and Shrinkage Aggregation Network for Inharmonious Region Localization**.

- Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11180079)
- DOI: [10.1109/TCSVT.2025.3614236](https://doi.org/10.1109/TCSVT.2025.3614236)
- Journal: IEEE Transactions on Circuits and Systems for Video Technology, 36(2):1795-1809, 2026

## Release scope

This repository is organized around the MSANet training and testing code. It includes the experiment scripts, model definitions, dataset loaders, losses, and utility code required by the released MSANet family.

Included:

- Primary training entry: `MSANet_train.py`
- Primary validation/testing entry: `MSANet_test.py`
- Additional MSANet variants: `MSANet_*.py`
- Archived/alternative MSANet variants: `qita/MSANet_*.py`
- Model code: `src/`
- Dataset loaders and transforms: `dataset/`
- Training and evaluation helpers: `train_utils/`, `train_utilsx/`
- Losses and metric helpers: `loss/`, `pytorch_iou/`, `pytorch_ssim/`
- Config files: `config/`, `configs/`

Not included:

- Model weights and checkpoints (`*.pth`, `*.pt`, `*.ckpt`, `*.pth.tar`, `checkpoint*`)
- Runtime cache files and generated Python bytecode

## Pretrained Weights

Final TCSVT/MSANet weights are available on [Google Drive](https://drive.google.com/drive/folders/10q_GoF1yPN4DBhQDmdQ_jZmfPz7vvyfJ?usp=sharing).

Download the checkpoint locally and pass it to the testing script with `--resume`.

## Usage

Install the required Python packages in an environment with PyTorch and CUDA support. The scripts use packages such as `torch`, `torchvision`, `numpy`, `Pillow`, `opencv-python`, and `albumentations`.

Training:

```bash
python MSANet_train.py --device cuda --batch-size 16 --epochs 35
```

Validation/testing with a local checkpoint:

```bash
python MSANet_test.py --device cuda --resume /path/to/model-34.pth
```

Before running on a new machine, update the dataset paths in the dataset loader or scripts to match your local HDobe5K, HCOCO, HFlickr, and Hday2Night locations.

## Citation

```bibtex
@article{chen2026msanet,
  title={A Novel Multi-view Perception and Shrinkage Aggregation Network for Inharmonious Region Localization},
  author={Chen, Shenghao and Ma, Chunjie and Zhao, Yibo and Liu, Meng and Xue, Yanbing and Gao, Zan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={36},
  number={2},
  pages={1795--1809},
  year={2026},
  doi={10.1109/TCSVT.2025.3614236}
}
```
