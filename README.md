# MVFDA: Generalisable Multi-View Detection via Frequency Domain Augmentation (https://doi.org/10.1007/s00138-026-01917-7)
Official code for **MVFDA**, published in *Machine Vision and Applications* (2026).

```bibtex
@article{Zhang2026MVFDA,
  title   = {MVFDA: generalisable multi-view detection via frequency domain augmentation},
  author  = {Zhang, Jining and Xu, Ming and Ling, Yuchen and Al-Nuaimy, Waleed and Pan, Xiaonan},
  journal = {Machine Vision and Applications},
  volume  = {37},
  number  = {6},
  pages   = {152},
  year    = {2026},
  doi     = {10.1007/s00138-026-01917-7}
}
```

Training is recommended on a GPU with memory comparable to or larger than an RTX 4090.

## Dependencies

The programme uses the following libraries:  
python 3.7+  
pytorch 1.7.1 & tochvision 0.8.2  
numpy  
matplotlib  
pillow  
opencv-python  
kornia  
motmetrics  
matlab & matlabengine

Also you should go to `multiview_detector/models/ops` and run `bash mask.sh` to build the deformable transformer (forked from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)).

## Data Preparation

Download the datasets from their official sources and place them under `Data/`:

| Dataset | Link |
|---------|------|
| Wildtrack | https://www.epfl.ch/labs/cvlab/data/data-wildtrack/ |
| MultiviewX | https://github.com/hou-yz/MVDet/ |
| MVPerception (Day-to-Night) | https://drive.google.com/file/d/110ZbfWw2oDv0arC1Dzha_O7J-rjzZVvq/view?usp=sharing|

Expected layout:
```
Data/
├── MultiviewX/
├── Wildtrack/
└── MVPerception/
```

## Training

```bash
# Wildtrack
python main.py -d wildtrack --mvfda_beta 0.1 --mvfda_prob 0.75 --mvfda_ref_mode maxdiff

# MultiviewX
python main.py -d multiviewx --mvfda_beta 0.1 --mvfda_prob 0.75 --mvfda_ref_mode maxdiff

# MVPerception
python main.py -d mvperception --mvfda_beta 0.1 --mvfda_prob 0.75 --mvfda_ref_mode maxdiff
```

Key MVFDA arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--mvfda_beta` | `0.1` | Low-frequency replacement ratio (`0` disables MVFDA) |
| `--mvfda_prob` | `0.75` | Per-view probability of applying MVFDA (`0` disables) |
| `--mvfda_ref_mode` | `maxdiff` | Reference strategy: `random`, `maxdiff`, or `maxdiff_ssim` |
| `--save_augmented_imgs` | `False` | Save augmented images under the run log directory |

Logs and checkpoints are written to `logs/<dataset>-<timestamp>/`.

## Testing & Cross-Dataset Evaluation

After training, `main.py` evaluates the trained model on the corresponding test split.

To evaluate MultiviewX-trained checkpoints on Wildtrack (cross-dataset):

```bash
# Place MultiviewX epoch checkpoints under multiviewx_epoch/
python cross_dataset_test_wildtrack.py
```

Results are saved under `test_results/`.

## Pre-Trained Models
The pre-trained models can be download from the [https://drive.google.com/file/d/1ajChzNd_DpGr_v4VMTg8jVj1pX_o4rxO/view?usp=drive_link].
