# UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior

Official repository for APGCC: Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance.

[Project Page](https://unirestore.github.io) | [Paper](https://arxiv.org/abs/2501.13134) | [Video](https://www.youtube.com/watch?v=Jm1NkDDXN90) | [Code](https://github.com/unirestore/UniRestore)

## Updates
- June 2025: ✨ Source code has been released!
- June 2025: ✨ UniRestore was accepted into CVPR 2025 Highlight!

## UniRestore
**UniRestore** leverages diffusion prior to unify ![PIR](https://img.shields.io/badge/Perceptual_Image_Restoration-(PIR)-blue?style=flat-square)
and ![TIR](https://img.shields.io/badge/Task_oriented_Image_Restoration-(TIR)-yellow?style=flat-square), achieving both high visual fidelity and task utility.
- **PIR** enhances visual clarity, but its outputs may not benefit recognition tasks.  
- **TIR** optimizes features for tasks like classification or segmentation, but often compromises visual appeal.
![UniRestore Demo](./assets/teaser.gif)

## Setup
1) Create a conda environment and activate it.
```
conda create -n unirestore python=3.11 -y
conda activatre unirestore
```
2) Clone and enter into repo directory.
```
git clone https://github.com/unirestore/UniRestore.git
cd UniRestore
```
3) Install remaining dependencies
```
pip install -r requirements.txt
```
4) Download pretrained UniRestore checkpoints and place them into path (./UniRestore/logs/).
- [Stage1 checkpoint](https://drive.google.com/file/d/1a7c8zL8XXd7m3dDEQWZMpQagnkBpdGnK/view?usp=share_link)
- [Stage2 checkpoint](https://drive.google.com/file/d/1m2a-8SUtVZaG5ovysJKWqc_mzgGCtDrx/view?usp=share_link)
- You can also complete by the command:
```
cd UniRestore
wget --no-check-certificate 'https://drive.google.com/file/d/1a7c8zL8XXd7m3dDEQWZMpQagnkBpdGnK/view?usp=share_link' -O ./logs/unirestore_stage1.ckpt
wget --no-check-certificate 'https://drive.google.com/file/d/1m2a-8SUtVZaG5ovysJKWqc_mzgGCtDrx/view?usp=share_link' -O ./logs/unirestore_stage2.ckpt
```

## Prepare Dataset
We use a JSON file to collect all the degraded images, ground truth images, and the corresponding annotations.
Each entry in the JSON contains a row formatted as:
"path_to_lq path_to_hq annotation".
Please download the dataset into the corresponding task folder, and then use the following command to create the data list:
```
cd UniRestore/dataset/$[task]
python process_$[dataset].py
```
where \[task\] refers to the task directory (e.g., 'PIR', 'Classification', 'Segmentation'), and the \[dataset\] specifies the dataset to be processed.

## Training Pipeline
Our method consists of two main training stages:
### 📌 Stage 1: Feature Restoration
Focus on restoring features by training the `CFRM`, `Controller`, and `SC-Tuner` modules.
```
python src/main.py fit --config ./configs/train_stage1.yaml
```
### 📌 Stage 2: Task Adaptation (recommend)
Adapt restored features and diffusion priors for specific downstream tasks by training the `TFA` module.
```
python src/main.py fit --config ./configs/train_stage2.yaml
```
<mark>💡 It is recommended to initialize Stage 2 with the provided pretrained checkpoint unirestore_s1.ckpt for better adaptation to each task.<mark>

### ➕ Add New Tasks Easily (recommend)
To introduce a new task, simply define new task-specific prompts and fine-tune without modifying the main model or accessing previous training data.
1. Add your custom task objective at: ./UniRestore/src/core/base
2. Modify the `tedit: task` field to add your new task-specific prompt words at configuration.
3. Implement a new data loader at: ./UniRestore/src/data
4. Run fine-tuning with:
```
python src/main.py fit --config ./configs/train_stage3.yaml
```
<mark>💡 This process enables flexible and efficient task extension without retraining the full model or accessing previous data.<mark>

## Inference
- For validation dataset:
```
python ./src/main.py validate --config ./configs/val.yaml --trainer.logger null
```
By default, the results will be saved to "./logs/unirestore/test". You can customize the inference detail by modifying the configuration file.

## Reference
If you find this work useful, please consider citing us!
```
@inproceedings{chen2025unirestore,
  title={UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior},
  author={Chen, I and Chen, Wei-Ting and Liu, Yu-Wei and Chiang, Yuan-Chun and Kuo, Sy-Yen and Yang, Ming-Hsuan and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17969--17979},
  year={2025}
}
```