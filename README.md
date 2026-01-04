# MMLoRAT: A Multi-Modal Version of LoRAT

## News
**[Jan. 4, 2026]**
* Add online template enable/disable support.

**[Dec. 23, 2025]**
* We have released the code of MMLoRAT.

**[Dec. 11, 2025]**
* Hunting for an ultra-efficient, high-performance RGB-T tracker? We have released the code for our [GOLA](https://github.com/MelanTech/GOLA), which is accepted by AAAI 2026!

## Performance
This framework is currently being refined, and the trained model will be released soon.

## Prerequisites
### Environment
Assuming you have a `Python 3.10.15` environment with pip installed.

#### system packages (ubuntu)
```shell
apt update
apt install -y libturbojpeg
```
#### install pytorch
```shell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

#### extra python packages
```shell
pip install -r requirements.txt
```
This codebase should also work on Windows and macOS for debugging purposes.

### Dataset
The paths should be organized as follows:
```
-- LasHeR/trainingset
    |-- 1boygo
    |-- 1handsth
    ...
```

### Prepare ```consts.yaml```
Fill in the paths.
```yaml
LasHeR_PATH: '/path/to/LasHeR0428/'
RGBT234_PATH: '/path/to/RGBT234/'
RGBT210_PATH: '/path/to/RGBT210/'
GTOT_PATH: '/path/to/GTOT/'
```

## Quick Start
Note: Our code performs evaluation automatically when model training is complete.

* **Model weight** is saved in ```/path/to/output/run_id/checkpoint/epoch_{last}/model.bin```.
* **Performance metrics** can be found on terminal output.
* **Tracking results** are saved in ```/path/to/output/run_id/eval/epoch_{last}/```.
  
### Preparation for pretrained models
* Download [pretrained models](https://drive.google.com/drive/u/0/folders/1pFGegBmVqPZ2iEYYTAJCIiW5RZOxLqjN) and put them in the `pretrained_models` folder.
* If you want to train the tracker without online template, download weights with the `wo_ot` suffix.

### Training
* Using `run.sh` for training and evaluation (Linux with NVIDIA GPU only)

```shell
# MMLoRAT-B-224
bash run.sh MMLoRAT dinov2 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# MMLoRAT-B-378
bash run.sh MMLoRAT dinov2 --mixin_config base_378 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# MMLoRAT-L-224
bash run.sh MMLoRAT dinov2 --mixin_config large --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# MMLoRAT-L-378
bash run.sh MMLoRAT dinov2 --mixin_config large_378 --mixin_config base_378 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# MMLoRAT-G-224
bash run.sh MMLoRAT dinov2 --mixin_config giant --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# MMLoRAT-G-378
bash run.sh MMLoRAT dinov2 --mixin_config giant_378 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Evaluate all datasets after training.
bash run.sh MMLoRAT dinov2 --mixin_config test-full --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Evaluate RGBT234 after training.
bash run.sh MMLoRAT dinov2 --mixin_config test-rgbt234 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Evaluate RGBT210 after training.
bash run.sh MMLoRAT dinov2 --mixin_config test-rgbt210 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Evaluate GTOT after training.
bash run.sh MMLoRAT dinov2 --mixin_config test-gtot --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb
```

### Evaluation
You can use the following command to inference and evaluate separately:

```shell
# Inference and evaluate on LasHeR.
bash run.sh MMLoRAT dinov2 --mixin_config evaluation --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Inference and evaluate on RGBT234.
bash run.sh MMLoRAT dinov2 --mixin_config evaluation --mixin_config test-rgbt234 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Inference and evaluate on RGBT210.
bash run.sh MMLoRAT dinov2 --mixin_config evaluation --mixin_config test-rgbt210 --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Inference and evaluate on GTOT.
bash run.sh MMLoRAT dinov2 --mixin_config evaluation --mixin_config test-gtot --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb

# Inference and evaluate all datasets.
bash run.sh MMLoRAT dinov2 --mixin_config evaluation --mixin_config test-full --output_dir '/path/to/output' --weight_path '/path/to/weights' --device_ids 0,1 --disable_wandb
```

You can evaluate the output results separately through the following steps:
1. Unzip the tracking results to a folder of your choice.

2. Run the evaluation script `evaluation.py`.
```shell
# Evaluate on LasHeR
python evaluation.py lasher --tracker_names MMLoRAT --result_paths /path/to/tracking/results

# Evaluate on RGBT234
python evaluation.py rgbt234 --tracker_names MMLoRAT --result_paths /path/to/tracking/results

# Evaluate on RGBT210
python evaluation.py rgbt210 --tracker_names MMLoRAT --result_paths /path/to/tracking/results

# Evaluate on GTOT
python evaluation.py gtot --tracker_names MMLoRAT --result_paths /path/to/tracking/results
```

### Profile Model
* Using `profile_model.py` for model profiling.
```shell
# MMLoRAT-B-224
python profile_model.py MMLoRAT dinov2 --device cuda
# MMLoRAT-B-378
python profile_model.py MMLoRAT dinov2 --mixin_config base_378 --device cuda
...
```

### Disable Online Template
You can add `--mixin_config disable_online_template` in the command to disable the online template.

## Acknowledgements
- This repo is based on [LoRAT](https://github.com/LitingLin/LoRAT), we thank for it's `trackit` framework, which helps us to quickly implement our ideas.
- We thank the [rgbt](https://github.com/opacity-black/RGBT_toolkit) library for facilitating evaluation in a Python environment.

## Citation
If you find this code useful in your research, please consider citing:
```
@inproceedings{gola,
  title={Group Orthogonal Low-Rank Adaptation for RGB-T Tracking},
  author={Shao, Zekai and Hu, Yufan and Liu, jingyuan and Fan, Bin and Liu, Hongmin},
  booktitle={AAAI},
  year={2026}
} 
```