## Multistage Modular Medical Models (M4)

[ENGLISH](README_EN.md) | [中文](README.md)

This project focuses on "Beyond Scaling Law, emphasizing data efficiency and reliability" as its core research theme. It aims to establish a modular visual foundation model (M4, Modular Medical Foundation Models) suitable for medical imaging, addressing critical challenges such as data scarcity, high annotation costs, and privacy constraints in real-world medical scenarios.

Traditional AI development relies on massive data and model scaling, which is often impractical in the medical field. This project adopts a "strategy over scale" approach, combining Self-Supervised Learning, Multi-stage Transfer Learning, and Task-specific Fine-Tuning to enable AI to gradually learn and develop clinically valuable medical semantic understanding capabilities with minimal expert-annotated data.

## Notice and License Statement

This project is based on the following code:

Source: https://github.com/rmaphoh/RETFound

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Modifications:
- Added support for Pixio-L/16, mae_pretrain_vit_large, and vit-large-patch16-224.
- Added scripts for automated experiment execution.
- Changed to use 5-fold validation for experimental evaluation.
- Added Supervised Fine-Tuning scripts.

## Install environment

1. Create a conda environment

```
conda create -n M4 python=3.11.0 -y
conda activate M4
```

2. Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Baseline Models

The following 10 models are used as baselines. Except for the RETFound model, please download the model weights and place them in the `baseline_models` folder. For the RETFound model, refer to the [official GitHub](https://github.com/rmaphoh/RETFound).

- [Dinov2 ViT-L/14 distilled (Meta, 2023)](https://github.com/facebookresearch/dinov2)
- [dinov3-vitl16-pretrain-lvd1689m (Meta, 2025)](https://github.com/facebookresearch/dinov3)
- [Pixio-L/16 (Meta, 2025)](https://github.com/facebookresearch/pixio)
- [mae_pretrain_vit_large (Meta, 2021)](https://github.com/facebookresearch/mae)
- [vit-large-patch16-224 (Google, 2020)](https://github.com/huggingface/pytorch-image-models)
- [RETFound_mae_natureCFP (Zhou et al., 2023)](https://github.com/rmaphoh/RETFound)
- [RETFound_mae_meh (Zhou et al., 2023)](https://github.com/rmaphoh/RETFound)
- [RETFound_mae_shanghai (Zhou et al., 2023)](https://github.com/rmaphoh/RETFound)
- [RETFound_dinov2_meh (Zhou et al., 2025)](https://github.com/rmaphoh/RETFound)
- [RETFound_dinov2_shanghai (Zhou et al., 2025)](https://github.com/rmaphoh/RETFound)

### Adaptation pipelines

![[assets/Adaptation-pipelines.png]](assets/Adaptation-pipelines.png)

In addition to directly applying natural image pre-trained models to downstream tasks, we also evaluate three adaptation strategies:
1. DAP (DAP directly uses model weights released by the official RETFound repository)
2. SFT (SFT additionally performs supervised fine-tuning on medical image or natural image datasets)
3. DAP + SFT (apply DAP first, then SFT)


## Dataset

### Fundus Datasets

| Dataset Name       | Size         | Source                                                                                             |
| :----------------- | :----------- | :------------------------------------------------------------------------------------------------ |
| APTOS2019          | 3,662 images | [Kaggle Link](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)             |
| MESSIDOR2          | 1,748 images | [ADCIS Link](https://www.adcis.net/en/third-party/messidor2/)                                     |
| Glaucoma_fundus    | 1,544 images | [Harvard Dataverse Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC) |
| Retina (Cataract)  | 601 images   | [Kaggle Link](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)                             |
| IDRID              | 516 images   | [IEEE Dataport Link](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| PAPILA             | 488 images   | [Figshare Link](https://figshare.com/articles/dataset/PAPILA/14798004/1)                          |

[Download 5-fold split method](https://drive.google.com/drive/folders/11euRYSptEpS9UQDAVP3UJ4vi4RINHvzx?usp=sharing)

After downloading the dataset, split it according to the provided 5-fold list and place it in the `data` folder. For example, place fold1 of the ATPOS2019 dataset in `data/5_fold_APTOS2019/APTOS2019_seed42_fold0`.

### Supervised Fine-Tuning Datasets

| Dataset Name       | Size         | Source                                                                                             |
| :----------------- | :----------- | :------------------------------------------------------------------------------------------------ |
| Augmented Ocular Disease (AOD) | 14,105 images | [GitHub Link](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/AOD.md) |
| Imagenet-1k        | 1,281,167 images | [Huggingface Dataset Link](https://huggingface.co/datasets/ILSVRC/imagenet-1k)                   |

After downloading, place the datasets in the `data` folder. Preprocess Imagenet-1k into the following format. By default, all data is used as training data for SFT experiments.

```
imagenet/  
├── train/  
│ ├── class_0/  
│ │ ├── img_0001.jpg  
│ │ └── img_0002.jpg  
│ └── class_1/  
│ ├── img_0005.jpg  
│ └── img_0006.jpg  
├── val/  
│ ├── class_0/  
│ │ ├── img_0007.jpg  
│ │ └── img_0008.jpg  
│ └── class_1/  
│ ├── img_0011.jpg  
│ └── img_0012.jpg  
└── test/  
	├── class_0/  
	│ ├── img_0013.jpg  
	│ └── img_0014.jpg  
	└── class_1/  
	├── img_0017.jpg  
	└── img_0018.jpg
```

## Run training

```python
# Run the following command to test downstream tasks
python script.py

# Run the following command for SFT training
python SFT_script.py
```

## Exp1: Performance of baseline models

In Exp1, we compare the performance of Natural Image pre-trained models and RETFound DAP models across 6 datasets.

Models without adaptation include:
- Dinov2
- Dinov3
- Pixio
- MAE_pretrain_vit_large
- Vit-large-patch16-224

Models with DAP include:
- RETFound_mae_natureCFP
- RETFound_mae_meh
- RETFound_mae_shanghai
- RETFound_dinov2_meh
- RETFound_dinov2_shanghai

### Experimental Setup

- **5-fold:** Each dataset undergoes 5-fold testing. Each model is trained once for each of the 5 folds. The model with the highest validation score for each fold is used for test set evaluation. The average and standard deviation of the test scores across the 5 folds are used as the evaluation criteria.
- **Validation details:** During fine-tuning on downstream tasks, validation is performed once per epoch. The checkpoint with the highest f1+auroc+kappa score is selected as the best validation and used for test set evaluation.
- **Hyper-parameters:** The following three hyper-parameter settings are used for testing, and the best-performing results are reported in the experimental results table.
   1. [RETFound default hyper-parameters](https://github.com/rmaphoh/RETFound)
   2. [RetFound paper settings (2025 version)](https://arxiv.org/abs/2509.03421)
   3. [Meta MAE fine-tune settings](https://github.com/facebookresearch/mae)

Key differences in hyper-parameters are as follows:

| Hyper-parameter | RETFound Default | RETFound Paper | MAE Default |
| :-------------- | :--------------- | :------------- | :---------- |
| **Batch Size**  | 24              | 24            | 64          |
| **Base Learning Rate** | 5e-3       | 5e-4         | 1e-3        |
| **Drop Path**   | 0.2             | 0.2           | 0.1         |
| **Layer Decay** | 0.65            | 0.65          | 0.75        |

The effects of different hyper-parameters using RETFound_dinov2_meh are compared in the table below. **Smaller learning rate settings** may yield better performance.

| Model / Setting            | APTOS2019 (3662) | MESSIDOR2 (1744) | Glaucoma Fundus (1544) | Retina (601)    | IDRiD Data (516) | PAPILA (488)    |
| :------------------------- | :--------------- | :--------------- | :--------------------- | :-------------- | :--------------- | :-------------- |
| **RETFound Default**       | 85.13 ±0.56      | 76.31 ±1.09      | 86.45 ±0.73            | **73.04 ±3.28** | 62.33 ±3.02      | 81.84 ±3.34     |
| **RETFound Paper**         | **85.24 ±0.08**  | 76.84 ±1.08      | 82.84 ±0.88            | 70.50 ±3.28     | **65.63 ±2.96**  | **84.08 ±3.19** |
| **MAE Default**            | 85.16 ±0.67      | **77.07 ±0.88**  | **86.92 ±0.63**        | 71.16 ±1.72     | 62.91 ±4.63      | 82.65 ±3.31     |

### Experimental results

**Baseline performance comparison (1150521)**
![[assets/Exp1-1.png]](assets/Exp1-1.png)

**Baseline performance comparison (1150521)**
![[assets/Exp1-2.png]](assets/Exp1-2.png)

## Summary

- RetFound dinov2 meh significantly outperforms other models.
- Among natural image models, dinov3 performs the best, followed by dinov2.
- Models pre-trained using MAE-based methods, whether natural image models or RetFound's continual pretraining, perform worse than the Dino series.

## Exp2: Supervised Fine-Tuning (SFT)

In Exp2, we add SFT to fine-tune Natural Image pre-trained models and RETFound DAP models on the Imagenet-1k and AOD datasets, and observe how SFT affects downstream performance under different pre-training workflows.

SFT models include:
- Dinov2
- Dinov3
- Pixio
- MAE_pretrain_vit_large

DAP + SFT models include:
- RETFound_dinov2_meh
- RETFound_dinov2_shanghai


### Experimental Setup

- Downstream task testing is the same as Exp1, but only **RETFound default hyper-parameters** are used.
- **Meta MAE fine-tune** hyper-parameters are used for SFT fine-tuning.
- The AOD dataset is fine-tuned for 30 epochs, with downstream task testing every 2 epochs. The Imagenet-1k dataset is fine-tuned for 1 epoch, with only one downstream task test after fine-tuning.
- The comparison values below are all accuracies.

### Experimental results

- SFT on AOD Dataset

	**Supervised Fine-Tuning on AOD dataset (1150521)**
	![[Exp2-1_AOD.png]](assets/Exp2-1_AOD.png )

- SFT on Imagenet-1k Dataset

	**Comparison of SFT on Imagenet-1k dataset (1150521)**
	| Model\Dataset                    | APTOS2019  | Glaucoma fundus | MESSIDOR2 | Retina     | IDRID_Data | PAPILA     |
	| -------------------------------- | ---------- | --------------- | --------- | ---------- | ---------- | ---------- |
	| **RETFound dinov2 (meh) w/o sft** | **0.8513** | **0.8572**      | 0.7631    | 0.7304     | 0.5942     | 0.8184     |
	| **DINOv2**                       | 0.8456     | 0.8013          | **0.7681**| 0.7138     | 0.5786     | 0.7592     |
	| **DINOv3**                       | 0.8396     | 0.8370          | 0.7452    | 0.7227     | 0.5845     | 0.7918     |
	| **Pixio**                        | 0.8347     | 0.7871          | 0.7308    | 0.6475     | 0.4796     | 0.7306     |
	| **RetFound (meh)**               | 0.8451     | 0.8417          | 0.7669    | **0.7448** | **0.6019** | **0.8204** |
	| **RetFound (shanghai)**          | 0.8484     | 0.8529          | 0.7297    | 0.6895     | 0.5942     | 0.7959     |

### Computational cost

According to the [RETFound paper](https://www.researchsquare.com/article/rs-6080254/v1), the resource costs for continual pretraining and SFT are compared below (using dinov2 large as the backbone model):

|Method|Dataset|Training epoch|GPU|Cost Time|
|-|-|-|-|-|
|Continual pretraining|Moorfields Eye Hospital (MEH)|unknown|A100\*4|16 days|
|SFT|Augmented Ocular Disease (AOD)|30|V100\*1|6 hours|
|SFT|Imagenet-1k|1|V100\*1|4 hours|

## Summary

- Overall, whether fine-tuned on the AOD or Imagenet-1k datasets, dinov2 and RetFound (meh) outperform other models.
- Fine-tuning on the AOD dataset achieves higher scores across all datasets compared to Imagenet-1k, and **models trained on Imagenet-1k do not surpass SOTA performance**.
- Although dinov3 performs better than other natural image models without SFT, after SFT, it performs significantly worse than dinov2 on all datasets except Glaucoma_fundus.
- Comparing the original RETFound_dinov2_meh (epoch 0) with the dinov2 model, even using the significantly lower-cost SFT method, better performance can be achieved on all datasets except Glaucoma fundus and PAPILA, demonstrating that the SFT method can achieve results close to continual pretraining with lower costs.