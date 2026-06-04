## Multistage Modular Medical Models (M4)

[ENGLISH](README_EN.md) | [中文](README.md)

本計畫以「超越 Scaling Law，以資料效率與可信賴性為核心」為研究主軸，致力於建立一套適用於醫療影像的模組化視覺基礎模型（M4, Modular Medical Foundation Models），回應真實醫療場域中資料稀缺、標註成本高與隱私限制等關鍵挑戰。

傳統 AI 發展仰賴海量資料與模型規模擴張，但在醫療領域中，此路徑往往難以落實。本計畫採取「策略優於規模」的思維，結合自監督學習（Self-Supervised Learning）、多階段轉移學習（Multi-stage Transfer Learning）與任務導向微調（Task-specific Fine-Tuning），讓 AI 能在僅仰賴少量專家標註資料的情況下，逐步學習並建立具臨床價值的醫學語意理解能力。

## Notice and License Statement

本專案基於以下程式碼：

來源：https://github.com/rmaphoh/RETFound

授權條款：Creative Commons Attribution-NonCommercial 4.0 International（CC BY-NC 4.0）

修改內容：
- 新增支援Pixio-L/16、mae_pretrain_vit_large及vit-large-patch16-224。
- 新增自動執行實驗的腳本。
- 本專案改為使用 5-fold validation 評估實驗結果。
- 新增Supervised Fine-Tuning腳本。

## Install environment

1. 建立conda環境

```
conda create -n M4 python=3.11.0 -y
conda activate M4
```

2. 安裝相依套件

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Baseline Models

總共以下10個模型做為Baseline，除RETFound模型以外請下載模型權重後放到`baseline_models`資料夾中，RETFound模型請參考[官方github](https://github.com/rmaphoh/RETFound)

- [Dinov2 ViT-L/14 distilled(Meta, 2023)](https://github.com/facebookresearch/dinov2)
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

除了直接利用Natural image pre-trained模型進行下游任務以外，我們也測試3種不同Adaptation方法的效果，分別為：
1. DAP (DAP將直接使用RETFound官方提供的模型權重)
2. SFT (SFT將額外在醫療影像或自然影像資料集上進行監督式微調)
3. DAP + SFT (先進行DAP，再進行SFT)


## Dataset

### Fundus Datasets

| 資料集名稱 (Dataset)   | 資料量 (Size) | 資源連結 (Source)                                                                                             |
| :---------------- | :--------- | :-------------------------------------------------------------------------------------------------------- |
| APTOS2019         | 3,662 張圖片  | [Kaggle 連結](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)                       |
| MESSIDOR2         | 1,748 張圖片  | [ADCIS 連結](https://www.adcis.net/en/third-party/messidor2/)                                               |
| Glaucoma_fundus   | 1,544 張圖片  | [Harvard Dataverse 連結](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC)   |
| Retina (Cataract) | 601 張圖片    | [Kaggle 連結](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)                                       |
| IDRID             | 516 張圖片    | [IEEE Dataport 連結](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| PAPILA            | 488 張圖片    | [Figshare 連結](https://figshare.com/articles/dataset/PAPILA/14798004/1)                                    |

[5-fold切分方式下載](https://drive.google.com/drive/folders/11euRYSptEpS9UQDAVP3UJ4vi4RINHvzx?usp=sharing)

dataset請於下載後依照提供的5-fold列表切分後放入`data`資料夾中，例如: ATPOS2019資料集的fold1請放入`data/5_fold_APTOS2019/APTOS2019_seed42_fold0`

### Supervised Fine-Tuning Datasets

| 資料集名稱 (Dataset) | 資料量 (Size) | 資源連結 (Source) |
| :-------------- | :--------- | :---------------------------------------------------------------------------------- |
|Augmented Ocular Disease (AOD)|14,105 張圖片|[github 連結](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/AOD.md)|
|Imagenet-1k|1,281,167 張圖片|[huggingface dataset 連結](https://huggingface.co/datasets/ILSVRC/imagenet-1k)|

請於下載後放入`data`資料夾中，Imagenet-1k請預處理成以下格式，SFT實驗時預設使用所有資料為training data
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
# 執行以下指令進行下游任務測試
python script.py

# 執行以下指令進行SFT訓練
python SFT_script.py
```


## Exp1: Performance of baseline models

在exp1我們比較Natural Image pre-trained model與RETFound的DAP模型在6個資料集的表現

其中Natural Image pre-trained model模型包含：
- Dinov2
- Dinov3
- Pixio
- MAE_pretrain_vit_large
- Vit-large-patch16-224

DAP的模型包含：
- RETFound_mae_natureCFP
- RETFound_mae_meh
- RETFound_mae_shanghai
- RETFound_dinov2_meh
- RETFound_dinov2_shanghai

### Experimental Setup

- **5-fold:** 每個資料集皆進行5-fold測試，每個模型對5個fold各訓練一次，每個fold得到一個validation分數最高的模型，並用validation表現最好的模型進行test set測試，最後計算5個fold的test分數之平均值與標準差，作為評斷模型好壞之標準
- **validation details:** Fine-tune於下游任務時每個epoch進行一次validation，取f1+auroc+kappa分數最高的checkpoint做為best validation，並用於計算test set的分數
- **hyper-parameters:** 測試時使用以下3種超參數微調模型，取表現最好的結果回報於實驗結果表格中
   1. [RETFound 程式預設超參數](https://github.com/rmaphoh/RETFound)設定
   2. [RetFound論文敘述(2025版)](https://arxiv.org/abs/2509.03421)設定
   3. [Meta MAE fine-tune](https://github.com/facebookresearch/mae)設定

超參數主要差異如下

| 超參數 | RETFound 程式預設超參數 | RETFound 論文超參數 | MAE 預設超參數 |
| :--- | :--- | :--- | :--- |
| **Batch Size** | 24 | 24 | 64 |
| **Base Learning Rate** | 5e-3 | 5e-4 | 1e-3 |
| **Drop Path** | 0.2 | 0.2 | 0.1 |
| **Layer Decay** | 0.65 | 0.65 | 0.75 |

使用RETFound_dinov2_meh測試不同超參數效果，比較如下表，**學習率較小的設定**可能較容易出現更好的表現

| 模型 / 設定                    | APTOS2019 (3662) | MESSIDOR2 (1744) | Glaucoma Fundus (1544) | Retina (601)    | IDRiD Data (516) | PAPILA (488)    |
| :------------------------- | :--------------- | :--------------- | :--------------------- | :-------------- | :--------------- | :-------------- |
| **RETFound 程式預設超參數** | 85.13 ±0.56      | 76.31 ±1.09      | 86.45 ±0.73            | **73.04 ±3.28** | 62.33 ±3.02      | 81.84 ±3.34     |
| **RETFound 論文超參數**         | **85.24 ±0.08**  | 76.84 ±1.08      | 82.84 ±0.88            | 70.50 ±3.28     | **65.63 ±2.96**  | **84.08 ±3.19** |
| **MAE 預設超參數**              | 85.16 ±0.67      | **77.07 ±0.88**  | **86.92 ±0.63**        | 71.16 ±1.72     | 62.91 ±4.63      | 82.65 ±3.31     |

### Experimental results

Baseline效果比較如下表

**Baseline performance comparison (2026/05/21)**
![[assets/Exp1-1.png]](assets/Exp1-1.png)

**Baseline performance comparison (2026/05/21)**
![[assets/Exp1-2.png]](assets/Exp1-2.png)

## Summary

- RetFound dinov2 meh表現明顯超越其他模型
- 在自然影像模型中dinov3表現最好，dinov2表現次之
- 透過MAE based預訓練的模型不論自然影像模型或是Retfound的continual pretraining表現都較Dino系列更差

## Exp2: Supervised Fine-Tuning (SFT)

在exp2我們加入SFT將Natural Image pre-trained model與RETFound的DAP模型微調於imagenet-1k與AOD資料集，觀察SFT對於不同預訓練流程的模型在下游任務的影響

其中執行SFT的模型包含：
- Dinov2
- Dinov3
- Pixio
- MAE_pretrain_vit_large

DAP + SFT的模型包含：
- RETFound_dinov2_meh
- RETFound_dinov2_shanghai


### Experimental Setup

- 下游任務測試方式與Exp1相同，但超參數僅使用**RETFound 程式預設超參數**
- SFT微調時使用**Meta MAE fine-tune**超參數設定
- AOD資料集總共微調30個epoch，每2個epoch進行一次下游任務測試; imagenet-1k資料集微調1個epoch，僅於微調完後進行1次下游任務測試
- 以下比較的數值皆為accuracy

### Experimental results

- SFT on AOD Dataset

	**Supervised Fine-Tuning on AOD dataset (2026/05/21)**
	![[Exp2-1_AOD.png]](assets/Exp2-1_AOD.png )

- SFT on Imagenet-1k Dataset

	**Comparison of SFT on Imagenet-1k dataset (2026/05/21)**
	| Model\Dataset                    | APTOS2019  | Glaucoma fundus | MESSIDOR2 | Retina     | IDRID_Data | PAPILA     |
	| -------------------------------- | ---------- | --------------- | --------- | ---------- | ---------- | ---------- |
	| **RETFound dinov2 (meh) w/o sft** | **0.8513** | **0.8572**          | 0.7631    | 0.7304     | 0.5942     | 0.8184     |
	| **DINOv2**                       | 0.8456     | 0.8013          | **0.7681**    | 0.7138     | 0.5786     | 0.7592     |
	| **DINOv3**                       | 0.8396     | 0.8370          | 0.7452    | 0.7227     | 0.5845     | 0.7918     |
	| **Pixio**                        | 0.8347     | 0.7871          | 0.7308    | 0.6475     | 0.4796     | 0.7306     |
	| **RetFound (meh)**               | 0.8451     | 0.8417          | 0.7669    | **0.7448** | **0.6019** | **0.8204** |
	| **RetFound (shanghai)**          | 0.8484     | 0.8529     | 0.7297    | 0.6895     | 0.5942     | 0.7959     |

### Computational cost

根據[RETFound論文](https://www.researchsquare.com/article/rs-6080254/v1)內容continual pretraining與SFT資源開銷比較如下(以下以dinov2 large為backbone model進行比較)

|Method|Dataset|Training epoch|GPU|Cost Time|
|-|-|-|-|-|
|Continual pretraining|Moorfields Eye Hospital(MEH)|unknown|A100\*4|16 days|
|SFT|Augmented Ocular Disease (AOD)|30|V100\*1|6 hour|
|SFT|Imagenet-1k|1|V100\*1|4 hour|

## Summary

- 整體來說，不論是透過AOD或imagenet-1k資料集進行SFT訓練，dinov2與RetFound(meh)表現皆比其他模型好
- 透過AOD進行SFT在各個資料集的最高分都比imagenet-1k更高，並且**透過imagenet-1k訓練的模型沒有超越SOTA的表現**
- 雖然dinov3在未進行SFT時效果比其他自然影像模型更好，但在SFT後除Glaucoma_fundus資料集外，在其他資料集Dinov3明顯比Dinov2更差
- 比較原始RETFound_dinov2_meh(epoch 0的點)與dinov2模型，即使使用開銷顯著更低的SFT方法，在6個資料集的效果除了Glaucoma fundus與PAPILA以外皆能取得更好的表現，顯示出SFT方法在開銷較低的情況下，也能取得與continual pretraining接近的效果

