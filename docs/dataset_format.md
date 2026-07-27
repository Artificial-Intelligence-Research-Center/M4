# MedClaw 資料集格式規範與匯入指南

- 狀態：v1.0 — **已實作**（描述現行程式行為）
- 適用範圍：`agent/` 自動微調 Agent（MedClaw）與 `main_finetune.py` 下游微調
- 相關文件：[auto_finetune_agent_design.md](auto_finetune_agent_design.md)、[data_firewall_design.md](data_firewall_design.md)、[per_fold_parallel_design.md](per_fold_parallel_design.md)
- 相關程式：`agent/dataset_registry.py`（掃描 + 驗證）、`agent/dataset_ingest.py`（上傳解壓）、`agent/web/app.py`（`/datasets*` 路由）、`agent/loop_controller.py`（`discover_folds`）、`util/datasets.py`（`build_dataset`）

---

## 1. 目錄結構（唯一支援的格式）

資料集必須是 torchvision **ImageFolder** 結構，且 `train` / `val` / `test` 三個 split
**缺一不可**——`main_finetune.py` 會對三者各呼叫一次 `build_dataset()`：

```
<資料集目錄>/
├── train/
│   ├── <類別A>/  img001.jpg  img002.jpg  …
│   └── <類別B>/  …
├── val/
│   ├── <類別A>/  …
│   └── <類別B>/  …
└── test/
    ├── <類別A>/  …
    └── <類別B>/  …
```

實際範例（本專案既有資料）：

```
data/5_fold_Glaucoma_fundus/Glaucoma_fundus_seed42_fold0/
├── train/{anormal_control, bearly_glaucoma, cadvanced_glaucoma}/*.png
├── val/  {anormal_control, bearly_glaucoma, cadvanced_glaucoma}/*.png
└── test/ {anormal_control, bearly_glaucoma, cadvanced_glaucoma}/*.png
```

### 1.1 支援的影像副檔名

`.jpg` `.jpeg` `.png` `.bmp` `.tif` `.tiff` `.webp`
（定義於 `dataset_registry._IMG_EXT`；不在清單內的檔案在統計與訓練時都會被忽略。）

灰階影像可以使用——loader 一律 `convert("RGB")` 成 3 通道；但預訓練 encoder 以 RGB
自然影像為主，驗證器會給提醒。

---

## 2. 硬性規則（不符合就無法訓練）

`agent/dataset_registry.py` 的 `validate()` 會把下列情況列為 **error**：

| # | 規則 | 為什麼 |
| --- | --- | --- |
| 1 | `train/`、`val/`、`test/` 三個目錄都必須存在 | `main_finetune` 同時載入三個 split |
| 2 | 每個 split 底下都要有類別子目錄（`<split>/<class>/*.jpg`） | ImageFolder 以子目錄名為類別 |
| 3 | **三個 split 的類別資料夾名稱必須完全相同** | 每個 split 各自建立 ImageFolder，label index = 該 split 內排序後的類別名。名稱對不上 → 標籤錯位，指標全錯且不會報錯 |
| 4 | 至少 2 個類別 | 分類任務的下限 |
| 5 | 不能有空的類別目錄（0 張影像） | ImageFolder 會讓類別數與實際資料不一致 |
| 6 | 整個資料集至少要有 1 張可辨識副檔名的影像 | — |
| 7 | 抽樣的影像必須讀得開（PIL 可 decode） | 毀損檔會讓訓練中途炸掉 |

> **規則 3 是最常見的坑。** 例如 test 集缺少某個罕見類別時，train 的
> `{A, B, C}` 對上 test 的 `{A, C}`，test 的 `C` 會被標成 label 1（train 是 2）。
> 若某個類別在某 split 真的沒有樣本，請建立空目錄以外的解法：重新切分，
> 讓每個 split 都有該類別的樣本（空目錄本身也違反規則 5）。

---

## 3. 建議規則（可訓練，但會出 warning）

| 情況 | 訊息與建議 |
| --- | --- |
| `train` < 100 張 | 資料量偏少，建議加強增強或用 linear probe 起手 |
| `val` 或 `test` < 10 張 | 指標會非常不穩定 |
| train 類別不平衡 ≥ 3:1 | 主要指標建議看 F1 / kappa 而非 accuracy |
| train 類別不平衡 ≥ 10:1 | 建議 class weight / focal loss / 平衡取樣 |
| 影像短邊 < 224 px | 小於預設 `input_size`，放大後細節有限 |
| 抽樣影像全為灰階 | 預訓練 encoder 以 3 通道 RGB 為主（載入時會自動轉 RGB） |
| 類別目錄底下還有子目錄 | ImageFolder 會**遞迴收進來並攤平為同一類**——若那是想區分的次分類，資料會被誤併 |

抽樣檢查會從「每個 split × 每個類別」平均取樣，總共最多 30 張
（`dataset_registry._SAMPLE_N`），只 open 不 decode 全部像素，成本很低。

---

## 4. 多 fold（cross-validation）的命名慣例

想使用 `eval.aggregation = mean_std`（單一最佳 recipe 套所有 fold 重跑）或
`per_fold`（每個 fold 各自獨立搜尋最佳 recipe，見
[per_fold_parallel_design.md](per_fold_parallel_design.md)），目錄命名**必須**符合
`agent/loop_controller.py` 的 `discover_folds()`：

- 你指定的 `data_path` 目錄名要以 `_fold<數字>` 結尾（正則 `(.*_fold)(\d+)$`）
- 同一個父目錄下、前綴相同的 `…_fold0`、`…_fold1`… 會被自動視為 sibling fold
- 不符合命名 → 只回傳 `[data_path]` 自己，靜默地退化成「單一 fold」

慣用佈局（與既有資料集一致）：

```
data/5_fold_<資料集名>/
├── <資料集名>_seed42_fold0/{train,val,test}/…
├── <資料集名>_seed42_fold1/…
├── <資料集名>_seed42_fold2/…
├── <資料集名>_seed42_fold3/…
└── <資料集名>_seed42_fold4/…
```

設定時指向**任一個** fold 即可，其餘由 `discover_folds()` 補齊：

```yaml
data_path: ./data/5_fold_MyDataset/MyDataset_seed42_fold0
eval:
  aggregation: mean_std      # single | mean_std | per_fold
```

### 4.1 從未切分的資料產生 fold

`data/new_fold.py` 可把一份既有的 `{train,val,test}` 資料集重切成 N 個 fold：

```bash
cd data
python new_fold.py --data_path MyDataset --seed 42 --fold 5
# → MyDataset_seed42_fold0 … MyDataset_seed42_fold4
```

注意：該腳本以 `root.split('/')` 判斷目錄層級，**必須在 `data/` 目錄下用相對路徑
執行**；且輸出目錄已存在時會直接報錯（換 seed 或先刪除）。

---

## 5. 三種匯入方式

### 5.1 Web UI 上傳（推薦）

```bash
conda activate M4
python -m agent.web.app --port 5000
```

開啟 `http://127.0.0.1:5000/datasets-page` → 「上傳新資料集」。

| 項目 | 限制 |
| --- | --- |
| 支援格式 | `.zip` `.tar` `.tar.gz` `.tgz` `.tar.bz2` `.tbz2` |
| 單次上傳大小 | 16 GB（`app.config["MAX_CONTENT_LENGTH"]`） |
| 解壓後總容量 | 64 GB（`dataset_ingest._MAX_TOTAL_BYTES`，zip bomb 防護） |
| 資料集名稱 | 留空則取檔名；只保留 `[\w.\-]`，其餘字元轉 `_`，上限 80 字元 |
| 同名處理 | 預設拒絕；要覆蓋需勾選「覆蓋同名資料集」（會先整個刪除舊目錄） |

行為說明：

- **自動找根目錄**：壓縮檔常多包一層（`MyData.zip` → `MyData/MyData/train/…`）。
  `locate_root()` 會從解壓目錄往下找最多 3 層、找出真正含 `train/` 的那層，
  再把內容攤平到 `data/<name>/` 頂層，並在回報中以 `lifted_from` 告知攤平了哪一層。
- **安全性**：只信任壓縮檔內的相對路徑。含 `..`、絕對路徑、或會跳出目標目錄的成員
  一律中止（zip-slip 防護）；symlink 與 hardlink 一律略過。
- **失敗不留半成品**：任何錯誤都會把自己建立的目錄整個清掉。
- **上傳後立刻驗證**：回傳 §2/§3 的 errors / warnings 給前端顯示。

### 5.2 直接放進 `data/`

把整理好的目錄複製或 symlink 到 `data/` 底下即可，掃描器（`dataset_registry.scan()`）
會自動列出。掃描規則：

- 從 `data/` 往下最多 **3 層**（足以涵蓋 `5_fold_X/X_fold0`）
- 一旦某層目錄本身就是合法資料集（含 `train/<class>/`），就不再往下找
- **略過**以 `.` 或 `_` 開頭的目錄（例：`_resize_cache`、上傳用的 `_upload_*` 暫存）
  與 split 目錄本身
- 掃描結果快取 300 秒；剛放進去沒看到時，用資料集頁的重新整理或
  `GET /datasets?refresh=1` 強制重掃

### 5.3 用絕對路徑直接指定

資料不在 `data/` 底下也可以訓練，設定 `data_path` 為絕對路徑即可；
只是不會出現在 Web UI 的下拉選單裡（選單來源是 `data/` 掃描結果）。
`POST /datasets/validate` 不限 `data/` 底下，可用來驗證任意路徑。
（`/datasets/delete` 則有圍欄，只允許刪除 `data/` 底下的目錄。）

---

## 6. 匯入前後的驗證

### CLI

```bash
python -c "import json; from agent import dataset_registry as r; \
print(json.dumps(r.validate('data/5_fold_PAPILA/PAPILA_seed42_fold0'), \
                 ensure_ascii=False, indent=2))"
```

### Web

資料集頁的「檢查既有路徑」，或直接呼叫：

```bash
curl -X POST -F 'path=data/5_fold_PAPILA/PAPILA_seed42_fold0' \
     http://127.0.0.1:5000/datasets/validate
```

回傳格式：

```jsonc
{
  "ok": true,                 // errors 為空才是 true
  "errors": [],               // 非空 → 現在無法訓練，必須修正
  "warnings": [],             // 非空 → 可以訓練，但結果可能受影響
  "summary": {
    "path": "...", "classes": ["a", "b"], "num_classes": 2,
    "counts": {"train": {"a": 100, "b": 80}, "val": {...}, "test": {...}},
    "n_train": 180, "n_val": 40, "n_test": 40, "total": 260,
    "n_sampled": 30,
    "image_size": {"min_width": 1024, "max_width": 2576,
                   "min_height": 768,  "max_height": 1934}
  }
}
```

---

## 7. 使用資料集

### Web UI

實驗頁的資料集下拉選單自動列出 `data/` 底下所有合法資料集，依上層目錄分組
（`group` = 相對 `data/` 的第一層目錄名，例 `5_fold_PAPILA`）、依 fold 編號排序。
不合法的資料集仍會列出，但標記 `ready: false` 與具體 `issues`。

### 設定檔

```yaml
# agent/config.example.yaml
data_path: ./data/5_fold_MyDataset/MyDataset_seed42_fold0
task:
  type: classification
task_template: fundus_classification
eval:
  aggregation: mean_std
```

```bash
python -m agent.auto_finetune --config agent/config.example.yaml
# 或旗標覆寫
python -m agent.auto_finetune --data_path ./data/5_fold_MyDataset/MyDataset_seed42_fold0
```

---

## 8. 效能相關（選用）

高解析度資料集（如 2576×1934 的眼底照）每次 `__getitem__` 都要 decode 整張大圖，
是 GPU 利用率偏低的主因之一。`main_finetune.py` 提供預縮圖磁碟快取：

```bash
--cache_resized 448 --cache_dir ./data/_resize_cache   # 0 = 關閉（預設，行為同原版）
```

- 快取 key 取自影像 `realpath` 的 sha1 → **多個 fold 以 symlink 指向同一實體檔時會共用快取**
- lazily 建立、多 worker 併發安全（tmp 檔 + `os.replace` 原子替換），毀損時 fallback 原圖
- 快取目錄以 `_` 開頭，不會被資料集掃描器誤認為資料集

---

## 9. 隱私（資料圍欄）

依 [data_firewall_design.md](data_firewall_design.md)，在預設的 `privacy.mode: strict`
下，**影像內容、檔名、影像路徑、資料集名稱與真實類別名都不會進入 LLM prompt**；
決策層看到的是假名化後的 `DatasetFacts`。因此放入敏感資料集**不需要**事先改名或
去識別化類別資料夾——但類別名仍會顯示在你自己的 Web UI 與報告中，若連本機畫面
都需要遮蔽，請自行在匯入前改名。

---

## 10. 疑難排解

| 症狀 | 原因與處置 |
| --- | --- |
| 上傳後回報「壓縮檔裡找不到 train/ 目錄」 | 壓縮檔多包超過 3 層，或根本沒有 `train/`。重新打包成 `<資料集>/{train,val,test}/<類別>/*.jpg` |
| 上傳被拒：「不支援的檔案格式」 | 只收 §5.1 列出的副檔名；`.rar` / `.7z` 請先轉成 zip |
| 上傳被拒：「壓縮檔含有不安全的路徑」 | 打包時用了絕對路徑（`tar -cf x.tar /abs/path`）。改成 `cd 父目錄 && tar -cf x.tar 資料集名` |
| 資料集沒出現在下拉選單 | 目錄名以 `_` 或 `.` 開頭／缺 `train/<class>/`／埋得比 3 層深／掃描快取未過期（`?refresh=1`） |
| 選單列出但無法選取 / 標紅 | `ready: false`，看 `issues` 或跑 §6 的 `validate()` |
| 明明有 5 個 fold，卻只跑了 1 個 | 目錄名結尾不是 `_fold<N>`，或 `eval.aggregation` 仍是 `single` |
| 指標異常低且各 split 表現矛盾 | 極可能是規則 3 的標籤錯位——先跑 `validate()` 確認三個 split 類別完全一致 |
| 影像數量比預期多 | 類別目錄底下還有子目錄，被 ImageFolder 遞迴攤平（見 §3） |
