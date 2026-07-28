# MedClaw 模型目錄 (Model Registry) — 設計文件

- 狀態：v0.2 — **載入層已實作**（§2–§5、§8、§9 已落地；§6 web UI 端點待做）
- 適用範圍：`agent/` 自動微調 Agent（MedClaw）的 encoder 目錄管理
- 相關文件：[auto_finetune_agent_design.md](auto_finetune_agent_design.md)、[data_firewall_design.md](data_firewall_design.md)
- 相關程式：`agent/encoder_registry.py`、`agent/migrate_model_registry.py`、`agent/schemas.py`（`EncoderCard`）、`tests/test_encoder_registry.py`、`models_vit.py`、`main_finetune.py`、`agent/web/app.py`

---

## 1. 目標與動機

目前「可用 encoder 目錄」硬編碼在 [agent/encoder_registry.py](../agent/encoder_registry.py) 的 `_CARDS` list，權重檔則平鋪在 `baseline_models/` 下：

```
baseline_models/
  dinov2_vitl14_pretrain.pth
  mae_pretrain_vit_large.pth
  vit_large_patch16_224.pth
```

新增一個 model 必須改 Python 程式，而且權重、metadata 分散在不同地方，難以由 UI 管理。

**目標：把每個 model 變成一個「自帶所有檔案的目錄」，registry 由掃描目錄產生，UI 只需管理這些目錄，不需碰程式碼。**

設計原則（延續本專案「開放擴充、核心不動」的第一原則）：

| 原則 | 說明 |
| --- | --- |
| **自描述目錄** | 一個 model = 一個目錄，manifest（`model.yaml`）+ 權重 + 其他相關檔案全在裡面，可整包搬移／複製／備份 |
| **資料驅動** | 新增 model = 新增一個目錄，不改 `encoder_registry.py` 以外的任何程式 |
| **UI 友善** | 所有操作對應單純的檔案系統動作（建目錄、放檔、改 YAML），UI 與 CLI 共用同一份 on-disk 契約 |
| **介面不變** | `encoder_registry` 對外的 `all_cards() / available_cards() / get() / weight_path()` 簽章不變，下游（`LLMAdvisor`、dataset picker、`main_finetune`）零改動 |
| **驗證前置** | manifest 以 `EncoderCard` pydantic 驗證；`model` 架構家族比對白名單；錯誤在載入／存檔當下就擋，而非訓練啟動時才炸 |

---

## 2. 目錄結構

```
baseline_models/
  <model_key>/                    # 目錄名 = model_key（唯一識別）
    model.yaml                    # manifest（必要）— EncoderCard 欄位
    weights.pth                   # 權重檔（本地權重時必要）
    README.md                     # 人類說明（可選）
    ...                           # 未來其他 model 相關檔案（config、前處理器…）
```

範例：

```
baseline_models/
  dinov2_vitl14/
    model.yaml
    weights.pth
  mae_vit_large/
    model.yaml
    weights.pth
  fundus_dinov2_vitl14/           # 新增的醫療 encoder
    model.yaml
    weights.pth
    README.md
  RETFound_dinov2_meh/            # gated（HF 下載）— 目錄可只有 manifest
    model.yaml
```

規則：

- **目錄名即 `model_key`**，是唯一識別碼（也用於 task_id）。目錄名須符合 `[A-Za-z0-9_.-]+`。
- 只有**含 `model.yaml` 的目錄**才視為一個 model；其他檔案／目錄忽略。
- 以 `.` 或 `_` 開頭的目錄（如 `_archive`、`.trash`）一律略過，方便 UI 做暫存／軟刪除。

---

## 3. `model.yaml` 契約（manifest）

manifest 就是一張 `EncoderCard`（欄位定義見 [agent/schemas.py](../agent/schemas.py) `EncoderCard`），另加少數「目錄相對」的解析規則。

```yaml
# baseline_models/fundus_dinov2_vitl14/model.yaml
# model_key 省略時，以目錄名為準；若填寫必須與目錄名一致
model: Dinov2                       # 架構家族（必須在白名單內，見 §5）
model_arch: dinov2_vitl14           # 傳給 models_vit 選變體
weight: weights.pth                 # 相對「本目錄」的檔名；或 HF id（gated）
embed_dim: 1024
patch_size: 14
input_size: 224
domain: medical_dap                 # natural | medical_dap
available: true                     # 權重本機可取得？gated 未取得填 false
notes: 在 XX 眼底資料集(約 N 張)上以 DINOv2 自監督預訓練; 適合眼底/視網膜分類。
```

### 3.1 欄位一覽

| 欄位 | 必要 | 說明 | 誰使用 |
| --- | --- | --- | --- |
| `model_key` | 否 | 省略＝目錄名；填寫須與目錄名相符 | registry 查找、task_id、UI |
| `model` | 是 | 架構家族，決定 `main_finetune` 建模與載權重分支 | `main_finetune.py` |
| `model_arch` | 是 | 家族內變體（如 `dinov2_vitl14`） | `models_vit.py` |
| `weight` | 是 | **本目錄相對檔名**（本地）或 HF id（gated） | `weight_path()` → `--finetune` |
| `embed_dim` / `patch_size` / `input_size` | 否 | 架構規格；進 prompt 供 LLM 判斷容量／解析度 | LLMAdvisor |
| `domain` | 否 | `natural`／`medical_dap`；結構化特性，選擇規則直接用 | Advisor |
| `available` | 否 | 權重是否可在本機取得（gated 未取得＝false） | `available_cards()` |
| `notes` | 否 | 自由文字：預訓練來源、擅長任務、已知限制 | LLMAdvisor 選擇依據 |

### 3.2 `weight` 路徑解析規則

`weight_path(card)` 依序判定（取代現行以 repo 根為基準的 `_abs()`）：

1. **HF id**（不含路徑分隔符、且非 `.pth/.pt` 結尾，如 `RETFound_dinov2_meh`）→ 原樣回傳，交給 `main_finetune` 走 HF 下載分支。
2. **本目錄相對檔名**（如 `weights.pth`）→ 解析為 `baseline_models/<model_key>/weights.pth` 的絕對路徑。**這是建議且預設的作法**（權重跟著目錄走）。
3. **絕對路徑**（相容舊資料）→ 原樣回傳。

> 與現況差異：現行 `weight` 是「相對 repo 根」（`baseline_models/xxx.pth`）；新制改為「相對 model 目錄」（`weights.pth`）。遷移方式見 §8，向後相容見 §9。

---

## 4. Registry 載入與解析

`encoder_registry.py` 由「硬編碼 list」改為「掃描 `baseline_models/` 目錄」，對外介面完全不變。

### 4.1 載入流程

```
scan baseline_models/*/model.yaml
  → 逐檔 yaml.safe_load → EncoderCard.model_validate
  → 補 model_key（省略時用目錄名）、記住 model_dir
  → 驗證（§5）：失敗的 card 跳過並收集 warning，不讓單一壞檔擋掉整個 registry
  → 依 model_key 建索引；重名以「先掃到者為準」並發 warning
快取結果（首次存取時載入一次；提供 reload() 供 UI 改檔後刷新）
```

- **找不到 `baseline_models/` 或目錄為空** → 回退到內建 `_BUILTIN_CARDS`（即現行那幾張卡），確保開發環境與測試不因缺目錄而壞掉。
- 載入以 **fail-soft** 為原則：個別 manifest 壞掉只跳過該 model 並記 warning（UI 可讀），不 raise。唯一會硬失敗的是被明確 `get(model_key)` 卻不存在（維持現行 `KeyError`）。

### 4.2 對外介面（不變）

| 函式 | 行為 |
| --- | --- |
| `all_cards(include_unavailable=False)` | 同現行語意 |
| `available_cards()` | 權重檔（`weight_path`）確實存在才回傳；HF id 視為可用 |
| `get(model_key)` | 未知 key 拋 `KeyError`（訊息列出可用 keys） |
| `weight_path(card)` | 依 §3.2 解析 |
| `reload()`（新增） | 重新掃描目錄；供 UI 增刪改後刷新，不必重啟程序 |

`EncoderCard` 需新增一個**非序列化**的內部欄位記住來源目錄（如 `model_dir: str = Field(default="", exclude=True)`），供 `weight_path` 解析與 UI 定位檔案；它不進 `model_dump()` 給 LLM 的 prompt。

---

## 5. 驗證規則

載入 manifest 與 UI 存檔前皆套用：

1. **`model_key` 一致性**：填寫時須等於目錄名；目錄名須符合 `[A-Za-z0-9_.-]+`。
2. **架構白名單**：`model` 必須是 `models_vit` 已註冊的家族之一——目前為
   `{Dinov2, Dinov3, MAE, SL_VIT, RETFound_mae, RETFound_dinov2, GastroNet, Pixio}`
   （對照 [main_finetune.py](../main_finetune.py) 建模與載權重分支）。不在白名單 → 拒絕，並提示「新架構需先在 `models_vit.py`／`main_finetune.py` 註冊」。
3. **權重可達性**：`available: true` 且 `weight` 為本地檔時，檔案必須存在，否則 UI 標記為「權重缺失」、`available_cards()` 不回傳（與現行 `available_cards()` 的存在性檢查一致）。
4. **唯一性**：`model_key`（＝目錄名）在 `baseline_models/` 下唯一。

> 白名單集中定義成一份常數 `encoder_registry.ARCH_WHITELIST`，供 registry 驗證與 UI 表單下拉共用，避免兩處不同步。

---

## 6. UI 管理契約

UI 對 `baseline_models/` 的操作全部對應單純的檔案系統動作，因此 CLI 直接手動放目錄、與 UI 操作可以並存。

| 操作 | 檔案系統動作 | 驗證 |
| --- | --- | --- |
| 列出 models | 掃描 `*/model.yaml` → `all_cards()` | — |
| 檢視 model | 讀 `model.yaml` + 列目錄檔案 | — |
| 新增 model | 建 `<model_key>/`、寫 `model.yaml`、上傳權重 | §5 全套；權重上傳後才可 `available: true` |
| 編輯 metadata | 覆寫 `model.yaml` | §5；存檔後 `reload()` |
| 更換權重 | 覆寫（或新增）目錄內權重檔並更新 `weight` | 檔案存在性 |
| 刪除 model | 移到 `_archive/`（軟刪）或刪目錄 | 確認無進行中的 run 引用 |

建議 UI 的表單即以 `EncoderCard` schema 動態生成（`model` 用白名單下拉、`domain` 用 enum、`notes` 用多行文字），存檔時走同一套 pydantic 驗證，錯誤即時回饋。

> **權重上傳大小**：現有權重約 1.2 GB／個。UI 需支援大檔上傳（分塊／串流），或提供「填 HF id 由後端下載」與「指定伺服器上既有檔路徑」兩種替代方式，避免瀏覽器直傳巨檔。

---

## 7. 資料圍欄考量

- 模型目錄屬於**決策平面的靜態目錄**，不含使用者資料。`model.yaml` 的內容（含 `notes`）本就會進 LLM prompt（見 [llm_advisor.py](../agent/llm_advisor.py) `_registry_context()`，列為 `exempt_text`），本設計不改變此性質——**但也因此 `notes` 不得填入任何使用者資料集的識別資訊**（路徑、真實類別名、機構名）。UI 應在 `notes` 欄位旁明示這點。
- 權重檔（`.pth`）是二進位模型參數，不經 LLM、不外送，維持在資料平面之外的一般檔案，無圍欄疑慮。
- UI 的檔案管理端點需做路徑防護（限制在 `baseline_models/` 內、拒絕 `..` 穿越），屬一般 web 安全，非圍欄範疇。

---

## 8. 遷移計畫

現有三檔轉為三個目錄（一次性 migration script `python -m agent.migrate_model_registry`，加 `--dry-run` 可先看動作）：

| 現況 | 遷移後 |
| --- | --- |
| `baseline_models/dinov2_vitl14_pretrain.pth` | `baseline_models/dinov2_vitl14/weights.pth` + `model.yaml` |
| `baseline_models/mae_pretrain_vit_large.pth` | `baseline_models/mae_pretrain_vit_large/weights.pth` + `model.yaml` |
| `baseline_models/vit_large_patch16_224.pth` | `baseline_models/vit_large_patch16_224/weights.pth` + `model.yaml` |
| `RETFound_dinov2_meh`（HF, `available:false`） | `baseline_models/RETFound_dinov2_meh/model.yaml`（僅 manifest） |

各 `model.yaml` 的欄位直接取自現行 `_BUILTIN_CARDS`（[encoder_registry.py](../agent/encoder_registry.py)）對應卡片。migration script 用 `os.rename`（同檔案系統零成本），不複製 1.2 GB；可重複執行（已有 `model.yaml` 的目錄會跳過）。

> 目錄名一律沿用**原本的 `model_key`**（MAE 那張維持 `mae_pretrain_vit_large`，而非本文件 v0.1 草案寫的 `mae_vit_large`）。因為目錄名即 `model_key`，而 `model_key` 已寫進既有 `runs/` 歷史的 trial_id 與 `--encoder` 參數，改名會讓舊 run 的 `reg.get()` 直接 `KeyError`。

---

## 9. 向後相容

- **內建 fallback**：`baseline_models/` 無任何 `model.yaml` 目錄時，回退內建 `_BUILTIN_CARDS`（現行卡片，`weight` 仍為 repo 根相對路徑）。既有測試與尚未遷移的環境照常運作。
- **`weight` 解析相容**：`weight_path` 同時支援「目錄相對檔名」（新制）、「repo 根相對／絕對路徑」（舊制）、「HF id」，三者並存，遷移可漸進。
- 下游（`LLMAdvisor`、dataset picker、`main_finetune`、`trainer`）不需修改，因為它們只透過 `encoder_registry` 的既有函式取值。

---

## 10. 未來擴充（非本次範圍）

若要**減少引入新權重時仍需改 `main_finetune.py`** 的情況，可把權重載入的差異也放進 manifest：

```yaml
checkpoint_key: teacher          # checkpoint 內取哪個 key（teacher / model / null=整包）
weight_source: local             # local | hf
key_rename:                      # state_dict key 前綴替換
  - { from: "backbone.", to: "" }
```

這能涵蓋 [main_finetune.py](../main_finetune.py) 現有的 `checkpoint["teacher"] vs ["model"]`、key 前綴清理等分支，讓「同架構、不同 checkpoint 格式」也能純 YAML 完成。但**建立新網路架構**（`nn.Module`）本質上仍需在 `models_vit.py` 寫 Python，無法 YAML 化——這是刻意的邊界。

---

## 11. 實作步驟

1. ✅ `EncoderCard` 加內部 `model_dir`（`exclude=True`）；`weight_path` 依 §3.2 解析。
2. ✅ `encoder_registry`：新增目錄掃描載入 + `_BUILTIN_CARDS` fallback + `reload()`；保留現有對外函式簽章（另加 `warnings()` 供 UI 顯示載入問題）。
3. ✅ 架構白名單集中為共用常數 `encoder_registry.ARCH_WHITELIST`，供 registry 驗證與 UI 表單使用。
4. ✅ migration script：`agent/migrate_model_registry.py`（`os.rename` + 產生 `model.yaml`）。
5. ⬜ web 端點：list / view / create / edit / delete + 權重上傳（大檔策略見 §6）。
6. ✅ 測試：`tests/test_encoder_registry.py`（載入／驗證／fallback／`available_cards` 存在性／壞檔 fail-soft／`weight_path` 三種解析）。
7. ⬜ 更新 [auto_finetune_agent_design.md](auto_finetune_agent_design.md) §5.3（EncoderRegistry）指向本文件。

---

## 附錄 A：完整 `model.yaml` 範例

```yaml
# baseline_models/fundus_dinov2_vitl14/model.yaml
model: Dinov2
model_arch: dinov2_vitl14
weight: weights.pth
embed_dim: 1024
patch_size: 14
input_size: 224
domain: medical_dap
available: true
notes: >
  在 XX 眼底資料集（約 N 張）上以 DINOv2 自監督預訓練；
  適合眼底／視網膜分類，對自然影像域外任務效果不明。
```
