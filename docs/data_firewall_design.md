# MedClaw 資料圍欄 (Data Firewall) — 設計文件

- 狀態：v0.2 — **P0–P2 與 P5 已實作**（見 §15 實作現況）
- 適用範圍：`agent/` 自動微調 Agent（MedClaw）
- 相關文件：[auto_finetune_agent_design.md](auto_finetune_agent_design.md)
- 相關程式：`agent/llm_advisor.py`、`agent/loop_controller.py`、`agent/qa_agent.py`、`agent/dataset_analyzer.py`、`agent/code_workspace.py`、`agent/web/app.py`

---

## 1. 目標

**核心不變式 (Invariant)：LLM 永遠看不到使用者的輸入資料。**

「輸入資料」在本文件中定義為：

1. **像素/檔案內容** — 任何影像的位元組、張量值、縮圖、base64；
2. **個別樣本的識別資訊** — 檔名、影像相對路徑、單筆預測記錄；
3. **資料集的識別性中繼資料** — 資料集絕對路徑、資料集名稱、類別資料夾名稱、任何可回推來源機構／收案批次的字串。

LLM 若需要知道資料的特性，只有兩條合法管道：

- **管道 A（程式分析）**：由我們事先寫好、經過審核的分析器 (Analyzer) 在資料平面執行，輸出**受 schema 約束的數值型事實**，經消毒後放進 prompt。
- **管道 B（詢問使用者）**：LLM 提出結構化問題，由 Web UI 呈現給使用者，**使用者自己輸入**答案；答案才進 prompt。

除此之外沒有第三條路。任何試圖繞過的呼叫都必須**中止 (fail-closed)**，而不是靜默過濾。

### 1.1 設計原則

| 原則 | 說明 |
| --- | --- |
| **單一出口 (single chokepoint)** | 所有送往 Claude API 的 payload 只能經由 `agent/privacy/egress.py`；其他路徑在執行期被 sentinel 攔截並拋錯。 |
| **允許清單優先 (allowlist over blocklist)** | 不是「過濾掉壞東西」，而是「只放行明確列舉的欄位」。未知欄位預設不外送。 |
| **schema 即圍欄** | 分析器輸出型別限定為數值／列舉／布林。**自由字串預設禁止**——沒有自由字串就沒有夾帶通道。 |
| **失敗即中止 (fail-closed)** | 圍欄偵測到疑似洩漏 → 拋 `EgressViolation`，中止該次決策並記錄，不自動改寫後放行。 |
| **可證明性 (auditable)** | 使用者能在 UI 上逐字看到「LLM 到底收到了什麼」。圍欄的說服力來自可檢視，不是來自宣稱。 |
| **能力最小化** | 能造成洩漏的能力（改程式、回饋原始 log）在 strict 模式下直接關閉，而不是靠檢查器把關。 |

### 1.2 非目標

- 不處理 Anthropic 端的資料留存政策（那是合約層問題，不是程式層）。
- 不阻止使用者**自願**把資料貼進討論框——但會明確警示且留稽核紀錄（§7.3）。
- 不涵蓋 `advisor.type=heuristic`（純規則式，本來就不外送任何東西）。

---

## 2. 現況稽核：目前的洩漏面

以下是逐條追查現有程式後確認的通道。**目前沒有任何像素進入 prompt**（現況已滿足最基本要求），但中繼資料與一條回饋迴路是實際破口。

| # | 通道 | 位置 | 現況 | 風險 |
| - | --- | --- | --- | --- |
| **L1** | `DatasetProfile.root`（絕對路徑）隨 `profile.model_dump_json()` 整包進 prompt | `agent/llm_advisor.py` 的 `select_encoders` / `compose_recipe` / `propose_next` / `review_and_decide` / `propose_debug` / `answer_question` | **直接外送** | 路徑常含機構名、收案批次、`5_fold_PAPILA` 等資料集識別 |
| **L2** | `class_names` / `class_counts` 的 key（類別資料夾名） | 同上 | **直接外送** | 類別資料夾名是使用者自由命名，可能含病人／診斷／院內編碼 |
| **L3** | `modality_hint` 由**路徑字串**比對關鍵字推導 | `agent/dataset_analyzer.py:53` | **直接外送** | 等同 L1 的衍生洩漏 |
| **L4** | 失敗 trial 的 **raw log 尾端 120 行**送給 LLM 診斷 | `agent/loop_controller.py:362-372` → `propose_debug(log_tail)` | **直接外送** | 已驗證：log 第 3 行起就是 `Namespace(... data_path='/home/jovyan/M4/data/5_fold_PAPILA/...' ...)`；traceback 會帶出個別影像檔路徑 |
| **L5** | **`code_edits` → 訓練程式 → log → 下一輪回饋 LLM** | `agent/llm_advisor.py`（`_CodeEdit`、`propose_debug`、`_apply_mutation`）+ `agent/code_workspace.py` | **開放（`allow_code_edit=True` 時）** | **最嚴重**。LLM 可插入 `print(samples[0].tolist())`，該 trial 失敗後 log tail 於下一輪回到 LLM ⇒ 構成完整的任意讀取通道。`code_workspace` 只檢查「改哪個檔」，不檢查「改成什麼」 |
| **L6** | `trial_id` / `task_id` 內嵌資料集名 | `agent/loop_controller.py:667`（`f"{model_key}_{ds_name}_t{k}"`），經 `_hist()` 外送 | **直接外送** | 等同 L1 |
| **L7** | `Recipe.provenance` 自由字典整包外送 | `_hist()`、`review_and_decide` | **直接外送** | 目前內容安全，但無 schema 約束 ⇒ 未來任何人塞進路徑就會洩漏 |
| **L8** | 使用者討論訊息 / `advisor.guidance` | `agent/conversation.py`、`agent/web/app.py:471,552` | **直接外送** | 使用者自願，但目前**無任何警示**，也無稽核 |
| **L9** | `SkillAdvisor` 交握 JSON 含完整 profile / log_tail | `agent/skill_advisor.py` | 寫入 `runs/_skill_handoff/` 供 Claude Code 讀 | 這條路徑完全繞過 `llm_advisor`，圍欄若只做在 LLMAdvisor 會被繞開 |
| L10 | `metrics` / `epoch_curve` / `gpu_stats` | `_hist()` | 直接外送 | **聚合統計，判定為安全**，維持外送 |

> 結論：破口集中在「路徑與名稱字串」（L1/L2/L3/L6）與「原始 log 回饋迴路 + 程式修改權」（L4/L5）。後者必須優先處理，因為它把靜態洩漏升級成**主動可控的讀取能力**。

**修補對照**（本表描述的是修補前的狀態，保留作為威脅分析）：L1/L2/L3 由 §4 的 `DatasetFacts`
取代整包 `DatasetProfile`；L4 由 §8.1 的 `ErrorFacts` 取代原始 log；L5 由 §8.2 的 strict
模式強制關閉 `allow_code_edit` 切斷；L6 由 §4.3 的假名 `trial_id` 處理；L7 由
`ProvenanceFacts` 白名單處理；L8 由 §6.3 的警示 + 稽核處理；L9 由
`SkillAdvisor._write_request()` 寫檔前的 `guard_payload()` 處理。L10 維持外送（聚合統計，判定安全）。

---

## 3. 架構總覽

```
┌────────────────────── 資料平面 (Data Plane) ─────────────────────┐
│  可觸碰檔案系統與像素。禁止直接持有 anthropic client。           │
│                                                                  │
│  dataset_analyzer / dataset_registry / analyzers/*               │
│  trainer → main_finetune (subprocess) → logs/, predictions.csv   │
│  evaluator / log_curves                                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │  只允許通過這一條管道
                   ┌─────────▼──────────┐
                   │  privacy/facts.py  │  DatasetFacts / AnalysisFacts /
                   │  privacy/redact.py │  ErrorFacts / UserFacts
                   │  (消毒 + 假名化)    │  ── 全部是受 schema 約束的物件
                   └─────────┬──────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                  決策平面 (Control Plane)                        │
│  llm_advisor / qa_agent / skill_advisor                          │
│  **禁止 open()、glob、PIL、讀 runs/*/logs/**                     │
│                                                                  │
│                    ┌──────────────────┐                          │
│                    │ privacy/egress.py│ ← 唯一出口               │
│                    │  guard() 掃描    │                          │
│                    │  audit 落地      │                          │
│                    └────────┬─────────┘                          │
│                    ┌────────▼─────────┐                          │
│                    │privacy/sentinel.py│ 執行期包裝 anthropic     │
│                    │  非 egress 呼叫 → raise                     │
│                    └────────┬─────────┘                          │
└─────────────────────────────┼────────────────────────────────────┘
                              ▼
                        Claude API
```

新增目錄：

```
agent/privacy/
  __init__.py
  facts.py        # 對 LLM 開放的資料契約 (DatasetFacts / AnalysisFacts / ErrorFacts / UserFacts)
  redact.py       # DatasetProfile → DatasetFacts；假名化；guard() 掃描器
  egress.py       # 唯一出口：guard + audit + 呼叫 API
  sentinel.py     # 執行期包裝 anthropic client，堵住繞道
  alias.py        # HMAC 假名 (資料集/類別/檔名) 與反查表（只存本機）
  errors.py       # EgressViolation 等
agent/analyzers/
  __init__.py     # 註冊表 + 執行器
  basic_counts.py
  image_stats.py
  class_balance.py
  split_leakage.py
  corrupt_files.py
  near_dup.py
```

---

## 4. Layer 1 — 對 LLM 開放的資料契約

現有 `DatasetProfile`（`agent/schemas.py:16`）**不再直接進 prompt**。改為衍生一個 LLM-safe 視圖；原 schema 不動，向後相容既有 `runs/*/dataset_profile.json`。

### 4.1 `DatasetFacts`

```python
class DatasetFacts(BaseModel):
    """唯一允許進入 prompt 的資料集描述。所有欄位皆為數值/列舉/布林。"""
    dataset_ref: str          # 假名，例 "DS-a1b2c3"；由 HMAC(salt, realpath)[:6] 產生
    task_type: Literal["classification"]
    num_classes: int
    class_labels: list[str]   # 一律 ["C0", "C1", ...]；除非使用者明確授權（§4.2）
    class_counts: list[int]   # 與 class_labels 同序，不用字典 key
    class_ordinal: bool | None  # 類別是否具序數關係 → 由使用者回答，不猜
    n_train: int
    n_val: int
    n_test: int
    has_kfold: bool
    n_folds: int | None
    imbalance_ratio: float
    is_grayscale: bool
    image_size: ImageSizeFacts    # min/median/max 的 w/h，純整數
    modality: Modality | None     # 列舉；由使用者選，不從路徑猜
    anatomy: Anatomy | None       # 列舉；由使用者選
```

被移除的欄位與替代方案：

| 原欄位 | 處置 |
| --- | --- |
| `root` | **刪除**。改為 `dataset_ref` 假名。訓練指令仍用真實路徑，但那是資料平面的事 |
| `class_names` | 預設假名化為 `C0..Cn`；`class_counts` 改為與之同序的 list |
| `modality_hint` | **刪除路徑猜測邏輯**（`dataset_analyzer.py:53`）。改為使用者在 UI 下拉選擇（fundus / OCT / X-ray / CT / MRI / dermoscopy / pathology / endoscopy / other）。這比從路徑猜更準，決策品質是**提升**不是下降 |

### 4.2 類別名的三段式政策

`privacy.class_names` 設定：

- `hashed`（strict 預設）— 一律 `C0..Cn`。
- `user_approved` — UI 逐一列出真實類別名，使用者勾選哪些可以外送（例如 `normal` / `glaucoma` 這種純醫學術語通常無妨，`ptn_batch3_wang` 則不勾）。未勾選者維持假名。
- `plain` — 全部照送（僅 `mode: off` 可用）。

假名對照表存在 `runs/<run>/alias_map.json`，**永不外送**，僅供 UI 把 LLM 回覆中的 `C2` 還原成使用者看得懂的名稱。

### 4.3 `trial_id` / `provenance` 的收斂

- `task_id` 改為 `f"{model_key}_{dataset_ref}_t{k}"`（`loop_controller.py:667`），資料集名不再進 id。真實可讀名稱只存在於 UI 與本機檔名。
- `Recipe.provenance` 從 `dict` 收斂為 `ProvenanceFacts` Pydantic 模型（欄位：`template`、`preset`、`advisor`、`mutation`、`reason`、`hparam_changes`、`gpu_opt`、`search`）。自由 dict 外送在 strict 模式下被 `guard()` 擋下。

---

## 5. Layer 2 — 分析器註冊表（管道 A：程式告知）

> 需求原文：「如果大模型需要知道資料的一些特性時，只能用我們事先寫好的程式將分析的結果告知。」

### 5.1 機制

LLM **不能執行任意程式**，但可以在決策回應中**點名**要跑哪個分析器：

```python
class _RecipeDecision(BaseModel):
    ...
    request_analysis: list[str] = Field(default_factory=list)  # 只接受註冊表 key
    request_reason: str = ""
```

`LoopController` 收到 `request_analysis` 後：

1. 過濾掉不在註冊表中的 key（拒絕並在對話中告知使用者）；
2. 檢查 `cost` 與本輪預算（`near_dup` 之類昂貴分析需要使用者同意才跑）；
3. 在**資料平面**執行分析器；
4. 輸出以各自的 `output_schema` 驗證 → 經 `redact.guard()` → 加入下一輪 prompt 的 `AnalysisFacts`；
5. 分析結果快取在 `runs/<run>/analysis/<key>.json`，同一 run 不重跑。

### 5.2 註冊表契約

```python
@dataclass
class Analyzer:
    key: str
    description: str          # 給 LLM 看的說明（寫在 system prompt 的資源目錄裡）
    output_schema: type[BaseModel]
    cost: Literal["cheap", "moderate", "expensive"]
    needs_consent: bool
    fn: Callable[[str], BaseModel]   # fn(dataset_root) -> 已驗證的輸出
```

**關鍵約束**：`output_schema` 的所有欄位型別必須是 `int` / `float` / `bool` / `Literal[...]` / 上述型別的 `list` 或巢狀模型。**不得出現 `str`**（除非是 `Literal` 列舉）。這條由 `tests/privacy/test_analyzer_schema.py` 在 CI 強制執行——沒有自由字串，就沒有夾帶通道。

### 5.3 初版分析器清單

| key | 說明 | 輸出（示意） | cost | 狀態 |
| --- | --- | --- | --- | --- |
| `class_balance` | 各類別樣本數、不平衡比、Gini、有效樣本數、建議 class weight | `counts[]`、`effective_n[]`、`suggested_weights[]`、`recommend_focal_loss` | cheap | ✅ |
| `image_stats` | 抽樣量測尺寸／長寬比百分位、通道模式張數、亮度直方圖（**bin 計數，非個別值**） | `width_p5/p50/p95`、`aspect_p50`、`brightness_hist[16]`、`mean_saturation` | cheap | ✅ |
| `corrupt_files` | 抽樣可讀性檢查 | `n_sampled`、`n_unreadable`、`n_truncated`、`blocking` | cheap | ✅ |
| `split_leakage` | train/val/test 的影像雜湊交集**數量** | `n_dup_train_test`、`leak_ratio_test`、`n_dup_cross_class` | moderate | ✅ |
| `near_dup` | perceptual hash（dHash）近重複**群數與比例** | `n_clusters`、`max_cluster_size`、`dup_ratio`、`n_clusters_cross_split` | expensive | ✅ |
| `label_noise_proxy` | 以 baseline linear probe 的 per-class 混淆**比例** | `confusion_ratio[][]` | expensive | ⬜ 未實作 |

`basic_counts` 未獨立成分析器 —— split × 類別樣本數已經是 `DatasetFacts` 的固定欄位，
不需要 LLM 點名。

注意 `split_leakage` / `near_dup` 只回**數量**，永不回哪些檔案。使用者若要知道是哪些檔案，在本機 UI 看，不經 LLM。

---

## 6. Layer 3 — 詢問使用者（管道 B）

> 需求原文：「或是直接向使用者詢問，讓使用者自己輸入這些資訊。」

### 6.1 契約

LLM 可在任一決策回應中附帶：

```python
class UserQuestion(BaseModel):
    key: str                              # 存進 user_facts.json 的欄位名
    question: str                         # 給使用者看的問題
    kind: Literal["choice", "multi", "number", "bool", "short_text"]
    options: list[str] = []               # kind=choice/multi 時
    why: str                              # 為什麼需要這個資訊（顯示給使用者）
    blocking: bool = False                # True = 沒答案就不決策；False = 可用預設繼續
```

### 6.2 流程

1. LoopController 收到 `questions` → `conversation.append(role="llm", kind="question", payload=...)`。
2. Web UI 在討論頻道渲染成一張**表單卡**（下拉／勾選／數字框／短文字），而不是純文字訊息。
3. 使用者送出 → 存入 `runs/<run>/user_facts.json`（append-only，含 ts 與問題原文）。
4. 下一輪 prompt 帶入 `UserProvidedFacts`，並標明「此為使用者親自填寫」。
5. `blocking=True` 且逾時（預設 30 分鐘）→ 在對話中說明「未取得回覆，改用保守預設繼續」，不卡死實驗。

### 6.3 使用者輸入的圍欄

使用者自己填的內容**仍會經過 `guard()`**，但行為不同：

- 命中疑似洩漏樣式（絕對路徑、檔名、base64）→ **顯示警示並要求二次確認**，不直接拒絕。使用者有權自願揭露。
- `short_text` 欄位長度上限 200 字元，且在 strict 模式下預設停用（只允許 choice / number / bool），把自由文字通道收窄。
- 所有經此管道送出的內容一律記入 `privacy_audit.jsonl`，UI 明確標示「這段文字會送到 Claude API」。

同樣的警示套用在既有的討論框與 `guidance` 欄位（L8）。

---

## 7. Layer 4 — 出口管制

### 7.1 `egress.call()` — 單一出口

`llm_advisor.py` 內所有 `client.messages.create` / `client.messages.stream`（`llm_advisor.py:244`、`llm_advisor.py:545`）改為：

```python
from .privacy import egress

resp = egress.call(
    label="compose_recipe",
    model=self.model,
    system=system,
    prompt=prompt,
    schema=_RecipeDecision,
    run_dir=self.run_dir,
)
```

`egress.call()` 依序做：

1. **guard**：對 `system + prompt` 全文執行掃描器（§7.2）。命中 → 拋 `EgressViolation`，**不送出**。
2. **audit**：把 `{ts, label, sha256, n_chars, verdict, matched_rules, payload}` 寫入 `runs/<run>/privacy_audit.jsonl`（既有的 `llm_calls.jsonl` 合併進來，避免兩份）。
3. **送出**並回傳。
4. 對回應也掃一次（低優先，防模型把不該回的東西複述回來造成落檔）。

### 7.2 `guard()` — 出口掃描器

這是**縱深防禦**，不是主要防線（主要防線是 §4 的允許清單）。規則：

| 規則 | 行為 |
| --- | --- |
| payload 含 `data_root` 的實際絕對路徑或其任一祖先路徑 | **block** |
| payload 含 `data/` 底下實際存在的目錄名或檔名（比對 `dataset_registry` 快取 + 抽樣檔名集合） | **block** |
| 影像副檔名樣式 `\.(jpg\|jpeg\|png\|tif\|tiff\|bmp\|webp)\b` | **block** |
| base64 影像 magic（`/9j/`、`iVBORw0KGgo`、`R0lGOD`） | **block** |
| 連續數值陣列長度 > 64（張量傾印） | **block** |
| 未經假名化的類別名（比對 `alias_map.json` 的原始名） | **block** |
| 疑似身分識別（身分證字號／病歷號常見樣式） | **block** |
| 使用者親自輸入的區段命中上述規則 | **warn + 需確認**（§6.3） |

掃描以「run 相關的真實字串集合」為基準做**精確比對**，而非泛用 PII 正則——這樣誤判率低且不可能漏掉本 run 的資料識別字串。

### 7.3 `sentinel.py` — 堵住繞道

光靠「請大家都走 egress」不構成圍欄。`agent/privacy/sentinel.py` 在 `agent/__init__.py` import 時安裝：

```python
def install():
    import anthropic
    orig = anthropic.resources.messages.Messages.create
    def guarded(self, *a, **kw):
        if not _called_from_egress():      # 檢查 call stack 是否含 privacy/egress.py frame
            raise EgressViolation(
                "偵測到未經資料圍欄的 Claude API 呼叫。所有呼叫必須經由 agent.privacy.egress。")
        return orig(self, *a, **kw)
    anthropic.resources.messages.Messages.create = guarded
    # stream / 其他 client 同理
```

如此一來，未來任何人（包含 LLM 自己改程式）新增一條 prompt 建構路徑，都會在第一次執行時就爆掉，而不是靜默洩漏。

### 7.4 `SkillAdvisor` 交握（L9）

`skill_advisor.py` 寫給 Claude Code 的 `req_*.json` 是**同等級的出口**。處理方式：

- payload 一律改用 `DatasetFacts` / `ErrorFacts`（與 LLMAdvisor 共用同一組消毒函式）；
- 寫檔前呼叫 `egress.guard_payload()`，命中即拒寫；
- 交握目錄的每次寫入同樣記入 `privacy_audit.jsonl`。

---

## 8. Layer 5 — 關掉「主動讀取」能力（L4 / L5）

這是最重要的一節。單純消毒欄位擋不住「LLM 改程式把資料印進 log、下一輪讀回來」。

### 8.1 原始 log 不再回饋

`loop_controller._debug_recipe`（`loop_controller.py:356-372`）目前把 log 最後 120 行原文交給 LLM。改為：

```
raw log ──▶ analyzers/error_extract.py ──▶ ErrorFacts ──▶ prompt
```

```python
class ErrorFacts(BaseModel):
    exc_type: str | None            # 限定於已知例外型別的列舉，未知者標 "Unknown"
    error_class: Literal["oom", "shape_mismatch", "nan_loss", "dataloader",
                         "checkpoint", "cuda", "config", "unknown"]
    message_template: str           # 去除所有路徑/數字後的樣板（白名單抽取，非黑名單過濾）
    tensor_shapes: list[list[int]]  # 從 shape mismatch 訊息抽出的形狀
    at_epoch: int | None
    at_step: int | None
    n_lines_scanned: int
    last_train_loss: float | None
    last_val_loss: float | None
```

`message_template` 採**允許清單抽取**：只有比對到已知錯誤樣板（CUDA OOM、`RuntimeError: shape ... is invalid`、`FileNotFoundError`、`ValueError: num_samples should be a positive integer` 等）才產生對應樣板字串，其餘一律 `unknown`。**不是**把原文做正則替換後放行——那種做法永遠有漏網。

`privacy.log_feedback` 設定：`none` | `structured`（預設）| `raw`（僅 `mode: off`）。

### 8.2 程式修改權

| 模式 | `allow_code_edit` | 說明 |
| --- | --- | --- |
| `strict` | **強制 False**，UI 灰掉且不可勾 | 沒有寫入能力，就沒有主動讀取通道。這是唯一能給出「證明」的組態 |
| `standard` | 可開，但受 §8.3 AST 檢查 + `log_feedback` 強制為 `structured` | 寫入受限 **且** 讀回路被切斷，需同時突破兩層 |
| `off` | 不受限 | 需在 UI 打字輸入資料集名稱確認，並在每頁顯示紅色橫幅 |

### 8.3 `code_edits` 的 AST 檢查器（standard 模式）

在 `code_workspace.create()` 套用編輯前，對**編輯後的檔案**做 AST 分析，命中即拒絕該編輯：

- 任何 `print` / `logging.*` / `sys.stdout.write` 的引數運算式中，出現對 `samples`、`images`、`targets`、`batch`、`data_loader`、`dataset` 等名稱的引用；
- 呼叫 `open`、`glob`、`os.walk`、`os.listdir`、`PIL.Image.open`、`np.save`、`torch.save`（輸出目錄白名單以外）；
- `.tolist()`、`.numpy()`、`.cpu()` 的結果流向任何輸出函式；
- 新增 `import` 非白名單模組（`socket`、`requests`、`urllib`、`subprocess`、`http`）；
- 修改 `util/datasets.py` 的 `__getitem__` / `loader`。

文件必須誠實記載：**AST 檢查是 best-effort，不是證明**。真正的保證來自 strict 模式的「不給寫入權」+「不回饋 log」。UI 上要如實呈現這個差別，不能讓使用者以為 standard = 安全。

---

## 9. 設定

`agent/config.py` 新增 `PrivacyConfig`：

```yaml
privacy:
  mode: strict              # strict | standard | off
  dataset_alias: true       # 路徑/名稱假名化
  class_names: hashed       # hashed | user_approved | plain
  log_feedback: structured  # none | structured | raw
  allow_free_text_questions: false   # 是否允許 LLM 向使用者要 short_text
  egress_audit: true
  guard_on_user_input: warn # warn | block | off
  salt_file: ~/.medclaw/privacy_salt   # 假名 HMAC salt，不進版控
  expensive_analyzers_need_consent: true
```

`mode` 是**巨集**，會覆寫個別欄位：

| | `strict` | `standard` | `off` |
| --- | --- | --- | --- |
| `dataset_alias` | true | true | false |
| `class_names` | hashed | user_approved | plain |
| `log_feedback` | structured | structured | raw |
| `advisor.allow_code_edit` | **強制 false** | 可開（受 AST 檢查） | 不限 |
| `guard()` 命中 | block | block | 僅記錄 |
| `allow_free_text_questions` | false | true | true |

`mode` 預設 `strict`。`AgentConfig.model_validator` 在載入時執行覆寫，讓「config 裡寫了 strict 卻同時開 code_edit」不可能成立。

---

## 10. Web UI

### 10.1 隱私分頁（`/privacy`）

新增分頁，是整個圍欄「可證明」的核心：

- **本 run 送出的所有 payload 清單** — 逐筆可展開看全文（那已是消毒後的內容，讓使用者親眼確認）；
- **攔截紀錄** — 哪一次呼叫被 `guard()` 擋下、命中哪條規則；
- **假名對照表** — `DS-a1b2c3 = 5_fold_PAPILA/PAPILA_seed42_fold0`、`C0 = normal`…（僅本機顯示）；
- **目前生效的隱私模式**與各項開關實際值。

### 10.2 其他 UI 變更

| 位置 | 變更 |
| --- | --- |
| 資料集頁 / 工作生成頁 | 新增 **modality / anatomy 下拉**（取代路徑猜測）與「類別名授權」勾選區 |
| 討論框、guidance 欄位 | 送出前顯示「這段文字會送到 Claude API」；命中 guard 規則時彈出二次確認 |
| `allow_code_edit` 勾選框 | strict 模式下灰掉，附說明「strict 模式不允許 LLM 修改訓練程式（避免經由 log 回讀資料）」 |
| 討論頻道 | 支援 `kind="question"` 的表單卡渲染（§6.2） |
| 全域 | `mode: off` 時顯示常駐紅色橫幅 |

---

## 11. 測試與驗證

圍欄的價值等於它的測試強度。

| 測試 | 內容 |
| --- | --- |
| `test_canary_leak.py` | 建立 canary 資料集：目錄 `CANARY_HOSP_9527/`、類別 `patient_wang_dr3` / `ptn_lin_normal`、檔名含 `CANARYTOKEN`。跑完整 dry-run 迴圈（含 debug、QA、improve），斷言 `privacy_audit.jsonl` 全文**不含任何 canary token**。這是主驗收測試 |
| `test_sentinel.py` | 直接呼叫 `anthropic.Anthropic().messages.create` 必須拋 `EgressViolation` |
| `test_analyzer_schema.py` | 遍歷註冊表，斷言所有 `output_schema` 欄位型別不含自由 `str` |
| `test_guard_rules.py` | 對 guard 餵入含路徑／檔名／base64／長數列的 payload，逐條驗證攔截 |
| `test_error_extract.py` | 對真實 log（`runs/*/logs/*.txt`）跑抽取器，斷言輸出的 `message_template` 不含 `/` 與資料集名 |
| `test_ast_guard.py` | 對一組惡意 `code_edits` 樣本（印張量、開檔、外連）驗證全數被拒 |
| `test_mode_macro.py` | `mode=strict` + `allow_code_edit=true` 的 config 載入後必須是 `False` |

CI 另加一條 **grep 檢查**：`agent/` 底下除 `privacy/egress.py` 外不得出現 `messages.create` / `messages.stream` 字樣。

---

## 12. 分階段落地

| 階段 | 內容 | 產出 | 狀態 |
| --- | --- | --- | --- |
| **P0 — 圍欄骨架** | `privacy/` 模組、`DatasetFacts`、`egress` 單一出口、`sentinel`、`PrivacyConfig` + mode 巨集、canary 測試。strict 下強制關 `allow_code_edit`、`log_feedback=structured` | 不變式成立，可驗證 | ✅ |
| **P1 — 結構化錯誤與分析器** | `analyzers/` 註冊表 + 5 個分析器、`error_extract`、`request_analysis` 回路 | 管道 A 完成；debug 能力回復 | ✅ |
| **P2 — 詢問使用者** | `UserQuestion` 契約、conversation `kind="question"`、Web 表單卡、`user_facts.json`、modality/anatomy 下拉 | 管道 B 完成 | ✅ |
| **P3 — 可證明性 UI** | 獨立 `/privacy` 分頁、假名對照表、逐筆 payload 全文 | 使用者能自行稽核 | ⬜ 工作台已有稽核摘要列 |
| **P4 — standard 模式** | `code_edits` AST 檢查器、類別名逐一授權 UI、`label_noise_proxy` | 在可控前提下放寬 | ⬜ |
| **P5 — SkillAdvisor 對齊** | 交握 payload 改用 facts、寫檔前 guard | 補上 L9 | ✅ |

P0 完成即滿足需求的核心；P1–P2 補回被圍欄拿掉的資訊管道；P3 讓圍欄可被檢驗。

---

## 13. 影響評估與取捨

### 13.1 決策品質

拿掉的資訊與影響：

| 拿掉 | 對決策的實際影響 |
| --- | --- |
| 資料集路徑／名稱 | **無**。encoder 選擇本來就依 `num_classes` / `n_train` / `imbalance_ratio` / `image_size` / modality |
| 類別名稱 | **輕微**。少數情境（判斷是否為序數任務、是否適合 ordinal loss）需要 → 由 §6 直接問使用者，答案更可靠 |
| modality 路徑猜測 | **正向**。`dataset_analyzer.py:53` 的關鍵字比對本來就脆弱（路徑沒關鍵字就是 `None`）。改由使用者選是嚴格改善 |
| 原始 log | **中等**。debug 品質取決於 `error_extract` 的樣板覆蓋率。緩解：初版覆蓋 OOM / shape / NaN / dataloader / checkpoint 五大類（實測佔失敗案例絕大多數），未覆蓋者標 `unknown` 並在對話中提示使用者自行查看 log |
| 程式修改權（strict） | **中等**。這是明確的能力換安全。standard 模式保留此能力給不需要最高保證的情境 |

### 13.2 已知限制（必須誠實記載於 README）

1. **AST 檢查器不是證明**。standard 模式的保證強度低於 strict。
2. **使用者自願揭露無法阻止**。圍欄只警示與記錄。
3. **側通道未涵蓋**。LLM 理論上可經由「選擇哪個 hyperparameter」對外傳遞極低頻寬的資訊；在本威脅模型（防意外洩漏與防 prompt 夾帶，非防蓄意惡意模型）下不處理。
4. **圍欄保護的是「送出去的內容」**，不涵蓋 Anthropic 端的處理與留存政策。

### 13.3 相容性

- `DatasetProfile` schema 不變 → 既有 `runs/*/dataset_profile.json` 可正常載入、`resume` 不受影響。
- 既有 run 的 `llm_calls.jsonl` 保留；新欄位（verdict）只出現在新紀錄。
- `advisor.type=heuristic` 完全不受影響。

---

## 14. 已決事項

1. **假名 salt 的生命週期** — 採 per-machine 固定，存於 `~/.medclaw/privacy_salt`（0600，不進版控）。同一份資料在不同 run 得到相同假名，LLM 可跨 run 關聯，但外人無法反推。
2. **`error_extract` 樣板覆蓋率** — 初版覆蓋 8 類 14 個樣板（OOM ×2、shape ×3、checkpoint ×2、NaN、dataloader ×3、CUDA ×2、config）。對 repo 內既有 `runs/*/logs/` 實測，唯一的失敗案例（CUDA-capable device busy）被正確歸類。未比對到樣板者一律 `unknown` 並標 `truncated=true`，讓決策層知道自己資訊不全。
3. **`request_analysis` 是否需要使用者逐次核准** — 只有 `cost="expensive"` 且 `needs_consent=True` 的分析器需要（目前只有 `near_dup`）。同意與否本身走管道 B 的問答卡，答案記在 `user_facts.json` 的 `consent:<key>`。
4. **多 fold 情境的 `dataset_ref`** — 採共用前綴 `DS-a1b2c3#f0` / `#f1`，讓 LLM 知道它們同源。`trial_id` 中的 `#` 會被換成 `_`（檔案系統可讀性）。

---

## 15. 實作現況與設計差異

程式已落地 P0–P2 與 P5。以下是實作與本文件初稿不同、值得記錄的幾點：

| 項目 | 設計初稿 | 實作 | 理由 |
| --- | --- | --- | --- |
| 資訊蒐集入口 | `request_analysis` 掛在每個決策的回應上 | **兩者都有**：新增 `Advisor.plan_information()` 作為實驗開跑前的專門階段，同時每個決策仍可附帶 `request_analysis` / `questions` | 開跑前集中問一次，比每輪零星要求更省 token 也更好懂；但 improve 階段發現新問題時仍需要能追問 |
| `NextAction` | 未變動 | 新增 `info: InfoRequest` 欄位 | 讓「決策 + 資訊需求」在同一次回應裡帶回，不必多一次呼叫 |
| `propose_debug` 介面 | 概念上換成 ErrorFacts | 簽章改為 `propose_debug(profile, encoder, trial, error_facts, workspace_dir, log_tail=None)`，三個 Advisor 實作全部對齊 | `log_tail` 只在 `privacy.log_feedback="raw"`（僅 `mode: off` 可設）時才會被填 —— 讓設定真的有意義，而不是偷偷忽略 |
| 稽核檔 | 與 `llm_calls.jsonl` 合併 | 分成兩份：`privacy_audit.jsonl`（verdict / 違規 / 全文，圍欄權威紀錄）與既有的 `llm_calls.jsonl`（web UI 相容） | 合併會動到現有 UI 的讀取路徑；payload 重複落地的磁碟成本（每 run 數 MB）可接受 |
| 出口掃描的數值傾印門檻 | 「長 numeric array」 | 單一連續數值序列 ≥ 512 個 | 逐 epoch 曲線最長數百點，224×224 影像傾印是 5 萬點 —— 中間有足夠餘裕，不會誤攔正常 prompt |
| 掃描的誤判防護 | 未提 | `Guard.build(exempt_text=...)` 以 encoder registry / preset / task template 的靜態文字建立豁免集合 | 否則資料集若叫 `fundus`，`task_template: fundus_classification` 會被自己的圍欄擋下來 |
| 可證明性 UI | `/privacy` 分頁 | 工作台聊天區上方的稽核摘要列（送出次數 / 攔截 / 警示 / 最近一次攔截的規則） | 分頁留到 P3；摘要列已足以讓使用者察覺攔截 |

### 15.1 落地檔案

```
agent/privacy/     facts.py（唯一可進 prompt 的契約）/ redact.py（消毒 + Guard）
                   egress.py（唯一出口 + 稽核）/ sentinel.py（執行期哨兵）
                   alias.py（假名）/ context.py（PrivacyContext + user_facts IO）
agent/analyzers/   __init__.py（註冊表 + schema 檢查）/ class_balance / image_stats
                   corrupt_files / split_leakage / near_dup / error_extract
tests/privacy/     19 項測試，`python -m tests.privacy.run_all`
runs/<run>/        privacy_audit.jsonl · alias_map.json · user_facts.json · analysis/*.json
```

### 15.2 驗收

`tests/privacy/test_canary_leak.py` 建立 canary 資料集（目錄 `CANARYHOSP9527/PTNSET7788_seed42_fold0`、
類別 `PATIENTWANG_dr3` / `LINMEIHUA_normal`、檔名 `CANARYTOKEN0001_*.jpg`），跑遍決策層
**全部八條** prompt 建構路徑，斷言沒有任何 canary 字串進入 payload；並反向驗證圍欄
確實會攔（故意讓 prompt 帶真實路徑 → `EgressViolation` + 稽核紀錄）。

本 repo 目前沒有 pytest，`tests/privacy/run_all.py` 是零依賴的執行器；測試函式命名相容
pytest，日後裝了直接 `pytest tests/privacy` 也能跑。
