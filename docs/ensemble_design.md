# MedClaw 模型集成 (Ensemble) — 設計文件

- 狀態：v0.2 — **方案 A 最小可用版已實作**（等權 / val 加權 + heuristic 選成員 + 收尾集成 + 報告）；LLM 選成員、stacking、方案 B 樹整合待做
- 適用範圍：`agent/` 自動微調 Agent（MedClaw）的多模型集成
- 相關文件：[auto_finetune_agent_design.md](auto_finetune_agent_design.md)、[data_firewall_design.md](data_firewall_design.md)、[model_registry_design.md](model_registry_design.md)
- 相關程式：`agent/evaluator.py`、`agent/metric_registry.py`、`agent/loop_controller.py`、`agent/journal.py`、`agent/llm_advisor.py`、`agent/schemas.py`、`engine_finetune.py`

---

## 1. 目標與動機

樹搜尋（AIDE 式）本來就會在**多個 encoder、多種 recipe** 上訓練出一批模型，且決策層刻意追求來源多樣（MAE vs Dino、自然 vs 醫療）。這正是集成（ensemble）最理想的素材：**多樣且各自夠強的成員，平均後通常勝過任一單一模型**。

**目標：讓 agent 能把已訓練的多個 trial 組成 ensemble，作為可比較、可交付的一種「動作」，在單模型改良趨於平坦時進一步提升下游效能。**

### 1.1 核心洞察：集成幾乎零成本

每個 trial 訓練結束時，`engine_finetune.evaluate` 已把**逐樣本 softmax 機率**落地：

- `predictions_val.csv` / `predictions_test.csv`，欄位 `true_label` + `<cls>_score …`（[engine_finetune.py:146](../engine_finetune.py#L146)）
- `evaluator.read_predictions(task_dir, mode)` 讀成 `(y_true, y_prob)`（[evaluator.py:43](../agent/evaluator.py#L43)）
- 所有 metric 都能從 `(y_true, y_prob)` 重算（[evaluator.py:60](../agent/evaluator.py#L60) `_custom_metric`、`metric_registry`）

因此**機率層級的軟投票集成不需要重新訓練、不需要 GPU**：把選定成員的 `y_prob` 加權平均，再用同一套 metric 函式評分即可。這是本方案優先採用軟投票的根本理由——用「已經付出的訓練成本」換取額外效能。

---

## 2. 資料圍欄定位（關鍵）

集成天然地把工作分成兩個平面，正好落在圍欄兩側：

| 動作 | 平面 | 說明 |
| --- | --- | --- |
| **選哪些成員、用什麼集成法、（可選）權重方向** | 決策平面（LLM 可見） | 只看 `TrialFacts`（trial_id 假名、metrics、encoder provenance、曲線），**選 trial_id**，不碰資料 |
| **讀 predictions、加權平均、算 metric** | 資料平面（本地程式） | `predictions_*.csv` 是**逐樣本記錄＝資料**，LLM 永遠看不到；由 `ensembler` 在本地執行 |
| **回饋給 LLM 的結果** | 決策平面 | 只有**集成後的彙整 metric**（與單一 trial 同格式的 facts）+ 使用的權重（少數係數，非逐樣本資料） |

換言之：**LLM 決定「集成策略」（metadata），本地程式執行「集成計算」（data），只有彙整指標回流**——與現有 `propose_next` / `evaluator` 的分工完全一致，不新增任何資料外洩通道。

> 權重最佳化（§4.2）若需要，是在 **val predictions** 上由本地程式求解，回傳的是少量係數（如 `[0.4, 0.35, 0.25]`），屬彙整量，可安全回饋給 LLM，等同 metric。

---

## 3. 集成成員的來源

成員候選 = journal 中 `status == "done"` 的 trial（跨整棵解答樹、跨 encoder）。選擇準則：

1. **效能門檻**：val primary 不低於「最佳單模型 − δ」（排除明顯弱者拖累）。
2. **多樣性**：優先納入**不同 encoder／不同 domain／不同 recipe** 的成員（軟投票的增益主要來自成員間的預測不相關）。`TrialFacts.provenance` 已含 encoder 與變異來源，足供 LLM 判斷。
3. **成員數**：預設 2–5 個；過多會攤薄強成員、邊際遞減。

選擇可由兩種決策層執行（與現有 advisor 架構一致）：

- **HeuristicAdvisor**：規則式——取 val top-K，並強制涵蓋不同 encoder（貪婪去相關）。
- **LLMAdvisor**：看 `TrialFacts` 挑成員（可解讀多樣性與曲線健康度），輸出 `EnsembleSpec`。

---

## 4. 集成方法

三種，複雜度遞增；v1 先做前兩種。**所有「調權重／學參數」一律在 val 上進行，test 只用於最終報告，杜絕洩漏。**

### 4.1 等權軟投票（equal soft-vote）— v1 預設
`y_prob_ens = mean_k(y_prob_k)`，成員機率算術平均。零參數、最穩健。

### 4.2 驗證集加權軟投票（val-weighted）— v1
在 `predictions_val.csv` 上尋找非負權重 `w`（`sum=1`）最大化 val primary（座標下降／小型網格搜尋，本地程式）；再套到 `predictions_test.csv` 報告。權重回饋給 LLM/使用者作為 provenance。

### 4.3 堆疊（stacking / meta-learner）— v2（可選）
以各成員的 val 機率為特徵，訓練一個輕量 meta-learner（如多類 logistic regression）；在 test 特徵上預測。仍是資料平面的小型 fit，無 GPU。過擬合風險較高，需 val 內再切一折或交叉驗證，列為進階選項。

### 4.4 對齊前提（正確性關鍵）
軟投票要求各成員的 predictions **列對列對應同一批樣本、欄位對應同一類別順序**：

- **樣本對齊**：目前 CSV 只有 `true_label + scores`，靠評估 loader 的**確定性順序**（eval 不 shuffle）保證。**建議在 predictions CSV 增加一個穩定 `sample_id`/index 欄位**，`ensembler` 以此 join，把「順序一致」從假設變成驗證（不一致即拒絕集成並告警）。
- **類別對齊**：`<cls>_score` 欄位順序須一致；同一資料集同一 split 天然一致，`ensembler` 仍應校驗欄位集合相同。
- **前提檢查失敗即中止該次集成**（fail-soft：不影響其餘實驗），不做「猜測性對齊」。

---

## 5. 資料結構（schema）

```python
# schemas.py 新增
class EnsembleSpec(BaseModel):
    member_trial_ids: list[str]                         # 選中的成員 (≥2)
    method: Literal["equal", "val_weighted", "stacking"] = "equal"
    weights: Optional[list[float]] = None               # method=val_weighted 時由本地求得
    rationale: str = ""

class EnsembleResult(BaseModel):
    ensemble_id: str                                    # 例: ens_t7_t4_t9
    spec: EnsembleSpec
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_score: float = 0.0
    member_ckpts: list[str] = Field(default_factory=list)   # 各成員 checkpoint-best.pth
    pred_path: Optional[str] = None                     # 落地的集成 predictions_test.csv
    status: Literal["done", "failed"] = "done"
    message: str = ""
```

`EnsembleResult` 刻意與 `TrialResult` 同構（有 `metrics` / `primary_score`），因此能**用同一套比較邏輯**進 `self.best`、進報告、進樹。決策層看到的是它的 facts 版（沿用 `TrialFacts`，`provenance` 標記 `mutation="ensemble"`、列出成員假名 id）。

---

## 6. 新模組：`agent/ensembler.py`（資料平面）

```python
def combine(run_dir, members: list[TrialResult], spec: EnsembleSpec,
            cfg: EvalConfig) -> EnsembleResult:
    """讀各成員 predictions → 對齊校驗 → 加權平均 → 落地 → 用 evaluator 算 metric。"""
```

- 讀取：對每個成員 `evaluator.read_predictions(task_dir, mode)`（val 用於定權，test 用於報告）。
- 對齊：§4.4 校驗；不通過回 `status="failed"`。
- 定權：`equal` 直接均權；`val_weighted` 在 val 上最佳化；`stacking` 擬合 meta-learner。
- 落地：寫 `runs/<run>/ensembles/<ensemble_id>/predictions_test.csv`（同格式，供報告與復現）。
- 評分：**重用 `metric_registry` / `evaluator.compute_primary`**，與單模型指標同口徑、可直接比較。
- 回傳 `EnsembleResult`。**全程不外送任何逐樣本資料**。

---

## 7. 決策層接口

於三種 advisor（`advisor.py` 抽象、`llm_advisor.py`、`skill_advisor.py`）新增一致方法：

```python
def propose_ensemble(self, profile, history: list[TrialResult]) -> Optional[EnsembleSpec]: ...
```

- **LLMAdvisor**：prompt 帶 `history` 的 `TrialFacts`（既有 `_hist`），指示「若有 ≥2 個夠強且多樣的成員，選出成員與方法；否則回 None」。輸出經 `EnsembleSpec` 驗證，成員 id 必須在 `history` 內（白名單，防幻覺）。權重不由 LLM 憑空給；`val_weighted` 交本地求解。
- **HeuristicAdvisor**：val top-K + encoder 去重的規則版。
- 回 `None` 表示「此刻不值得集成」，迴圈照常。

---

## 8. 與樹搜尋的整合

提供兩種整合強度，建議 **A 先落地、B 作為擴充**。

### 方案 A：收尾綜合步驟（predictions-level，低風險，預設）
在 StopPolicy 觸發停止**之後**（或每達一定輪數的檢查點），自動呼叫 `propose_ensemble`：

1. 若回傳 spec → `ensembler.combine` 產生 `EnsembleResult`；
2. 以同口徑 primary 比較，**勝過最佳單模型才採用**為最終交付，並在報告說明成員與權重；
3. 不改樹的搜尋語意——集成是**葉節點式的最終合成**，不再被 `improve`。

優點：改動小、零 GPU、不觸碰既有 draft/improve/debug/resume 流程；風險低。

### 方案 B：一等公民的 `ensemble` 節點（擴充）
把 `ensemble` 加入 `Stage`，成為 policy／決策層可在**搜尋中途**選擇的動作：單模型改良連續 `patience` 輪無提升時，policy 以一定機率提出集成節點；`EnsembleResult` 作為節點進 journal，可再被納入更高階集成（集成的集成需防過擬合，設深度上限）。

優點：更自動、更強；代價：`journal.Node`／`_search_policy`／`review_search_choice` 需支援無單一 encoder 的節點型別，改動較大。建議在方案 A 驗證有效後再做。

---

## 9. 交付與推論

集成勝出時，交付物需讓使用者能復現推論：

- 報告（`report.py`）新增一段：成員 trial、各自 encoder／score、集成方法與**權重**、集成 test 指標 vs 最佳單模型。
- 落地 `ensembles/<id>/`：`spec.json` + 集成 `predictions_test.csv` + 指向各成員 `checkpoint-best.pth` 的清單。
- 可選：產生 `inference_ensemble.py`——載入 N 個成員 checkpoint、對輸入各自 forward、依權重平均機率。這是一段固定模板程式，非 LLM 生成，無圍欄疑慮。

> 注意：集成推論需**同時載入 N 個模型**，記憶體與延遲為單模型的 ~N 倍。報告應標示此成本，讓使用者權衡「效能增益 vs 部署成本」。

---

## 10. 風險與邊界

| 風險 | 對策 |
| --- | --- |
| test 洩漏（在 test 上調權重） | 定權／學參數一律在 **val**；test 只報告（§4） |
| 樣本／類別未對齊 | §4.4 以 `sample_id` join + 欄位校驗；不通過即中止該次集成 |
| 弱成員拖累 | 效能門檻 + 多樣性選擇（§3）；`val_weighted` 可自動壓低弱成員權重 |
| 機率未校準（不同 encoder 尺度不一） | v2 可加 temperature scaling（在 val 上）再平均 |
| 集成過擬合 val（stacking） | stacking 列為進階、需交叉驗證；預設用零參數等權 |
| 部署成本 N 倍 | 報告標示；提供「集成 vs 最佳單模型」讓使用者選 |

**邊界**：本方案是**預測層級**集成（重用已訓練模型），不涉及聯合訓練或知識蒸餾——那需要新的訓練流程，屬另案。

---

## 11. 實作步驟（待辦）

1. `schemas.py`：`EnsembleSpec` / `EnsembleResult`。
2. `engine_finetune.py`：predictions CSV 增加穩定 `sample_id` 欄（對齊保證）。
3. `agent/ensembler.py`：`combine()`（讀取／對齊／定權／落地／評分），重用 `evaluator`＋`metric_registry`。
4. advisor 三處新增 `propose_ensemble`（facts 驅動、成員白名單）；`TrialFacts.provenance` 標記 ensemble。
5. `loop_controller`：方案 A——停止後的集成綜合步驟 + 與 `self.best` 比較 + 落地。
6. `report.py`：集成成員／權重／對比區塊；可選 `inference_ensemble.py` 模板。
7. 測試：對齊校驗、等權／加權指標重算、洩漏防護（權重只由 val 求）、回 None 路徑、圍欄（`ensembler` 不外送逐樣本）。
8.（擴充）方案 B：`ensemble` 納入 `Stage` 與樹搜尋。

---

## 附錄 A：一次集成的資料流

```
決策層 (LLM)                     資料平面 (ensembler)              決策層
  看 TrialFacts                    讀 predictions_val/test.csv
  選成員 t4,t7,t9  ── spec ──▶     校驗對齊 (sample_id)
  選 method=val_weighted           在 val 上求權重 w=[.4,.35,.25]
                                   test: y_prob_ens = Σ w_k·y_prob_k
                                   evaluator/metric_registry 評分
                                   落地 ensembles/ens_t4_t7_t9/    ── EnsembleResult(facts) ──▶  比較 best、寫報告
             （逐樣本 predictions 全程留在資料平面，未外送）
```
