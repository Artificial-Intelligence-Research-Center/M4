---
name: finetune-advisor
description: >-
  MedClaw 的決策層 (Advisor) 模板。當需要「根據使用者提供的資料集，建議
  一個或多個預訓練 encoder、組出 decoder / 訓練 Recipe、提出超參數，或提出改良方案
  以提升下游任務效能」時使用。也用於 M4 fine-tune 的 model/encoder/decoder selection、
  hyperparameter 建議、recipe 變異 (regularizer / augmentation / multi-task 等) 決策。
  This skill is the decision layer (SkillAdvisor) for the M4 auto-finetuning agent:
  given a dataset profile, recommend encoders, compose a training recipe, propose
  hyperparameters, and suggest performance-improving mutations.
---

# finetune-advisor (M4 自動微調決策層 — 模板)

> 狀態：**Template v0（最低限制）**。這是佔位骨架，決策範圍刻意保持寬鬆，將來再視需要收斂。
> 完整架構與契約定義見 [`docs/auto_finetune_agent_design.md`](../../../docs/auto_finetune_agent_design.md)。

## 這個 skill 做什麼

作為 M4 自動微調 Agent 的 **Advisor（決策層）**，根據資料集特性做出以下決策：

1. **選 encoder（多個）** — 依 `DatasetProfile` 從 encoder 目錄挑 1 個以上預訓練 backbone。
2. **組 Recipe** — 為選定 encoder 組出可組合的訓練配方：decoder head、pooling、regularizer、augmentation、loss、超參數。
3. **提改良 / 變異** — 看已完成 trial 的結果，提出對 Recipe 的改良動作（加 regularizer / 換 head / 調 augmentation / 加輔助任務 / 調超參）或決定停止。
4. **（未來）提示下游任務起點** — 依資料建議不同 `TaskTemplate` 供選擇。

## 何時使用

- 使用者要對某資料集自動挑模型 / 組微調配方 / 調參 / 想辦法提升效能時。
- 被 Agent 主程式（`Advisor` 介面的 `SkillAdvisor` 實作）呼叫時。

## 輸入

- `DatasetProfile`（由 `DatasetAnalyzer` 產出；或直接給資料集路徑，由本 skill 觸發分析）。
- 可用資源目錄：`EncoderRegistry`、`ComponentRegistry`、`TaskTemplate` 清單。
- （改良時）已完成的 `TrialResult` 歷史。

## 輸出（契約，schema 見設計文件 §6）

依被要求的決策，回傳對應物件：
- `list[EncoderChoice]` — 建議的 encoder（含 `adaptation: finetune|lp` 與理由）。
- `Recipe` — 完整可組合訓練配方。
- `NextAction` — 改良動作或停止決定。
  ⚠ `stop` 與 `prune_branch` 別混用：`stop=true` **結束整個實驗**（所有分支都收斂、
  使用者要求、或再跑任何 trial 都不值得）；只是這一輪的變異基準節點走不通（例如
  凍結特徵無訊號、權重載入不完整）時請用 `prune_branch=true`，搜尋會自動改從樹上
  其他節點繼續。

**輸出即契約**：只回結構化決策物件，不直接產生訓練程式碼；實體建構交給 Agent 的 `RecipeBuilder`。

## 決策指引（最低限制版 — 待補）

目前**不施加硬性規則**，僅列鬆散原則，將來再收斂為明確政策：

- 小資料 / 高類別不平衡 → 傾向 linear-probe 或較小學習率、較強 regularization / weighted loss。
- 醫療影像領域 → 優先考慮醫療 DAP 預訓練 encoder（若目錄中可取得）。
- 多 encoder 時求「來源多樣」（自然影像 vs 醫療、MAE-based vs Dino-based）以利比較。
- 改良動作應**有限、正交、可驗證**：一次只變異一個面向，並記錄 provenance。

> ⚠️ 以上為佔位建議，非強制。實際政策待專案後續定義（見設計文件 §11 待決問題）。

## 邊界

- 建議必須落在 `EncoderRegistry` / `ComponentRegistry` 白名單內（避免產生不存在的元件）。
- Recipe 的 Head × Task × Loss × Metric 需相容（由 `RecipeBuilder` 驗證，不相容則退回重議）。
- 除此之外，本模板**不預設其他限制**；範圍將來再調整。
