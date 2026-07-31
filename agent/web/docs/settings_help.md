<!-- key: anthropic_api_key -->
供 `advisor=llm/skill` 呼叫 Claude API 使用。留空＝沿用現有值（不清除）。也可改用環境變數 `ANTHROPIC_API_KEY` 或 `ant auth login`。存於 `~/.medclaw/settings.json`（權限 600，不進 git）。

**經 OpenRouter 等代理端點時這個欄位不用填**：改設環境變數 `ANTHROPIC_BASE_URL`（例 `https://openrouter.ai/api`）+ `ANTHROPIC_AUTH_TOKEN`（`sk-or-...`）+ `MEDCLAW_LLM_MODEL`（例 `anthropic/claude-opus-4.5`）。有設這兩個變數時本欄位的 key 會被忽略（改用 bearer 認證）。範例見 repo 根目錄的 `.env`。

<!-- key: advisor_type -->
決策層：**llm**（Claude 決策，最聰明，需金鑰）／**heuristic**（規則式，免金鑰、可離線、可重現）／**skill**（委派 Claude Code skill）。新實驗的預設，工作台表單可覆寫。

<!-- key: model -->
`advisor=llm` 使用的 Claude 模型 id（例：`claude-opus-4-8`）。沒有表單欄位，完全由這裡決定。

<!-- key: num_drafts -->
一開始跨 encoder 各開幾條**新起點（draft）**——多樣性來源。越多越廣、但越花時間。

<!-- key: max_trials -->
全域 **trial 總數上限**（＝樹搜尋總輪數）。達到即停。

<!-- key: min_trials -->
至少跑滿這麼多輪，才允許 **patience 提早停**（保護早期探索）。

<!-- key: patience -->
連續這麼多輪 **improve/resume 無提升**就停；draft/debug 不計入。

<!-- key: improve_temperature -->
選「要改良哪個節點」時的抽樣溫度。越大越探索（給落後節點機會）；`≤0` = greedy 只選目前最佳。

<!-- key: debug_prob -->
每輪以此機率優先去**除錯失敗的葉節點**（而非改良成功節點）。

<!-- key: max_debug_depth -->
連續除錯鏈的長度上限；超過就放棄該分支，避免在壞掉的節點上無限重試。

<!-- key: resume_epochs -->
續訓策略：選中節點的曲線還沒收斂時，從其 checkpoint **多跑這麼多 epoch**。

<!-- key: max_resumes -->
同一分支連續續訓的次數上限（仍不收斂就放棄續訓）。

<!-- key: primary_metric -->
迴圈最佳化與排名的**主要指標**。`score`＝`(f1+roc_auc+kappa)/3`；不平衡資料建議 `f1`／`balanced_accuracy`／`kappa`，重視漏診用 `recall`。

<!-- key: privacy_mode -->
資料圍欄：**strict**（LLM 完全看不到資料、不可改程式；醫療建議）／**standard**（可改 `main_finetune.py`，原始 log 仍不回饋）／**off**（關閉，不建議）。

<!-- key: ensemble_enabled -->
搜尋停止後，把多個已訓練 trial 以**機率軟投票**組成 ensemble（重用既有 predictions，不重訓）。勝過最佳單模型才採用。

<!-- key: ensemble_method -->
集成方法：**equal**（等權）／**val_weighted**（在 val 上求權重）／**stacking**（val 上訓練 meta-learner）。權重／參數一律在 val 上求，test 只報告。

<!-- key: ensemble_llm_select -->
成員選擇是否用 **LLM**（獨立於 advisor）。關＝一律規則式（val top-K + encoder 去重）；開＝LLM 依 TrialFacts 選（不可用時自動退回規則式）。

<!-- key: ensemble_min_members -->
組 ensemble 至少要幾個成員；不足就不集成。

<!-- key: ensemble_max_members -->
最多納入幾個成員；過多會攤薄強成員、邊際遞減。

<!-- key: ensemble_member_delta -->
成員門檻：primary 需 ≥「最佳單模型 − 此值」才納入（排除明顯落後者拖累）。

<!-- key: ensemble_in_search -->
方案 B：單模型改良進入**平坦期**時，於搜尋『中』就先組 ensemble，並隨模型池成長重試（不干擾單模型樹搜尋）。

<!-- key: ensemble_search_patience -->
搜尋中集成的觸發門檻：連續這麼多輪 improve/resume 無提升就觸發一次。

<!-- key: ensemble_max_search_ensembles -->
整個 run 中「搜尋中集成」的次數上限（避免每輪重跑）。

<!-- key: gpu_pack_low_util -->
逐 fold 平行（`aggregation=per_fold`）時：偵測到某卡 **util 低且記憶體也低**，就在該卡**加開第二個 fold** 提高利用率——取代加大 batch（後者會拖慢收斂）。

<!-- key: gpu_max_folds_per_gpu -->
每顆 GPU 最多同時幾個 fold。`1`＝嚴格一 fold 一卡（不共用）；`2`＝低利用率時最多兩個共用。

<!-- key: gpu_pack_util_below -->
共卡觸發條件之一：該卡即時 GPU 利用率低於此百分比。

<!-- key: gpu_pack_mem_below -->
共卡觸發條件之二：該卡已用記憶體比例低於此值（0–1），確保塞得下第二個 fold。
