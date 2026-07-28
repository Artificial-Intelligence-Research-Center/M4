# M4 自動微調 Agent

設計文件：[`../docs/auto_finetune_agent_design.md`](../docs/auto_finetune_agent_design.md)
資料圍欄：[`../docs/data_firewall_design.md`](../docs/data_firewall_design.md)
資料集格式與匯入：[`../docs/dataset_format.md`](../docs/dataset_format.md)
決策層 skill 模板：[`../.claude/skills/finetune-advisor/`](../.claude/skills/finetune-advisor/)

> **資料圍欄 (Data Firewall)**：LLM 永遠看不到使用者的輸入資料 —— 影像、檔名、路徑、
> 真實類別名都不會進入 prompt。決策層看到的是假名化後的 `DatasetFacts`；它需要知道
> 資料特性時，只能 (a) 點名執行 `agent/analyzers/` 裡**事先寫好**的分析器，或
> (b) 提出問題請使用者親自回答。所有 API 呼叫必須經 `agent/privacy/egress.py`，
> `privacy/sentinel.py` 會在執行期擋下任何繞道。預設 `privacy.mode: strict`。

給定資料集與預測需求，自動完成 encoder 選擇 → Recipe 組合 → 訓練 → 評估 →
改良迭代，並回報效能。Agent 是既有 `main_finetune.py` pipeline 的
**編排層 + 決策層 + 元件組合層**；訓練/評估核心沿用不變。

## 進度 (路線圖 §12)

| 階段 | 內容 | 狀態 |
| --- | --- | --- |
| **P0** 骨架 | 契約 schema + Analyzer + EncoderRegistry + HeuristicAdvisor + Trainer 串單一 trial | ✅ |
| **P1** 多 encoder | `LoopController` 外層廣度比較 + 全域最佳 + `report.md` | ✅ |
| **P2** LLM 決策 | `LLMAdvisor` (Claude structured outputs) 選 encoder / 組 Recipe / 改良 | ✅ |
| **P3** Recipe 元件化 | `ComponentRegistry` + RecipeBuilder；mlp head / weighted_ce / focal / mixup…；`main_finetune` head/loss 組態化 | ✅ |
| **P4** 改良迴圈 | `propose_next` 變異階梯 + provenance + StopPolicy | ✅ |
| **P5** 評估自訂 | `EvalConfig` + `MetricRegistry` 自訂 metric + 多 fold 彙整 (mean_std) | ✅ |
| **P6** 任務擴充 | `TaskTemplate` 目錄 (`fundus_classification` + `regression` 抽象) + `suggest_task_templates` | ✅ |
| **P7** skill 接口 | `SkillAdvisor` (file-based handoff, 介面同 LLM/Heuristic, 可整組抽換) | ✅ |

## 資料圍欄進度 (`../docs/data_firewall_design.md` §12)

| 階段 | 內容 | 狀態 |
| --- | --- | --- |
| **P0** 圍欄骨架 | `privacy/` 模組、`DatasetFacts`、`egress` 單一出口、`sentinel` 執行期哨兵、`PrivacyConfig` + mode 巨集、canary 測試 | ✅ |
| **P1** 分析器通道 | `analyzers/` 註冊表 (5 個分析器) + `error_extract` 結構化失敗事實 + `request_analysis` 回路 | ✅ |
| **P2** 詢問使用者 | `UserQuestion` 契約、`kind="question"` 表單卡、`user_facts.json`、modality/anatomy 下拉 | ✅ |
| **P3** 可證明性 UI | 獨立 `/privacy` 分頁（逐筆 payload 全文、假名對照表） | ⬜ 工作台已有稽核摘要列 |
| **P4** standard 模式 | `code_edits` AST 檢查器、類別名逐一授權流程 | ⬜ |
| **P5** SkillAdvisor 對齊 | 交握 payload 改用 facts + 寫檔前 guard | ✅ |

測試：`python -m tests.privacy.run_all`（19 項；canary 資料集 + 哨兵 + 出口掃描 +
schema 無自由字串 + mode 巨集）。

## 模組對應

| 檔案 | 角色 |
| --- | --- |
| `privacy/` | **資料圍欄**：`facts`（唯一可進 prompt 的契約）/ `redact`（消毒 + 出口掃描）/ `egress`（唯一出口 + 稽核）/ `sentinel`（執行期擋繞道）/ `alias`（假名對照）/ `context` |
| `analyzers/` | **管道 A**：LLM 可點名執行的分析器註冊表（`class_balance` / `image_stats` / `corrupt_files` / `split_leakage` / `near_dup`）+ `error_extract`（log → `ErrorFacts`） |
| `schemas.py` | 資料契約 (§6)：`DatasetProfile / EncoderChoice / Recipe / TrialResult / NextAction / InfoRequest / EvalConfig`。⚠ `DatasetProfile` 屬資料平面，不得直接進 prompt |
| `config.py` | `AgentConfig` — YAML 設定檔契約與載入 (§7) |
| `dataset_analyzer.py` | §5.1 掃 ImageFolder → `DatasetProfile` |
| `encoder_registry.py` | §5.3 encoder 目錄 — 掃描 `baseline_models/<model_key>/model.yaml` 產生 `EncoderCard`（[model_registry_design.md](../docs/model_registry_design.md)）；新增 model = 新增一個目錄，不改程式 |
| `migrate_model_registry.py` | 一次性遷移：`baseline_models/*.pth` 平鋪 → `<model_key>/{model.yaml,weights.pth}` |
| `component_registry.py` | §5.4 可組合元件目錄 + Recipe→CLI 映射 (head/loss/regularizer/aug) |
| `advisor.py` | §5.2 `Advisor` 介面 + `HeuristicAdvisor`（規則式 + P4 變異階梯）+ `build_advisor` 工廠 |
| `llm_advisor.py` | §5.2 `LLMAdvisor` — Claude API structured outputs 決策；prompt 分段與 cache 佈局見 [prompt_structure_design.md](../docs/prompt_structure_design.md) |
| `recipe_builder.py` | §5.4 `Recipe` → `main_finetune.py` 指令（經 ComponentRegistry） |
| `trainer.py` | §5.5 subprocess 包裝 + GPU 可用性判斷（實際配置 CUDA） |
| `evaluator.py` | §5.6 讀 metrics/predictions；EvalConfig 選優 + 自訂 metric |
| `metric_registry.py` | §5.6 自訂 `callable(y_true,y_prob)->float` + 多 fold 彙整 |
| `loop_controller.py` | §5.7 兩層迴圈（外層多 encoder；內層 AIDE 式樹搜尋）+ 多 fold 擴展 |
| `journal.py` | 內層解答樹（`Node`/`Journal`，參考 [WecoAI/aideml](https://github.com/wecoai/aideml)），落地 `search_tree.json` |
| `code_workspace.py` | LLM 修改訓練程式的隔離副本（`<run_dir>/src/`，原始程式不動） |
| `report.py` | §9 `report.md`：各 encoder 最佳 + 全域最佳 + 多 fold 彙整 |
| `ledger.py` | §5.9 `ledger.jsonl` 落地 |
| `task_template.py` | §5.8 下游任務起點目錄（Recipe 骨架 + 相容白名單 + 預設 EvalConfig） |
| `skill_advisor.py` | §5.2 `SkillAdvisor` — 委派給 `finetune-advisor` skill（file-based handoff） |
| `auto_finetune.py` | 設定檔驅動入口 (§7) |
| `run.py` | P0 單一 trial 入口（保留） |
| `dataset_registry.py` | 掃描 `data/` 列出可用資料集 + `validate()` 格式檢查（MedClaw 可否訓練） |
| `dataset_ingest.py` | 上傳資料集：壓縮檔安全解壓（zip-slip 防護）+ 自動找出資料集根目錄 |
| `web/` | 最小 Flask 界面 |

## 用法

環境：`conda activate M4`（本機 python：`/opt/conda/envs/M4/bin/python`）。

### 設定檔驅動（P1+，推薦）

```bash
# dry-run：只組指令、不訓練（無 GPU 可驗證整條 pipeline）
python -m agent.auto_finetune --config agent/config.example.yaml --dry_run

# 實際訓練（需 GPU）
python -m agent.auto_finetune --config agent/config.example.yaml

# 旗標覆寫
python -m agent.auto_finetune --data_path ./data/5_fold_PAPILA/PAPILA_seed42_fold0 \
    --advisor heuristic --num_drafts 3 --max_trials 12 --min_trials 6 --patience 4
```

設定檔範例見 [`config.example.yaml`](config.example.yaml)。關鍵欄位：
`advisor.type` = `heuristic | llm | skill`；`advisor.allow_code_edit`（允許 LLM 修改訓練程式，見下）；
`loop.{max_trials,min_trials,patience,min_delta}`（StopPolicy，見下）；
`loop.{num_drafts,debug_prob,max_debug_depth}`（內層樹搜尋，見下）；
`eval.{primary_metric,report_metrics,aggregation}`（`aggregation=mean_std` 觸發多 fold 彙整）。

### 全域樹搜尋（參考 [WecoAI/aideml](https://github.com/wecoai/aideml) 的 agentic tree search）

整個 run 是**單一全域解答樹**（`journal.py`；不再是每個 encoder 各自的迴圈）：
每個 trial 是一個節點，每輪由 `_search_policy` 決定下一步，並在對話中明說
「本輪從哪個節點開始」——

1. **draft** — 起手先開滿 `num_drafts` 個起點（無 parent），輪流分配給 Advisor
   選出的各 encoder（同一 encoder 再次 draft 時換下一組超參 preset）；
2. **debug** — 以 `debug_prob` 機率挑一個失敗的 leaf（連續除錯鏈 ≤ `max_debug_depth`）
   修正重跑：`Advisor.propose_debug` 讀失敗 log 提出最小修正（heuristic 為規則式：
   OOM→減半 batch、NaN→降 lr）；
3. **resume（繼續訓練策略）** — 抽中的節點訓練完但 curve 未收斂
   （`log_curves.unconverged`：結束時 train_loss 仍下降 / val_score 仍上升，
   且 val_loss 未回升）→ 從其 `checkpoint-best.pth` 續訓 `resume_epochs` 個
   epoch（`--resume` + `--more_epochs`，恢復 optimizer/scaler 完整狀態）。
   防壟斷護欄：符合條件也只以 `resume_prob`（0.5）機率選 resume（其餘落到
   improve）；同一節點**至多一個** resume child（重複續訓＝重複計算）；上一段
   續訓沒帶來提升就不往下追；同分支連續 ≤ `max_resumes`；全 run 連續
   `max_failed_resumes` 次 resume 無提升 → 自動停用 resume 並告知；
4. **improve** — 對所有成功節點的 primary 分數做 **softmax 加權抽樣**
   （`improve_temperature` 控制，≤0 即 greedy）選出本輪改善起點，被選機率會顯示
   給使用者；交 `Advisor.review_and_decide(base=選中節點)` 做一個正交變異長出 child。

metric 回饋自然修剪差的分支；整棵樹落地 `<run_dir>/search_tree.json`
（節點含 encoder/stage/parent/metric/select_prob/overridden/mutation）。

**決策層可以改選節點（`loop.select_override`，預設開）。** policy 是規則＋依指標抽樣，
看不懂討論內容；而 `review_and_decide` 只能在**已選定的節點上**做變異，換不了節點也
開不了 draft。所以討論裡的【QA 結論】若說「下一輪開一條 DINOv2 finetune 的 draft」，
在原本的流程裡是無法落實的。現在 policy 選完後會多一步
`Advisor.review_search_choice(tree, proposal, discussion)`：把整棵樹（每個節點附
`allowed` = 該節點現在允許的階段）和這一輪的選擇交給決策層過目，它可以維持原議，或改選
`draft`（可指定 encoder/adaptation/preset）／`improve`／`debug`／`resume` 的任一節點。

改選一律經 LoopController 驗證——節點要存在、improve 需有成功分數、debug 需為可除錯的
失敗葉節點、resume 需通過續訓護欄（`force=True` 只跳過 `resume_prob` 那道機率關卡，
重複計算與正確性的護欄照舊）、draft 的 encoder/preset 要在目錄裡。不合法就沿用 policy
的選擇，並把原因寫進討論。維持或改選的理由都會顯示給使用者，節點也會標記
`overridden`（解答樹 tooltip 會寫「起點由決策層改選」）。

代價是每輪多一次 LLM 呼叫；`advisor=heuristic` 一律沿用 policy，不想要可設
`loop.select_override: false`。

**StopPolicy（聯集）**：`max_trials` 硬上限、時間預算、使用者/Advisor 停止，以及
「已完成 ≥ `min_trials` 且連續 `patience` 輪**改良嘗試**（improve/resume）無提升
（增幅 > `min_delta` 才算提升）」。draft/debug 屬探索/修復，不計入 patience
（但帶來提升照樣歸零），避免探索期被過早切斷。`encoders_per_run` 已停用——
draft 輪替的 encoder 上限由 `num_drafts` 決定。web 工作台的「🌳 解答樹」cell
即時畫出全域樹（方形=draft、圓形=improve、菱形=debug、三角形=resume；
節點上方=encoder；✗ 虛線框=失敗、★ 實線框=全域最佳；游標移到節點顯示完整資訊，
含被選機率）。

### 允許 LLM 修改程式（`advisor.allow_code_edit`，預設關閉）

生成工作時可勾選（web 表單）或在設定檔開啟。開啟後，debug 階段 LLM 除了調 Recipe
超參，還可對訓練程式（`main_finetune.py`/`engine_finetune.py`/`models_vit.py`/
`util/*.py`）提出 exact find/replace 編輯；由 `code_workspace.py` 把原始碼複製到
`<run_dir>/src/<tag>/`、在副本上套用編輯並執行該副本——**原始程式永遠不會被修改**。
每個修改版本連同 `edits.json` 落地保存，可重現、可審查。

### 單一 trial（P0，保留）

```bash
python -m agent.run --data_path ./data/5_fold_PAPILA/PAPILA_seed42_fold0 --dry_run
```

### Web 界面

```bash
python -m agent.web.app --host 0.0.0.0 --port 5000
```

分頁：**工作台**（決策討論 + notebook）、**新實驗**、**資料集**、**超參起點**、
**背景工作**、**歷史 runs**。

### primary 是 test 成績，曲線是 val

一個 trial 裡有兩組數字，UI 上都已標明 split：

| | 來源 | 用途 |
| --- | --- | --- |
| **逐 epoch 曲線**（loss / score 圖） | 訓練期間每個 epoch 在 **val** 上評估 | 挑最佳 checkpoint、判斷收斂與過擬合 |
| **primary / metrics** | 訓練結束後用 val 最佳 checkpoint 在 **test** 上重跑一次 | trial 之間比較、histogram、解答樹分數 |

所以「圖上 best 0.861」與「primary 0.765」不一樣是正常的：不同 split，且 checkpoint
是照 val 挑的，val 分數本來就樂觀偏高。進行中的 trial 還沒跑 test，卡片上會標
`primary (val)` 表示那是目前的 val 最佳分數。

解析上有兩個坑，`log_curves.split_final_eval()` / `METRIC_RE` 已處理，web 與 advisor
共用同一份實作（避免兩邊 regex 各自演化）：

- 最終 test 的輸出與逐 epoch val **完全同格式**（連 `val loss:` 字樣都一樣），不切掉的話
  曲線末端會多一個其實是 test 的「epoch」，看起來像每個 trial 最後一個 epoch 都固定跳升，
  也會讓 `unconverged()` 誤判成 val_loss 回升而不續訓。
- kappa 低於隨機水準時是負數（score 也跟著變負），指標 regex 少了 `-?` 會讓整個區塊匹配
  失敗，該 epoch 從 score 曲線消失、之後每個點的 epoch 位置左移。

### 資料集（選擇 / 上傳 / 格式檢查）

「工作台」與「新實驗」的資料集欄位是**下拉選單**，內容由 `dataset_registry.scan()`
掃描 `data/` 而來（依上層目錄分組、標示類別數 / 張數 / fold，並可切換為自訂路徑）。
無法直接訓練的資料集會標 ⚠ 並列出原因。

「資料集」分頁可**上傳新資料集**（`.zip` / `.tar` / `.tar.gz`，單次上限 16 GB），
解壓到 `data/<名稱>/` 後立刻檢查格式。壓縮檔若多包一層外層目錄會自動攤平；
含 `..`／絕對路徑／symlink 的成員一律拒收。資料集太大不方便上傳時，可以直接把目錄
放到伺服器上，再用「檢查既有路徑」驗證。

格式檢查（`dataset_registry.validate()`）對照的是 `main_finetune.py` 的實際載入方式
（三個 split 各自 `ImageFolder`，label index = 該 split 內排序後的類別資料夾名）：

| 判定 | 條件 |
| --- | --- |
| 錯誤（無法訓練） | 缺 `train`/`val`/`test` 任一；split 底下沒有類別子目錄；三個 split 類別名稱不一致（會標籤錯位）；類別數 < 2；有空的類別目錄；抽樣影像讀不開 |
| 警告（可訓練但需留意） | train < 100 張；val/test < 10 張；類別不平衡 ≥ 3:1；類別目錄下還有子目錄（會被攤平為同一類）；影像短邊 < 224；抽樣影像全為灰階 |

API：`GET /datasets`（清單，`?refresh=1` 重掃）、`POST /datasets/validate`（`path=`）、
`POST /datasets/upload`（`archive=` 檔案、`name=`、`overwrite=`）、
`POST /datasets/delete`（`path=`，限 `data/` 底下）。

### 資料管線效能（GPU 利用率低的根因修正）

高解析度小資料集（如 PAPILA：488 張 2576×1934 JPEG、每 epoch 僅 12 iters）的
GPU 利用率瓶頸在資料管線：每 step decode 整張 5M-pixel 圖 + 每 epoch 重建
worker pool。修正（實測資料管線穩態 **9.3× 加速**）：

- `--persistent_workers` / `--prefetch_factor`（main_finetune）：跨 epoch 保留
  worker pool。agent 跑的 trial **一律**帶 `--persistent_workers`（統計等價、
  不影響期望值；epoch≥1 增強亂數流非逐位相同故預設關）。
- `--cache_resized N` + `--cache_dir`（main_finetune / util/datasets.py）：預縮圖
  磁碟快取——原圖短邊縮到 N 後快取（lazily、多 worker 安全、跨 fold 去重），
  訓練從快取讀，decode 成本降 ~20×；`samples`/metrics CSV 的路徑仍是原圖。
- agent 於 **run 起手**依 `profile.image_size_stats` 決定（`gpu.cache_auto`，
  中位短邊 ≥ 1.5×`gpu.cache_short_side` 才啟用），全 run 所有 trial 一致以保
  可比性，記入 provenance 並發 ⚡ 訊息；`_apply_gpu_boost` 的 num_workers
  上限會 clamp 到每 epoch iteration 數。

## `main_finetune.py` 的擴充（P3，最小侵入）

新增 CLI 參數，**不帶旗標時與原版行為完全一致**（保護 baseline 重現，§11-1）：
- `--head_type {linear,mlp}`（+ `--head_hidden_dims`、`--head_dropout`）：mlp 時以 `MLPHead` 覆寫 `model.head`。
- `--loss {cross_entropy,weighted_ce,focal}`（+ `--focal_gamma`）：`build_criterion` 依此建 criterion；weighted 由 train 類別數推 inverse-frequency 權重。
- 既有的 `--mixup / --cutmix / --smoothing / --drop_path / --layer_decay / --weight_decay` 由 Recipe 的 regularizer/hparams 驅動。

## 決策層 (Advisor) 三實作

- **`HeuristicAdvisor`**（預設、離線退路）：規則式選 encoder / 組 Recipe；P4 變異階梯（mixup → mlp head → label_smoothing…），一次一個正交變異，記錄 provenance。
- **`LLMAdvisor`**（P2）：Claude API `messages.parse()` structured outputs，模型 `claude-opus-4-8`，adaptive thinking，穩定目錄加 `cache_control`。LLM 只吐決策，Recipe 由純程式建構。**需 `pip install anthropic` + 金鑰**（`ANTHROPIC_API_KEY` 或 `ant auth login`）；不可用時各方法退回 `HeuristicAdvisor`。
- **`SkillAdvisor`**（P7）：以 file-based handoff 委派給 `finetune-advisor` skill（寫 `req_*.json`、讀 `resp_*.json`）；介面與其他兩者相同，故決策層可整組抽換（`advisor.type=skill`）。resp 未就緒時退回 `fallback`（預設 Heuristic），讓迴圈仍可推進。未來把 `_invoke_skill` 接上實際 Claude Code skill 呼叫即可。

### GPU 利用率最佳化（`gpu.*`，預設開啟）

每個 trial 訓練期間，`trainer` 以背景 thread 每 `sample_interval_s` 秒取樣
`nvidia-smi`（`gpu_monitor.py`；扣掉暖身樣本），彙整成 `TrialResult.gpu_stats`
（平均/中位利用率、記憶體峰值）。當某節點的平均利用率低於 `util_target`（預設
60%），從它衍生的下一輪 recipe 會自動採取措施提高利用率：

- **dataloader 瓶頸**（log 的 `time:/data:` 解析出資料載入佔比 `data_frac` ≥ 0.3）
  → 增加 `num_workers`（上限 `max_num_workers`）；
- **記憶體有餘裕** → `batch_size` 加倍（利用率低於目標一半且記憶體足夠時 ×4；
  `accum_iter`>1 時折算維持有效 batch）；兩者可同時採用，
  受實測記憶體峰值與 `max_batch_size` 自我限制。

適用於 draft/debug/improve 每一輪（draft 無 parent 時以「最近一個 trial」的
量測為依據）；resume 節點不適用（沿用 checkpoint 內的 args）。訓練中另有
watchdog：開始約 3 分鐘後取樣平均若低於目標，當下就在對話發 ⚠ 警示（不必等
trial 結束）。調整記錄在 `provenance.gpu_opt` 並在對話中告知（⚡ 訊息）；
完成訊息附 GPU 利用率；trial 卡片顯示利用率與記憶體峰值，LLM 決策時也看得到
`gpu_stats`。batch/worker 都到頂仍偏低時會明確告知「無計可施」。

## 現況與界線

- **多 fold**：`aggregation=mean_std` 時，先在設定的單一 fold 篩選出全域最佳 Recipe，再把它擴到所有 sibling fold 重跑並彙整（§5.7 screen-then-expand）。
- **GPU 判斷**：`trainer.gpu_allocatable` 以「實際嘗試配置 CUDA」判斷，修正 `script.py` 的 `memoryUtil<0.1` 在 Exclusive_Process 下誤判（§5.5 / §11-3）。
- **尚未接上**：非分類 head（segmentation/regression，P6）、多任務多 head、`SkillAdvisor`（P7）、部分預留元件（ema/r_drop/自訂 aug）— dry-run 會以 unsupported 標示。
- **依賴**：`pydantic>=2`、`pyyaml`、`flask`（M4 env 已具）；`anthropic`（僅 LLMAdvisor 需要）。
