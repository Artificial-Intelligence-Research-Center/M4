# 逐 fold 平行搜尋 — 工作分配到多 GPU 設計文件

- 狀態：v0.2 — **已實作（本機多 GPU + 自適應共卡）**；多節點 PyTorchJob 為預留介面
- 適用範圍：`aggregation = per_fold` 時，把每個 fold 的完整搜尋平行分配到多顆 GPU
- 相關程式：`agent/fold_fleet.py`、`agent/loop_controller.py`、`agent/trainer.py`、`agent/web/app.py`、`agent/web/templates/workspace.html`
- 相關文件：[ensemble_design.md](ensemble_design.md)（per_fold 模式的評估語意）、[data_firewall_design.md](data_firewall_design.md)

---

## 1. 目標

`aggregation = per_fold` 時，每個 sibling fold（`..._fold0/1/…`）各自跑一次**完整的樹搜尋**，獨立找出自己的最佳 recipe。本文件說明**如何把這些 fold 的工作平行分配到多顆 GPU**：

1. 決定可以同時跑幾個 fold（平行度）。
2. 把每個 fold 指派到某一顆 GPU。
3. 當某個工作因 GPU 記憶體不足（OOM）失敗時，如何等待資源釋出後再繼續。

**核心設計：一個 fold = 一個完全獨立的子執行**（`runs/<parent>/foldN/`），有自己的對話、ledger、解答樹、LLM 記錄與 report。父 run 只做協調 + 彙整。這讓「徹底分開」自然成立，也讓每個 fold 能被單獨指派 GPU、單獨等待資源、單獨在 UI 檢視。

---

## 2. 平行度如何決定

**平行度 = 可用 GPU 數量**，且**每顆 GPU 同時最多跑一個 fold**。

- `detect_gpus()`（[fold_fleet.py](../agent/fold_fleet.py)）決定 GPU 清單：
  1. `nvidia-smi --query-gpu=index`（不必載入 torch，最輕量）；
  2. 退回 `torch.cuda.device_count()`；
  3. 皆失敗 → `[0]`（單一裝置）。
- 執行緒池 `ThreadPoolExecutor(max_workers=len(devices))`：最多 `#GPU` 個 fold 同時進行。
- 為什麼用執行緒就夠：**真正的訓練是子程序**（`trainer.run_trial` 以 `CUDA_VISIBLE_DEVICES=<device>` 起 `main_finetune.py`），GPU 計算不在 Python 行程內，故執行緒不受 GIL 影響；LLM 決策是 I/O bound，執行緒亦適合。

### 決策準則與取捨
- **起始一 fold 一卡**：先把每張卡各鋪一個 fold（`acquire` 永遠選目前負載最少的卡）。
- **fold 數 ≤ GPU 數**：全部同時開跑，一 fold 一卡。
- **fold 數 > GPU 數**：多出來的 fold 進入佇列。**是否讓它們與別的 fold 共用一卡由「自適應共卡」決定（§3.1）**，而不是無條件塞。
- **共卡 vs 加大 batch**：當一張卡被單一 fold 低度利用時，與其在該 fold 內**加大 batch**（會拖慢收斂、且與同卡另一 fold 搶記憶體），不如**加開第二個 fold** 用掉閒置的計算與記憶體。因此開啟共卡時，fleet 內各 fold 的 `gpu.grow_batch` 會被設為 `false`（見 §3.1）。

---

## 3. GPU 指派過程（裝置池）

指派用一個**執行緒安全的裝置池**（`queue.Queue`）做，語意是「一顆 GPU 一個令牌」：

```
pool = Queue(); for d in devices: pool.put(d)      # 每顆 GPU 放一個令牌

run_one(fold i):
    device = pool.get()          # 阻塞直到有空閒 GPU（= 取得該卡的獨占使用權）
    try:
        等待 device 可配置記憶體 (§4)
        以 child.device = device 跑這個 fold 的完整搜尋
    finally:
        pool.put(device)         # 跑完歸還，佇列中的下一個 fold 立刻遞補
```

- **round-robin 且自負載平衡**：`pool.get()` 先到先得，快跑完的 fold 先歸還 GPU，下一個 fold 立刻補上——自然把工作平均分散到所有 GPU，不需要預先靜態切分。
- 每個 fold 子執行的 `cfg.device` 設為取得的 device；`trainer` 對其訓練子程序設 `CUDA_VISIBLE_DEVICES=<device>`，因此不同 fold 落在不同實體 GPU。
- 指派結果即時寫入 `folds.json`（`{index, name, subdir, device, status, primary}`），UI 讀取後顯示「Fold k · GPU d · 狀態」。

範例（3 folds、2 GPUs）：fold0→GPU0、fold1→GPU1，兩者跑；先跑完的（假設 fold1）歸還 GPU1 → fold2→GPU1。實測見 `fold_fleet` 測試。

### 3.1 自適應共卡：低利用率時加開第二個 fold（本次新增）

排程器 `_Scheduler`（[fold_fleet.py](../agent/fold_fleet.py)）每張卡有一個「容量」`cap`，初始 `cap=1`（一 fold 一卡）。一個**監看執行緒**每 `_PACK_POLL_S`（15s）以 `nvidia-smi` 讀各卡的**即時利用率與記憶體佔比**：

```
若某卡  active==1（只跑一個 fold） 且 util < pack_util_below(35%) 且 mem 佔比 < pack_mem_below(0.5)
   連續 _PACK_LOW_STREAK(2) 次   →  cap[該卡] += 1  (至多 max_folds_per_gpu)
                                     並喚醒佇列 → 下一個 fold 立即加開到這張卡
```

- **觸發條件正是使用者要的**：偵測到「GPU 使用率低**且**記憶體也低」——利用率低代表有閒置計算、記憶體低代表塞得下第二個 fold——才加開，兼顧「提高利用率」與「不 OOM」。
- **立即遞補**：每個 fold 都有自己的執行緒，未取得卡位時阻塞在 `acquire()`；容量一提高，等待中的 fold 立刻搶到該卡（不必等別的 fold 全部跑完）。
- **共卡時不加大 batch**：開啟共卡（`pack_low_util` 且 `max_folds_per_gpu>1`）時，每個 fold 子執行的 `gpu.grow_batch=false`，`LoopController._apply_gpu_boost` 因此**跳過加大 batch**（仍保留 dataloader worker 的調整）。提高利用率的手段改為「共卡」。
- **回復**：某卡的一個 fold 結束、`active` 降回 1 時，`cap` 收回為 1 並重新累計——**下一次共卡需再度量到低利用率才會發生**（持續自適應，不會殘留過度共卡）。
- 連續 streak 要求（2 次）避免被瞬間的 util 抖動誤觸。

**相關設定**（`gpu.*`）：`max_folds_per_gpu`(2) · `pack_low_util`(true) · `pack_util_below`(35%) · `pack_mem_below`(0.5) · `grow_batch`(true，共卡時自動關)。設 `max_folds_per_gpu=1` 或 `pack_low_util=false` 即回到「嚴格一 fold 一卡」。

> 取捨：`nvidia-smi` 的 util 是瞬時值，故用「連續 streak + 記憶體雙條件」降低誤判；真正的 OOM 保護仍由 §4 的資源閘門與單 trial debug 承接。未來可加「共卡後仍 OOM → 自動退回獨占」的回饋。

---

## 4. 記憶體不足（OOM）／資源等待與續跑

分兩個層級處理，因為 OOM 可能發生在「單一 trial」或「整個 fold 啟動時」。

### 4.1 單一 trial 的 OOM（既有機制，最常見）
訓練期間某個 trial OOM 時，該 trial 標記失敗，樹搜尋的 **debug 階段**接手：`HeuristicAdvisor.propose_debug` / LLM 依 `ErrorFacts.error_class ∈ {oom, cuda}` 產生修正——**batch_size 減半、accum_iter 加倍**（等效 batch 不變）再重試（`max_debug_depth` 為上限）。這一層完全在單一 fold 的子執行內完成，不需要跨 fold 協調。

### 4.2 fold 啟動前的資源閘門（本次新增）
一顆 GPU 剛被前一個 fold（或外部程序）釋放時，記憶體可能還沒回收乾淨。故在 `run_one` 於某 device 上啟動 fold **之前**，先等它真的能配置記憶體：

```
_wait_allocatable(device):
    while not trainer.gpu_allocatable(device, timeout=30):   # 實際試配一小塊 CUDA 記憶體
        每 _GPU_POLL_S(20s) 檢查一次；在對話回報「等待 GPU d 釋出中…」
        逾時 _GPU_WAIT_MAX_S(1800s) 仍嘗試啟動（避免永久卡住）
```

`trainer.gpu_allocatable(device)` 以子程序實際 `torch.randn(...).cuda()` 試配，是「能不能真的用」而非只看 nvidia-smi 數字，較可靠。

### 4.3 fold 層級的失敗重試（本次新增）
若整個 fold 啟動/執行拋出**疑似資源錯誤**（訊息含 `out of memory / cuda error / cublas …`），在同一顆 GPU 上**等待其釋出後重試**，上限 `_OOM_RETRIES(2)`：

```
while True:
    try: out = _launch_fold(child, sub); break
    except e:
        if _is_resource_error(e) and attempt < _OOM_RETRIES:
            attempt++; 回報「等待 GPU 釋出後重試…」; _wait_allocatable(device); continue
        raise
```

因裝置池保證一顆 GPU 同時只有一個 fold，「等待釋出」通常是等**外部**程序或前一輪殘留；重試前的閘門確保記憶體真的可用才再啟動。

### 設計取捨與未來
- 目前重試是**在原 GPU 上等待**（簡單、可預測）。未來可加「改派到另一顆空閒 GPU」的策略（把該 fold 重新丟回裝置池）。
- 參數（`_GPU_POLL_S / _GPU_WAIT_MAX_S / _OOM_RETRIES`）目前是模組常數；需要時可提升為 `GpuOptConfig` 欄位由 UI 調整。
- 跨 fold 的**全域記憶體壓力**（多顆卡同時吃滿主機記憶體/IO）不在本層處理；由「一 GPU 一 fold」與各自的 dataloader worker 上限自然約束。

---

## 5. 隔離與資料圍欄

- 每個 fold 子執行由 `auto_finetune.run(child_cfg, run_dir=sub)` 啟動，**自建 advisor 與 PrivacyContext**（per-fold 假名／出口守衛），彼此完全獨立——一個 fold 的對話、LLM 記錄、隱私稽核不會混入另一個。
- 父 run 只寫「協調訊息」（哪個 fold 在哪顆 GPU 開始/完成）與彙整 report；父層不接觸資料。
- 使用者提供的資料特性（模態/部位/類別序數）由父 run 的 `user_facts.json` 複製給每個 fold，讓各 fold 的 LLM 都看得到。

---

## 6. Web UI

- **run 選擇器下方新增 fold 選擇器**（`#rs-fold`）：當選到的 run 是逐 fold 平行的父 run（`run_status` 回傳 `parallel_folds`）時出現，列出每個 fold 的「名稱 · GPU · 狀態 · 最佳分數」。
- 選「總覽」→ 顯示父層的**各 fold 進程總覽表**（GPU/狀態/最佳 + 「查看」按鈕）與彙整 report；父層對話顯示協調訊息。
- 選某個 fold → 有效 run 變成 `parent/foldN`，**對話與 notebook 徹底切換到該 fold 的獨立子執行**（自己的解答樹、histogram、trial、LLM 記錄）。討論/中斷/回答問題都作用在該 fold。
- `run_status` 接受巢狀 `run=<parent>/foldN`（限制在 `runs/` 內，擋 `..` 穿越）。

---

## 7. 檔案佈局

```
runs/<parent>/
  folds.json            # {mode:"per_fold_parallel", devices:[...], done, folds:[{index,name,subdir,device,status,primary,best}]}
  config.yaml           # 父 run 設定 (aggregation=per_fold)
  conversation.jsonl    # 父層協調訊息
  report.md             # 各 fold 最佳 recipe/超參 + mean±std 彙整
  user_facts.json
  fold0/                # 一個完整、獨立的 run
    config.yaml (aggregation=single, device=0, data_path=..._fold0)
    conversation.jsonl · ledger.jsonl · search_tree.json · llm_calls.jsonl · report.md · privacy_audit.jsonl · trials/ · logs/
  fold1/  (device=1)  …
```

---

## 8. 未來：多節點（PyTorchJob）

本機多 GPU 與多節點的差別只在「怎麼啟動一個 fold」，協調/彙整/UI 契約完全不變。已預留 `PyTorchJobLauncher`（[fold_fleet.py](../agent/fold_fleet.py) 檔尾）：

1. 把 `child_cfg` dump 到 `sub_dir/config.yaml`（`sub_dir` 需在共享儲存 PVC 上，Web UI 才讀得到各 fold 的對話/ledger/解答樹）。
2. 依樣板產生 PyTorchJob CRD：image、`command = python -m agent.auto_finetune --config <sub>/config.yaml`、`resources.limits.nvidia.com/gpu = 1`、掛載 PVC。
3. `kubectl apply` 提交，輪詢 job 狀態至完成，從 `sub_dir` 讀回 best。
4. `run_fleet` 的裝置池此時退化為「叢集排程配額」（例如同時最多 N 個 job），其餘協調（folds.json、彙整、UI）不動。

---

## 9. 實作對應

| 功能 | 位置 |
| --- | --- |
| GPU 偵測 | `fold_fleet.detect_gpus` |
| 平行度 / 指派 / 自適應共卡 | `fold_fleet._Scheduler`（負載感知 acquire + 監看執行緒 `monitor`）+ `_gpu_util_mem` |
| 共卡時關閉加大 batch | `config.GpuOptConfig.grow_batch` + `loop_controller._apply_gpu_boost` |
| 資源閘門 / OOM 等待重試 | `fold_fleet._wait_allocatable` / `_is_resource_error` / `run_one` |
| 單 trial OOM 修復 | `advisor.HeuristicAdvisor.propose_debug` / `llm_advisor.propose_debug` |
| fold=獨立子執行 | `fold_fleet._launch_fold` → `auto_finetune.run(run_dir=sub)` |
| 父索引 manifest | `fold_fleet._write_manifest` → `folds.json` |
| 路由 per_fold → 艦隊 | `web/app.run_full` |
| UI 子 run 位址 / parallel_folds | `web/app.run_status`（巢狀路徑 + `parallel_folds`） |
| UI fold 選擇器 / 父總覽 | `web/templates/workspace.html`（`effRun/onRunChange/updateFoldSelector/renderParentOverview`） |
| 多節點預留 | `fold_fleet.PyTorchJobLauncher` |
