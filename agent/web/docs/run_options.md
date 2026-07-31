<!-- key: overview -->
## 新完整流程 — 設定說明

把游標移到左邊任一個選項，或點選它，這裡就會顯示該選項的詳細說明。

- **決策層 (advisor)**：由誰做決策（挑 encoder、組 recipe、改良、選集成成員）。
- **超參起點 / 主要指標 / 多 fold 彙整**：訓練與評估的基本設定。
- **迴圈參數**：樹搜尋跑多少輪、何時停。
- **資料圍欄**：LLM 能看到多少資料（預設完全看不到）。
- **影像模態 / 拍攝部位 / 類別是否有序**：你告訴系統的資料特性（LLM 不從路徑猜）。

> 這份說明存在 `agent/web/docs/run_options.md`，直接編輯即可改善描述，重新整理頁面就生效。

<!-- key: advisor.llm -->
## 決策層：LLM（預設）

用 Claude API 做每一步決策：依 `DatasetFacts` 挑 encoder、組訓練 recipe、每輪檢視結果並提出一個正交變異、判斷何時停止。

- **最聰明**、能讀懂你的引導方向與討論訊息。
- 需要 `ANTHROPIC_API_KEY`（在「設定」頁或環境變數）。環境不可用時各決策會自動退回 heuristic。
- 也可改走 Anthropic-compatible 的代理端點（如 OpenRouter）：設 `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` + `MEDCLAW_LLM_MODEL` 三個環境變數即可，見 `.env`。
- 仍受**資料圍欄**約束：LLM 看不到影像、檔名、路徑、真實類別名。

<!-- key: advisor.heuristic -->
## 決策層：Heuristic（規則式）

不呼叫任何 LLM，用固定規則決策：小資料→linear probe、不平衡→weighted/focal loss、依固定變異階梯（mixup→mlp head→…）逐一嘗試。

- **免金鑰、快、可離線**、完全可重現。
- 不會讀你的自然語言引導，彈性較低。
- 適合當基準線，或在沒有 API 時使用。

<!-- key: advisor.skill -->
## 決策層：Skill（委派 Claude Code skill）

把決策請求以檔案交握（handoff）交給 Claude Code 的 `finetune-advisor` skill 產生，再讀回。

- 介面與其他 advisor 相同，可整組抽換。
- 尚未接上實際 skill 呼叫時，會退回 heuristic 讓迴圈繼續。
- 進階／實驗性選項。

<!-- key: preset.default -->
## 超參起點：default

平衡的一組微調起始超參（learning rate、layer decay、drop path、weight decay、epochs…），適合大多數眼底/醫療影像分類。之後每輪可由決策層在有理由時微調。

<!-- key: preset.paper -->
## 超參起點：paper

貼近原始論文設定的超參，通常 epochs 較多、正則較保守。想重現論文式訓練或資料量較大時可選。

<!-- key: preset.mae -->
## 超參起點：mae

針對 MAE 類 encoder 調校的起點（例如較高 layer decay）。當你主要比較 MAE 預訓練骨幹時較合適。

<!-- key: primary_metric.score -->
## 主要指標：score（預設）

`(f1 + roc_auc + kappa) / 3` 的綜合分數。同時兼顧不平衡下的 F1、排序能力（AUC）與一致性（kappa），是穩健的預設最佳化目標。

<!-- key: primary_metric.accuracy -->
## 主要指標：accuracy

整體答對比例。**類別不平衡時會誤導**（全猜多數類也可能很高），不建議在不平衡資料當唯一目標。

<!-- key: primary_metric.f1 -->
## 主要指標：f1（macro）

各類別 F1 的平均，對少數類敏感。**類別不平衡**時比 accuracy 合適。

<!-- key: primary_metric.roc_auc -->
## 主要指標：roc_auc（macro, OVR）

排序/區辨能力，與閾值無關。適合關心「分數排序」而非硬分類的情境；二元與多類（one-vs-rest）皆可。

<!-- key: primary_metric.kappa -->
## 主要指標：kappa（Cohen's κ）

扣除隨機一致後的一致性。**有序分級**（如 DR 0–4）常用；對不平衡也較穩健。

<!-- key: primary_metric.balanced_accuracy -->
## 主要指標：balanced_accuracy

各類別 recall 的平均（等於巨集平均 recall）。在不平衡資料上比 accuracy 公平地看待少數類。

<!-- key: primary_metric.precision -->
## 主要指標：precision（macro）

巨集平均精確率。當**偽陽性代價高**（誤報要少）時作為目標。

<!-- key: primary_metric.recall -->
## 主要指標：recall（macro）

巨集平均召回率。當**偽陰性代價高**（漏診要少）時作為目標——醫療篩檢常見。

<!-- key: primary_metric.average_precision -->
## 主要指標：average_precision

PR 曲線下面積（巨集）。在**高度不平衡**、關心正類排序時，比 roc_auc 更有鑑別力。

<!-- key: aggregation.single -->
## 多 fold 彙整：single（預設）

只在你選的單一 fold 上訓練與評估，報告單 fold 成績。**最快**，適合快速探索與比較。

<!-- key: aggregation.mean_std -->
## 多 fold 彙整：mean_std

搜尋結束後，把**全域最佳 recipe** 依檔名樣式 `..._foldN` 自動找到的其他 fold 上**各重訓一次**，報告跨 fold 的 **平均 ± 標準差**。

- 給出更可信的泛化估計，但**成本 = 額外 (N−1) 次完整訓練**。
- 只驗證最終贏家，不是用交叉驗證來選模型；選模型仍在單一 fold 上進行。
- 需要路徑以 `_fold0/1/…` 命名且同層有其他 fold，否則自動略過。

<!-- key: privacy_mode.strict -->
## 資料圍欄：strict（預設，最嚴）

LLM **完全看不到**影像、檔名、路徑、真實類別名（資料集以假名 `DS-…`、類別以 `C0…Cn` 表示）。它要資料特性時只能點名執行本地分析器，或直接問你。

- **強制關閉「允許 LLM 改程式」**——因為「可改程式 + log 回饋」合起來等於一條完整的資料讀取通道。
- 醫療資料的建議預設。

<!-- key: privacy_mode.standard -->
## 資料圍欄：standard

LLM 仍看不到影像與路徑，但**可以修改 `main_finetune.py`**（隔離副本執行）。原始訓練 log **不會**回饋給 LLM（只回結構化 ErrorFacts）。真實類別名需你逐一授權才會外送。

<!-- key: privacy_mode.off -->
## 資料圍欄：off（關閉，不建議）

**關閉圍欄**：資料集路徑、真實類別名、原始訓練 log 都可能送到 Claude API。只在非敏感的公開資料、且你清楚後果時使用；UI 會常駐紅色警示。

<!-- key: class_ordinal -->
## 類別是否有序

類別之間有沒有**程度順序**（例：糖尿病視網膜病變分級 0–4 是有序；貓/狗是名目）。

- 有序 → 決策層可能偏好 kappa、有序感知的處理。
- 只有你知道，LLM 不會從資料夾名稱猜。留「未指定」也可以。

<!-- key: modality -->
## 影像模態

影像的成像方式（fundus 眼底、OCT、X-ray、CT、MRI、超音波、皮膚鏡、病理、內視鏡…）。

由**你**指定而非從路徑猜測（資料圍欄）。它會影響決策層偏好哪類 encoder（例如醫療 DAP）。不確定可留「未指定」。

<!-- key: anatomy -->
## 拍攝部位

影像的解剖部位（eye 眼、chest 胸、brain 腦、skin 皮膚、gi 消化道…）。同樣由你指定，供決策層參考，非必填。

<!-- key: loop -->
## 迴圈參數（樹搜尋）

- **起手 drafts**：一開始跨 encoder 各開幾條新起點（多樣性）。
- **max_trials**：全域 trial 總數上限（= 樹搜尋總輪數）。
- **min_trials**：至少跑滿這麼多輪才允許 patience 提早停（保護探索期）。
- **patience**：連續這麼多輪 improve/resume 無提升就停（draft/debug 不計）。

停止條件是這些的聯集，另有時間預算與你隨時可按的「中斷」。

<!-- key: guidance -->
## 引導方向 (guidance)

給決策層的一段自然語言方向，例：「資料量小且不平衡，優先醫療 DAP encoder + focal loss；先 linear probe 快速篩選。」

- 只有 **advisor=llm/skill** 會採用。
- ⚠ 這段文字會**原文送到 Claude API**，請勿貼上病人資訊或檔案路徑。

<!-- key: allow_code_edit -->
## 允許 LLM 修改訓練程式

開啟後，LLM 可在 debug 失敗 trial、以及 improve 階段修改 `main_finetune.py`（修改版隔離在 run 目錄執行，原始程式不動）。

- **strict 圍欄下強制關閉**（改程式 + log 回饋 = 資料讀取通道）。要用需先把資料圍欄設為 standard/off。

<!-- key: class_ordinal.ordinal -->
類別有**程度順序**（例：DR 分級 0–4、病變嚴重度）。決策層可能偏好 kappa 等對順序敏感的評估。

<!-- key: class_ordinal.nominal -->
類別**無順序**（名目，例：貓/狗、病灶型態）。一般以 f1/accuracy 等評估即可。

<!-- key: loop.num_drafts -->
一開始跨 encoder 各開幾條**新起點（draft）**——多樣性的來源。越多越廣但越花時間。

<!-- key: loop.max_trials -->
全域 **trial 總數上限**（＝樹搜尋總輪數）。達到就停。

<!-- key: loop.min_trials -->
至少跑滿這麼多輪，才允許 **patience 提早停**（保護早期探索）。

<!-- key: loop.patience -->
連續這麼多輪 **improve/resume 無提升**就停；draft/debug 不計入。

<!-- key: modality.fundus -->
眼底（視網膜）彩色照——青光眼、糖尿病視網膜病變等。

<!-- key: modality.oct -->
光學同調斷層（OCT）——視網膜分層結構斷面。

<!-- key: modality.xray -->
X 光平片（如胸腔 X 光）。

<!-- key: modality.ct -->
電腦斷層（CT）——橫斷面影像。

<!-- key: modality.mri -->
磁振造影（MRI）——軟組織對比佳。

<!-- key: modality.ultrasound -->
超音波影像。

<!-- key: modality.dermoscopy -->
皮膚鏡——色素性皮膚病灶。

<!-- key: modality.pathology -->
病理切片（顯微／全玻片 WSI）。

<!-- key: modality.endoscopy -->
內視鏡（消化道等腔道）。

<!-- key: modality.other -->
其他／未列出的成像模態。

<!-- key: anatomy.eye -->
眼睛。

<!-- key: anatomy.chest -->
胸腔。

<!-- key: anatomy.brain -->
腦部。

<!-- key: anatomy.skin -->
皮膚。

<!-- key: anatomy.gi -->
消化道（GI）。

<!-- key: anatomy.breast -->
乳房。

<!-- key: anatomy.bone -->
骨骼。

<!-- key: anatomy.abdomen -->
腹部。

<!-- key: anatomy.other -->
其他部位。

<!-- key: aggregation.per_fold -->
## 多 fold 彙整：per_fold（每個 fold 各自搜尋）

對每個 sibling fold（`..._fold0/1/…`）**從頭獨立跑一次完整搜尋**，各自找出該 fold 的最佳 encoder／超參——不同 fold 可能得到不同的最佳配方。

- 與 `mean_std` 不同：`mean_std` 只把**單一**最佳 recipe 套到各 fold 重跑；`per_fold` 是**各 fold 獨立選模型**。
- **迴圈預算每個 fold 各自獨立**：`max_trials` 與時間預算對每個 fold 分別重新起算（fold 0 用完不會排擠後面的 fold）。只有「中斷實驗」會結束整個逐 fold 流程。
- **成本最高**：≈ N 倍的完整搜尋（5-fold＝5 次）。
- 報告列出每個 fold 的最佳 recipe／關鍵超參／分數，並給各 fold 最佳分數的 **mean ± std**；每個 fold 的解答樹另存 `search_tree_foldN.json`。
- 需要路徑以 `_fold0/1/…` 命名且同層有其他 fold。
