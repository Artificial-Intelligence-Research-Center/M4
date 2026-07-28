# MedClaw Prompt 結構與快取佈局 — 設計文件

- 狀態：v0.1 — §2～§4 是**已實作並量測**的現況；§5 起是**提案，尚未實作**
- 適用範圍：`agent/llm_advisor.py` 送往 Claude API 的 prompt 組成與 prompt cache 佈局
- 相關文件：[auto_finetune_agent_design.md](auto_finetune_agent_design.md)、[data_firewall_design.md](data_firewall_design.md)
- 相關程式：`agent/llm_advisor.py`（`_data_block` / `_user_content` / `_breakpoint_indices`）、`agent/privacy/redact.py`（`discussion_facts` / `user_segments` / `scrub`）、`agent/loop_controller.py`（`_tree_view`）
- 量測來源：`runs/run_20260726_211716/fold4/llm_calls.jsonl`（2026-07-28，27～32 個節點的解答樹）

---

## 1. 目標與動機

決策迴圈每輪固定發兩次 API 呼叫（`review_search_choice` → `review_and_decide`）。修好 prompt cache 之後，**穩定前綴**（資料集事實、分析結果、trials 歷史）已經幾乎全部命中，但每輪仍有約 3.1 萬 tokens 走全額付費，其中**四分之三是「與使用者的討論記錄」**——而它每輪重送兩份、從來沒有命中過。

目標：**把「本質上只會增長」的內容全部移進可快取的穩定區，讓每輪只為真正新增的那幾百 tokens 付費。**

本文件統一以 base input price = `1x` 為計價單位：

| 項目 | 倍率 |
| --- | --- |
| 未快取 input | 1x |
| cache read | 0.1x |
| cache write（5m） | 1.25x |
| cache write（1h） | 2x |

---

## 2. 現況：一次決策輪的 prompt 結構

一次請求由 `system` + `user` 兩部分組成，各自切成數個 content block：

```
system:
  [0] 決策規則 _RULES                          固定
  [1] 可用資源目錄 _registry_context()         固定           ← breakpoint

user:
  [0]   資料集事實 DatasetFacts                一個 run 內固定
  [1]   追加分析標頭                           只會 append
  [2..] 每個分析器一段                         只會 append
  [k]   trials 標頭                            只會 append
  [k+1..] **每個 trial 一段**                  只會 append     ← breakpoint ×2
  [-2]  使用者親自提供的事實                   會改寫           ← breakpoint
  [-1]  本輪 volatile（指令 + schema + …）     每輪全變
```

段的切法（`_data_block`）依「變動頻率由低到高」排列，breakpoint 只下在最後三段（`_breakpoint_indices`；system 佔掉 1，總額度 4）。

### 2.1 量測

| | `review_search_choice` | `review_and_decide` |
| --- | ---: | ---: |
| 穩定前綴 | 2,505 字元 / 5 段 | 125,374 字元 / 32 段 |
| volatile | 25,191 字元 | 25,432 字元 |
| cache read | 4,444 | 87,860 |
| cache write（1h） | 0 | 6,425 |
| 未快取 input | 16,638 | 14,256 |
| **成本** | **17,082 單位** | **35,892 單位** |

`review_search_choice` 的 `write = 0` 是**正確且理想**的結果：它那 5 段每輪逐字相同，早已在快取裡，這次是純命中（命中會順帶刷新 TTL），沒有任何新東西需要寫。它只有 4,444 是因為它呼叫 `_data_block(profile)` **不帶 history**——prompt 裡本來就沒有那 12 萬字元的 trials。

`review_and_decide` 修好 block 邊界前是 **140,694 單位**（read 3,065 / write 63,972 @2x），現在是 35,892，降約 75%。

---

## 3. 快取的三條硬規則（實測得出，不要再踩）

這三條都是在這個專案上用 Console 帳單反覆對出來的，記在這裡免得下次重來。

### 3.1 比對以 content block 邊界為單位，不是任意位元組前綴

**內容是前綴還不夠，邊界要對得上。**

早期版本把整段 trials 併成一個會長大的 block。內容確實是乾淨的 append-only 前綴鏈（實測 66,512 → 74,325 → 80,729 → 87,527 → 94,619 字元，每一輪都是上一輪的逐字延伸），TTL 是 1h，間隔只有 12 分鐘——但**一次都沒有命中**，每輪照樣重寫六萬多 tokens。

決定性證據：同一批位元組放在**固定不變**的 block 裡（system 那 3,065、不帶 history 的小呼叫那 2,505）每次都命中；小呼叫的第一段（1,536 字元）是大呼叫開頭的逐字前綴、而且確實在快取裡，大呼叫卻連那 1,536 的額度都拿不到——因為大呼叫在那個位移上沒有 block 邊界。

**推論：已經寫出去的內容就不要再改動它所在的 block；新內容一律另起一個 block。** trials 改成每筆一段之後，上一輪的 breakpoint 這一輪仍落在段邊界上，於是讀 87,860 / 只寫 6,425。

> breakpoint 只會往回找 20 個 block。每輪只多 1 段，遠在額度內。

### 3.2 最小可快取前綴依模型而定，且**不隨世代單調遞減**

低於下限即使下了 breakpoint 也只會付 write 卻永遠讀不到（API 不報錯，只是靜靜地不快取）。實作見 `_CACHE_MIN_TOKENS`：opus-5 是 512、opus-4-8 是 1024、**opus-4-7 是 2048**（比 4.8 高）、4-6/4-5/haiku-4-5 是 4096。**不能用「越新越小」去推**，換模型時要查表。

門檻要用**累計**長度判斷而非單段長度：最小前綴是從 prompt 開頭算起的，所以接在 50k 之後的一小段照樣值得下 breakpoint。

### 3.3 TTL 要看「最長」間隔，而且兩個修正必須同時到位

1h write 是 2x、5m 是 1.25x，只有在「下次呼叫時 5m 早就過期」時才划算。判斷依據是**本 run 觀察到的最長呼叫間隔**（`_next_cache_ttl`）——用平均會被連發的呼叫（一輪內兩三次呼叫只隔幾秒、重試更是連三發）稀釋到門檻以下，於是訓練空檔再長也永遠選 5m。

超過 1h 的間隔**不列入證據**：那種間隔連 1h 都活不過，反而該用比較便宜的 5m 寫入，否則一次隔夜中斷就會讓之後每輪都多付 2x。

⚠ **順序警告**：在 §3.1 的邊界問題修好之前，把 TTL 改成 1h 會讓帳單**更貴**（每輪重寫 64k × 2x = 128k 單位，而不是 × 1.25x）。這兩件事要一起到位才有意義。

同一請求裡所有 breakpoint 的 ttl 必須一致——render 順序（tools → system → messages）上，ttl 長的 block 不得排在 ttl 短的之後，否則直接 400。

---

## 4. 剩餘成本拆解

把兩個呼叫的 volatile 拆開看（最新一輪）：

| 內容 | `review_search_choice` | `review_and_decide` | 可否快取（現況） |
| --- | ---: | ---: | --- |
| 與使用者的討論記錄 | 19,087 字元 | 19,426 字元 | ✗ 滑動窗，前綴不穩定 |
| 解答樹（含 `allowed`） | 10,431 字元 | — | ✗ 整包 `indent=2` 重送 |
| 本輪指令 + schema | ~2,400 字元 | ~2,433 字元 | ✗ 本來就該每輪變 |
| 變異基準 / 目前 encoder | — | ~3,336 字元 | ✗ 每輪變 |

**討論記錄佔了 `review_and_decide` volatile 的 76%**，而且同一輪的兩個呼叫**各送一份**。它是目前最大的一筆，也是最容易改的一筆。

---

## 5. 提案 A — 討論記錄：滑動窗 → 累積前綴

### 5.1 問題

`redact.discussion_facts(entries, alias, limit=20)` 取的是 `entries[-limit:]`——**尾端的滑動窗**。每多一則訊息，最舊的那則就從前面掉出去，於是這一段的開頭每輪都不一樣，永遠不可能是上一輪的前綴。它也用 `json.dumps(..., indent=2)` 整包輸出，但實測 indent 只佔 2%（內容是中文自由文字，不是結構），**壓縮沒有意義，關鍵在讓它可快取**。

### 5.2 提案

改成**由頭累積、每則一段**，放進穩定區（排在 trials 之後、使用者事實之前）：

```
[trials …]
[討論標頭]
[討論 #1]   ← 寫出後不再變動
[討論 #2]
 …
[討論 #N]
[使用者親自提供的事實]
```

每輪只有新增的幾則需要寫入，其餘 0.1x 讀回。

### 5.3 取捨與上限

| | 滑動窗（現況） | 累積前綴（提案） |
| --- | --- | --- |
| prompt 長度 | 有界（20 則） | 隨 run 增長 |
| 可快取 | ✗ 永不命中 | ✓ 只付新增的 |
| LLM 看到的內容 | 只有最近 20 則 | 全部 |

cache read 是 0.1x，所以**累積到現況的 10 倍長度都還比較便宜**。但兩件事要留意：

1. **這是行為改變，不只是成本改變**——決策層會看到整段歷史而非最近 20 則。討論裡的舊指示會不會被過度沿用，需要實測。建議保留「最近 N 則另外重述一次於 volatile」的作法，讓近期發言仍有 recency。
2. 真的需要設上限時，**只能從前面凍結、不能從前面丟棄**：把最舊的 K 則壓成一段「早期討論摘要」寫死，之後不再變動。從前面丟棄等於回到滑動窗。

### 5.4 ⚠ 出口掃描必須同步

`redact.user_segments(entries, alias, limit=20)` 用**同一個窗**取出「使用者親自輸入」的片段，`privacy.egress._enforce` 據此把命中判為警示而非中止（fail-closed）。

**兩個 limit 一旦不同步就會炸 run**：prompt 裡帶著第 1 則的使用者原文、而 `user_segments` 只涵蓋最後 20 則時，那段文字裡的任何命中都會被判成違規 → 中止整個 run。改 `discussion_facts` 的視窗時**必須同步改 `user_segments`**，兩者要吃同一個參數。

---

## 6. 提案 B — 解答樹：不變核心 + 變動疊加

### 6.1 問題

`_tree_view` 每個節點輸出 11 個欄位，整包 `json.dumps(indent=2)` 每輪重送。但這些欄位的變動頻率差很多：

| 欄位 | 變動 |
| --- | --- |
| `trial_id` / `stage` / `parent` / `encoder` / `adaptation` / `mutation` | 建立後不變 |
| `status` / `primary_score` / `epochs` | `_tree_view` 只輸出 `evaluated` 的節點 → 已定案 |
| `is_leaf` | 該節點長出子節點時翻轉 |
| `allowed` | 隨階段開關而變（`improve` / `resume` / `debug`） |

也就是說**只有 `is_leaf` 和 `allowed` 會變**，其餘都是寫死的。

### 6.2 提案

拆成兩塊：

```
穩定區（每個節點一段，append-only）:
  {"trial_id":…, "stage":…, "parent":…, "encoder":…, "adaptation":…,
   "status":…, "primary_score":…, "epochs":…, "mutation":…}

volatile（只有變動的部分）:
  目前各節點允許的階段: {"t3":["improve","resume"], "t7":["debug"], …}
  目前的葉節點: ["t9","t12", …]
```

### 6.3 實測效益（27 個節點）

```
現況 (indent=2 整包)          10,431 字元   ← 每輪全額付費
提案 不變核心 (JSONL, 可快取)   7,624 字元
提案 變動疊加 allowed + is_leaf  1,726 字元   ← 只有這段留在 volatile
→ volatile 減少 8,705 字元 (83%)
```

### 6.4 與 trials 歷史的重疊

穩定區已經有 `TrialFacts`（`trial_id` / `status` / `primary_score` / `epochs` / `provenance.mutation` / 逐 epoch 曲線）。解答樹的不變核心真正**新增**的只有 `stage` / `parent` / `encoder` / `adaptation`——也就是**圖結構**。

若把不變核心縮到只剩圖結構（4 個欄位），穩定區還能再省一半以上。代價是決策層要自己把兩張表對起來；是否值得，取決於實測決策品質有沒有變差。**建議先做 §6.2 的完整版，確認省下來的量之後再考慮縮欄位。**

---

## 7. 提案 C — 同一輪兩次呼叫的重複

`review_search_choice`（`loop_controller.py` 選完節點後）與 `review_and_decide`（improve 階段）在同一輪、相隔約 7 秒，**各送一份完整的討論記錄**（約 19,000 字元 × 2）。

三個選項：

| 選項 | 說明 | 評估 |
| --- | --- | --- |
| **C1 讓兩者共用同一組穩定段** | 小呼叫也帶 history | ✗ 已量化，見 §8 |
| **C2 把討論放進兩者共同的穩定前綴** | 兩個呼叫的段列表在「討論」之前必須逐字相同 | 需要小呼叫也帶 trials → 退化成 C1 |
| **C3 合併成一次呼叫** | 一次同時回「要不要改選節點」與「這一輪怎麼變異」 | 省掉一整份討論；但兩者 schema 與時機不同，耦合度上升 |

C3 是唯一真正能消掉重複的作法，但它改的是迴圈結構而非 prompt 結構，**建議等 §5、§6 落地量測完再評估**——那時討論已經可快取，重複的代價只剩 0.1x，C3 的效益可能就不值得那個耦合。

---

## 8. 刻意不做

### 8.1 不要讓 `review_search_choice` 也帶 history

直覺上「讓它也去命中那筆 87,860 的大快取」會省錢，實際上**貴 53%**：

```
現在（穩定前綴只有 2,505 字元）        17,082 單位
帶 history 共用大前綴（94,285 tokens）  26,066 單位
```

因為 cache read 雖然只要 0.1x，但 94,285 tokens 的 0.1x（9,429）已經比它現在整個 volatile 全額付費（16,638）的一半還多，而 volatile 該付的一毛都沒少。**小呼叫維持小 prompt 才是對的。**

### 8.2 不要為了省 token 去壓縮中文自由文字

討論記錄改 JSONL 只省 2%。自由文字的成本在字數本身，壓縮格式沒有意義——要省只能靠讓它可快取（§5）或減少重送（§7）。

### 8.3 不要把 `model_dir` 之類的內部欄位放進 `model_dump()`

`_registry_context()` 的內容會逐字進 system block。任何新增的內部欄位都會讓那 3,065 tokens 的快取失效一次，也會多送無意義的 token 給 LLM。`EncoderCard.model_dir` 用 `Field(exclude=True)` 正是為此。

---

## 9. 預估效益

以最新一輪的量測為基準，**估算**（標示為估算：假設 1 字元 ≈ 0.66 token，取自本輪 volatile 的實測比值）：

| | 現況 | §5 + §6 之後（估） |
| --- | ---: | ---: |
| `review_search_choice` | 17,082 | ~7,400 |
| `review_and_decide` | 35,892 | ~23,000 |
| **每輪合計** | **~52,974** | **~30,400（-43%）** |

主要來自：討論記錄從「每輪兩份全額」變成「每輪只寫新增的幾則」，解答樹的 volatile 減少 83%。

實際數字要跑起來看 Console，本節純屬紙上推算。

---

## 10. 實作步驟

1. `redact.discussion_facts` 改為累積模式（新增 `cumulative: bool` 或把 `limit` 語意改成「凍結門檻」），**同步改 `redact.user_segments`**，兩者吃同一參數（§5.4）。
2. `_data_block` 在 trials 之後、使用者事實之前插入討論段（每則一段）。
3. `review_search_choice` / `review_and_decide` 的 volatile 移除討論、改為只重述最近 N 則（§5.3）。
4. `loop_controller._tree_view` 拆成 `_tree_core()`（不變）與 `_tree_state()`（`allowed` / `is_leaf`）。
5. `_data_block` 收下 `_tree_core()` 的每節點段；`review_search_choice` 的 volatile 只帶 `_tree_state()`。
6. 測試：延伸 `tests/test_llm_advisor_cache.py` — 新增一則討論 / 新增一個節點後，既有段必須逐字且逐段不變；`user_segments` 必須涵蓋 prompt 中所有使用者原文。
7. 跑一輪實測，比對 Console 的 cache read / write 與 `llm_calls.jsonl` 的 `cache_breakpoints`。

---

## 11. 驗證方法

不要靠推理判斷快取有沒有生效——**這個專案已經連續兩次推理錯誤**（第一次以為是 TTL，第二次以為內容是前綴就夠了）。一律用兩份證據對：

1. **Console 的 request 明細**：Input / Cache Read / Cache Write(5m) / Cache Write(1h)。三者相加 = 該筆的 Input Tokens。
2. **`runs/<run>/[fold*/]llm_calls.jsonl`**：每筆有 `cache_segments`（段數）、`cached_prefix_chars`（穩定前綴總長）、`cache_breakpoints`（各 breakpoint 的累計位移）、`cache_ttl`。

判讀規則：

- **上一輪的 `cache_breakpoints` 必須出現在這一輪的段邊界集合裡**，否則就是 §3.1 的邊界沒對上（不論內容多乾淨）。
- `Cache Read` 應該接近上一輪的最大 breakpoint 位移換算的 token 數。
- `Cache Write` 應該只有新增的量。**write 大、read 小 = 每輪重寫，最貴的失敗模式**。
- 小呼叫出現 `write = 0` 是**正確**的（純命中），不是異常。

離線比對的作法（不必發請求）：取兩輪的 `prompt` 與 `cache_breakpoints`，檢查前一輪的段列表是否為後一輪的前綴。

---

## 12. 資料圍欄考量

本文件的所有提案都**只改變內容的排列與分段，不改變送出去的內容**，因此不影響 [data_firewall_design.md](data_firewall_design.md) 的結論。但有三點要在實作時守住：

- **出口掃描涵蓋所有 block**：`egress._payload_text` 會走訪 `system` 與每則 message 的所有 content block，分再多段都會掃到。分段時**每一段必須是完整的語意單位**（一整行 JSON、一整則討論），不能把一個字串切成兩段——否則跨邊界的內容在掃描時會被 `"\n\n"` 隔開而漏掉。
- **`user_segments` 必須與 prompt 中的使用者原文逐字對應**（§5.4）。這是 fail-closed 的前提，不同步就會誤判成違規並中止 run。
- **討論記錄含使用者自由輸入**。移進穩定區只是換位置，仍走 `guard_on_user_input` 政策（預設 warn）；但因為它會被快取更久，`runs/<run>/privacy_audit.jsonl` 的稽核仍以「每次請求」為單位記錄，不受快取影響。

---

## 13. 已完成的部分（供對照）

以下在 2026-07-28 已實作，本文件的量測即基於此：

| 改動 | 位置 |
| --- | --- |
| trials 每筆自成一段（§3.1） | `_data_block` |
| 分析器結果排序 + 每筆一段 | `_facts_segments` |
| 使用者事實移到 trials 之後、自成一段 | `_user_facts_block` / `_data_block` |
| breakpoint 依累計長度、保留最後 3 個 | `_breakpoint_indices` |
| TTL 改用最長間隔 + 1h 上限排除 | `_next_cache_ttl` / `_note_gap` |
| resume 時從既有 log 接續間隔統計 | `log_path` setter / `_seed_gap_from_log` |
| 最小可快取前綴依模型查表 | `_CACHE_MIN_TOKENS` / `_cache_min_chars` |
| 落地 `cache_segments` / `cache_breakpoints` | `_cache_log_fields` |

測試見 `tests/test_llm_advisor_cache.py`。
