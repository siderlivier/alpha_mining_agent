# 使用教學與功能解說

> 2026-09-21：本文件只涵蓋**因子挖掘與逐因子評估**。組合回測、過擬合診斷套件與
> 市場狀態模型（HMM）／總經三章已移除——HMM 與總經整組移到 `research_archive/`
> 不進版控，回測與合成（`backtest.py`／`factor_lab.py`／`ml_diagnose.py`）程式仍在，
> 但現階段不用於任何結論（理由見 `README.md` 的「為什麼不報組合績效」與 `TODO.md` 的 R05）。
> audit 預設只回饋 validation；舊 [audit] 經驗不再注入 prompt，新驗證經驗標 [audit:validation]。

本文件是 `README.md` 的展開版：解釋**每個功能為什麼存在**、**參數怎麼調**、
**出事了怎麼辦**。設計理由的完整版在 `SPEC_架構設計規格書.md`。

---

## 目錄

1. [系統怎麼運作（一輪發生什麼事）](#1-系統怎麼運作一輪發生什麼事)
2. [三種 LLM 呼叫](#2-三種-llm-呼叫)
3. [記憶系統](#3-記憶系統)
4. [整理回合](#4-整理回合)
5. [預算計量與控制](#5-預算計量與控制)
6. [逾時與失敗處理](#6-逾時與失敗處理)
7. [運算子提案：從提案到上線](#7-運算子提案從提案到上線)
8. [config.yaml 完整參數說明](#8-configyaml-完整參數說明)
9. [日常操作指令速查](#9-日常操作指令速查)
10. [疑難排解](#10-疑難排解)

> 新增功能索引：[參考因子](#參考因子stage-2-的盲區修補)、[佇列消費機制](#佇列消費機制)、
> [運算子解鎖（第五步）](#批准後怎麼實作五步驟)、[三層前瞻檢測](#前瞻偏差檢測三層)

---

## 1. 系統怎麼運作（一輪發生什麼事）

```
                    ┌─ memory/learnings.md（經驗庫）
                    ├─ memory/library.json（已入庫因子摘要，不含績效）
                    └─ fields.yaml + DSL 運算子卡片
                              │
                    ① Generate（LLM，約 300~560s）
                              ↓  15 個候選：hypothesis / prediction / formula
                    ② Evaluate（本地 pandas，約 50s，不花 token）
                              ↓  四階段漏斗 + 八項診斷 → verdict
                    ③ Distill（LLM，約 200~390s）
                              ↓  逐候選歸因：diagnosis / failure_type / lesson
                    ④ attempts 落盤（15 筆，只進不出）
                    ⑤ passed 因子入庫（含密封 test 指標）
                    ⑥ learnings.md 追加（經驗 ≤3 條、佇列 ≤3 條）
                    ⑦ DSL 受限訊號記錄（整理回合的提案素材）
                              ↓
                    ⑧ 記帳 → budget.json
                    ⑨ 每 10 輪觸發整理回合
```

**關鍵設計：② 是純本地計算。**LLM 只負責「提假設」和「解釋為什麼失敗」，
所有績效數字都由本地 Python 算，LLM 從頭到尾看不到 test 期指標。這是防止
「LLM 對著測試集調參」的根本機制。

### 四階段漏斗（Stage 1~4）

| 階段 | 檢查 | 門檻（`config.funnel`） |
|---|---|---|
| Stage 1 | 訊號存在性 | `\|IC\| ≥ 0.02` 且方向與 `direction` 一致 |
| Stage 2 | 與**因子庫＋參考因子**去相關 | 對任一因子 `\|ρ\| ≤ 0.5`（見下方「參考因子」） |
| Stage 3 | 與**同輪其他候選**去相關 | `\|ρ\| ≤ 0.7` |
| Stage 4 | 穩健性 | sub-train ICIR ≥ 0.45、validation ICIR ≥ 0.30、衰減 ≤ 50%、多頭腿超額 > 0、覆蓋率 ≥ 0.60、月換手 ≤ 0.40 |
| Stage 4b | 產業限定通道 | 全池未達標時，檢查單一產業是否達到更嚴的門檻（0.60 / 0.35） |

### 參考因子（Stage 2 的盲區修補）

2026-09-19 起，正式參考池改為 `registered_base_fields_v1`：候選是 `fields.yaml` 登錄的全部 32 個基礎欄位，不使用曾依 test 選取的上游 DFS survivors 或舊候選 CSV。目前 18 個通過，舊 25 個 DFS 參考因子完整保存在提交前備份。

```powershell
python -B src/seed_reference.py --dry-run
python -B src/seed_reference.py --list
python -B src/seed_reference.py --apply
python -B src/seed_reference.py --clear
```

`--list`／`--dry-run` 會重算訓練與驗證指標，但不提交。`--apply` 一次替換參考池並備份；不需要先 clear。`--clear` 也走同一備份交易。舊 `--from-upstream`／`--all` 路徑已移除，避免繞回 test 篩選。

預設資格：訓練期 |t| > 2、|ICIR| > 0.3；驗證期同向且 |ICIR| > 0.2；衰減 ≤ 50%；有效月份至少 36／12。每個評估區間的最後一月不列入，避免標籤價格跨越區間。方向只由 sub_train 決定，負向欄位儲存前取負，名稱與 metadata 標明反向。ICIR 使用月 IC 均值／標準差，沒有乘年化係數。

去重使用 sub_train＋validation 的月內排名池化相關，預設 |ρ| > 0.95 剔除較弱者；不讀 test 值來決定保留。test 期值仍保存在正式值檔，供日後人類評估，但不參與資格或方向判定。可用 `--min-icir`、`--max-decay`、`--max-ref-corr` 指定規則，請勿依 test 結果調參。

`memory/transactions/<交易ID>/` 保留提交前完整 library／values；新 R-ID 不重用歷史 ID。中斷鎖須先確認工作已結束，再執行 `python -B src/factor_scope.py --recover`。這是故障回復，尚非任意歷史版本的一鍵切換。

mock 挖礦會把因子庫、attempts、經驗、預算及日誌放在系統暫存目錄；mock 整理同樣隔離，不覆寫正式經驗。啟動時印出挖礦 sandbox 路徑。

歷史「26 個 own 對 25 個 DFS」績效仍是舊版本研究，不適用於新的 18 個參考因子，也不因流程修正而成為全新 holdout。

---

## 2. 三種 LLM 呼叫

全系統只有三種 LLM 呼叫，全部走 `claude -p`（headless）：

| 呼叫 | 模板 | 實測耗時 | 逾時上限 |
|---|---|---|---|
| Generate | `prompts/generate.md` | 中位 418s、最大 561s | `timeout_sec`（1200s） |
| Distill | `prompts/distill.md` | 中位 250s、最大 469s | `timeout_sec`（1200s） |
| Consolidate | `prompts/consolidate.md` | 約 300~530s | `consolidate_timeout_sec`（1800s） |

**耗時主要由「輸出長度」決定，不是輸入長度。**實測把 `learnings.md` 從
24,459 字元壓回 4,400 字元，Generate 只快了 4%（遠低於 98s 的標準差）。
想讓一輪變快，該調的是 `candidates_per_round`（每個候選都要寫一整段中文假設），
不是清經驗庫。

### 進度輸出

每通呼叫前後都會 log，避免終端機看起來像當掉：

```
Generate: prompt 28,431 字元，timeout 1200s，呼叫中...
Generate: 回應 8,204 字元，耗時 485s，227,414 tokens / $1.4069
```

任何一通耗時超過 timeout 的 80% 會額外警告，讓你在真的爆掉之前先看到。

---

## 3. 記憶系統

### 兩層結構

| 檔案 | 性質 | 會不會進 prompt |
|---|---|---|
| `memory/attempts/A-XXXX.json` | 逐筆嘗試，**只進不出、永不刪改** | ❌ 只有整理回合會看摘要 |
| `memory/learnings.md` | 蒸餾後的經驗 | ✅ **每輪 Generate 全文注入** |
| `memory/library.json` | 因子庫：自有 `F-xxx` + 參考 `R-xxx` | ⚠️ 自有的注入 id/中文名/面向/公式/解釋；參考的只給名稱與描述、**不給公式**。兩者**絕不含績效數字** |

`library.json` 裡的 `test_metrics_sealed` 是密封欄位，只供人類查閱。任何把它
放進 prompt 的改動都等於毀掉整個實驗的可信度。

### learnings.md 的小節規則

Distill 追加經驗時，`section` 欄位只接受：

1. 白名單三節：`全域規則`、`禁忌方向`、`待驗證假設佇列`
2. **learnings.md 裡已經存在的小節**（整理回合會自行重組出「紅海地圖」
   「聚合審計」這類有用的小節，不該被擋掉）

憑空造的新小節（例如某個 category 名）會被歸入「全域規則」並在 log 警告。
這是為了防止長出一堆只有一兩條的孤兒小節——早期版本因為 `prompts/distill.md`
裡寫著「或 category名」，累積出 9 個垃圾小節。

### 每輪追加上限

| 項目 | 上限 | 理由 |
|---|---|---|
| `learnings_additions` | 全輪 3 條 | 一輪能學到的通則本來就不多 |
| `queue_suggestions` | 全輪 3 條 | 佇列每輪只該長幾條；消化速度有限 |
| 佇列總量 | 50 條（`memory.QUEUE_MAX_ITEMS`） | 安全網；滿了拒收新項而非丟舊的（頂端是整理回合排過序的高優先項） |
| 一輪消化 | 5 條 | `prompts/generate.md` 的軟性限制，其餘配額留給新假設 |

> 計數同時認得 `- x` 和 `1. x` 兩種清單格式。整理回合的 LLM 常把 `-` 改寫成
> 編號清單，只認 `-` 的話計數會歸零、上限形同失效。

### 佇列消費機制

佇列的每一條都有 `[Q-xxx]` 編號。完整循環：

```
① append_learning("待驗證假設佇列", ...) → 自動補 [Q-xxx] 編號
② Generate 讀到帶編號的佇列，若某候選是為了消化其中一條 →
   回傳 from_queue: "Q-012"
③ mining_loop 驗證完（不論成敗）→ mem.consume_queue_item("Q-012", {...})
   → 從 learnings.md 移除，全文＋verdict＋attempt 落到
     memory/queue_consumed.jsonl（只進不出，資訊不遺失）
④ 整理回合全文重寫後，mem.normalize_queue_ids() 幫新條目補編號
```

**消費與成敗無關**——佇列是「待辦清單」不是「願望清單」，試過就該移出，
否則每輪 Generate 都要重讀已經試過的東西。結果保存在 `queue_consumed.jsonl`。

一輪最多消化 5 條（`prompts/generate.md` 的軟性限制），其餘配額留給新假設。

防呆：`from_queue` 格式不符 `Q-\d{3,}`、或編號不在目前佇列中，一律忽略，
不會誤刪。整理回合被明確要求保留既有 `[Q-xxx]` 編號。

---

## 4. 整理回合

### 做什麼

1. **升級**：≥3 個獨立 attempt 支持的 single_case 經驗 → 合併標 `[promoted]`
2. **推翻**：與新證據矛盾的舊規則 → 標 `[失效]`（不刪除）
3. **壓縮**：全文壓到 `learnings_token_cap`（4000 tokens ≈ 8000 字元）內
4. **議程**：重排待驗證佇列、刪掉已被試過或已被禁忌覆蓋的項目
5. **審計消化**：把聚合審計的發現寫成 `[audit]` 規則
6. **運算子提案**：DSL 受限訊號中同類卡點 ≥2 次時提出新運算子

### 觸發條件

```python
by_round = round_id % consolidation_every_n_rounds == 0     # 每 10 輪
by_size  = len(learnings.md) > learnings_hard_cap_chars     # 超長安全網（24,000 字元）
# 且：距上次整理嘗試 ≥ consolidation_min_gap_rounds（3 輪）才允許 by_size 觸發
```

⚠️ **`by_size` 是安全網，不該變成常規機制。**整理回合的產出目標就是 8,000
字元，每輪增長約 1,000 字元——若把門檻設在 8,000，整理完 3 輪就又觸發，
每次 $0.63 + 5 分鐘。24,000 的門檻確保每 10 輪的常規觸發一定先發生。

`consolidation_min_gap_rounds` 是防重試風暴：整理失敗時 `by_size` 條件不會
消失，沒有間隔限制的話會每輪重試（實測 R57/R58 連燒兩次）。

### 整理回合看得到多少證據

```python
ATTEMPTS_DETAIL = 45                              # 最近 45 筆逐筆全文
ATTEMPTS_WINDOW = 10 × 15 = 150                   # 涵蓋範圍
# 45~150 筆之間只給聚合統計
```

聚合區塊長這樣（約 1,500 字元，取代 16,000 字元的逐筆列表）：

```
## 較早的 105 筆（A-0669~A-0773，聚合統計，不逐筆列出）
- verdict 分布：rejected_stage1 60；rejected_stage2 29；rejected_stage4 16
- failure_type 分布：hypothesis_wrong 46；expression_bad 30；duplicate 29
- category 分布：electronics_dual_confirmation 3；...
- 重複出現的 lesson（出現次數 × 前 45 字，升級 [promoted] 的候選）：
    (3×) 券資比短期變化速度未展現出比水準值更強的資訊含量...
```

「重複出現的 lesson」正是判斷該不該升級 `[promoted]` 需要的訊號，用 1/20 的
篇幅保住了證據基礎。若把 150 筆全逐筆列出，prompt 會從 34,030 漲到 53,714
字元（+58%），足以把呼叫推過逾時。

### 安全機制

- 重寫**前**自動備份到 `memory/learnings_history/learnings_<時間戳>.md`
- 新文本缺任何必要小節（`## 全域規則`/`## 禁忌方向`/`## 待驗證假設佇列`）
  → 拒絕採用，保留原文
- 原始 LLM 輸出**一律先落盤**到 `memory/consolidate_raw/<時間戳>.txt`
- JSON 解析失敗時先做**本地修復**（不花 token）再放棄

### JSON 本地修復是什麼

整理回合要求 LLM 把一份 ~8,000 字元的 markdown 塞進 `learnings_md` 這個 JSON
字串欄位，等於要它把 200+ 個換行全部寫成 `\n`。漏掉任何一個，`json.loads`
就噴 `Invalid control character`。

但**那份輸出其實是好的**，只是跳脫沒做乾淨——而且 token 在 `call_llm` 回傳
的當下就已經花掉了。所以流程是：

```
call_llm() 回來
  ① 原始輸出無條件落盤 → memory/consolidate_raw/
  ② json.loads() 直接試
  ③ 失敗 → escape_raw_controls() 就地補跳脫，再試一次（0 token）
  ④ 還是失敗 → 拋錯，訊息告訴你原始檔在哪，可手動挖出 learnings_md 貼回去
```

②③④ 完全不打 LLM。刻意**不做重試**——重試要再花一次完整 prompt 的錢。

---

## 5. 預算計量與控制

### 真實 token，不是估算

`config.llm.output_format: json` 會讓 `call_llm` 自動附上 `--output-format json`，
從回傳的信封拆出真實用量：

```json
{"result": "...", "total_cost_usd": 0.0731,
 "usage": {"input_tokens": 12000, "output_tokens": 3400,
           "cache_creation_input_tokens": 800, "cache_read_input_tokens": 25000}}
```

四項分開記錄。舊版用 `(prompt字數 + 輸出字數) / 3` 估算，實測同一通呼叫
**估算記 7 tokens、真實 41,200 tokens**——因為估算法看不到 Claude Code 自己
塞進去的系統提示、工具定義與 cache 讀取量。

CLI 版本太舊或格式改變時會自動退回估算（係數改為 `/2`），並在 `budget.json`
的 `estimated_calls` 記下有幾通是估的，不會讓整輪掛掉。

### budget.json 欄位

```json
{
 "week_of": "2026-08-17",          // 週一，跨週自動重置
 "rounds_used": 6,                  // 本週成功輪次（失敗不佔額度）
 "total_rounds": 59,                // 全期累計，不重置
 "last_consolidate_round": 59,      // 全期狀態，擋整理重試風暴
 "tokens_used": 1524980,
 "cost_usd": 8.412145,
 "tokens_breakdown": {"input": 42, "output": 366927,
                      "cache_creation": 415388, "cache_read": 742623},
 "llm_calls": 13,
 "estimated_calls": 0,              // 沒拿到真實 usage 的通數
 "failed_calls": 0                  // 失敗但仍計費的通數
}
```

### 失敗一律計帳

**token 在 `call_llm` 回傳的當下就已經花掉了**，成敗不影響這件事。所以記帳
與成敗解耦：

```python
meter = Meter()
try:
    run_round(ctx, mem, round_id, meter, mock_fn)
except Exception as e:
    ok = False
# ← 記帳在 try/except 外面，成敗都執行
b = load_budget(); commit_usage(b, meter); save_budget(b)
```

`Meter` 由呼叫端建立並傳入，所以中途拋例外時已花掉的量仍留在呼叫端手上。
涵蓋：JSON 解析重試的那通、最後放棄的那通、逾時的那通、整理回合失敗的那通。

失敗**不佔** `rounds_used`（維持原設計），但另記在 `failed_calls`。

### 三種週上限

```yaml
weekly_round_budget: 15       # 輪次
weekly_token_budget: 0        # 0 = 不限制
weekly_cost_budget_usd: 0     # 0 = 不限制
```

檢查點在迴圈開頭，**已經開始的那一輪一定會跑完並存檔**，不會半途中斷讓資料
處於不一致狀態。

> 建議先跑一週、用 `--budget` 看真實數字再設上限。不要沿用舊的
> `est_tokens_used`，那是估算值、低估 2~3 倍。

### 查看用量

```bash
python src/mining_loop.py --budget
```

```
本週（2026-08-17 起）用量：
  輪次    6/15
  tokens  1,524,980（未設上限）
  成本    $8.4121（未設上限）
  明細    in 42 / out 366,927 / cache 建立 415,388、讀取 742,623
  呼叫    13 通
  全期累計輪次 59
  ✅ 額度內
```

### 成本結構（實測）

一輪約 **$0.9~1.5**，其中每通呼叫平均輸出 **~30,000 tokens**。但 15 個候選的
JSON 本身只需要 4,000~6,000 tokens——**約 83% 的輸出是模型的思考推理**。

想降成本的三個槓桿：

| 做法 | 怎麼改 | 影響 |
|---|---|---|
| 降低推理深度 | `llm.command: "claude -p --effort low"` | 最大槓桿，但可能影響假設品質——**先跑 `claude --help` 確認你的 CLI 版本有這個 flag** |
| 減少候選數 | `candidates_per_round: 15 → 10` | 輸出量約略線性下降，Generate 也會變快 |
| 設成本上限 | `weekly_cost_budget_usd: 15` | 純保險，不影響行為 |

> `total_cost_usd` 是 CLI 的 client-side 估算，官方文件註明可能與實際帳單有
> 出入。當作相對用量的精確計量很好用，對帳到分請看 claude.ai 的 usage 頁面。

---

## 6. 逾時與失敗處理

### 逾時 = token 花掉但拿不到結果

`claude -p --output-format json` 會把整個回應緩衝到最後才一次吐出信封。被
timeout kill 掉時 pipe 裡是空的：

```
情境 A：CLI 逾時前已寫出部分內容 → TimeoutExpired.stdout 拿得到
情境 B：CLI 結尾才一次輸出（真實行為） → TimeoutExpired.stdout = None
```

所以逾時是**最貴的失敗方式**，timeout 要按「最壞情況的兩倍」抓，不能按平均
值抓。目前設定對歷史最大值有 2.1~3.2 倍餘裕。

逾時時：用估算值記帳、`Meter.timeouts` +1、若有片段則存到
`memory/llm_raw/<時間戳>_timeout.txt`、錯誤訊息直接告訴你該調哪個參數。

### 各階段失敗的影響範圍

| 失敗點 | 後果 |
|---|---|
| **Generate 失敗** | 整輪歸零（沒候選就沒東西可保），損失一通的 token |
| **Distill 失敗** | **降級繼續**：15 筆 attempts 與入庫照常，只缺歸因/lesson/佇列 |
| **Consolidate 失敗** | 挖礦資料完全不受影響，`learnings.md` 原封不動 |

Distill 降級是重要的省錢機制：走到 Distill 時 Generate 的 token 已經花掉、
本地漏斗也跑完了，而 attempts 落盤與入庫**都不需要 Distill 的輸出**（步驟 4
本來就用 `dist_map.get(id, {})` 容錯，入庫只看 `cands + diags`）。舊版會把
通過漏斗的因子一起丟掉。

降級的 attempt 會標 `"distill_failed": true`，整理回合與事後查帳看得出來。

實際發生過一次（Round 58 撞到月消費上限）：

```
⚠️ Distill 失敗（You've hit your monthly spend limit）
   → 降級繼續：15 筆 attempts 與入庫照常，但本輪沒有歸因、lesson 與佇列建議
完成（Distill 降級）。本輪 78,964 tokens，$0.6554
```

---

## 7. 運算子提案：從提案到上線

### 提案怎麼產生

整理回合看 `memory/dsl_limitations.jsonl`（Distill 每輪寫入的「假設成立但
DSL 表達不出來」訊號），同類卡點出現 ≥2 次時提出新運算子。

### 三道機械驗證

`consolidate.validate_proposal()` 檢查：

1. 名稱格式 `[a-z_][a-z0-9_]{2,20}`
2. 簽名在純函數白名單 `(x)` / `(x, n)` / `(x, y, n)`——**不得引入新常數參數**
3. 證據引用 ≥2 個且都是真實存在的 attempt id

⚠️ **機械驗證不檢查語意可用性，也不檢查範例公式是否合法。**`qtr_only` 通過
了全部三道驗證，但實際上在月頻面板下會產生 100% NaN（季底月只佔 1/3 樣本，
而 `MIN_PERIODS_FRAC = 2/3`），而且它自己給的範例 `ts_mean(..., 4)` 的窗口 4
根本不在白名單裡。**這就是人工閘門存在的理由。**

### 審核清單

```bash
python -c "import json;d=json.load(open('memory/operator_proposals.json',encoding='utf-8'));print([p['name'] for p in d['pending']])"
```

審核時該問的問題：

- **覆蓋率**：這個運算子會不會讓大量樣本變 NaN？過得了 `min_coverage: 0.60` 嗎？
- **窗口相容**：搭配 `MIN_PERIODS_FRAC = 2/3`，在白名單窗口 `[3,6,12,24]` 下
  還有有效值嗎？
- **前瞻風險**：語意是否只依賴 t 及以前的資料？
- **是否與既有運算子重疊**：例如 `cs_rank` 已經是產業內排名了
- **效能**：需要 `_roll_apply`（逐窗 Python 迴圈）的話會拉長 Evaluate 時間

### 批准後怎麼實作（五步驟）

**沒有自動機制**——`status: "pending_review"` 只是文字欄位，程式碼裡沒有任何
地方會讀它。批准 = 你自己寫。以 `rank_nz` 為例：

**① `src/dsl.py` 的 `_SIGNATURES` 加簽名**

```python
    # 截面
    "cs_rank": ("n", "n"), "cs_rank_all": ("n", "n"), "cs_z": ("n", "n"),
    "rank_nz": ("n", "n"),          # ← 新增（n=數值, w=窗口, b=布林）
```

**② `src/dsl.py` 的 `Engine.eval` 加實作**

```python
        if name == "rank_nz":
            v = self.eval(a[0])
            return self._per_group(v.where(v.notna() & (v != 0)),
                                   lambda s: s.rank(axis=1, pct=True))
```

> 簽名含 `w` 的運算子會被 `_TS_OPS` 自動收錄，實作要寫在 `if name in _TS_OPS:`
> 區塊內（窗口參數是 `a[-1][1]`）。

**③ `tests/test_dsl_no_lookahead.py` 的 `FORMULAS` 加一條用到它的公式**

這步**不能跳過**。那裡有守門測試：

```python
def test_all_operators_covered():
    missing = set(dsl._SIGNATURES) - covered
    assert not missing, f"未被前瞻測試覆蓋的運算子: {missing}"
```

加了運算子卻沒加對應公式，`pytest` 就會紅。

**④ 補正確性測試 + 更新提案狀態**

在 `tests/test_dsl_correctness.py` 加逐格對照測試，並把
`operator_proposals.json` 裡那筆從 `pending` 移到 `implemented`。

**⑤ 跑 `python src/unblock.py --apply` 解鎖被卡住的假設**

這步最容易被忘記，但省下的錢最多。原理：

```
提案的 evidence 欄位  →  就是「被這個 DSL 缺口卡住的 attempt 編號」
        ↓
       attempts/A-xxxx.json  →  裡面有完整的 hypothesis
        ↓
   （只取 failure_type = expression_bad 的：假設沒被證偽、只是表達不出來）
        ↓
   放回 learnings.md 的待驗證假設佇列，標【解鎖】
```

沒有這步的話，那些已經付過 Generate + Evaluate + Distill 成本、而且診斷結論是
「假設可能成立、只是寫不出來」的候選，會永遠躺在 attempts 裡沒人再看。
已解鎖的記在 `memory/unblocked.json`，不會重複放。

```bash
python src/unblock.py                        # 預覽
python src/unblock.py --apply                # 寫入佇列
python src/unblock.py --op streak_true --apply   # 只處理某個運算子
python src/unblock.py --limit 5 --apply          # 一次最多放回幾條
```

**然後下一輪就自動生效**——`operators_card()` 是從 `_SIGNATURES` 動態產生的，
`build_generate_prompt()` 每輪呼叫它，新運算子會自己出現在 DSL 規格卡裡。
**不需要改 `config.yaml`，也不需要改 prompt。**

### 已實作的三個運算子

| 運算子 | 語意 | 解決什麼 |
|---|---|---|
| `rank_nz(x)` | 產業內百分位排名，但把 0 與 NaN 排除在排名之外 | `if_else(cond,X,0)` 在 `cs_rank` 下大量同分並列稀釋 IC（`[promoted]` 全域規則第 3 條） |
| `industry_demean(x)` | 產業內去均值，**保留量級** | `cs_rank` 會丟掉量級資訊，無法表達「去除產業共同 driver 後的殘差」 |
| `streak(x, n)` | 從當期往回數、單期變化同號的連續期數（1..n） | 無法表達「改善是否有持續性」「連續兩季同向才算確認」 |
| `streak_true(cond, n)` | 布林條件連續為真的期數（0..n） | 模型三次用 `streak(布林, n)` 撞 type_error——它要的是「連續 12 期 ROE>0」這種語意 |
| `clip_std(x, n)` | 截尾至**前 n 期**（不含當期）的均值 ±3σ | `sdiv`/yoy 在低基期分母下比值爆量，排序被離群樣本主導 |

`streak` 有暖機期：sign 序列首格是 NaN，`_mp(6)=4`，所以前 4 格是 NaN。
當期無變化（0）或缺值時回 0——方向未定義不該算進趨勢。

⚠️ `clip_std` 的實作**刻意偏離提案偽代碼**：邊界用 `shift(1)` 取前 n 期，不含
當期。提案原本寫 `ts_mean(x,n)`（含當期），那樣離群值會撐高自己的上界而截不到
——實測 `[10]*15+[1000]` 的上界是 949.9，等於沒截。另加了歷史零變異保護
（`sd < eps` 就不截），否則會把序列硬壓成常數。

### 提案去重

`pending` / `implemented` / `rejected_log` 三個清單的名稱都會查。只查
`pending` 的話，被你否決的提案會在下次整理回合原封不動地再被提一次。

---

## 8. config.yaml 完整參數說明

### `dsl`

| 參數 | 預設 | 說明 |
|---|---|---|
| `window_whitelist` | `[3, 6, 12, 24]` | 窗口參數只能是這幾個（單位：月）。改動會使既有 `fhash` 的語意基準改變 |
| `max_depth` | 3 | 運算子嵌套深度上限。審計顯示深度 3 的 train→test 衰減 20% vs 深度 2 的 -3% |
| `max_fields` | 3 | 每式使用欄位數上限 |

### `split`

| 參數 | 說明 |
|---|---|
| `sub_train` | 2012-01 ~ 2017-12，用於 Stage 1/2/3/4 |
| `validation` | 2018-01 ~ 2019-12，用於衰減檢查 |
| `test` | 2020-01 ~ 2026-06 — ⛔ **逐因子指標永不回饋 LLM** |

### `funnel`

見第 1 節的漏斗表。`stage4b_industry.enabled: false` 可關閉產業限定通道。

### `budget`

| 參數 | 預設 | 說明 |
|---|---|---|
| `weekly_round_budget` | 15 | 每週成功輪次上限 |
| `candidates_per_round` | 15 | 每輪候選數。**降低這個是縮短耗時與成本最直接的槓桿** |
| `consolidation_every_n_rounds` | 10 | 常規整理間隔 |
| `learnings_token_cap` | 4000 | 告訴整理回合壓縮到多少 tokens（≈8000 字元） |
| `learnings_hard_cap_chars` | 24000 | 超長觸發門檻（安全網）。**不要設成 8000**，會變成每 3 輪整理一次 |
| `consolidation_min_gap_rounds` | 3 | 兩次整理嘗試的最小間隔（擋重試風暴） |
| `weekly_token_budget` | 0 | 0 = 不限制 |
| `weekly_cost_budget_usd` | 0 | 0 = 不限制 |

### `llm`

| 參數 | 預設 | 說明 |
|---|---|---|
| `command` | `"claude -p"` | 可自由加旗標，例如 `"claude -p --effort low"`。`--output-format json` 會自動附加 |
| `output_format` | `json` | `json` = 取真實 token/成本；`text` = 退回字數估算 |
| `retry_limit` | 1 | JSON 陣列解析失敗的重試次數（Generate/Distill 用；整理回合不重試） |
| `timeout_sec` | 1200 | Generate / Distill。歷史最大 561s → 2.14x 餘裕 |
| `consolidate_timeout_sec` | 1800 | 整理回合。歷史最大 534s → 3.24x 餘裕 |

### `backtest`（`src/backtest.py`）與 `ml`（`src/factor_lab.py`）

這兩個區塊設定的是**組合回測與因子合成**（`top_q`、`cost`、`weighting`、
`min_train_months`、`retrain_every`、`embargo`、`ridge_alpha`、`min_feature_coverage`）。
程式還在、測試也還在跑，但**現階段不用於任何結論**——因子評估走的是
`crosssec_oos.py` 的逐因子 IC，不經過回測引擎。參數意義直接看 `config.yaml` 的註解。

⚠️ 若之後重新啟用這條線，`embargo` **不可設成 0**：`fwd_ret_1m` 看的是下一個月，
設 0 會讓訓練期最後一個月的標籤與測試期第一個月重疊。
`tests/test_factor_lab.py::test_segments_embargo_gap` 會擋住這種改動。

---

## 9. 日常操作指令速查

```bash
# 挖礦
python src/mining_loop.py --rounds 5     # 連跑 5 輪
python src/mining_loop.py --mock         # 不耗額度跑通管線
python src/mining_loop.py --dry-run      # 只印 Generate prompt
python src/mining_loop.py --budget       # 只印用量，不呼叫 LLM

# 查看
python src/query_attempts.py --limit 40
python src/query_attempts.py --verdict passed
python src/report.py                     # 因子庫總覽（終端機）
python src/report.py --html              # 產生 memory/report.html 並自動開啟
python src/report.py --save              # 另存 memory/report.md
python src/audit.py                      # 聚合審計（可餵 agent 的版本）
python src/audit.py --human              # 含個體明細，勿餵 agent

# 維護
python src/seed_reference.py --list      # 參考因子：預覽篩選結果
python src/seed_reference.py --apply     # 參考因子：匯入因子庫
python src/seed_reference.py --clear     # 參考因子：移除
python src/unblock.py                    # 預覽可解鎖的假設
python src/unblock.py --apply            # 放回待驗證假設佇列
python src/consolidate.py                # 手動整理（會計入預算）
python src/consolidate.py --no-budget    # 手動整理但不計帳
python src/consolidate.py --mock         # 測試整理管線

# 因子評估（目前結論用的路徑）
python -B src/crosssec_oos.py --vs-reference    # 自有 vs 基準的分布對照
python -B src/crosssec_oos.py --pairwise        # 逐對 CSV
python -B src/crosssec_oos.py --deflate-icir    # 多重檢定校正
python -B src/crosssec_oos.py --span test       # 橫斷面樣本外
python -B src/mining_stats.py                   # 漏斗組成與自我進化

# 資料擴張後的產業範圍重測
python -B src/build_base.py                     # 先重建面板
python -B src/scope_ui.py --port 8765           # 本機介面（僅 localhost）
python -B src/factor_scope.py --check-only      # 或走 CLI
python -B src/factor_scope.py --apply RUN_ID

# 測試（AI 依規格書撰寫的功能性測試；「全數通過」＝規格被編碼成可執行斷言，
#       不等於有人逐條人工審閱過 227 個案例）
python -m pytest tests/ -q               # 227 passed（約 1.5 分鐘；缺 data/ 時會跳過數支）
python -m pytest tests/test_pipeline_no_lookahead.py -q  # 只跑全管線前瞻檢測
python -m pytest tests/test_dsl_no_lookahead.py -q   # 只跑前瞻偏差測試
python -m pytest tests/test_factor_lab.py -q         # 組合實驗室（保留，但不用於結論）
```

---

## 10. 疑難排解

### `Failed to authenticate: OAuth session expired and could not be refreshed`

CLI 登入憑證過期，不是程式問題。依序試：

1. 開一般 cmd 執行 `claude`，用 `/status` 看 `Login` 那列是否顯示
   `Expired — log in again`；是的話 `/login`
2. 互動模式正常但 `claude -p` 壞掉 → `claude logout` → `claude login`，
   並確認 `claude --version` 與互動介面顯示的版本一致
3. **長期解**：改用一年期 token
   ```cmd
   claude setup-token
   setx CLAUDE_CODE_OAUTH_TOKEN "<token>"
   ```

### `You've hit your monthly spend limit`

Anthropic 帳號層級的月消費上限。設 `weekly_cost_budget_usd` 讓系統在撞到
Anthropic 的上限之前先自己停，或降 `candidates_per_round` / 加 `--effort low`。

### `LLM 呼叫逾時`

調高 `timeout_sec` / `consolidate_timeout_sec`。若逾時前有收到片段，路徑會寫在
錯誤訊息裡（`memory/llm_raw/`）。逾時的 token 已用估算值計入預算。

### 整理回合 `JSON 物件未閉合` / `Invalid control character`

先看 `memory/consolidate_raw/<最新時間戳>.txt`——本地修復已經試過一次仍失敗，
但**原始輸出還在**。可以手動從中取出 `learnings_md` 的內容貼進
`memory/learnings.md`（`learnings.md` 本身未被更動）。

### `RuntimeWarning: invalid value encountered in divide`

已修。成因是 `xsec_corr()` 在某月因子為常數時，`rank` 全同分 → 標準差 0 →
`np.corrcoef` 除以零。結果本來就是 NaN 會被濾掉，只是會洗版。若再出現，
檢查是否有其他 `.corr()` 呼叫缺少 `nunique() < 2` 保護。

### 終端機好像卡住了

Generate 約 300~560s、Distill 約 200~390s、整理約 300~530s，這期間沒有輸出是
正常的。現在每通呼叫前後都有進度行，看 `mining.log` 最後一行是「呼叫中...」
就表示還在跑。真的卡住的話會在 timeout 後拋錯，不會無限等待。

### 佇列項目一直重複出現

檢查 `learnings.md` 的佇列條目有沒有 `[Q-xxx]` 編號。整理回合全文重寫時若把
編號改掉或刪掉，消費機制就認不得。跑 `python -c "import sys;sys.path.insert(0,'src');
from memory import Memory;print(Memory().normalize_queue_ids())"` 補回去。

### 實作了新運算子，但沒看到相關的新候選

跑 `python src/unblock.py` 看有沒有可解鎖的假設。若顯示 0 條，可能是那個運算子的
`evidence` 對應的 attempt 失因不是 `expression_bad`（`hypothesis_wrong` 代表機制
已被證偽，刻意不解鎖），或是已經解鎖過了（見 `memory/unblocked.json`）。

### 前瞻偏差檢測（三層）

```bash
python -m pytest tests/test_dsl_no_lookahead.py -q      # DSL 層：33 個運算子
python -m pytest tests/test_pipeline_no_lookahead.py -q # 全管線（資料 → 漏斗 → 診斷）
python -m pytest tests/test_factor_lab.py -q            # 下游（因子合成 → 回測）
```

**DSL 層**：對覆蓋全部運算子的公式集，把 `t_cut` 之後的資料換成 1e6 重算，
斷言 `t_cut` 以前逐格完全一致。`test_all_operators_covered` 確保新增運算子
時不會忘了加測試公式。

**全管線層**：造一份合成 `monthly_base`，複製一份把 sub_train 結束月之後的
**所有欄位**汙染，兩邊各跑一次完整的 `evaluate_batch`，斷言 sub-train 期的
診斷數字逐項相同。同時檢查：

- 切分不重疊（`max(sub_train) < min(validation) < max(validation) < min(test)`）
- `fwd_ret_1m`（label，本來就看未來）不在 `ctx.fields` 也不在 `ctx.data`
- 參考因子進入 Stage 2 後同樣不引入前瞻（造一個未來全汙染的 R-001 測）

> ⚠️ 汙染起點**從 `config.yaml` 的 `split.sub_train[1]` 讀**，不寫死。
> 這裡曾經寫死 index 導致汙染點落在 sub_train 內部，測試自己製造了假陽性。

**下游層**（因子合成 → 回測）：漏斗把因子放行之後，還有一段路才變成組合績效，
那段路有自己的三個洩漏面：

- `prep` 的產業內 z-score 只用當期橫斷面（改成全期 mean/std 會洩漏且不報錯）
- `walk_forward` 的訓練窗不含測試期，且留 `embargo` 個月空檔
- `backtest` 只吃算好的 score，不做任何跨期運算

同樣用「汙染未來 → 斷言過去位元級不變」的手法（`check_exact=True`）。
這套測試驗證過確實有效：把訓練窗改成全期、或把 embargo 拿掉，測試立刻轉紅。

> ⚠️ **這一層與上面兩層的用途不同。** 它保護的是 `factor_lab.py` 的合成模型與
> `backtest.py` 的組合回測，而那條路徑**現階段不用於任何結論**——本專案的因子評估
> 走 `crosssec_oos.py` 的逐因子 IC，不經過合成模型也不經過回測引擎。
> 測試保留著是因為程式還在，不是因為結論依賴它。同理，已封存的 HMM／總經測試
> （`test_hmm_regime.py`、`test_fetch_macro.py`，87 個）留在 `research_archive/`，
> 與本文的任何內容無關。

上游資料對齊也查過：`build_base.py` 與前置專案 `mine_dfs.py` 的月頻快照都是
`groupby(["stock_id","ym"]).tail(1)`（月底），`fwd_ret_1m` 都是下月報酬，
兩邊定義一致，所以參考因子的 `ym=M` 與 agent 的 `ym=M` 指的是同一個時點。

### 測試 `test_all_operators_covered` 失敗

你加了 DSL 運算子但沒在 `tests/test_dsl_no_lookahead.py` 的 `FORMULAS` 加一條
用到它的公式。這是刻意的守門測試，不要繞過。

### learnings.md 被整理回合改壞了

每次重寫前都會備份到 `memory/learnings_history/learnings_<時間戳>.md`，直接
複製回去即可。`learnings.md` 也可以隨時手動編輯——它是給人看也給人改的。


### 2026-09-20 呼叫失敗與預算說明

失敗CLI／錯誤信封／逾時會記錄可得用量；未回報金額顯示未知，不能理解為免費。token或美元上限設0表示未限制；若啟用美元上限且出現未知金額，核對帳單前不啟動下一通。額度在每通呼叫前檢查，已在進行的一通不保證不超支。預算檔損毀時先修復紀錄，不應刪掉它讓預算歸零。

本週已發生但CLI未回報的費用，請以供應商帳單核對；修復程式不會自動追溯補齊先前漏記。
