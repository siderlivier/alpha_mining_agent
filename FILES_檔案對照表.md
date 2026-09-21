# 檔案對照表

2026-09-20 挖礦／回測可靠性修正沿用既有檔案，未新增功能腳本。memory：入庫範圍驗證、同範圍歷史指標與 prompt 經驗過濾；audit：預設 validation，test 僅 human；report：同範圍展示、產業通過及 token／空庫修正；factor_lab：因果覆蓋選取、保留未知標籤預測列；backtest：缺價明確報錯與期初回撤。對應測試擴充 test_m4、test_industry_track、test_factor_lab。fetch_macro／hmm_regime／threshold_sweep 暫緩，尚未封存。

`tests/test_compare_upstream.py` 同時可獨立執行保存結果的選股稽核：`python tests/test_compare_upstream.py --audit-run logs/<run>`。它核對來源雜湊、重算覆蓋後的流動性排名、重播訊號持股，輸出 `selection_rank_evidence.csv`，逐筆列出缺報酬持股的流動性百分位、分數排名與新進／續抱狀態。歷史資料不在本機時，對應整合測試會明確 skip，其餘合成案例仍能執行。

2026-09-20 新增跨專案比較：`src/compare_upstream.py` 只讀前置專案的 DFS 產生器、月頻快取與成交價格，將自有／原 25 個 DFS／固定 8 因子／自有加 DFS 套用同一模型階梯、流動性與緩衝規則。`--universe upstream` 沿用前置 ML 篩選順序；`--universe common` 額外提供共同股票池敏感度對照。輸出至 `logs/dfs_comparison_<時間>/`，不替換正式因子庫；相容績效有未來報酬篩選限制，須一起閱讀選股稽核，不能當成已消除前瞻偏差的結果。

`tests/test_compare_upstream.py` 覆蓋產業內流動性、前 20% 續抱、股票池縮小、未成交不補選、缺未來報酬不改選股、跨月持股狀態及預測不讀當期未來標籤等案例。輸出中的 manifest 保存來源雜湊與設定，runner_snapshot.py 保存執行版本；大型 parquet 不進 Git。

> 本專案每一個檔案的中文名稱與職責。更新日期：2026-09-18
>
> 其他文件的分工：`SPEC_架構設計規格書.md` 講**為什麼這樣設計**、
> `GUIDE_使用教學.md` 講**怎麼操作**、`RESULTS.md` 講**數字與出處**、
> 本檔講**哪個檔案在做什麼**。

**四層架構**（資料單向流動，下層依賴上層）：

2026-09-18：第一批挖礦修正沿用既有檔案，未新增功能腳本。
eval_candidates 負責原精度裁判、定向與交易用途；admit／memory 保存定向與用途；
factor_lab 排除僅做空因子；crosssec_oos 重算時套用儲存定向；report 顯示用途；
build_base 以精確去年同月及前值絕對值計算 EPS／營業利益變化。
回歸案例擴充於既有 test_funnel、test_pipeline_no_lookahead、test_factor_lab。
資料遷移與剩餘工作見評估報告第 9 節。
後續快取重建已完成，最新結果與回復方法見評估報告第 10 節；
`data/rebuild_history/<run_id>/` 保存本機原檔、暫存與 manifest（不進 Git）。
產業擴張、自動分類、參考池換版及 Web UI 設計統一放在 SPEC 附錄 F，尚未實作，未新增功能腳本。

```
挖掘層  LLM 提假設 → DSL 寫成公式 → 四階段漏斗 → 記憶蒸餾
   ↑ 這是本專案的主軸，其餘三層是為了讓它的產出可信、可用
資料層  前置專案 panel.parquet → 月頻面板 / 參考因子 / 總經
評估層  因子值 → 合成分數 → 回測 → 過擬合診斷
應用層  用市場狀態切換因子組（支線，結論不顯著）
```

---

## 一、根目錄：設定與文件

| 檔案 | 中文名稱 | 做什麼 | 版控 |
|---|---|---|---|
| `config.yaml` | 全域設定（單一事實來源） | 所有門檻集中在此：DSL 複雜度、資料切分、四階段漏斗門檻、預算、回測參數、ML 參數、HMM 參數、LLM 逾時。**改任何 `funnel` 或 `dsl` 區塊前要先讀規格書第 9 章。** 註解裡記錄了調參的理由與踩過的坑（例如逾時從 600s 調到 1200s，因為實測 Generate 平均 454s） | ✅ |
| `fields.yaml` | 基礎欄位白名單 | 27 個可用欄位分五類（價量 10、品質 6、成長 5、價值 4、籌碼 2）。每個欄位的 `desc` **會注入 Generate prompt**——那不是註解，是 prompt engineering，LLM 的假設品質取決於它對欄位經濟含義的理解 | ✅ |
| `README.md` | 專案門面 | 對外說明：成果、方法論、但書、架構、安裝使用 | ✅ |
| `SPEC_架構設計規格書.md` | 架構設計規格書 | 每個設計決定的**理由**與驗收標準，11 章 | ✅ |
| `GUIDE_使用教學.md` | 使用教學 | 逐項功能解說、參數調校、疑難排解，13 章 | ✅ |
| `RESULTS.md` | 實測數據總表 | 17 節，所有數字＋產生它的指令。含第 16 節「結論被推翻」的完整紀錄 | ✅ |
| `TODO.md` | 待辦清單 | 跨兩個專案的待處理事項 | ✅ |
| `REVIEW_程式碼與數值正確性評估報告_2026-09-14.md` | 程式碼與數值正確性評估報告 | 27 項問題的嚴重程度、證據、數值影響、修正方向及驗收條件；含 test 資訊，僅供人類審閱，不放入挖礦 prompt | ✅ |
| `FILES_檔案對照表.md` | 檔案對照表 | 就是本檔 | ✅ |
| `.gitignore` | 版控排除規則 | 註解說明了**每個被排除的檔案怎麼重建**。核心原則：`memory/` 底下的 JSON 與 markdown **要**進版控（那是 agent 的記憶，是專案核心資產），只有 37MB 的 `factor_values.parquet` 因體積排除 | ✅ |

---

## 二、`src/` 挖掘層 — 本專案的主軸

| 檔案 | 行數 | 中文名稱 | 做什麼 |
|---|---|---|---|
| `dsl.py` | 509 | 受限 DSL 引擎 | 因子公式的語言。**用 Python `ast` 解析，絕不 `eval`**。33 個運算子帶型別簽章（`n` 數值／`b` 布林／`w` 窗口），窗口只准 {3,6,12,24}、深度 ≤3、欄位 ≤3、禁巢狀 `if_else`、禁魔術常數。前瞻偏差是**語言層面表達不出來**，不是事後檢查。`_canon()` 把交換律運算子正規化以偵測等價公式。每個人工核准的新運算子都在型別表旁註記它的證據 attempt 編號 |
| `eval_candidates.py` | 527 | 四階段漏斗評估器 | 候選批次 JSON → 每個候選一份約 300 token 的緊湊診斷。Stage 1 IC 門檻 → Stage 2 與因子庫去相關（ρ≤0.5）→ Stage 3 批次內去相關（ρ≤0.7）→ Stage 4 ICIR／衰減／多空腿／覆蓋率／換手；Stage 4b 是產業專屬通道（門檻**更嚴**，因為單產業樣本少又有跨產業多重比較）。`sealed_test_metrics()` 算出的 test 指標永不進入輸出 |
| `mining_loop.py` | 804 | 挖礦迴圈 orchestrator | 一輪的完整編排：預算檢查 → 組 Generate prompt → `claude -p` → 解析候選 → 本地漏斗 → 組 Distill prompt → `claude -p` → 解析歸因 → attempts 落盤 → passed 入庫 → lesson 追加。含 token 計量、成本記帳、逾時搶救（LLM 原始輸出先落地再解析） |
| `memory.py` | 413 | 記憶層 | 兩層記憶的讀寫。`attempts/` 只進不出、永不刪改；`library.json` 存入庫因子中繼資料（含密封 test 指標）。`admit()` 會擋下缺中文名的、未通過的、以及公式等價於既有因子的候選。`own_library()` / `reference_library()` 把自有 F-xxx 與參考 R-xxx 分開統計 |
| `consolidate.py` | 323 | 整理回合 | 每 10 輪觸發一次。聚合審計 → 組 prompt（經驗庫全文＋近期 attempts 摘要＋audit＋DSL 受限訊號）→ LLM 重寫 `learnings.md` → **驗證與備份後才落盤**（重寫得太短或結構壞掉會被拒絕）→ 運算子提案經機械驗證寫入 `operator_proposals.json` |
| `admit.py` | 72 | 入庫工具 | 把評估通過的候選寫進因子庫。重算因子值 → 計算密封 test 指標（不輸出）→ `memory.admit()`。**stdout 只顯示中文名／面向／公式，密封指標絕不列印** |
| `unblock.py` | 138 | 解鎖卡住的假設 | 當初因為「DSL 缺這個運算子」而失敗的假設，在新運算子上線後放回佇列重試。不做這件事，那些已經付過診斷成本的候選就白白浪費 |
| `query_attempts.py` | 59 | 嘗試紀錄檢索 | 用標籤檢索取代向量庫（零 token 成本）。供整理回合與人工審計調閱個案 |
| `seed_memory.py` | 31 | 記憶初始化 | 建立空的記憶骨架。**刻意不預載任何理論或前置專案的結論**——避免把預設想法當成記憶餵給模型、汙染它自己的歸納 |
| `mining_stats.py` | 134 | 挖礦運作統計 | 漏斗淘汰組成、入庫率隨時間的變化、記憶與提案規模、成本。存在理由：README 宣稱「入庫率下滑是因子庫飽和不是 agent 退步」，那是可否證的主張，需要一支能重跑的程式而不是一段文字。判準是**看 S1／S4 與 S2 的走向是否分歧** |

---

## 三、`src/` 資料層

| 檔案 | 行數 | 中文名稱 | 做什麼 |
|---|---|---|---|
| `build_base.py` | 191 | 月頻面板建構 | 前置專案的 `panel.parquet`（日頻 PIT 面板）→ `monthly_base.parquet`（月頻 36 欄快照）。分三階段可獨立重跑：`--stage prices` 價量技術面、`--stage chips` 籌碼面、`--stage final` 財務面＋合併＋算標籤。`fwd_ret_1m` 逐月 winsorize 到 1%/99% 壓制假極端 |
| `build_regime.py` | 96 | 市場狀態標註表 | 逐年 TAIEX 報酬／波動 → 多空盤整標籤（>+10% 多頭、<−10% 空頭）＋月頻大盤報酬。標籤由數字自動生成（客觀），風格備註是人工知識 |
| `seed_reference.py` | — | 參考池換版 | 使用全部登錄基礎欄位，train/validation選取、定向及去重；18個新參考。舊DFS版本留在交易備份，不再讀舊候選test指標 |
| `fetch_macro.py` | 859 | 總經資料抓取 | 景氣對策信號（分數＋燈號）、領先／同時指標、M1B/M2 年增率與黃金交叉、美債殖利率。只走政府開放資料與 FinMind 免費層。**最重要的設計決定：發布落後寫進資料結構本身**——每列都有 `ym`（描述的月份）與 `pub_ym`（已公開的月份），取值一律走 `as_of()`；新增欄位若忘了在 `PUB_LAG_MONTHS` 登記會**直接丟錯**而不是預設 0 |

---

## 四、`src/` 評估層

| 檔案 | 行數 | 中文名稱 | 做什麼 |
|---|---|---|---|
| `backtest.py` | 157 | 回測核心 | 由分數建構「逐月在各產業內取前 `top_q` 等權做多」的組合，算月報酬、換手、成本、績效與超額。移植自前置專案並改為讀 `config.yaml`——**移植而非重寫，因為那套邏輯已驗證過，重寫只會製造語意漂移**。算出 `long` / `long_short` / `benchmark` 三條腿，但目前所有 headline 都只用 `long` |
| `factor_lab.py` | 474 | 因子組合實驗室 | 回答三個問題：自有 vs 參考因子哪組強、哪些因子在組合裡是拖累、最佳組合要幾個因子。`--compare` 合成比較（equal/ridge/lgbm）、`--loo` 留一法、`--greedy` 貪婪前向選擇、`--cost-scan` 成本敏感度。`segments()` 是 walk-forward 唯一的切窗來源（可測試的接縫），`--select-span` 把「在哪裡挑」與「在哪裡報」分開，避免選擇偏誤 |
| `ml_diagnose.py` | 910 | 過擬合診斷套件 | 存在理由只有一個：**一個漂亮的 IR 有很多種假法，每一種要用不同的方法拆穿**。`--regime` 多空狀態分解、`--breadth` 逐月攤開與集中度、`--liquidity` 流動性歸因與過濾、`--importance` 特徵重要性＋SHAP＋規模中性、`--dsr` Deflated Sharpe＋PBO、`--regime-factors` 逐因子多空月 ICIR 差異、`--timing` 擇時上限與損益兩平準確率 |
| `audit.py` | 123 | 聚合審計 | 對入庫因子計算 sub-train → test 的 ICIR 衰減，但輸出**只按維度聚合**（category／AST 深度／運算子／欄位面向／窗口參數）。**絕不輸出任何單一因子的 test 數字**——agent 只能學到「哪類設計容易衰減」，學不到「哪個因子在 test 期表現如何」 |
| `report.py` | 263 | 人類專用報表 | 因子庫總覽＋挖礦統計，可輸出終端／HTML／markdown。⚠️ **含密封 test 指標，只供人類閱讀決策，嚴禁複製進任何 prompt 或 `learnings.md`** |

---

## 五、`src/` 應用層（支線）

| 檔案 | 行數 | 中文名稱 | 做什麼 |
|---|---|---|---|
| `hmm_regime.py` | 847 | 市場狀態模型 | Gaussian HMM 判斷下月多空態，用途**不是進出場而是切換因子組**。三道前瞻紀律：觀測值時點對齊（`obs_ret[t] = mkt[t−1]`）、**只用自己寫的前向濾波不用 hmmlearn 的 Viterbi／平滑**（那兩者會用到未來觀測）、參數與標籤只在訓練窗擬合。指令：`--fit` 擬合看狀態性質、`--predict` 樣本外方向準確率、`--apply` 切換因子組、`--ablate` 特徵消融、`--riskadj` 風險調整後三張表（原始指標／等 beta 對照／配對自助法 CI） |
| `threshold_sweep.py` | 58 | 門檻掃描 | 同時報 Sharpe 與 IR，看「換裁判會不會換結論」。存在理由：`--riskadj` 只比較預設門檻 0.5，若結論只在 0.5 成立就是巧合不是結論。⚠️ 這張表能支持的是「結論不依賴門檻」，**不是**「最佳門檻是 0.2」——在 test 期挑最好的門檻就是選擇偏誤 |

| `crosssec_oos.py` | 134 | 橫斷面樣本外檢定 | 因子是在 4 個產業上挖出來的，民生與製造與營建是 agent 從沒看過的產業。拿新資料測它們，得到與「往前走」互相獨立的第二種樣本外——**往旁邊走**。判讀：抓到真機制的因子在新產業應該還有 IC；只是原產業特性代理的會歸零。⛔ 只讀資料不寫 `memory/`，因子庫的歷史指標不該被重算覆蓋 |

---

## 六、`prompts/` — LLM 的三份模板

| 檔案 | 中文名稱 | 做什麼 |
|---|---|---|
| `generate.md` | 提假設模板 | 「你是台股量化因子挖掘研究員……提出**假設先行**的候選因子」。注入欄位卡片、運算子卡片、`learnings.md`、因子庫摘要（不含 test 指標）、待辦佇列 |
| `distill.md` | 歸因蒸餾模板 | 「對每個候選做誠實的歸因，並蒸餾出可累積的經驗」。要求對照 prediction 與實測、標注 `failure_type` 與 `lesson_confidence` |
| `consolidate.md` | 整理回合模板 | 「重寫經驗庫，讓它更精煉、更可信、更有方向性。**不生成新因子**」。可提出運算子提案 |

---

## 七、`tests/` — 227 個測試

> ⚠️ 這 227 個測試是 **AI 依規格書撰寫的功能性測試**，全數通過。
> 「全數通過」代表規格有被編碼成可執行的斷言，**不代表有人逐條人工審閱過 227 個案例**。
> 下面前四套另外做過變異驗證（故意把程式寫錯，確認測試會轉紅）。

### 前瞻偏差檢測（四套，這是整個系統可信度的基石）

| 檔案 | 行數 | 中文名稱 | 檢查什麼 |
|---|---|---|---|
| `test_dsl_no_lookahead.py` | 97 | DSL 層前瞻檢測 | 對覆蓋全部 33 個運算子的公式集，把 `t_cut` 之後的所有欄位注入 1e6 重算，斷言 cut 以前的因子值**逐格完全一致**（NaN 位置也一致）。`test_all_operators_covered` 確保新增運算子時必須補一條用到它的公式 |
| `test_pipeline_no_lookahead.py` | 235 | 全管線前瞻檢測 | 往上測到整條評估管線：`monthly_base` → 載入對齊 → 四階段漏斗 → 診斷數字。也檢查標籤沒被當成欄位暴露、參考因子不會洩漏未來 |
| `test_ml_diagnose.py` | 491 | 診斷套件正確性與方向鎖定 | 每個統計量都用「已知答案的合成資料」驗證；另含 R10 的方向鎖定（因子方向只能用選取窗決定）與 `crosssec_oos` 去膨脹 ICIR 的數學性質 |
| `test_factor_lab.py` | 488 | 組合實驗室前瞻檢測 | 橫斷面標準化**只用當期**、walk-forward 訓練窗不碰測試期、embargo 有留空隙。⚠️ 蓋的是目前**不用於結論**的合成模型路徑 |

### 其餘測試

| 檔案 | 行數 | 中文名稱 | 檢查什麼 |
|---|---|---|---|
| `test_dsl_correctness.py` | 265 | DSL 計算正確性 | 每個運算子的結果 vs pandas 手算逐格一致 |
| `test_dsl_validation.py` | 107 | DSL 語法與複雜度防線 | 深度上限、窗口白名單、魔術常數、巢狀 `if_else`、欄位數上限、注入攻擊、型別規則、等價哈希 |
| `test_funnel.py` | 211 | 四階段漏斗迴歸 | 行為快照：語法錯 → `rejected_syntax`、低波動因子死在 Stage 4 多頭腿、門檻用未捨入值、空頭專用因子只看空頭換手 |
| `test_industry_track.py` | 101 | Stage 4b 產業通道 | 造一個「只在金融產業完美預測、其他產業純噪音」的合成因子，斷言只有金融通過 |
| `test_factor_scope.py` | 182 | 產業範圍擴張可靠性 | 分類驗證、舊排除產業不補回、test 隔離、交易用途、提交與中斷回復 |
| `test_scope_ui.py` | 72 | 本機 UI 的 HTTP 防護 | Host／CSRF／Origin 檢查、路徑白名單、重複工作互斥 |
| `test_memory.py` | 140 | 記憶層 | attempt 編號序、必填欄位、不可竄改、入庫要有中文名、等價公式會被擋、摘要不洩漏 |
| `test_m4.py` | 195 | 審計／報表／整理回合 | 聚合審計不洩漏個別因子、報表對人類保留密封指標、整理回合會拒絕壞的重寫並備份、運算子提案驗證 |
| `test_seed_reference.py` | 114 | 參考因子去重 | 重點在第四道篩選：抓「同一條公式掛兩個名字」，保留 ICIR 較強的那個 |
| `test_priority_repairs.py` | 168 | 資料修復的正確性 | 日曆對齊與跨股隔離、實際流通股數忽略減資與無效值、去重與欄位重建 |
| `test_compare_upstream.py` | 157 | 與前置專案的相容對照 | 未來標籤不能改變當期預測；另含可獨立執行的選股稽核（`--audit-run`）。缺本機歷史產出時自動 skip |
| `conftest.py` | 36 | 測試共用夾具 | 合成面板資料與產業對照 |

> `test_hmm_regime.py`、`test_fetch_macro.py`（87 個測試）已隨 HMM／總經一起移到
> `research_archive/`，不在這 227 個之內。

---

## 八、`memory/` — agent 的全部狀態

> ⚠️ 除了 `factor_values.parquet`（37MB）之外**全部進版控**——這是 agent 的記憶，是本專案的核心資產。

| 檔案 | 中文名稱 | 內容 |
|---|---|---|
| `attempts/` | 逐筆嘗試紀錄 | **1043 個 JSON，只進不出、永不刪改**。每筆含：`round`、`category`、`hypothesis`（假設與機制）、`prediction`（事前預測）、`formula`、`name_zh`、`direction`、`complexity`、`result`（漏斗診斷）、`diagnosis`（LLM 對照預測的歸因）、`failure_type`、`lesson`、`lesson_confidence`、`verdict`、`id`、`ts` |
| `learnings.md` | 經驗庫 | 90 行、9.6KB。**唯一注入 Generate prompt 的記憶**。可直接手動編輯——蒸餾品質差時人可以立刻改，不必等下一輪 |
| `learnings_history/` | 經驗庫備份 | 4 份，每次整理回合改寫前自動備份原文 |
| `library.json` | 因子庫 | 51 個因子。自有 `F-xxx` 欄位：中文名／描述／面向／類別／公式／公式哈希／欄位／深度／建立時間／輪次／來源 attempt／sub_train／validation／覆蓋率／換手／**`test_metrics_sealed`**。參考 `R-xxx` 多帶 `reference`、`dfs_name`、`icir_train`、`icir_test`、`decay_pct`、產業範圍 |
| `factor_values.parquet` | 因子值 | 632 萬列（`factor_id`／`ym`／`stock_id`／`value`）。Stage 2 去相關與 `factor_lab` 的資料來源。**唯一不進版控的 memory 檔** |
| `operator_proposals.json` | 運算子提案 | 三個清單：`pending`（待審）0、`implemented`（已上線）5、`rejected_log`（否決）2 |
| `dsl_limitations.jsonl` | DSL 表達限制 | 52 筆（`attempt`／`note`）。agent 說「這個想法表達不出來」的紀錄，是運算子提案的證據來源 |
| `queue_consumed.jsonl` | 佇列已消化 | 20 筆。整理回合產出的待辦被 Generate 認領的紀錄 |
| `unblocked.json` | 已解鎖假設 | `done` 清單＋`last_run`，避免同一個假設重複放回佇列 |
| `budget.json` | 預算狀態 | 週用量與全期輪次（68 輪）。跨 session 自動接續、每週一重置 |
| `report.html` | 報表輸出 | `report.py --html` 的產物，含密封指標，不進版控 |
| `consolidate_raw/` | 整理回合原始輸出 | 3 個檔。解析失敗時可手動救回，不進版控 |
| `llm_raw/` | LLM 逾時搶救片段 | 逾時是最貴的失敗方式（token 花了卻拿不到結果），所以原始輸出先落地，不進版控 |

---

## 九、`data/` — 資料層（全部不進版控，可重建）

| 檔案 | 中文名稱 | 內容 | 怎麼重建 |
|---|---|---|---|
| `monthly_base.parquet` | 月頻基礎面板 | **239,919 列、36 欄、1,616 檔、6 產業**、2012-01 ~ 2026-06。`stock_id`／`ym`／`group`／`fwd_ret_1m` ＋ 27 個 `fields.yaml` 欄位 | `python src/build_base.py` |
| `tmp_prices.parquet` | 價量中間檔 | `build_base --stage prices` 的月底快照 | 同上 |
| `tmp_chips.parquet` | 籌碼中間檔 | `build_base --stage chips` 的月底快照 | 同上 |
| `dfs_snapshot.parquet` | 參考因子自足快照 | 395 萬列（`dfs_name`／`ym`／`stock_id`／`value`）。**存了這份之後就不必再跨專案讀取** | `python src/seed_reference.py --apply` |
| `dfs_candidates.csv` | 參考因子篩選指標表 | 前置專案 DFS 因子的 ICIR／衰減等指標，四道篩選的輸入 | 同上 |
| `market_monthly.parquet` | 月頻大盤報酬 | 174 列（`ym`／`taiex_ret`） | `python src/build_regime.py` |
| `regime_table.json` | 市場狀態標註表 | 逐年（2012~2026）的 `taiex_ret`／`taiex_vol`／`regime`／`note` | 同上 |
| `macro_monthly.parquet` | 總經月頻面板 | 3,992 列（`ym`／`field`／`value`／**`pub_ym`**）。長格式，8 個欄位 | `python src/fetch_macro.py --fetch` |
| `stock_names.json` | 股票代號對照 | 975 檔的代號 → 名稱，流動性歸因報表用 | 從前置專案抓 |

> ✅ **2026-09-17 已重建**：前置專案補齊逐檔抓取的資料表後重跑 `build_base.py`，
> 從 958 檔 / 4 產業擴到 **1,616 檔 / 6 產業**（新增民生與製造、營建）。
> 沒有任何欄位缺值率惡化超過 3pp。舊檔備份在 `data/_before_rebuild/`。
> ⚠️ 但 `memory/factor_values.parquet`（958 檔）與 `data/dfs_snapshot.parquet`
> 尚未重算，`library.json` 的指標仍是舊資料的結果。詳見 `TODO.md`。

---

## 十、執行產物與雜項（全部不進版控）

| 檔案 | 中文名稱 | 說明 |
|---|---|---|
| `mining.log` | 挖礦逐輪紀錄 | 每輪的時間、token、成本、逾時紀錄。`config.yaml` 的逾時設定就是讀這份實測數字調的 |
| `build_base.log` | 面板建構紀錄 | 上次 `build_base.py` 的輸出 |
| `logs/riskadj.txt` | 風險調整對照輸出 | `hmm_regime.py --riskadj` 的原始輸出存證 |
| `logs/ablate_sharpe.txt` | 消融實驗輸出 | `hmm_regime.py --ablate` 的原始輸出存證 |
| `logs/threshold_sweep.txt` | 門檻掃描輸出 | `threshold_sweep.py` 的原始輸出存證 |
| `logs/mining_stats.txt` | 挖礦統計輸出 | `mining_stats.py` 的原始輸出存證 |
| `riskadj.json` | 風險調整結果 | `--riskadj --save` 的結構化結果 |
| `ablate_sharpe.json` | 消融結果 | `--ablate --save` 的結構化結果 |
| `changes.diff` | 暫存 diff | 一次性的差異檔 |
| `備用/` | 個人備份 | `learnings.md` 的舊版、`library.json.預去重備份` |
| `python` | **空檔案（誤建）** | 0 bytes，應該是打指令時手滑產生的。已在 `.gitignore` 排除，**可以直接刪掉** |
| `.pytest_cache/` | pytest 快取 | 自動產生 |

---

## 資料擴張功能（2026-09-18）

| 檔案 | 中文名稱 | 說明 |
|---|---|---|
| `src/factor_scope.py` | 產業資格與版本提交核心 | CLI／UI 共用，引用既有產業 IC 判定；管理分類驗證、新產業資格、雜湊、備份與中斷回復。有合成資料可靠性測試 |
| `src/scope_ui.py` | 本機操作服務 | 固定 CLI 工作、單工作互斥、localhost／CSRF 保護，不接受任意命令 |
| `src/scope_ui.html` | 產業檢查介面 | 進度、範圍變化、原因明細及歷史；不包含另一套判定公式 |
| `tests/test_factor_scope.py` | 擴張可靠性測試 | 分類、排除規則、test 隔離、交易用途、提交及回復 |
| `tests/test_scope_ui.py` | UI HTTP 測試 | 請求保護、路徑限制及重複工作防護 |
| `SCOPE_資料擴張操作說明.md` | 操作說明 | 更新順序、CLI／UI、設定與限制 |
| `memory/scope_runs/`、`memory/scope_jobs/` | 執行產物 | 計畫、暫存值、提交前備份及工作日誌；不進版控 |

核心獨立成檔是為 CLI 和 UI 共用提交契約；原 `crosssec_oos.py` 保留相容轉入參數及原探索報告。新檔不重建 IC 計算邏輯。

## 附錄：常見的「我要改 X，該動哪個檔？」

| 想做的事 | 動這個檔 | 注意 |
|---|---|---|
| 調漏斗門檻 | `config.yaml` 的 `funnel` | 先讀規格書第 9 章 |
| 加一個可用欄位 | `fields.yaml` ＋ `build_base.py` | `desc` 要寫清楚經濟含義，它會進 prompt |
| 加一個 DSL 運算子 | `dsl.py` 的 `_SIGNATURES` ＋ 實作 ＋ `test_dsl_no_lookahead.py` 的 `FORMULAS` | 不補測試公式會直接 fail；上線後記得跑 `unblock.py --apply` |
| 改 LLM 的提問方式 | `prompts/generate.md` | 改完先跑 `mining_loop.py --dry-run` 看 prompt 長相 |
| 經驗庫品質變差 | 直接編輯 `memory/learnings.md`，或先修 `prompts/distill.md` | 不要讓垃圾經驗累積 |
| 改回測口徑 | `config.yaml` 的 `backtest` | `top_q`／`cost`／`weighting`／`ann` |
| 換合成模型 | `factor_lab.py` 的 `walk_forward` | 順序建議 equal → ridge → lgbm |
| 資料源更新 | 前置專案 `tw_alpha_strategy`，再重跑 `build_base.py` | 本專案不自己抓資料 |

## 2026-09-19 修復新增項目

| 檔案 | 職責 |
|---|---|
| `tests/test_priority_repairs.py` | 缺月、股數、故障／中斷恢復、參考test隔離及mock保護；合成資料驗證 |
| `memory/transactions/` | 一般入庫／參考換版交易備份，不進版控；非手動編輯區 |
| `data/rebuild_history/20260919_234830_priority_repairs/` | 本次正式資料遷移前備份、差異及驗證，不進版控 |

旧 `dfs_snapshot.parquet`／`dfs_candidates.csv` 現為歷史資料，新版 seed_reference 不再使用。正式資料已重建，歷史表格中的958檔及舊值檔筆數不代表現況。

| `logs/performance_20260920_013215/` | 2026-09-20績效重測：REPORT.md、原精度JSON、9組月報酬、特徵清單、成本／回撤複核、输入雜湊及設定快照；人類研究資料，不進挖礦prompt |

2026-09-20：`tests/test_priority_repairs.py`另涵蓋LLM字串括號、錯誤信封／非零退出用量、逾時回報、未知成本、預算損毀與呼叫前上限防護。
