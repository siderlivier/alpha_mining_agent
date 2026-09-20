# 待辦清單

> 更新日期：2026-09-20
> 版控判準見 `.gitignore` 開頭；檔案職責見 `FILES_檔案對照表.md`

---

## 程式碼審查的 27 個問題：結案盤點

來源是 `REVIEW_程式碼與數值正確性評估報告_2026-09-14.md`。**27 項中 26 項已結案**，
其中 4 項結案為「不需要解決」。

| 狀態 | 數量 | 編號 |
|---|---:|---|
| ✅ 已修正 | **22** | R01～R04、R06～R22、R27 |
| ⚪ 不需要解決（結案） | **4** | R23、R24、R25、R26 |
| 🔴 未結案 | **1** | R05 |

### 本次（2026-09-20）處理的三項

**R10｜過擬合診斷用全期 IC 決定因子方向 — 真的修了**

`ml_diagnose._active_returns()` 原本對整段資料算 mean IC 決定方向，**定完向才切 test**。
一個在 train 為負、在 test 強正的因子會被事後翻向，然後在 test 上量出漂亮的
Sharpe——那個 Sharpe 是自己造出來的。

```python
ORIENT_CUTOFF = fl.SPANS["validation"][1]      # 2019-12，鎖在選取窗末端
ic = _mean_ic(sub[sub["ym"] <= ORIENT_CUTOFF]) # 不再吃全期
```

照審查報告要求的方式驗收：造一個「train 期微弱為負、test 期強正」的因子，
確認選取窗 IC = −0.97、全期 IC 被 test 拉成 +0.37——**舊實作會在這裡不翻向**，
新實作照選取窗翻向。新增 3 個測試（`test_orientation_*`）。

**R11｜DSR 試驗池不足以代表整個搜尋 — 改成明確界定範圍**

這不是公式寫錯，是**統計結論的適用範圍**問題，所以修正方式是讓輸出自己講清楚。
`cmd_dsr` 現在會附上：

```
⚠️ 這個 DSR／PBO 涵蓋的候選宇宙（R11）
   納入：N 個已入庫因子的單因子策略 + M 個合成模型
   未納入：1043 筆 attempts 裡被漏斗刷掉的候選（入庫率僅約 2%）
           參考因子池的選取與去重、產業範圍核准、模型與超參數選擇
   → 可以說「在這個試驗池內，最佳策略不是多重測試的產物」；
     不能說「整個挖礦流程已經去偏」。
   ⚠️ 也不能草率把試驗數改成 attempts 總數——那些候選高度相關。
```

**R12｜換手定義與空頭成本 — 結案為「刻意保留」，但必須標示**

你的決定是保留與前置專案相同的名單換手定義（對稱差 ÷ 聯集），以便數字可直接對照。
這是設計選擇不是 bug，但代價要寫在程式碼裡而不是只存在對話中：

```python
# ⚠️ R12（刻意保留，非疏漏）：換手＝「多頭名單的對稱差 ÷ 聯集」，不是權重交易量。
#    固定 n 檔、替換比例 q 時此值為 2q/(1+q)——換一半得 66.67% 而不是 50%。
#    保留是為了與前置專案可對照；代價是：
#      (a) 權重漂移與 score 權重改變不反映在成本裡
#      (b) long_short 只扣了多頭換手，**空頭換倉完全沒扣費**
#    → `long_short` 一欄不得當成完整的多空淨績效引用。
```

### 結案為「不需要解決」的四項

| 編號 | 問題 | 為什麼不修 |
|---|---|---|
| R23 | 部分總經更新覆蓋完整快取 | 位置在 `fetch_macro.py`，**已封存**（見下節） |
| R24 | 殖利率日期排序與單位猜測 | 同上；且該端點本來就取不到資料 |
| R25 | BetaExcess 混用 CAGR 與算術年化 | 位置在 `hmm_regime.py`，**已封存** |
| R26 | 多頭月數與績效區間不一致 | 位置在 `hmm_regime.py` 與 `threshold_sweep.py`，**已封存** |

四項全部落在已移出主線的檔案裡。封存版本保留原狀並在 `research_archive/README.md`
註明——若未來重新啟用那條線，這四項要跟著一起處理。

### 唯一未結案：R05

**用未來報酬是否存在來篩選股票。**

防護已經到位——`backtest.portfolio_returns()` 不再默默跳過缺報酬的股票，改成直接丟
`MissingReturnError`。但這只是讓問題**停止靜默**，不等於解決：

```
backtest.MissingReturnError: 2020-02: missing/nonfinite returns for ['3219','5205'];
selection is fixed. Resolve valuation before reporting performance.
```

`ml_diagnose --liquidity` 的流動性過濾壓力測試現在三組（own／reference／all）
全部卡在這裡跑不完。要解決需要**持股／現金帳**處理停牌、恢復交易、下市／收購清算
與無法賣出，已知缺口是 29 個股票月、17 檔。

**在這之前，任何組合績效數字都不該進 README。** 這也是 README 這一版不寫績效章的原因。

---

## 本次完成的其他整理

### 研究封存：HMM 與總經移出主線

移到 `research_archive/macro_regime/`，不進版控：

```
src/{hmm_regime,fetch_macro,threshold_sweep}.py
tests/test_{hmm_regime,fetch_macro}.py      （87 個測試一起走）
data/macro_monthly.parquet
riskadj.json / ablate_sharpe.json / 三份原始輸出
```

`RESULTS.md` 整份移為 `research_archive/RESULTS_歷史存檔_至20260920.md`——
它的 17 節數字全部建在舊的 958 檔 / 4 產業面板上，資料修復後已失效。

⚠️ `data/market_monthly.parquet` 與 `data/regime_table.json` **不在封存內**，
它們仍被 `eval_candidates.py` 使用。

測試數 307 → **222**（封存帶走 87 個；R10 新增 3 個）。

### 版控瘦身：50.0 MB → 3.66 MB

發現 `data/_before_rebuild/` 的 48MB 已被追蹤——`.gitignore` 寫的是 `data/*.parquet`，
**單個 `*` 不跨目錄**，子資料夾整個漏掉。同一個坑還罩著 `data/rebuild_history/`（455MB）、
`memory/scope_runs/`（265MB）、`memory/transactions/`（83MB）。

`.gitignore` 改寫，開頭寫明判準：

> 進版控的條件只有兩個：(1) 重建專案的必要條件 —— 程式、設定、prompts；
> (2) 理解決策的必要條件 —— 文件、以及 agent 的記憶。
> **可以由 (1)+(2) 重跑出來的東西就是產物，產物不進版控。**

依此 `RESULTS.md` 與 `logs/` 不進版控（產物），
`REVIEW`／`SCOPE`／`TODO` 進版控（判斷與決策，重跑不出來）。

### 逐因子對照工具

`crosssec_oos.py` 新增兩個模式，未新增檔案：

```bash
python -B src/crosssec_oos.py --vs-reference   # 自有 vs 基準的分布對照
python -B src/crosssec_oos.py --pairwise       # 逐對 CSV：4,383 列 × 21 欄
```

每個因子**只在自己的 `approved_groups` 內**量 IC，所以產業限定因子不會被拿去
替它沒核准的產業打分。輸出 `logs/pairwise_own_vs_reference.csv`。

---

## 🔴 立即：推上 GitHub

本機累積 6 個 commit 未推送，`origin/main` 仍在 `0fdc622`。

```bash
cd C:\Users\User\Desktop\股匯操盤AI\alpha_mining_agent
git push -u origin main
```

- [ ] `git push -u origin main`
- [ ] 推完確認 GitHub 上**沒有** `logs/`、`research_archive/`、任何 parquet

---

## 待辦

### 🔴 R05 的落地處理

- [ ] 建持股／現金帳：停牌凍結、恢復交易、下市／收購清算、無法賣出
- [ ] 補足事件與可交易價格（已知缺口 29 個股票月、17 檔）
- [ ] 完成後重跑 `ml_diagnose --liquidity` 的 (B) 壓力測試與 `compare_upstream` 的 A/B
- [ ] **這三件做完才寫 README 的績效章**

### 🟠 README 改寫（進行中）

章節順序已定：系統在做什麼 → 系統結構（含流程圖）→ 漏斗與自我進化 →
逐因子 vs 基準 → 橫斷面樣本外 → 方法論 → 安裝使用 → 目錄。
**不寫績效章**（R05 未結案），**不提 HMM 與總經**（已封存）。

- [ ] 寫內文
- [ ] 同步更新 `FILES_檔案對照表.md`（封存、新模組、測試數 222）

### 🟡 前置專案 `tw_alpha_strategy`

- [ ] `fetch_extra.py` 的守衛改成「比對表內股票集合 vs 現行 universe，缺的就補抓」。
      目前是「檔案存在就跳過」，正是 10 個檔靜默沒重抓的原因。
- [ ] OHLC 內部矛盾 1,497 筆（幅度 ≥5%，342 檔）。不影響 mining agent
      （`build_base.py` 只讀 close/close_raw/ret/volume/amount）。低優先度。
- [ ] `benchmarks.parquet` 停在 06-30。*（你說這個專案先不理市場指數，備查。）*

### 🟢 挖礦

- [ ] 繼續 `python src/mining_loop.py --rounds 5`
- [ ] 資料池從 958 檔擴到 1,616 檔、產業 4 → 6 之後，Stage 2 的去相關基準等於換了
      一張牌桌。值得再跑幾輪看入庫率會不會從 0.9% 回升——若回升，那本身就是
      「飽和是相對於股票池而言」的證據。
