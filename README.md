# 台股 Alpha 因子挖掘 Agent

自我進化的因子挖掘系統：LLM 提假設 → 本地 Python 驗證 → 經驗蒸餾入庫。

- 完整設計理由見 `SPEC_架構設計規格書.md`
- **逐項功能解說、參數調校、疑難排解見 `GUIDE_使用教學.md`**

## 前置需求

- Python 3.10+：`pip install pandas numpy pyarrow pyyaml pytest`
- Claude Code CLI（`claude` 指令可用，用 Pro 訂閱登入）
- 前置專案資料：`../tw_alpha_strategy/data/processed/panel.parquet`

> 長時間無人值守跑的話，建議改用一年期 token 而非互動登入，避免 OAuth 過期：
> `claude setup-token` → `setx CLAUDE_CODE_OAUTH_TOKEN "<token>"`

## 首次設定（一次性）

```bash
python src/build_base.py      # panel → 月頻快照（約 1-2 分鐘）
python src/build_regime.py    # 市場狀態表 + 月頻大盤報酬
python src/seed_memory.py     # 初始化空記憶（不預載任何理論）
python -m pytest tests/ -q    # 應全數通過（建好資料後 164 passed，約 2 分鐘）
```

## 挖礦

```bash
python src/mining_loop.py --dry-run     # 先看 Generate prompt 長什麼樣
python src/mining_loop.py               # 跑 1 輪（真實 LLM，消耗訂閱額度）
python src/mining_loop.py --rounds 5    # 連跑 5 輪（受週預算限制自動停）
python src/mining_loop.py --mock        # 假 LLM 跑通管線（不耗額度，測試用）
python src/mining_loop.py --budget      # 只印本週用量，不呼叫 LLM
```

一輪約 8~15 分鐘、$0.9~1.5。輪次／token／成本三種週預算都在 `config.yaml`
的 `budget` 區塊，狀態存在 `memory/budget.json`，跨 session 自動接續、每週一重置。

## M3 人工監督（放行自動化前必做）

```bash
python src/query_attempts.py --limit 40          # 瀏覽全部嘗試
python src/query_attempts.py --verdict passed    # 看入庫因子的假設品質
type memory\learnings.md                         # 審核蒸餾出的經驗
```

審核重點：hypothesis 是否有機制（而非套套邏輯）、diagnosis 是否對照了
prediction、lesson 是否過度泛化。**如果 lesson 品質差，先修 prompts/ 再繼續，
不要讓垃圾經驗累積。** `learnings.md` 可直接手動編輯修正。

## 查看成果與定期維護（M4）

```bash
python src/report.py            # 因子庫總覽 + 統計（人類專用，含密封 test 指標）
python src/report.py --html     # 產生 memory/report.html 並自動開啟（推薦）
python src/report.py --save     # 另存 memory/report.md
python src/audit.py --human     # 過擬合審計（--human 含個體明細，勿餵 agent）
python src/consolidate.py       # 手動觸發整理回合（每 10 輪也會自動觸發）
python src/unblock.py           # 列出「因缺運算子而卡住、現已可重試」的假設
python src/unblock.py --apply   # 把它們放回待驗證假設佇列
```

### 參考因子（一次性設定，強烈建議）

```bash
python src/seed_reference.py --list     # 看前置專案有哪些 DFS 因子會被灌入
python src/seed_reference.py --dry-run  # 含去重報告，但不寫入
python src/seed_reference.py --apply    # 灌進 Stage 2 的去相關基準
python src/seed_reference.py --clear    # 還原
```

Stage 2 預設只跟 **agent 自己入庫的**因子去相關，不知道前置專案
`mine_dfs.py` 已經挖出 86 個因子——結果就是重新發明輪子（F-005 其實就是
`neg_vol_126`、F-006 幾乎等於 `neg_accruals`）。匯入後這類候選會在 Stage 2
就被擋下，省掉整輪診斷成本。

- 參考因子以 `R-xxx` 與自有的 `F-xxx` **同住 `library.json`**（單一因子庫），
  靠 `reference: true` 區分；`report` / `audit` / 統計走 `own_library()`，
  不會把它們算成 agent 的成果。
- 篩選四道：沿用前置專案的 survivors 判準、`|ICIR_train|` 門檻、
  **衰減 ≤ 50%**（與本專案 Stage 4 同標準），最後**參考因子彼此去重**
  （`--max-ref-corr`，預設 0.95）。86 → 31 → 25 個。
- 第四道是必要的：前四道之前都是**逐因子**判斷，看不到「同一條公式掛兩個名字」。
  上游 `mine_dfs.py` 就有 `net_margin`／`ni_to_rev`（ρ=1.000）、
  `op_margin`／`op_to_rev`（ρ=1.000）這種情形——兩個都收，等權合成時
  「淨利率」這個概念就被賦予兩倍權重。
- Generate prompt 會列出它們（中文名 + 一句話描述，**不給公式**——那些是原始
  財報欄位算的，本 DSL 沒有對應欄位，給了只會誘使模型拼出無效公式）。
- 第一次匯入需要前置專案；之後會存 `data/dfs_snapshot.parquet`
  **自足快照**，`--clear` 與重新匯入都不再跨專案。

整理回合會：升級/推翻/壓縮經驗（原文自動備份到 `memory/learnings_history/`）、
消化聚合審計、並在證據充分時提出**新運算子提案**。

## 因子組合實驗室（回測 / 合成 / 診斷）

因子庫裡的每個因子單獨看都過了漏斗，但**放在一起未必更好**。這一層回答三個問題：
agent 自己挖的（`F-xxx`）跟前置專案 DFS 的（`R-xxx`）哪組強、哪些因子其實在拖累、
最佳組合到底要幾個因子。

```bash
python src/factor_lab.py --compare                  # own / reference / all × equal / ridge
python src/factor_lab.py --compare --model equal,ridge,lgbm   # 加 LightGBM（需 pip install lightgbm）
python src/factor_lab.py --loo --set own            # 留一法：Δ貢獻為負的就是拖累者
python src/factor_lab.py --greedy --set own --select-span validation --span test  # 貪婪前向選擇
python src/factor_lab.py --factors F-001,F-005,R-003 --model ridge   # 指定子集
python src/factor_lab.py --loo --set own --rank Sharpe        # 改用 Sharpe 排序
python src/factor_lab.py --cost-scan --set all --model equal,ridge,lgbm  # 成本敏感度
python src/factor_lab.py --compare --save memory/lab_compare.json
```

實測結果（51 個因子、test 期 2020-01 起、單次換手成本 0.4%）：

| 因子組 | 模型 | 因子數 | CAGR | 超額 | IR | 換手 |
|---|---|---|---|---|---|---|
| own | ridge | 26 | 38.19% | 10.72% | 1.58 | 58.2% |
| own | lgbm | 26 | 39.32% | 11.70% | 1.83 | 67.5% |
| reference | lgbm | 25 | 41.65% | 13.74% | 2.07 | 64.3% |
| **all** | **lgbm** | **51** | **45.54%** | **16.36%** | **2.57** | 67.2% |

同期等權基準 CAGR 24.26%。LightGBM 在三組都勝出，合併全部因子最好——
但**換手也最高**，務必配 `--cost-scan` 看它在高成本下是否守得住。

> ⚠️ **別只看 CAGR。** 台股 test 期（2020-01 起）本身就是大多頭，等權全樣本
> 基準的 CAGR 就有 24%。多頭組合報 32% 聽起來很漂亮，但其中 24% 是 beta，
> 因子真正的貢獻是**超額（Excess）與資訊比率（IR）**——所以 `--loo` / `--greedy`
> 預設用 IR 排序，`--compare` 的表也會把超額、IR 與基準 CAGR 一併印出來。

- **資料全部在本專案內**：因子值讀 `memory/factor_values.parquet`（`F-` 與 `R-`
  同住），標籤讀 `data/monthly_base.parquet`。不跨專案匯入。
- **`--span` 預設 `test`**，而且只有 `test` 是真正的樣本外——agent 的因子是用
  `sub_train`+`validation` 選出來的，前置專案的 DFS 因子也是用 ≤2019-12 篩的，
  兩邊 test 期起點相同，比較才公平。選其他期會跳警告。
- **合成順序建議 equal → ridge → lgbm**。等權沒有參數可以過擬合，是誠實的基準；
  Ridge 若贏不過等權，就別急著上 LightGBM。（本專案的實測是 Ridge 大幅勝過
  等權、LightGBM 又再勝過 Ridge，所以三層都值得留著。）
- **LightGBM 需要 `pip install lightgbm`**，不裝也不影響其他功能。
- 回測口徑與 agent 的評估一致：逐月**在各產業內**取分數前 `top_q`，產業內不足
  `min_stocks_per_group` 就跳過該產業該月，換手按比例扣成本。

### 怎麼讀 `--loo` 的結果

`Δ貢獻 = 全集 IR − 去掉該因子後的 IR`。**負值代表拿掉它反而更好**，
那個因子在組合裡是拖累（通常是與別人高度相關、卻多帶了噪音）。它單獨的 IC 可能
還不錯——「單獨有用」和「加進組合有用」是兩件事，這正是要跑留一法的原因。

⚠️ **被標成拖累的比例本身就是訊號**。因子少而互補時，拖累者應該只有少數幾個；
若超過一半都被標成拖累，代表這組因子過度擁擠（互相高度相關），
此時該讀的是**排序**而不是正負號，並且該考慮先去重或改用 `--greedy` 重挑。

### `--greedy` 的選擇偏誤陷阱

walk-forward 保證的是**模型參數**沒看到未來，但「要選哪幾個因子」**本身也是一次
擬合**。用 test 期的 IR 挑因子、再用 test 期的 IR 當成績，就是拿答案卷挑答案。

實測的落差（26 個自有因子、Ridge）：

| 做法 | 挑選期 | test 期 IR | test 期超額 |
|---|---|---|---|
| 全部 26 個，不挑 | — | **1.583** | **10.72%** |
| 貪婪挑 k=3（在 validation 挑） | validation | 1.425 | 8.47% |
| 貪婪挑 k=10（在 test 挑） | ~~test~~ | ~~2.264~~ | ~~12.04%~~ ← 假的 |

結論是**別挑**：全部用比任何挑出來的子集都好。挑出來的組合在 validation 期
IR 高達 4.4，到 test 期只剩 1.43——那 4.4 是 24 個月樣本上的噪音。

所以 `--greedy` 一定要配 `--select-span validation --span test`；不加時程式會
在結尾印警告。`--loo` 同理：它是**診斷工具**（看誰跟誰重複），不是選股清單產生器。

### 前瞻紀律（三道，`tests/test_factor_lab.py` 逐項守著）

1. **切分**：headline 只看 `test` 期。
2. **walk-forward + embargo**：第 i 段用 `ym <= months[i-1-EMBARGO]` 訓練，預測
   `months[i : i+RETRAIN_EVERY]`。月頻標籤看下一個月，不留 embargo，訓練期最後
   一個月的標籤會和測試期第一個月重疊。
3. **標準化只用當期橫斷面**：產業內 z-score 是逐月 `groupby(["ym","group"])`。
   ⛔ **絕不可改成全期 mean/std**——那等於把未來的分布洩漏進歷史。

參數在 `config.yaml` 的 `backtest:`（`top_q` / `cost` / `weighting` / `ann`）與
`ml:`（`min_train_months` / `retrain_every` / `embargo` / `ridge_alpha` /
`min_feature_coverage`）兩個區塊。

## 過擬合／穩健性診斷（`src/ml_diagnose.py`）

`--compare` 說 LightGBM 的 IR 有 2.57。這支模組的存在理由只有一個：
**一個漂亮的 IR 有很多種假法**，每一種要用不同的方法拆穿。

| 假法 | 拆穿的方法 | 指令 |
|---|---|---|
| 只是搭上多頭順風車 | 分多空 régime 看 | `--regime` |
| 只是少數幾個月撐起來的 | 逐月攤開 + 拿掉最好 N 個月 | `--breadth` |
| 只是小型股／低流動股溢酬 | 流動性歸因與過濾 | `--liquidity` |
| 只是規模因子換個名字 | 規模中性版重跑 | `--importance` |
| 只是試了很多策略挑到的幸運兒 | Deflated Sharpe / PBO | `--dsr` |
| 報酬被換手成本吃光 | 成本敏感度 | `factor_lab --cost-scan` |
| （加狀態模型值不值得？） | 逐因子多空月 ICIR 差異 | `--regime-factors` |
| （加擇時值不值得？） | 上限與損益兩平準確率 | `--timing` |

```bash
python src/ml_diagnose.py --all                  # 五項全跑（約 18 分鐘）
python src/ml_diagnose.py --regime               # 多空 régime + 逐年超額
python src/ml_diagnose.py --breadth              # 逐月分布、集中度、產業組成
python src/ml_diagnose.py --liquidity            # 流動性歸因 + 過濾×成本
python src/ml_diagnose.py --importance           # 特徵重要性 + SHAP + 規模中性
python src/ml_diagnose.py --dsr --save memory/diag_dsr.json
python src/ml_diagnose.py --regime-factors        # 逐因子的多空月 ICIR 差異
python src/ml_diagnose.py --timing               # 擇時研究（不含在 --all，很慢）
```

### ⚠️ 目前的回測是「只做多」

`backtest.portfolio_returns()` 算出三條腿，但 `factor_lab` 與 `ml_diagnose`
報的**每一個數字都取 `long`**：

| 腿 | 內容 | 目前有沒有用 |
|---|---|---|
| `long` | 各產業內分數**前** `top_q` 等權做多，扣換手成本 | ✅ 所有 headline |
| `long_short` | 前 `top_q` 做多 − 後 `top_q` 做空 | ❌ 算了但沒人讀 |
| `benchmark` | 全池等權 | ✅ 只用來算超額 |

實測 `long_short`：CAGR 36.69%、Sharpe **2.63**、MaxDD **−9.82%**
（對照 `long` 的 45.54% / 1.92 / −16.66%）。**報酬較低但風險調整後明顯較好**，
因為對沖掉了 beta。空頭腿單獨看是年化 −7.87%——多頭市場裡放空是逆風，
它的價值在對沖，不在賺錢。台股放空另有借券成本與平盤下不得放空的限制，
這條腿目前沒有納入成本模型，要當真用還需要補。

**另一件相關的事：本策略完全沒有產業輪動，也沒有市場擇時。**
回測是逐月「在各產業內」取前 `top_q`，所以產業權重結構性地鎖在全池比例上
（實測持股 60.6% 電子 vs 全池 59.9%，四個產業偏離都在 ±0.7% 內），
而且永遠 100% 投資。全部的超額都來自產業內選股。

多頭腿對基準做迴歸：**beta 0.992、alpha 年化 +16.55%**。
45.54% 的 CAGR 裡約 23.7% 是 beta、16.6% 是 alpha——**一半以上是市場曝險**。

### 想加狀態模型（HMM）之前，先看這兩張表

**（一）用狀態決定「進出場」的門檻很高**（`--timing`）：test 期有 65% 的月份
上漲，「永遠猜漲」就有 65% 準確率且績效等同滿倉。要贏過滿倉的 CAGR，
需要 **80%** 的方向準確率——猜錯一個上漲月就整月踏空，代價極高。

**（二）用狀態決定「因子權重」的門檻低得多**（`--regime-factors`）：

低波動族群（`R-004/005/006`、`F-005`、`F-008`）呈現極強的不對稱，
而且**兩個期間都成立**（t = ICIR × √月數）：

| 因子 | 分類期 ICIR_漲 (t) | 分類期 ICIR_跌 (t) | test ICIR_漲 (t) | test ICIR_跌 (t) |
|---|---|---|---|---|
| R-006 低波動（月） | +0.34 (2.6) | +1.61 (**9.5**) | −0.12 (−0.9) | +1.89 (**9.8**) |
| R-004 低波動（半年） | +0.29 (2.2) | +2.14 (**12.1**) | −0.15 (−1.0) | +1.71 (**8.9**) |
| F-008 低調接近新高 | +0.02 (0.2) | +1.57 (**8.3**) | −0.31 (−2.2) | +1.60 (**8.3**) |

**下跌月的強度壓倒性且穩定（t = 8~12）；上漲月兩期都貼近零。**
所以 test 期的「12/51 變號」不是結構改變，只是上漲月的雜訊翻面——
真正可依賴的是**不對稱本身**，不是正負號。

**切換因子也確實會降 beta**（低波動因子挑的就是低 beta 的股票）：

| 組合 | beta | alpha 年化 | CAGR | IR |
|---|---|---|---|---|
| 一直用全部因子 | 0.932 | +13.27% | 39.32% | 1.74 |
| 一直用低波動防禦組 | **0.722** | +8.34% | 27.37% | 0.23 |
| 上漲月全部／下跌月防禦（完美預知） | 0.880 | **+16.95%** | 42.94% | **1.98** |

**關鍵差別在失敗的代價**：進出場猜錯 = 整月不在場；
因子權重猜錯 = 用了比較不合適的那組因子（IC 仍為正），只是少賺一點。
**低一個量級。** 所以狀態訊號建議先接在因子權重上，而不是進出場。

移植自前置專案 `src/ml/` 的四支腳本，改成完全在本專案內執行
（因子值讀 `memory/factor_values.parquet`、流動性讀 `data/monthly_base.parquet`）。
需要 `scipy`（`--dsr`）、`lightgbm`、`shap`（選配）。

### 51 因子 + LightGBM 的實測結論

| 檢定 | 結果 | 判讀 |
|---|---|---|
| Deflated Sharpe | **0.997** | 扣掉 54 次試驗的多重測試，超額仍顯著 |
| PBO | 0.001 | 見下方的但書 |
| 大盤下跌月超額 | +13.08%（IR 2.78, t 4.17） | 逆風時反而更好，不是多頭順風車 |
| 剔除底 70% 低流動股 | 超額 15.09%（原 16.28%） | edge 在買得到的股票上也成立 |
| 規模中性（排除 3 個規模代理） | IR 2.57 → 2.14（−16.9%） | 與「隨機拿掉一個重要因子」同量級 |
| 成本拉到 1.6% | 超額仍有 6.68% | 不是靠高換手買來的 |

**但書一：PBO 0.001 要打折扣看。** 試驗母體裡 51 個是單因子策略、只有 3 個
是合成模型，而合成模型在任何切塊都穩定勝過單因子——樣本內最佳幾乎總是模型，
樣本外也是，PBO 於是趨近 0。這個數字說明的是「合成勝過單因子」很穩健，
**不是**「這組超參數沒過擬合」。程式會在 PBO 過低時自動印出這段警告。

**但書二：可交易性有代價。** 只看最流動 30% 的股票時，年化超額仍有 15.10%，
但 IR 從 2.55 掉到 1.16、月勝率從 84% 掉到 58%，2022 與 2023 年小幅為負。
超額的**水準**守得住，**穩定度**守不住——因為只剩 24 檔持股，集中度上升。

### 下跌月的超額是廣泛的還是少數事件？（`--breadth`）

「大盤下跌月超額 +13%」有兩種完全不同的成因，`--breadth` 把它拆開：

| 版本 | 下跌月數 | 超額為正 | 中位超額 | IR | 去掉最好 3 個月的 IR | 最大 3 月佔總超額 |
|---|---|---|---|---|---|---|
| 全體股票池 | 27 | **85%** | +1.08% | 2.78 | **2.42** | 33% |
| 可交易版（最流動 30%） | 30 | 60% | +0.51% | 1.34 | **0.78** | **58%** |

**全體是廣泛的**：27 個下跌月有 23 個超額為正，中位數 +1.08% 與平均 +1.09%
幾乎相同（分布不偏斜），拿掉最好的 3 個月 IR 只從 2.78 掉到 2.42。

**可交易版是少數事件撐起來的**：只有 60% 的下跌月為正，最好的 3 個月佔了
58% 的總超額，拿掉之後 IR 從 1.34 崩到 0.78。這是整套診斷裡最該記住的一格。

## 市場狀態模型（`src/hmm_regime.py`）

Gaussian HMM 判斷下個月是多頭態還是空頭態。用途**不是決定進出場**
（`ml_diagnose --timing` 顯示那要 80% 準確率才損益兩平），而是**切換因子組**。

```bash
python src/hmm_regime.py --fit                # 訓練期擬合，看狀態性質
python src/hmm_regime.py --predict            # walk-forward 樣本外方向準確率
python src/hmm_regime.py --apply              # 用狀態切因子組，比較靜態
python src/hmm_regime.py --predict --features ret,vol,breadth --states 3
```

參數在 `config.yaml` 的 `hmm:` 區塊。需要 `pip install hmmlearn`。

### 只餵價格資料的實測結果（未加政府數據）

**方向預測：有訊號，但不足以擇時。**

| 門檻 | 預測漲的月數 | 準確率 | vs 基準 | 預測漲時平均報酬 | 預測跌時平均報酬 |
|---|---|---|---|---|---|
| 0.5 | 23 / 77 | 49.4% | −15.6% | **+4.07%** | **+1.11%** |

兩組報酬差異檢定 **t = 2.16、p = 0.036**——預測看多的月份，報酬確實顯著較高。
但方向準確率 49.4% 仍**輸給**「永遠猜漲」的 64.9%，因為模型太保守，
77 個月只喊 23 個月看多，漏掉大量上漲月。

**切換因子組（test 期）：降風險，但不加報酬。**

| 做法 | beta | alpha | CAGR | IR | MaxDD |
|---|---|---|---|---|---|
| 靜態：全部因子 | 0.932 | +13.27% | 39.32% | **1.74** | −15.59% |
| 靜態：只用防禦組 | 0.711 | +9.75% | 28.82% | 0.35 | −13.16% |
| **HMM 切換（樣本外）** | **0.791** | +13.52% | 35.80% | 1.08 | **−13.16%** |
| 完美預知（上限） | 0.874 | +16.97% | 42.82% | 1.98 | −13.16% |

**誠實的結論：目前這版沒有達標。** IR 從 1.74 掉到 1.08，落在靜態之下而非
靜態與上限之間。它確實把 beta 降了 15%（0.932 → 0.791）、回檔縮小 2.4 個
百分點，但代價是報酬穩定度。原因同上——模型太保守，70% 的時間待在防禦態，
而防禦組單獨的 IR 只有 0.35。

下一步是餵入景氣／貨幣數據看能否改善；`--apply` 那張表就是驗收標準：
**HMM 切換那一列要落在「靜態全部因子」與「完美預知」之間才算有貢獻。**

### 一個非直覺的發現：狀態標籤不能望文生義

訓練期擬合出來的兩個狀態：

| 狀態 | 觀測報酬均值 | 觀測波動 | → 下月市場報酬 | → 下月上漲比例 |
|---|---|---|---|---|
| 低報酬態 | +0.16% | 6.54% | **+2.56%** | **72.0%** |
| 高報酬態 | +1.24% | 3.35% | +0.40% | 61.8% |

**高波動狀態之後反而漲得多**（波動的風險溢酬／反彈效應）。
照直覺把「過去報酬高、波動低」當多頭態，方向預測會系統性地反向——
第一版就是這樣寫的，test 期準確率只有 35.1%（基準 64.9%）、t = −2.33。
那不是模型無效，是**標籤貼反了**。

修正方式是用資料決定：`bull_state_from_train()` 在**訓練窗內**比較各狀態的
後續市場報酬，取最高者。⛔ 只能用擬合當下已實現的月份配對
（觀測 `X[j]` 對應 `mkt[idx[j]]`，後者要下個月底才知道，所以配對只到 `j = n-2`）。

### 三道前瞻紀律（`tests/test_hmm_regime.py` 逐項守著）

1. **時點對齊**：`obs_ret[t] = mkt[t-1]`。`fwd_ret_1m` 是未來報酬，
   少 shift 一格就是直接看答案。
2. **只用前向濾波，不用平滑**：`hmmlearn.predict()`（Viterbi）與
   `predict_proba()`（平滑）都會用整段序列回推每個時點的狀態。
   本模組只用 hmmlearn 擬合參數，狀態推論自己寫遞迴。
3. **參數與標籤只在訓練窗擬合**：擴張窗重擬，測試期永遠只驗證；
   因子的攻擊／防禦分類只用 ≤ validation 末端的資料。

驗證過測試有效：把 `forward_filter` 換成 `predict_proba`、或把 `shift(1)`
拿掉，對應的測試立刻轉紅。

## 運算子提案審核

提案寫進 `memory/operator_proposals.json`，分三個清單：

| 清單 | 意義 |
|---|---|
| `pending` | 通過機械驗證、等你審核 |
| `implemented` | 你已批准並實作進 `src/dsl.py` |
| `rejected_log` | 機械驗證未過，或你人工否決 |

機械驗證只查名稱格式／純函數簽名／證據引用存在且 ≥2，**不保證語意可用**
（例如 `qtr_only` 通過了驗證，但在月頻面板上會產生 100% NaN）。
**人工批准前絕不自動實作**——批准後照 `GUIDE_使用教學.md` 的五步驟寫進
`dsl.py`、補齊測試、跑 `unblock.py`，下一輪 Generate 就會自動看到新運算子。

已實作的提案：`rank_nz(x)`、`industry_demean(x)`、`streak(x, n)`、
`streak_true(cond, n)`、`clip_std(x, n)`。

**實作完新運算子後記得跑 `python src/unblock.py --apply`**——當初因為表達不出來
而失敗的假設會被放回佇列重試，否則那些已經付過診斷成本的候選就白白浪費了。

## 目錄速覽

```
config.yaml / fields.yaml       全部門檻與欄位白名單（單一事實來源）
prompts/                        Generate / Distill / Consolidate 模板
src/dsl.py                      受限 DSL（前瞻偏差語言級封鎖），33 個運算子
src/unblock.py                  解鎖：把因缺運算子而卡住的假設放回佇列
src/seed_reference.py           把前置專案的 DFS 因子灌進 Stage 2 去相關基準
src/eval_candidates.py          四階段漏斗 + 八項診斷
src/backtest.py                 回測核心：產業內 top-q 組合、換手、成本、績效
src/factor_lab.py               因子組合實驗室：合成比較 / 留一法 / 貪婪前向選擇
src/ml_diagnose.py              過擬合診斷：régime / 流動性 / 規模中性 / DSR+PBO
src/hmm_regime.py               市場狀態模型（Gaussian HMM）：預測狀態 + 切換因子組
src/memory.py                   兩層記憶（attempts + learnings）
src/mining_loop.py              orchestrator（claude -p 編排、預算計量）
src/consolidate.py              整理回合（經驗壓縮 + 運算子提案）
src/audit.py / src/report.py    聚合審計 / 人類查閱報表
memory/                         agent 的全部狀態（可隨時人工檢視/修訂）
  ├ attempts/                   逐筆嘗試（只進不出）
  ├ learnings.md                唯一注入 prompt 的記憶
  ├ learnings_history/          每次整理前的自動備份
  ├ library.json                因子庫：自有 F-xxx + 參考 R-xxx（含密封 test 指標）
  ├ factor_values.parquet       入庫因子的逐月因子值（Stage 2 去相關 + factor_lab 的來源）
  ├ budget.json                 週用量與全期輪次
  ├ operator_proposals.json     運算子提案三清單
  ├ consolidate_raw/            整理回合的原始 LLM 輸出（解析失敗時可手動救回）
  ├ llm_raw/                    逾時搶救的片段
  ├ queue_consumed.jsonl        已消化的佇列項目（只進不出）
  └ unblocked.json              已解鎖過的假設，避免重複放回
data/                           資料層（全部自足，不需要前置專案）
  ├ monthly_base.parquet        月頻面板：欄位、fwd_ret_1m、group、amt_21
  ├ dfs_snapshot.parquet        參考因子的自足快照
  ├ dfs_candidates.csv          參考因子的篩選指標表
  └ stock_names.json            股票代號 → 名稱（流動性歸因報表用）
mining.log                      逐輪執行紀錄
tests/                          164 個測試（DSL、全管線、因子合成三套前瞻檢測 + 過擬合統計量驗證）
```

⚠️ `memory/attempts/` 只進不出；`library.json` 的 `test_metrics_sealed`
絕不可放進任何 prompt——修改相關程式前先重讀規格書第 9 章。
新增 DSL 運算子時**必須**同步在 `tests/test_dsl_no_lookahead.py` 的
`FORMULAS` 加一條用到它的公式，否則 `test_all_operators_covered` 會失敗。
