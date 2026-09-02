# 代辦清單

> 更新日期：2026-09-02
> 涵蓋兩個專案：`alpha_mining_agent`（本專案）與 `tw_alpha_strategy`（前置專案）

---

## 🔴 立即：把 repo 推上 GitHub

GitHub repo 已建好（<https://github.com/siderlivier/alpha_mining_agent>，public、空的），
本機 git 也已完成 2 個 commit、remote 指向正確位址。**只差最後一步推送**——
這一步必須在你自己的終端機做，因為這個 session 的 shell 沒有 GitHub 憑證。

```bash
cd C:\Users\User\Desktop\股匯操盤AI\alpha_mining_agent
git push -u origin main
```

就這一行。repo 是空的，不會有衝突。

**已排除在版控外的東西**（見 `.gitignore`）：`data/*.parquet`、
`memory/factor_values.parquet`（37MB）、`*.log`、`備用/`。
這些都能重建，方式寫在 `.gitignore` 的註解裡。

- [ ] `git push -u origin main`
- [ ] 確認 GitHub 上看得到 `memory/attempts/` 的 1043 筆紀錄（那是挖礦軌跡）

---

## 🟠 前置專案 `tw_alpha_strategy` 的待處理事項

完整分析見該專案的 `TODO_因子重複問題.md`。

### 已完成

- [x] **修掉 3 對同公式雙名因子**（`mine_dfs.py`）
      `gross_margin`/`gp_to_rev`、`op_margin`/`op_to_rev`、`net_margin`/`ni_to_rev`
      ρ = 1.000，連 ICIR 都逐位相同。已改成只保留 `d_*` 衍生因子。
- [x] **加同值防呆** `_assert_no_duplicate_factors()`：用整欄雜湊比對，
      `generate()` 回傳前若有任兩欄完全相同就直接丟錯。

### 待做

- [ ] **重跑 `mine_dfs.py`** → 更新 `data/processed/dfs_candidates.csv`
      （因子數會從 85 降到 82）
- [ ] **重跑 `src/ml/ml_model.py`**，確認「等權 baseline」的數字**有變**
      —— 沒變就代表某個環節沒吃到修正
- [ ] 檢查 `mine_gp.py` 是否共用同一份因子池（尚未確認）
- [ ] 通知 `alpha_mining_agent` 重新匯入參考因子快照：
      `python src/seed_reference.py --clear && python src/seed_reference.py --apply --from-upstream`
- [ ] （選配）B 節那 4 對「公式不同但高度重合」的因子
      （`roa`/`roe`、`d_roa`/`d_roe`、`pretax_to_px`/`ni_to_px`、`pretax_to_rev`/`ni_to_rev`），
      在 `ml_model.py` 的 SHAP／重要性報告裡**合併呈現**，否則單一概念的
      重要性會被拆成兩半而低估。**不要刪**，它們在某些情境會分開。

---

## 🟡 HMM 市場狀態模型：只餵價格資料的版本已完成，但**沒有達標**

`src/hmm_regime.py` 已可用，164 個測試通過。目前的實測結果：

| 做法 | beta | alpha | CAGR | IR | MaxDD |
|---|---|---|---|---|---|
| 靜態：全部因子 | 0.932 | +13.27% | 39.32% | **1.74** | −15.59% |
| **HMM 切換（樣本外）** | 0.791 | +13.52% | 35.80% | **1.08** | −13.16% |
| 完美預知（上限） | 0.874 | +16.97% | 42.82% | 1.98 | −13.16% |

**驗收標準是「HMM 那列要落在靜態與完美預知之間」，目前 IR 落在靜態之下。**
它有降 beta（−15%）與縮小回檔（−2.4pp），但報酬穩定度掉了。
原因是模型太保守：77 個月只喊 23 個月看多，70% 的時間待在防禦態。

### 待做

- [ ] **接入總經特徵**（見下一節），目標是讓模型只在真正的空頭月才切防禦
- [ ] 重跑 `python src/hmm_regime.py --apply`，用同一張表驗收
      ⚠️ **不要換別的指標來讓數字好看**
- [ ] 若加了總經仍不達標，考慮：
      - 改用連續部位配重（依 `p_bull` 線性調整防禦組權重）而非二元切換
      - 狀態數從 2 改 3（但 ~170 個月的樣本很容易過擬合，要看 DSR）
      - 接受「只用它降 beta」這個較小的目標，別強求提高報酬

---

## 🟡 總經資料抓取：三個 bug 已修，待實測

`src/fetch_macro.py` 第一次跑 `--probe` 三個來源全失敗，但**全部是 client 端的 bug，
不是沒有 API**：

| 來源 | 原本的錯 | 真正的原因 | 已修 |
|---|---|---|---|
| 國發會景氣指標 | 「CSV/XLS/XLSX 都讀不起來」 | 下載回來的是 **ZIP** | ✅ 自動解壓，且逐張表解析 |
| 央行貨幣總計數 | `UnicodeEncodeError: ascii` | 網址含中文路徑 `/經研處/`，沒做百分比編碼 | ✅ `_encode_url()` |
| FinMind 殖利率 | `{'status': 200, 'data': []}` | 沒帶 token（成功但空的，不會報錯） | ✅ 自動從 `.env` 找 token |

### 待做

- [ ] 重跑 `python src/fetch_macro.py --probe`，把輸出貼回來
- [ ] 依實際欄名微調 `parse_ndc()` / `parse_cbc()`（政府資料欄名常改版）
- [ ] `python src/fetch_macro.py --fetch` → 產生 `data/macro_monthly.parquet`
- [ ] `python src/fetch_macro.py --verify` 確認 `pub_ym` 對齊正確
- [ ] 把 `as_of()` 接進 `hmm_regime.market_series()` 當額外特徵
- [ ] 補一支前瞻測試：汙染 cut 之後的景氣資料，斷言 cut 之前的狀態序列位元級不變

> ⚠️ **發布落後是這一段最大的風險。** 景氣對策信號是國發會每月 27~30 日
> 發布**上個月**的資料，M1B 約每月 25 日發布上月資料。在月底 t 換股時，
> 最多只能用 `t−1` 月的數字。直接把「3 月燈號」對上「3 月底決策」就是前瞻偏差，
> **而且不會報錯，只會讓回測變好看**。
> `fetch_macro.py` 已經把落後寫進資料結構（`ym` vs `pub_ym`），
> 新增欄位若忘了在 `PUB_LAG_MONTHS` 登記會**直接丟錯**而不是預設 0。

---

## 🟢 挖礦本身：可以繼續跑

- [ ] 繼續 `python src/mining_loop.py --rounds 5`（一輪約 8~15 分鐘、$0.9~1.5）
- [ ] 定期 `python src/report.py --html` 看因子庫成長
- [ ] 留意第 70 輪的整理回合成本（上次有一次 $2.38 / 1106s 的異常，需觀察是否重現）
- [ ] 成本槓桿（目前 ~$1.29/輪，83% 的輸出是推理 token）：
      `--effort low`、`candidates_per_round` 15→10、設 `weekly_cost_budget_usd`

---

## 🟢 已完成的里程碑（存檔用）

<details>
<summary>展開</summary>

- 四階段漏斗 + 產業專屬通道（Stage 4b）
- 兩層記憶：`attempts/` 只進不出 + `learnings.md` 蒸餾
- 整理回合：經驗壓縮、運算子提案、佇列消費機制（`[Q-xxx]`）
- 5 個運算子從提案到上線：`rank_nz`、`industry_demean`、`streak`、`streak_true`、`clip_std`
- 參考因子匯入 + **四道篩選**（含彼此去重，86 → 31 → 25）
- 因子組合實驗室：`--compare` / `--loo` / `--greedy` / `--cost-scan`
- 過擬合診斷六項：régime / breadth / liquidity / importance / DSR+PBO / timing
- **四套前瞻偏差檢測**：DSL 層、全管線、因子合成、HMM
- 164 個測試通過

**目前最佳成果**（test 期 2020-01 起，77 個月，51 個因子 + LightGBM）：
年化超額 **16.36%**、IR **2.57**、Deflated Sharpe **0.997**、PBO 0.001。

**但要記得的三個但書**：
1. 可交易版（最流動 30%）IR 從 2.55 掉到 1.16，且下跌月超額有 58% 來自最好的 3 個月
2. PBO 0.001 要打折扣看（試驗母體以單因子為主，合成模型必然穩定勝出）
3. 六項檢定共用同一段 test 期歷史——真正的樣本外只有一種：**往前走**

</details>
