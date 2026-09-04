# 代辦清單

> 更新日期：2026-09-04（HMM 線結案，專案暫停）
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

- [ ] `git push -u origin main`（本機有 4 個 commit 待推）
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

## ⚫ HMM 市場狀態模型：**這條線已結案，結論是負面的**

`src/hmm_regime.py` 完成、`src/fetch_macro.py` 完成、消融實驗跑完。
最終驗收表（test 期 2020-01 起，77 個月）：

| 做法 | beta | alpha | CAGR | IR | MaxDD |
|---|---|---|---|---|---|
| **靜態：完全不切換** | 0.932 | +13.27% | 39.32% | **1.74** | −15.59% |
| HMM 切換（純價格特徵） | 0.791 | +13.52% | 35.80% | 1.08 | −13.16% |
| HMM 切換（＋總經特徵） | — | — | — | ≤1.08 | — |
| 完美預知（上限） | 0.874 | +16.97% | 42.82% | 1.98 | −13.16% |

**結論（請不要誤讀）**：純價格版是「所有切換方案裡最不糟的」，
**不是**「比不切換好」。所有切換版本都輸給 1.74。
正確的敘述是：**在這份資料上，任何形式的狀態切換都不如不切換。**

加入總經特徵（景氣對策信號、領先指標、M1B/M2）後 IR **沒有改善**，
多組 seed 的中位數仍在純價格版之下。詳細的消融矩陣、門檻掃描與
`p_bull` 分布見 `RESULTS.md` 第 11 節。

為什麼失敗，三個可驗證的原因：

1. **樣本太小**：可用月份約 170 個，其中真正的空頭月更少。
   2 狀態 HMM 要從這麼少的轉折學到穩定的轉移矩陣，本來就很勉強。
2. **因子本身已經有防禦性**：下跌月超額的 t 值在 train/test 都是 8~12，
   模型想切換的方向，因子早就自己做完了——切換只是把它切掉。
3. **總經資料的發布落後吃掉了所有領先性**：能用的只有 `t−1` 月的燈號，
   而燈號本身就是落後指標的合成。

### 若未來要重啟，該從哪裡改（不是從哪裡調參）

- [ ] 不要再調 HMM 的門檻或狀態數去追 IR——那是在對同一段 test 期過擬合
- [ ] 改用**連續配重**（依 `p_bull` 線性調整防禦組權重）而非二元切換，
      並且要先在 train 期證明它有效，再碰 test 期
- [ ] 或者放棄「提高報酬」這個目標，只保留「降 beta」——
      切換版 beta 0.791 vs 靜態 0.932 是真的，這個效果站得住

---

## ⚫ 總經資料抓取：可用，但用不上

`src/fetch_macro.py` 完成、67 個測試通過。三個抓取來源的最終狀態：

| 來源 | 狀態 | 備註 |
|---|---|---|
| 國發會 景氣指標（對策信號分數＋燈號、領先/同時指標） | ✅ 可抓 | 下載回來是 ZIP，已自動解壓逐張表解析 |
| 央行 貨幣總計數（M1B / M2 年增率、黃金交叉） | ✅ 可抓 | 欄名與值都是全形，已做半形正規化 |
| FinMind 美債殖利率曲線 | ❌ 空資料 | `data_id` 應為 `United States 10-Year`（官方文件寫錯）。修正後仍回空陣列，判定為方案層級/額度限制，非程式 bug。標記為選配。 |

發布落後已寫進資料結構本身（`ym` vs `pub_ym`、`as_of()`），
新增欄位若忘了在 `PUB_LAG_MONTHS` 登記會**直接丟錯**而不是預設 0。
這份基礎設施保留下來——若之後接別的總經來源，前瞻防護不用重寫。

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
