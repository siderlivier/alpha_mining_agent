# 資料擴張：因子自動檢查與產業範圍

## 操作

先更新上游資料，再執行 `python -B src/build_base.py` 重建面板。股票產業採用 `config.yaml` 的 `scope_control.classification_source` 指定之上游分類；本工具不根據公司名稱猜產業。

啟動本機介面：

```powershell
python -B src/scope_ui.py --port 8765
```

開啟 http://127.0.0.1:8765 ，選「開始檢查」產生待套用計畫，或選「檢查並自動套用」依設定完成檢查及提交。檢查會寫報告與暫存值，不更動正式因子庫。套用前先停止挖礦及其他資料更新工作，避免使用先前已載入記憶體的版本。

CLI 使用相同核心：

```powershell
python -B src/factor_scope.py --check-only
python -B src/factor_scope.py --auto-apply
python -B src/factor_scope.py --apply RUN_ID
python -B src/factor_scope.py --recover
```

`RUN_ID` 換成檢查報告的實際 ID。`crosssec_oos.py --refresh --check-only` 也會轉入相同流程。原本不帶這些參數的橫斷面報告仍是探索用途，不負責核准產業。

## 分類與判定規則

- `scope_control.groups` 登錄允許的產業。值為祖先產業清單；分類更名或拆分必須登錄來源，不能藉此把舊產業當作新產業。
- 股票與上游分類不一致、未知分類、重複股票月份、歷史分類變動或因子來源紀錄不明，都會阻擋整批提交。
- 只評估尚未見過的新產業；舊排除產業永不補回。既有核准範圍不因本次重測縮減。
- 沿用 Stage 4b 的 train／validation 標準，以完整精度判斷；另外檢查覆蓋率、交易用途與 `max_monthly_turnover`。不讀 test 表現來核准。
- 多頭、空頭、雙邊用途固定，不能為了通過新產業任意換方向。純空頭因子可保留純空頭用途。
- 通過及不通過都記錄為已見；樣本不足保留待測，資料補足後可以重試。
- 全市場舊因子也先固定於入庫時的產業，新增產業通過後才擴大。

`approved_groups` 是現行使用範圍；原始 `industry_scope` 與入庫績效保留為歷史，不代表擴張後重新估計的績效。參考因子本次不換版。

## 紀錄與恢復

每次報告位於 `memory/scope_runs/RUN_ID/report.json`，包含逐因子逐產業的原因、原始指標及輸入雜湊。套用前再次核對資料、設定、程式及暫存輸出雜湊；變動後舊計畫失效，須重新檢查。

套用前備份 metadata 與因子值，使用互斥鎖和中斷標記；發生中斷時讀取會被阻擋，執行 `--recover` 依備份回復。不要手動刪除鎖或中斷標記。UI 僅綁定 localhost，一次執行一個工作；關閉瀏覽器不取消工作，請保留伺服器程序直到完成。第一版沒有取消按鈕。

## 驗證範圍

`tests/test_factor_scope.py` 與 `tests/test_scope_ui.py` 涵蓋分類衝突、舊排除產業、樣本不足、方向用途、test 資料隔離、重跑、過期計畫、提交失敗回復與 HTTP 請求保護（全套 227 個測試全數通過；這些是 AI 依規格書撰寫的功能性測試）。這不是「不存在任何 bug」的保證，也不代表先前評估報告的所有資料／回測問題已修正。上游分類本身的業務正確性仍由來源負責。

Web UI 已實作且 HTTP 測試通過；本次瀏覽器工具因分頁工作階段錯誤，未完成視覺及點擊驗收。
