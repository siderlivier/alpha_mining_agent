"""
總體經濟資料抓取：景氣指標、貨幣總計數、殖利率曲線。

給 `hmm_regime.py` 當額外特徵用。只走**政府開放資料**與 FinMind 免費層，
不需要付費訂閱。

⛔ 這支程式最重要的設計決定：**發布落後寫進資料結構，不是事後才補**
─────────────────────────────────────────────────────────────
每一列都有兩個時間欄位：

    ym      這筆資料**描述**的月份（例如 2020-03 的景氣燈號）
    pub_ym  這筆資料**已經公開**的月份（例如 2020-04）

在月底 t 做決策時，可以用的資料是 `pub_ym <= t` 的那些。
`as_of(panel, t)` 就是在做這件事。

為什麼要這樣設計，而不是「抓進來之後統一 shift(1)」：
  - 各來源的落後不同（景氣指標 1 個月、殖利率 0 個月），統一 shift 會錯
  - 落後是**資料的屬性**，跟在來源旁邊才不會在下游被忘記
  - 忘了對齊不會報錯，只會讓回測變好看——這是最貴的一種 bug

發布時程（依官方公告，見 PUB_LAG_MONTHS 的註解）：
  景氣對策信號 / 領先指標  國發會每月 27~30 日發布**上個月**資料 → 落後 1 個月
  貨幣總計數 M1B / M2      央行約每月 25 日發布**上個月**資料     → 落後 1 個月
  美債殖利率 / 各國利率    日頻，當月即可得                       → 落後 0 個月

用法
----
    python src/fetch_macro.py --probe          # 只測試各來源通不通，不寫檔
    python src/fetch_macro.py --fetch          # 抓取並寫入 data/macro_monthly.parquet
    python src/fetch_macro.py --fetch --source ndc,cbc
    python src/fetch_macro.py --manual 景氣指標.xls   # 手動下載的檔案轉進來
    python src/fetch_macro.py --verify         # 檢查已存檔案的時間對齊

抓不到時的退路：國發會的查詢頁是 JS 動態產生的，若 API 走不通，
從 https://index.ndc.gov.tw/n/zh_tw/data/eco 手動下載，再用 --manual 匯入。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "macro_monthly.parquet"

# ---------------------------------------------------------------------------
# 發布落後（單位：月）。ym + 落後 = pub_ym
#
# ⚠️ 改這裡等於改回測的前瞻紀律，動之前先確認官方公告的發布時程。
#    寧可高估落後（保守）也不要低估——低估不會報錯，只會讓績效虛高。
# ---------------------------------------------------------------------------
PUB_LAG_MONTHS = {
    # 國發會每月 27~30 日發布上個月的景氣指標與對策信號
    "monitoring_score": 1,      # 景氣對策信號綜合分數（9~45）
    "monitoring_color": 1,      # 燈號（藍/黃藍/綠/黃紅/紅，轉成 1~5）
    "leading": 1,               # 領先指標綜合指數
    "leading_yoy": 1,
    "coincident": 1,            # 同時指標
    # 央行約每月 25 日發布上個月的貨幣總計數
    "m1b_yoy": 1,
    "m2_yoy": 1,
    "m1b_minus_m2": 1,          # 黃金交叉：M1B 年增率 − M2 年增率
    # 日頻資料，當月底就知道當月的值
    "us10y": 0,
    "us2y": 0,
    "us_curve": 0,              # 10Y − 2Y，殖利率曲線斜率
}

COLOR_MAP = {"藍燈": 1, "黃藍燈": 2, "綠燈": 3, "黃紅燈": 4, "紅燈": 5,
             "藍": 1, "黃藍": 2, "綠": 3, "黃紅": 4, "紅": 5}

# 政府資料開放平台的資料集編號（CKAN 風格 REST API）
DATA_GOV_API = "https://data.gov.tw/api/v2/rest/dataset/{}"
DATASETS = {
    "ndc": "6099",      # 景氣指標及燈號（國家發展委員會）
    "cbc": "6024",      # 貨幣總計數（中央銀行）
}
NDC_PAGE = "https://index.ndc.gov.tw/n/zh_tw/data/eco"


# ---------------------------------------------------------------------------
# 共用
# ---------------------------------------------------------------------------

def _encode_url(url: str) -> str:
    """
    把網址裡的非 ASCII 字元做百分比編碼。

    央行的開放資料網址含中文路徑：
        https://www.cbc.gov.tw/public/data/OpenData/經研處/EF15M01.csv
    `urllib` 送出前會用 ascii 編碼標頭，遇到中文直接丟
    `UnicodeEncodeError: 'ascii' codec can't encode characters`。
    這不是「沒有 API」，是 client 端沒編碼。
    """
    import urllib.parse
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((
        parts.scheme, parts.netloc,
        urllib.parse.quote(parts.path, safe="/%"),
        urllib.parse.quote(parts.query, safe="=&%?"),
        parts.fragment))


def _http_get(url: str, timeout: int = 30, **kw) -> bytes:
    """單一的 HTTP 出口，方便測試時整支換掉。"""
    import urllib.request
    req = urllib.request.Request(_encode_url(url), headers={
        "User-Agent": "Mozilla/5.0 (compatible; alpha_mining_agent/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def to_ym(s) -> str | None:
    """
    把各種民國／西元的月份寫法統一成 'YYYY-MM'。

    政府資料常見的格式：'11303'（民國112年3月）、'2024/03'、'2024-03'、
    '2024年3月'、'202403'。錯一種就整批資料對不上，所以集中處理。
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    # 先全形轉半形。央行的 CSV 連數字都是全形（`１１３０３`），
    # 不轉的話 isdigit() 對全形數字雖然為 True，int() 也還能處理，
    # 但長度判斷與分隔符比對全都會錯。
    t = _halfwidth(str(s)).strip()
    t = t.replace("年", "/").replace("月", "").replace(".", "/")
    t = t.replace("-", "/").replace("M", "/").replace("m", "/")
    if "/" in t:
        parts = [p for p in t.split("/") if p]
        if len(parts) < 2:
            return None
        y, m = parts[0], parts[1]
    elif t.isdigit() and len(t) == 6:            # 202403
        y, m = t[:4], t[4:]
    elif t.isdigit() and len(t) == 5:            # 11303 = 民國112年3月
        y, m = str(int(t[:3]) + 1911), t[3:]
    else:
        return None
    try:
        yi, mi = int(y), int(m)
    except ValueError:
        return None
    if yi < 1911:                                # 民國年
        yi += 1911
    if not (1 <= mi <= 12) or not (1900 < yi < 2200):
        return None
    return f"{yi:04d}-{mi:02d}"


def shift_ym(ym: str, months: int) -> str:
    p = pd.Period(ym, freq="M") + months
    return str(p)


def add_pub_ym(df: pd.DataFrame) -> pd.DataFrame:
    """
    依欄位查 PUB_LAG_MONTHS，長格式加上 pub_ym。

    輸入是 tidy 長格式（ym, field, value），因為不同欄位的落後不同，
    寬表沒辦法只用一個 pub_ym 欄位表達。
    """
    miss = sorted(set(df["field"]) - set(PUB_LAG_MONTHS))
    if miss:
        raise ValueError(
            f"這些欄位沒有登記發布落後：{miss}\n"
            f"→ 請先在 PUB_LAG_MONTHS 補上，不要讓沒對齊的資料混進面板。")
    out = df.copy()
    out["pub_ym"] = [shift_ym(y, PUB_LAG_MONTHS[f])
                     for y, f in zip(out["ym"], out["field"])]
    return out


# ---------------------------------------------------------------------------
# 來源 1：國發會景氣指標（政府資料開放平台）
# ---------------------------------------------------------------------------

def _resource_urls(dataset_id: str) -> list[str]:
    """從開放平台的資料集 metadata 取出實際的檔案下載連結。"""
    raw = _http_get(DATA_GOV_API.format(dataset_id))
    meta = json.loads(raw.decode("utf-8"))
    body = meta.get("result", meta)
    urls = []
    for d in (body.get("distribution") or []):
        u = d.get("resourceDownloadUrl") or d.get("downloadUrl") or d.get("url")
        if u:
            urls.append(u)
    return urls


def _unzip(raw: bytes) -> list[bytes]:
    """
    ZIP 檔就解開，回傳裡面每個表格檔的位元組。

    國發會開放資料的下載連結給的是 **.zip**（`&icon=.zip`），
    直接丟給 pandas 當然讀不起來。這是「抓得到但解不開」，不是沒有 API。
    """
    import zipfile
    if raw[:2] != b"PK":
        return [raw]
    out = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in z.namelist():
            if n.endswith("/") or n.startswith("__MACOSX"):
                continue
            if Path(n).suffix.lower() in (".csv", ".xls", ".xlsx", ".txt", ""):
                out.append(z.read(n))
    if not out:
        raise ValueError("ZIP 裡沒有可讀的表格檔")
    return out


def _read_one(raw: bytes) -> pd.DataFrame:
    """單一檔案：CSV / XLS / XLSX 都試一次，回傳第一個讀得起來的。"""
    # 政府報表常見：前幾列是標題與說明，真正的表頭在第 2~4 列。
    # 所以每種編碼都從 skiprows=0 試到 3，先窄後寬。
    attempts = []
    for skip in (0, 1, 2, 3):
        for enc in ("utf-8-sig", "big5", "cp950"):
            attempts.append((pd.read_csv, {"encoding": enc, "skiprows": skip,
                                           "on_bad_lines": "skip"}))
        attempts.append((pd.read_excel, {"header": skip}))
    for reader, kw in attempts:
        try:
            df = reader(io.BytesIO(raw), **kw)
            if len(df.columns) > 1 and len(df) > 0:
                return df
        except Exception:
            continue
    raise ValueError("這份檔案 CSV/XLS/XLSX 都讀不起來")


def _read_tabular(raw: bytes) -> pd.DataFrame:
    """讀成表格。ZIP 會先解開，逐個檔案嘗試直到讀得起來。"""
    errs = []
    for member in _unzip(raw):
        try:
            return _read_one(member)
        except Exception as e:
            errs.append(str(e))
    raise ValueError(f"這份檔案（或 ZIP 內的 {len(errs)} 個檔）都讀不起來")


def _read_all_tabular(raw: bytes) -> list[pd.DataFrame]:
    """ZIP 裡可能有多張表（景氣指標與燈號常分開放），全部讀出來。"""
    out = []
    for member in _unzip(raw):
        try:
            out.append(_read_one(member))
        except Exception:
            continue
    return out


def _pick(df: pd.DataFrame, *keywords, exclude=()) -> str | None:
    """
    在欄名裡找包含關鍵字的欄位。政府資料的欄名常改版，所以用模糊比對。

    兩個細節都是被實際資料咬過才加的：

    1. **關鍵字優先於欄位順序**（外層迴圈跑關鍵字，不是跑欄位）。
       欄名 `景氣對策信號分數(綜合判斷分數)` 同時含有「分數」與「信號」，
       若照欄位順序比對，找「燈號/信號」時會先撞上這個分數欄，
       把分數當成燈號去查 COLOR_MAP，結果整欄變 NaN 被默默丟掉。
    2. **`exclude` 排除已認領的欄位**，避免同一欄被兩個欄位共用。
    """
    for k in keywords:
        for c in df.columns:
            if c in exclude:
                continue
            if k in str(c):
                return c
    return None


def parse_ndc(df: pd.DataFrame) -> pd.DataFrame:
    """把國發會的景氣指標表轉成 tidy 長格式。"""
    ymc = _pick(df, "年月", "時間", "期間", "date", "Date", "月份")
    if ymc is None:
        ymc = df.columns[0]
    # 依序認領欄位，認過的就排除——否則「分數」與「燈號」會搶同一欄
    claimed = {ymc}
    mapping = {}
    for key, kws in (("monitoring_score", ("綜合判斷分數", "對策信號分數", "分數")),
                     ("monitoring_color", ("燈號", "信號")),
                     ("leading", ("領先指標綜合指數", "領先指標")),
                     ("coincident", ("同時指標綜合指數", "同時指標"))):
        col = _pick(df, *kws, exclude=claimed)
        mapping[key] = col
        if col is not None:
            claimed.add(col)

    out = []
    for field, col in mapping.items():
        if col is None:
            continue
        for _, r in df.iterrows():
            ym = to_ym(r[ymc])
            if ym is None:
                continue
            v = r[col]
            if field == "monitoring_color":
                v = COLOR_MAP.get(str(v).strip())
            else:
                v = pd.to_numeric(v, errors="coerce")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            out.append({"ym": ym, "field": field, "value": float(v)})
    d = pd.DataFrame(out)
    if len(d) and "leading" in set(d["field"]):
        lead = d[d["field"] == "leading"].set_index("ym")["value"].sort_index()
        yoy = (lead / lead.shift(12) - 1).dropna()
        d = pd.concat([d, pd.DataFrame({"ym": yoy.index, "field": "leading_yoy",
                                        "value": yoy.values})], ignore_index=True)
    return d


def fetch_ndc() -> pd.DataFrame:
    """
    國發會景氣指標。下載回來的是 ZIP，裡面可能有多張表
    （指標一張、燈號一張），所以每張都解析、最後合起來。
    """
    errs, parts = [], []
    for url in _resource_urls(DATASETS["ndc"]):
        try:
            sheets = _read_all_tabular(_http_get(url))
            if not sheets:
                errs.append(f"{url} → ZIP 內沒有讀得起來的表")
                continue
            for i, df in enumerate(sheets):
                try:
                    d = parse_ndc(df)
                    if len(d):
                        parts.append(d)
                        print(f"  ✅ 國發會景氣指標（第 {i+1} 張表）："
                              f"{len(d)} 列，欄位 {sorted(set(d['field']))}")
                except Exception as e:
                    errs.append(f"{url}[表{i+1}] → {type(e).__name__}: {e}")
            if parts:
                return pd.concat(parts, ignore_index=True).drop_duplicates(
                    ["ym", "field"], keep="first")
        except Exception as e:
            errs.append(f"{url} → {type(e).__name__}: {e}")
    raise RuntimeError("國發會景氣指標全部來源都失敗：\n  " + "\n  ".join(errs))


# ---------------------------------------------------------------------------
# 來源 2：央行貨幣總計數
# ---------------------------------------------------------------------------

def _halfwidth(s: str) -> str:
    """
    全形轉半形。政府 CSV 的欄名常寫成 `Ｍ１Ｂ` 而不是 `M1B`，
    直接做子字串比對會完全找不到。
    """
    return "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
                   for c in str(s))


# M1B / M2 的官方定義（央行）：
#   M1A = 通貨淨額 + 支票存款 + 活期存款
#   M1B = M1A + 活期儲蓄存款
#   M2  = M1B + 準貨幣
# EF15M01 這張表給的是**組成項目**，沒有直接的 M1B / M2 欄位，所以自行加總。
M1B_PARTS = ("持有通貨", "支票存款", "活期存款", "活期儲蓄存款")
M2_EXTRA = ("準貨幣",)


def _level_col(df, keyword, used):
    """
    找某個組成項目的「原始值」欄（不是年增率欄）。

    表格是 `<項目>-原始值` / `<項目>-年增率` 成對出現，
    抓錯就會把百分比當成餘額去加總。
    """
    for c in df.columns:
        n = _halfwidth(c)
        if c in used or keyword not in n:
            continue
        if "年增率" in n or "增減" in n or "%" in n:
            continue
        if "原始值" in n or "餘額" in n or "-" not in n:
            return c
    return None


def _crosscheck_deposits(df, cols, ymc) -> None:
    """
    交叉驗證：三項存款加總應該等於表裡的「存款貨幣-計」。

    這是抓「欄位認錯」最有效的一道。若不小心把年增率欄當成餘額欄、
    或把「活期存款」對到「活期儲蓄存款」，加總就會對不上總計——
    而沒有這個檢查的話，錯誤會安靜地變成一組看起來合理的 M1B 數字。
    """
    tot = _level_col(df, "存款貨幣-計", set())
    if tot is None:
        tot = next((c for c in df.columns
                    if "存款貨幣" in _halfwidth(c) and "計" in _halfwidth(c)
                    and "年增率" not in _halfwidth(c)), None)
    if tot is None:
        print("     （表裡沒有「存款貨幣-計」，略過加總交叉驗證）")
        return
    parts = [cols[k] for k in ("支票存款", "活期存款", "活期儲蓄存款") if k in cols]
    if len(parts) < 3:
        return
    s = sum(pd.to_numeric(df[c], errors="coerce") for c in parts)
    t = pd.to_numeric(df[tot], errors="coerce")
    ok = (s.notna() & t.notna() & (t != 0))
    if not ok.any():
        return
    rel = ((s[ok] - t[ok]).abs() / t[ok].abs()).max()
    if rel > 0.01:
        raise ValueError(
            f"三項存款加總與「{tot}」對不上（最大相對誤差 {rel:.1%}）。\n"
            f"    可能認錯欄位——加總用的是 {parts}。\n"
            f"    寧可在這裡停下來，也不要產出一組看起來合理的錯誤 M1B。")
    print(f"     （交叉驗證通過：三項存款加總 ≈「{tot}」，最大誤差 {rel:.2%}）")


def parse_cbc(df: pd.DataFrame) -> pd.DataFrame:
    """
    M1B / M2 年增率，並算出黃金交叉（M1B 年增率 − M2 年增率）。

    兩條路徑：
    1. 表裡直接有 M1B / M2 欄 → 直接用（欄名先做全形轉半形）
    2. 只有組成項目（EF15M01 就是這樣）→ 依央行定義自行加總
    """
    ymc = _pick(df, "年月", "時間", "期間", "date", "月份") or df.columns[0]

    # ── 路徑 1：直接找 M1B / M2 ──
    norm = {c: _halfwidth(c) for c in df.columns}
    m1c = next((c for c, n in norm.items()
                if "M1B" in n and c != ymc and "年增率" not in n), None)
    m2c = next((c for c, n in norm.items()
                if "M2" in n and c not in (ymc, m1c) and "年增率" not in n), None)

    rows = []
    if m1c is not None and m2c is not None:
        source = f"直接欄位（{m1c} / {m2c}）"
        for _, r in df.iterrows():
            ym = to_ym(r[ymc])
            if ym is None:
                continue
            rows.append({"ym": ym,
                         "m1b": pd.to_numeric(r[m1c], errors="coerce"),
                         "m2": pd.to_numeric(r[m2c], errors="coerce")})
    else:
        # ── 路徑 2：由組成項目加總 ──
        used, cols = set(), {}
        for k in M1B_PARTS + M2_EXTRA:
            c = _level_col(df, k, used)
            if c is None:
                raise ValueError(
                    f"表裡既沒有 M1B / M2 欄，也湊不齊組成項目（缺「{k}」）。\n"
                    f"    實際欄名：{[str(c) for c in df.columns][:20]}")
            cols[k] = c
            used.add(c)
        source = "組成項目加總（M1B=通貨+支票+活期+活儲；M2=M1B+準貨幣）"
        for _, r in df.iterrows():
            ym = to_ym(r[ymc])
            if ym is None:
                continue
            vals = {k: pd.to_numeric(r[c], errors="coerce")
                    for k, c in cols.items()}
            if any(pd.isna(v) for v in vals.values()):
                continue
            m1b = sum(vals[k] for k in M1B_PARTS)
            rows.append({"ym": ym, "m1b": m1b,
                         "m2": m1b + sum(vals[k] for k in M2_EXTRA)})
        _crosscheck_deposits(df, cols, ymc)
    print(f"     （M1B/M2 來源：{source}）")
    if not rows:
        # 一列都沒解析成功時，把實際的期間值印出來——否則下一行會丟
        # `KeyError: None of ['ym'] are in the columns`，那個訊息完全看不出
        # 是月份格式沒認出來。
        sample = [repr(v) for v in df[ymc].head(6).tolist()]
        raise ValueError(
            f"「{ymc}」欄一列都認不出月份格式。實際值：{sample}\n"
            f"    → 請在 to_ym() 補上這種寫法。")
    w = pd.DataFrame(rows).dropna().drop_duplicates("ym").set_index("ym").sort_index()
    if len(w) < 13:
        raise ValueError(f"只解析出 {len(w)} 個月，不足以算 12 個月年增率")

    # 欄位可能已經是年增率(%)，也可能是餘額。用量級判斷：餘額是兆元等級。
    is_level = w["m2"].abs().median() > 1000
    if is_level:
        print("     （偵測為餘額資料，自行計算 12 個月年增率）")
        yoy = pd.DataFrame({"m1b_yoy": w["m1b"] / w["m1b"].shift(12) - 1,
                            "m2_yoy": w["m2"] / w["m2"].shift(12) - 1}).dropna()
    else:
        print("     （偵測為年增率資料，除以 100 轉成小數）")
        yoy = pd.DataFrame({"m1b_yoy": w["m1b"] / 100.0,
                            "m2_yoy": w["m2"] / 100.0}).dropna()
    yoy["m1b_minus_m2"] = yoy["m1b_yoy"] - yoy["m2_yoy"]
    return (yoy.reset_index().melt(id_vars="ym", var_name="field",
                                   value_name="value").dropna())


def fetch_cbc() -> pd.DataFrame:
    errs = []
    for url in _resource_urls(DATASETS["cbc"]):
        try:
            d = parse_cbc(_read_tabular(_http_get(url)))
            if len(d):
                print(f"  ✅ 央行貨幣總計數：{url}\n     取得 {len(d)} 列")
                return d
        except Exception as e:
            errs.append(f"{url} → {type(e).__name__}: {e}")
    raise RuntimeError("央行貨幣總計數全部來源都失敗：\n  " + "\n  ".join(errs))


# ---------------------------------------------------------------------------
# 來源 3：美債殖利率曲線（FinMind 免費層）
# ---------------------------------------------------------------------------

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
# .env 的搜尋順序：本專案 → 前置專案（token 原本放在那裡）
ENV_PATHS = (ROOT / ".env", ROOT.parent / "tw_alpha_strategy" / ".env")


def find_finmind_token() -> str:
    """
    依序找 FINMIND_TOKEN：環境變數 → 本專案 .env → 前置專案 .env。

    會找到前置專案是刻意的——token 本來就設在那裡，要求使用者複製一份
    只會製造兩份會不同步的祕密。**只讀不寫，也不會把 token 印出來。**
    """
    tok = os.environ.get("FINMIND_TOKEN", "").strip()
    if tok:
        return tok
    for p in ENV_PATHS:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("FINMIND_TOKEN") and "=" in line:
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    print(f"     （token 讀自 {p}）")
                    return v
    return ""


def finmind_datalist_hint(dataset: str, token: str) -> str:
    """
    問 FinMind「這個資料集有哪些 data_id」，把答案放進錯誤訊息。

    「status=200 但 data=[]」有三種可能，光看回應分不出來：
      (a) data_id 寫錯     (b) token 無效或額度用完     (c) 需要付費層
    datalist 端點能把 (a) 直接排除——它會回傳合法的 data_id 清單。
    有清單就照著改；連 datalist 都空，那就是 (b) 或 (c)。
    """
    import urllib.parse
    url = ("https://api.finmindtrade.com/api/v4/datalist?"
           + urllib.parse.urlencode({"dataset": dataset, "token": token}))
    try:
        raw = json.loads(_http_get(url, timeout=20).decode("utf-8"))
        ids = raw.get("data") or []
    except Exception as e:
        return f"     （查詢 datalist 也失敗：{type(e).__name__}: {e}）"

    if not ids:
        return ("     datalist 也回空 → 不是 data_id 寫錯，而是 **token 無效／額度用完，"
                "\n     或這個資料集需要付費層**。可用 FinMind 網站的用量頁確認。")

    # datalist 認得 data_id、資料端點卻回空 —— 這個組合幾乎只有兩種解釋：
    # 額度用完，或這個資料集要付費層。把用量查出來就能分辨。
    head = [str(x) for x in ids[:6]]
    return (f"     datalist 認得這個資料集（{len(ids)} 個合法 data_id，"
            f"例如 {head}）。\n"
            f"     **合法 data_id 查得到、資料端點卻回空**，代表問題不在寫法：\n"
            f"{finmind_usage_hint(token)}")


def finmind_usage_hint(token: str) -> str:
    """查 FinMind 的 API 用量，分辨「額度用完」與「需要付費層」。"""
    try:
        raw = json.loads(_http_get(
            "https://api.finmindtrade.com/api/v4/user_info?token=" + token,
            timeout=20).decode("utf-8"))
        d = raw.get("user_count"), raw.get("api_request_limit"), raw.get("level")
        used, limit, level = d
        if limit is not None:
            pct = f"（{used / limit:.0%}）" if used and limit else ""
            note = ("\n       → 額度已滿，等下個時段重置即可，資料集本身沒問題。"
                    if used is not None and limit and used >= limit else
                    "\n       → 額度還有剩，所以是**這個資料集需要付費層**。"
                    "\n         景氣指標與貨幣總計數已經夠用，殖利率可以先跳過。")
            return (f"       用量：{used}/{limit} {pct}　會員層級：{level}{note}")
        return f"       user_info 回應：{str(raw)[:200]}"
    except Exception as e:
        return (f"       （查用量也失敗：{type(e).__name__}）\n"
                f"       → 請到 FinMind 網站的用量頁確認是額度還是層級問題。")


def fetch_yields(token: str | None = None, start: str = "2010-01-01") -> pd.DataFrame:
    """
    美債 10Y / 2Y 與曲線斜率。日頻取月底值。

    殖利率曲線倒掛（10Y − 2Y < 0）是經典的衰退領先訊號，而且
    **日頻、當月即可得、完全免費**——沒有發布落後的問題，
    是這四組特徵裡最乾淨的一組。
    """
    import urllib.parse
    token = token or find_finmind_token()
    if not token:
        raise RuntimeError(
            "找不到 FINMIND_TOKEN。FinMind 對沒帶 token 的請求會回\n"
            "     {'status': 200, 'data': []}——**成功但空的**，不會報錯，\n"
            "     很容易被誤判成「這個資料集沒有資料」。\n"
            "     請在專案根目錄放 .env（FINMIND_TOKEN=...）或設環境變數。")
    frames = {}
    # FinMind 的 data_id 是完整國名 + 年期（用 datalist 端點查出來的），
    # 不是文件上寫的 "10-Year"。後面幾個是舊寫法的備援。
    for tag, data_ids in (("us10y", ("United States 10-Year", "10-Year", "10Y")),
                          ("us2y", ("United States 2-Year", "2-Year", "2Y"))):
        d, used, tried = None, None, []
        # 免費層有時會限制可回溯的歷史深度：要 2010 年起會回空，
        # 但只要近幾年就給得出來。所以每個 data_id 都試「完整歷史」與
        # 「近三年」兩種起點，才不會把「歷史太深」誤判成「沒有這個資料集」。
        recent = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        starts = [start] if start >= recent else [start, recent]
        for data_id in data_ids:      # 官方文件的寫法偶有變動，多試幾種
            for st in starts:
                q = urllib.parse.urlencode({"dataset": "GovernmentBondsYield",
                                            "data_id": data_id,
                                            "start_date": st, "token": token})
                raw = json.loads(_http_get(f"{FINMIND_URL}?{q}").decode("utf-8"))
                n = len(raw.get("data") or [])
                tried.append(f"{data_id}@{st}→{n}筆")
                if raw.get("status") == 200 and n:
                    d, used = pd.DataFrame(raw["data"]), f"{data_id}（起 {st}）"
                    break
            if d is not None:
                break
        if d is None:
            raise RuntimeError(
                f"FinMind GovernmentBondsYield 取不到 {tag}：{'、'.join(tried)}\n"
                f"{finmind_datalist_hint('GovernmentBondsYield', token)}")
        print(f"     {tag}: data_id='{used}'，{len(d)} 筆，欄位 {list(d.columns)}")
        dcol = next((c for c in ("date", "Date", "日期") if c in d.columns), None)
        vcol = next((c for c in ("value", "Value", "yield", "close", "price")
                     if c in d.columns), None)
        if vcol is None:      # 只剩一個數值欄的話就用它
            nums = [c for c in d.columns
                    if c != dcol and pd.api.types.is_numeric_dtype(d[c])]
            vcol = nums[0] if len(nums) == 1 else None
        if dcol is None or vcol is None:
            raise RuntimeError(
                f"FinMind 回了 {len(d)} 筆 {tag}，但認不出日期／數值欄。\n"
                f"     實際欄位：{list(d.columns)}\n"
                f"     前兩筆：{d.head(2).to_dict('records')}\n"
                f"     → 把欄名補進 fetch_yields() 的候選清單。")
        d["ym"] = pd.to_datetime(d[dcol]).dt.to_period("M").astype(str)
        s = pd.to_numeric(d[vcol], errors="coerce").groupby(d["ym"]).last()
        # 殖利率通常以「%」給（4.2 = 4.2%）。若中位數 < 0.5 就當它已經是小數，
        # 免得把 0.042 又除以 100 變成 0.00042——這種錯不會報錯，只會讓
        # 曲線斜率縮成噪音。
        if s.abs().median() > 0.5:
            s = s / 100.0
        else:
            print(f"     （{tag} 看起來已是小數形式，不再除以 100）")
        frames[tag] = s
    w = pd.DataFrame(frames).dropna()
    if not len(w):
        raise RuntimeError(
            f"10Y 與 2Y 各自有資料，但沒有共同的月份。\n"
            f"     10Y：{frames['us10y'].index.min()} ~ {frames['us10y'].index.max()}\n"
            f"     2Y ：{frames['us2y'].index.min()} ~ {frames['us2y'].index.max()}")
    w["us_curve"] = w["us10y"] - w["us2y"]
    print(f"  ✅ 美債殖利率：{len(w)} 個月，"
          f"倒掛月份 {int((w['us_curve'] < 0).sum())} 個")
    return (w.reset_index().melt(id_vars="ym", var_name="field",
                                 value_name="value").dropna())


# ---------------------------------------------------------------------------
# 手動下載的檔案
# ---------------------------------------------------------------------------

def load_manual(path: Path) -> pd.DataFrame:
    """
    匯入手動下載的檔案。國發會的查詢頁是 JS 動態產生的，API 走不通時的退路。

    自動判斷是景氣指標表還是貨幣總計數表。
    """
    raw = Path(path).read_bytes()
    df = _read_tabular(raw)
    print(f"  讀到 {len(df)} 列，欄位：{list(df.columns)[:10]}")
    if _pick(df, "M1B") is not None:
        return parse_cbc(df)
    return parse_ndc(df)


# ---------------------------------------------------------------------------
# 面板組裝與時點查詢
# ---------------------------------------------------------------------------

def build_panel(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """把各來源的長格式併起來，加上 pub_ym，去重後排序。"""
    d = pd.concat([p for p in parts if p is not None and len(p)],
                  ignore_index=True)
    d = d.dropna(subset=["ym", "field", "value"])
    d = add_pub_ym(d)
    d = d.drop_duplicates(["ym", "field"], keep="last")
    return d.sort_values(["field", "ym"]).reset_index(drop=True)


def as_of(panel: pd.DataFrame, decision_ym: str) -> dict:
    """
    在月底 `decision_ym` 做決策時，**當下真的知道**的最新總經數值。

    ⛔ 這是整支模組的重點。只取 `pub_ym <= decision_ym` 的列，
       再對每個欄位取 ym 最大的那筆。任何繞過這個函式直接讀 panel 的
       下游程式，都可能在不報錯的情況下用到還沒公布的數字。
    """
    avail = panel[panel["pub_ym"] <= decision_ym]
    if not len(avail):
        return {}
    latest = avail.sort_values("ym").groupby("field").tail(1)
    return dict(zip(latest["field"], latest["value"]))


def as_of_frame(panel: pd.DataFrame, months) -> pd.DataFrame:
    """對一串月份逐月做 as_of，組成可以直接餵給 HMM 的寬表。"""
    rows = {m: as_of(panel, m) for m in months}
    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------
# 指令
# ---------------------------------------------------------------------------

# yield 標成選配：它是四組特徵裡唯一非必要的，而且 FinMind 免費層
# 對這個資料集的支援不穩。ndc + cbc 已足以餵給 HMM。
SOURCES = {"ndc": ("國發會景氣指標", fetch_ndc),
           "cbc": ("央行貨幣總計數", fetch_cbc),
           "yield": ("美債殖利率曲線（選配）", fetch_yields)}
REQUIRED_SOURCES = ("ndc", "cbc")


def cmd_probe(names):
    print("探測各來源是否可用（不寫檔）…\n")
    ok, bad = [], []
    for n in names:
        label, fn = SOURCES[n]
        print(f"[{n}] {label}")
        try:
            d = fn()
            print(f"     欄位 {sorted(set(d['field']))}，"
                  f"月份 {d['ym'].min()} ~ {d['ym'].max()}\n")
            ok.append(n)
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:300]}\n")
            bad.append(n)
    print(f"可用：{ok or '（無）'}　失敗：{bad or '（無）'}")
    missing_required = [n for n in REQUIRED_SOURCES if n in bad]
    if not missing_required:
        opt = [n for n in bad if n not in REQUIRED_SOURCES]
        print(f"✅ 必要來源都通了，可以 --fetch。"
              + (f"（{opt} 是選配，缺了不影響）" if opt else ""))
    else:
        print(f"❌ 必要來源 {missing_required} 還沒通。")
    if "ndc" in bad or "cbc" in bad:
        print(f"\n政府開放平台走不通時的退路：")
        print(f"  1. 到 {NDC_PAGE} 手動下載景氣指標")
        print(f"  2. python src/fetch_macro.py --manual <下載的檔案>")
    return ok


def cmd_fetch(names, manual=None):
    parts = []
    for n in names:
        label, fn = SOURCES[n]
        print(f"[{n}] {label}")
        try:
            parts.append(fn())
        except Exception as e:
            print(f"  ❌ 略過：{type(e).__name__}: {str(e)[:200]}")
    if manual:
        for p in manual:
            print(f"[manual] {p}")
            parts.append(load_manual(Path(p)))
    if not parts:
        raise SystemExit("沒有任何來源成功，未寫檔。先跑 --probe 看失敗原因。")

    panel = build_panel(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"\n✅ 已寫入 {OUT.relative_to(ROOT)}：{len(panel)} 列")
    _summarise(panel)
    return panel


def _summarise(panel: pd.DataFrame):
    print(f"\n{'欄位':<18}{'月份範圍':<22}{'筆數':>6}{'發布落後':>10}")
    print("-" * 58)
    for f, g in panel.groupby("field"):
        print(f"{f:<18}{g['ym'].min()} ~ {g['ym'].max():<10}{len(g):>6}"
              f"{PUB_LAG_MONTHS[f]:>8} 個月")


def cmd_verify():
    """檢查已存檔案的時間對齊，並示範 as_of 的效果。"""
    if not OUT.exists():
        raise SystemExit(f"找不到 {OUT}，請先跑 --fetch")
    panel = pd.read_parquet(OUT)
    _summarise(panel)

    bad = panel[panel["pub_ym"] < panel["ym"]]
    if len(bad):
        raise SystemExit(f"❌ 有 {len(bad)} 列的 pub_ym 早於 ym——資料會看到未來")
    print("\n✅ 每一列的 pub_ym 都不早於 ym")

    demo = sorted(panel["ym"].unique())[-6:]
    print(f"\n=== as_of 示範：在各月底真正知道的最新數值 ===")
    for m in demo:
        got = as_of(panel, m)
        lag = {f: sorted(panel[(panel['field'] == f)
                               & (panel['pub_ym'] <= m)]['ym'])[-1]
               for f in got}
        print(f"  {m} 決策時：" + "、".join(
            f"{f}={got[f]:.3f}(資料月 {lag[f]})" for f in list(got)[:3]))
    print("\n注意「資料月」都早於「決策月」——落後越大的欄位差距越大，"
          "\n這正是 PUB_LAG_MONTHS 在做的事。")


def main():
    ap = argparse.ArgumentParser(description="總經資料抓取（政府開放資料）")
    ap.add_argument("--probe", action="store_true", help="只測試來源是否可用")
    ap.add_argument("--fetch", action="store_true", help="抓取並寫檔")
    ap.add_argument("--verify", action="store_true", help="檢查已存檔案的對齊")
    ap.add_argument("--source", default=",".join(SOURCES),
                    help=f"逗號分隔：{'/'.join(SOURCES)}")
    ap.add_argument("--manual", nargs="*", help="手動下載的檔案路徑")
    a = ap.parse_args()

    names = [s.strip() for s in a.source.split(",") if s.strip()]
    bad = [n for n in names if n not in SOURCES]
    if bad:
        raise SystemExit(f"未知來源 {bad}，可用：{list(SOURCES)}")

    if a.verify:
        cmd_verify()
    elif a.fetch or a.manual:
        cmd_fetch(names if a.fetch else [], a.manual)
    else:
        cmd_probe(names)


if __name__ == "__main__":
    main()
