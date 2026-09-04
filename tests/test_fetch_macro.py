"""
總經資料抓取（fetch_macro）的解析與**時點對齊**測試。

這支模組的核心風險不在「抓不抓得到」——抓不到會噴錯，一眼就看得出來。
真正危險的是**抓到了但對齊錯了**：景氣燈號要到次月底才公布，若把
「3 月的燈號」對上「3 月底的決策」，回測不會報錯，只會變好看。

所以測試的重心全放在 ym / pub_ym 的關係與 `as_of()` 的行為上。
網路部分一律用假的 `_http_get` 取代，測試不依賴外部連線。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fetch_macro as fm        # noqa: E402


# ---------------------------------------------------------------------------
# 月份格式正規化
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("2024/03", "2024-03"),
    ("2024-3", "2024-03"),
    ("202403", "2024-03"),
    ("2024年3月", "2024-03"),
    ("11303", "2024-03"),          # 民國 113 年 3 月
    ("10001", "2011-01"),          # 民國 100 年 1 月
    (" 2024/03 ", "2024-03"),
])
def test_to_ym_handles_common_government_formats(raw, want):
    """
    政府資料的月份格式很雜，民國年尤其容易錯。

    `11303` 若被當成西元 11303 或直接截成 113，整批資料就會對到錯的月份，
    而且**不會報錯**——只會讓因子跟總經資料錯開好幾年。
    """
    assert fm.to_ym(raw) == want


@pytest.mark.parametrize("raw", ["", "abc", None, np.nan, "2024", "2024/13",
                                 "1800/05"])
def test_to_ym_rejects_garbage(raw):
    """認不出來要回 None 讓上游丟掉，不能猜。"""
    assert fm.to_ym(raw) is None


def test_shift_ym_crosses_year_boundary():
    assert fm.shift_ym("2024-12", 1) == "2025-01"
    assert fm.shift_ym("2024-01", -1) == "2023-12"


# ---------------------------------------------------------------------------
# 發布落後：寫進資料結構
# ---------------------------------------------------------------------------

def test_add_pub_ym_applies_per_field_lag():
    """
    每個欄位的落後不同，不能統一 shift。

    景氣燈號落後 1 個月、美債殖利率落後 0 個月——統一處理必然有一邊錯。
    """
    d = pd.DataFrame({"ym": ["2024-03", "2024-03"],
                      "field": ["monitoring_score", "us_curve"],
                      "value": [30.0, 0.005]})
    out = fm.add_pub_ym(d)
    got = dict(zip(out["field"], out["pub_ym"]))
    assert got["monitoring_score"] == "2024-04", "景氣資料要到次月才公布"
    assert got["us_curve"] == "2024-03", "日頻資料當月就知道"


def test_add_pub_ym_refuses_unregistered_field():
    """
    沒登記發布落後的欄位必須擋下來，不能預設 0。

    預設 0 等於預設「沒有落後」——那是最危險的預設值，
    新增一個總經欄位卻忘了查發布時程時，回測會靜默地看到未來。
    """
    d = pd.DataFrame({"ym": ["2024-03"], "field": ["未登記的新指標"],
                      "value": [1.0]})
    with pytest.raises(ValueError, match="發布落後"):
        fm.add_pub_ym(d)


def test_every_registered_lag_is_non_negative():
    """落後不可能是負的——負的代表資料在描述的月份之前就公布了。"""
    assert all(v >= 0 for v in fm.PUB_LAG_MONTHS.values())


# ---------------------------------------------------------------------------
# as_of：時點查詢
# ---------------------------------------------------------------------------

@pytest.fixture
def panel():
    rows = []
    for i, ym in enumerate([str(p) for p in
                            pd.period_range("2024-01", periods=6, freq="M")]):
        rows.append({"ym": ym, "field": "monitoring_score", "value": 20.0 + i})
        rows.append({"ym": ym, "field": "us_curve", "value": 0.01 * i})
    return fm.build_panel([pd.DataFrame(rows)])


def test_as_of_never_returns_unpublished_data(panel):
    """
    在 2024-03 月底做決策時，能拿到的景氣分數只到 2024-02
    （2024-03 的資料要 2024-04 才公布）。
    """
    got = fm.as_of(panel, "2024-03")
    feb = panel[(panel["field"] == "monitoring_score") & (panel["ym"] == "2024-02")]
    assert got["monitoring_score"] == feb["value"].iloc[0]

    mar = panel[(panel["field"] == "monitoring_score") & (panel["ym"] == "2024-03")]
    assert got["monitoring_score"] != mar["value"].iloc[0], \
        "拿到了當月還沒公布的景氣分數"
    # 零落後的欄位則可以拿到當月
    assert got["us_curve"] == panel[(panel["field"] == "us_curve")
                                    & (panel["ym"] == "2024-03")]["value"].iloc[0]


def test_as_of_is_monotone_in_time(panel):
    """
    決策月份越晚，能看到的資料月份只會更新、不會倒退。

    這條擋的是排序或去重寫錯導致「拿到更舊的資料」的情況。
    """
    seen = []
    for m in [str(p) for p in pd.period_range("2024-02", periods=5, freq="M")]:
        got = fm.as_of(panel, m)
        seen.append(got.get("monitoring_score", -np.inf))
    assert seen == sorted(seen), f"as_of 隨時間倒退了：{seen}"


def test_as_of_empty_before_first_publication(panel):
    """第一筆資料公布之前，什麼都拿不到（而不是拿到最舊的那筆）。"""
    assert fm.as_of(panel, "2023-12") == {}


def test_as_of_frame_matches_row_by_row(panel):
    months = [str(p) for p in pd.period_range("2024-02", periods=4, freq="M")]
    wide = fm.as_of_frame(panel, months)
    for m in months:
        row = wide.loc[m].dropna().to_dict()
        assert row == pytest.approx(fm.as_of(panel, m))


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------

def test_parse_ndc_extracts_score_and_colour():
    df = pd.DataFrame({
        "年月": ["11301", "11302", "11303"],
        "綜合判斷分數": [22, 25, 31],
        "景氣對策信號燈號": ["黃藍燈", "綠燈", "綠燈"],
        "領先指標綜合指數": [100.5, 101.2, 102.0],
    })
    d = fm.parse_ndc(df)
    got = d.set_index(["field", "ym"])["value"]
    assert got[("monitoring_score", "2024-01")] == 22
    assert got[("monitoring_color", "2024-02")] == 3, "綠燈要對應到 3"
    assert got[("monitoring_color", "2024-01")] == 2, "黃藍燈要對應到 2"
    assert got[("leading", "2024-03")] == 102.0


def test_parse_ndc_survives_renamed_columns():
    """政府資料的欄名會改版，解析要靠關鍵字模糊比對而不是精確欄名。"""
    df = pd.DataFrame({"統計期間": ["2024/01"],
                       "景氣對策信號分數(綜合判斷分數)": [22],
                       "燈號": ["綠燈"]})
    d = fm.parse_ndc(df)
    assert set(d["field"]) >= {"monitoring_score", "monitoring_color"}


def test_parse_cbc_computes_yoy_from_levels():
    """餘額資料要自行算年增率；量級判斷不能把年增率誤當餘額。"""
    ym = [str(p) for p in pd.period_range("2023-01", periods=24, freq="M")]
    lvl = np.linspace(20000, 24000, 24)          # 兆元等級 → 判定為餘額
    df = pd.DataFrame({"年月": ym, "M1B": lvl, "M2": lvl * 2})
    d = fm.parse_cbc(df)
    got = d.set_index(["field", "ym"])["value"]
    assert ("m1b_yoy", "2024-01") in got.index
    # M1B 與 M2 同比例成長 → 年增率相同 → 交叉為 0
    assert got[("m1b_minus_m2", "2024-01")] == pytest.approx(0.0, abs=1e-9)


def test_parse_cbc_treats_small_numbers_as_percentages():
    ym = [str(p) for p in pd.period_range("2023-01", periods=14, freq="M")]
    df = pd.DataFrame({"年月": ym, "M1B": [6.0] * 14, "M2": [4.0] * 14})
    d = fm.parse_cbc(df).set_index(["field", "ym"])["value"]
    assert d[("m1b_yoy", "2023-05")] == pytest.approx(0.06)
    assert d[("m1b_minus_m2", "2023-05")] == pytest.approx(0.02)


def test_parse_cbc_reports_missing_columns():
    df = pd.DataFrame({"年月": ["2024/01"], "貨幣": [1.0]})
    with pytest.raises(ValueError, match="M1B"):
        fm.parse_cbc(df)


# ---------------------------------------------------------------------------
# 抓取流程（用假的 HTTP）
# ---------------------------------------------------------------------------

def test_fetch_ndc_uses_metadata_resource_urls(monkeypatch):
    """從開放平台 metadata 取下載連結 → 讀檔 → 解析，整條串起來。"""
    meta = {"result": {"distribution": [
        {"resourceDownloadUrl": "https://example.invalid/bad.csv"},
        {"resourceDownloadUrl": "https://example.invalid/good.csv"}]}}
    good = pd.DataFrame({"年月": ["11301"], "綜合判斷分數": [22],
                         "燈號": ["綠燈"]}).to_csv(index=False).encode("utf-8-sig")

    def fake_get(url, **kw):
        if url.endswith("/6099"):
            return json.dumps(meta).encode()
        if "bad" in url:
            raise RuntimeError("404")
        return good

    monkeypatch.setattr(fm, "_http_get", fake_get)
    d = fm.fetch_ndc()
    assert len(d) and "monitoring_score" in set(d["field"])


def test_fetch_ndc_raises_when_all_sources_fail(monkeypatch):
    """全部失敗要丟明確的錯，不能回空 DataFrame 讓上游以為抓到了。"""
    monkeypatch.setattr(fm, "_http_get", lambda url, **kw: (
        json.dumps({"result": {"distribution": [
            {"resourceDownloadUrl": "https://example.invalid/x.csv"}]}}).encode()
        if url.endswith("/6099") else (_ for _ in ()).throw(RuntimeError("boom"))))
    with pytest.raises(RuntimeError, match="全部來源都失敗"):
        fm.fetch_ndc()


def test_build_panel_deduplicates_and_sorts():
    a = pd.DataFrame({"ym": ["2024-01"], "field": ["us_curve"], "value": [0.01]})
    b = pd.DataFrame({"ym": ["2024-01"], "field": ["us_curve"], "value": [0.02]})
    p = fm.build_panel([a, b])
    assert len(p) == 1 and p["value"].iloc[0] == 0.02, "重複的月份要保留後來的"
    assert set(p.columns) >= {"ym", "field", "value", "pub_ym"}


# ---------------------------------------------------------------------------
# 實測抓不到時暴露出來的三個問題（都是 client 端的 bug，不是沒有 API）
# ---------------------------------------------------------------------------

def test_url_with_chinese_path_is_percent_encoded():
    """
    央行的開放資料網址含中文路徑（.../OpenData/經研處/EF15M01.csv）。
    不編碼的話 urllib 送出前會丟 UnicodeEncodeError——看起來像「API 掛了」，
    其實是 client 沒編碼。
    """
    url = "https://www.cbc.gov.tw/public/data/OpenData/經研處/EF15M01.csv"
    enc = fm._encode_url(url)
    enc.encode("ascii")                      # 不可以再丟 UnicodeEncodeError
    assert "%E7%B6%93" in enc, f"中文沒被編碼：{enc}"
    assert enc.startswith("https://www.cbc.gov.tw/")
    assert enc.endswith("EF15M01.csv")


def test_url_encoding_is_idempotent():
    """已經編碼過的網址不可以被二次編碼（%E7 變成 %25E7 就抓不到了）。"""
    once = fm._encode_url("https://x.invalid/a/經研處/b.csv")
    assert fm._encode_url(once) == once


def test_read_tabular_unzips_government_archives():
    """
    國發會的下載連結給的是 ZIP。直接丟給 pandas 會說「讀不起來」，
    看起來像資料格式有問題，其實只是沒解壓。
    """
    import zipfile
    csv = pd.DataFrame({"年月": ["11301"], "綜合判斷分數": [22],
                        "燈號": ["綠燈"]}).to_csv(index=False).encode("utf-8-sig")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("景氣指標.csv", csv)
    df = fm._read_tabular(buf.getvalue())
    assert "綜合判斷分數" in "".join(str(c) for c in df.columns)


def test_read_all_tabular_returns_every_sheet_in_zip():
    """ZIP 裡常常指標一張表、燈號另一張表，不能只讀第一個。"""
    import zipfile
    a = pd.DataFrame({"年月": ["11301"], "綜合判斷分數": [22]}).to_csv(index=False)
    b = pd.DataFrame({"年月": ["11301"], "領先指標綜合指數": [100.0]}).to_csv(index=False)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.csv", a.encode("utf-8-sig"))
        z.writestr("b.csv", b.encode("utf-8-sig"))
        z.writestr("__MACOSX/._a.csv", b"junk")     # 要被忽略
    sheets = fm._read_all_tabular(buf.getvalue())
    assert len(sheets) == 2


def test_read_tabular_handles_report_style_header_rows():
    """政府報表常在真正的表頭上面放幾列標題，要能跳過。"""
    raw = ("景氣指標統計表\n說明：本表由國發會編製\n"
           "年月,綜合判斷分數\n11301,22\n").encode("utf-8-sig")
    df = fm._read_tabular(raw)
    assert "綜合判斷分數" in [str(c) for c in df.columns]


def test_finmind_empty_response_raises_actionable_error(monkeypatch):
    """
    FinMind 對沒帶 token 的請求回 {'status': 200, 'data': []}——
    **成功但空的**。若照字面判斷會以為「這個資料集沒資料」，
    實際上是認證問題。錯誤訊息必須把這件事講明白。
    """
    monkeypatch.setattr(fm, "find_finmind_token", lambda: "")
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="成功但空的"):
        fm.fetch_yields()


def test_finmind_tries_alternate_data_ids(monkeypatch):
    """官方文件的 data_id 寫法偶有變動，要多試幾種再放棄。"""
    seen = []

    def fake_get(url, **kw):
        import urllib.parse as up
        q = up.parse_qs(up.urlsplit(url).query)
        did = q["data_id"][0]
        seen.append(did)
        data = ([{"date": "2024-01-31", "value": 4.0}] if did in ("10Y", "2Y")
                else [])
        return json.dumps({"status": 200, "msg": "success", "data": data}).encode()

    monkeypatch.setattr(fm, "_http_get", fake_get)
    d = fm.fetch_yields(token="dummy")
    assert "10-Year" in seen and "10Y" in seen, f"沒有試過備用寫法：{seen}"
    assert set(d["field"]) == {"us10y", "us2y", "us_curve"}


def test_find_finmind_token_prefers_env_and_never_returns_placeholder(monkeypatch):
    monkeypatch.setenv("FINMIND_TOKEN", "from-env")
    assert fm.find_finmind_token() == "from-env"
    monkeypatch.setenv("FINMIND_TOKEN", "   ")
    monkeypatch.setattr(fm, "ENV_PATHS", ())
    assert fm.find_finmind_token() == ""


# ---------------------------------------------------------------------------
# 央行貨幣總計數：實際的表沒有 M1B/M2 欄，只有組成項目
# ---------------------------------------------------------------------------

def _cbc_like(n=30, scale=1.0):
    """仿 EF15M01 的欄位結構：每個項目都有「-原始值」與「-年增率」兩欄。"""
    ym = [str(p) for p in pd.period_range("2022-01", periods=n, freq="M")]
    g = np.linspace(1.0, 1.2, n)
    cur, chk, dem, sav, qm = (2000 * g, 500 * g, 3000 * g, 8000 * g, 30000 * g)
    return pd.DataFrame({
        "期間": ym,
        "貨幣機構以外各部門持有通貨-原始值": cur * scale,
        "貨幣機構以外各部門持有通貨-年增率": np.full(n, 5.0),
        "存款貨幣-計-原始值": (chk + dem + sav) * scale,
        "存款貨幣-計-年增率": np.full(n, 4.0),
        "存款貨幣-支票存款-原始值": chk * scale,
        "存款貨幣-支票存款-年增率": np.full(n, 3.0),
        "存款貨幣-活期存款-原始值": dem * scale,
        "存款貨幣-活期存款-年增率": np.full(n, 4.0),
        "存款貨幣-活期儲蓄存款-原始值": sav * scale,
        "存款貨幣-活期儲蓄存款-年增率": np.full(n, 6.0),
        "準貨幣-原始值": qm * scale,
        "準貨幣-年增率": np.full(n, 2.0),
    })


def test_parse_cbc_builds_m1b_from_components():
    """
    實際的 EF15M01 沒有叫 M1B / M2 的欄位，只有組成項目。
    依央行定義自行加總：M1B = 通貨+支票+活期+活儲，M2 = M1B+準貨幣。
    """
    df = _cbc_like(30)
    d = fm.parse_cbc(df)
    assert {"m1b_yoy", "m2_yoy", "m1b_minus_m2"} == set(d["field"])
    # 各項同比例成長 → M1B 與 M2 的年增率相同 → 交叉為 0
    x = d[d["field"] == "m1b_minus_m2"]["value"]
    assert x.abs().max() < 1e-9


def test_parse_cbc_never_sums_the_percentage_columns():
    """
    每個項目都有「-原始值」與「-年增率」兩欄，抓錯就會把百分比當餘額加總。
    這裡讓餘額是兆元等級、年增率是個位數——若誤用年增率欄，
    量級判斷會把它當成「已經是年增率」而走錯分支，結果完全不同。
    """
    df = _cbc_like(30)
    d = fm.parse_cbc(df).set_index(["field", "ym"])["value"]
    # 餘額線性成長 1.0→1.2，12 個月的年增率應為正且遠小於 1（不是 5.0 那種百分數）
    v = d["m1b_yoy"]
    assert (v > 0).all() and v.max() < 0.5, f"年增率量級不對：{v.head().tolist()}"


def test_parse_cbc_prefers_explicit_m1b_column_when_present():
    """有些表就直接給 M1B / M2，這時不必自己加總。"""
    ym = [str(p) for p in pd.period_range("2022-01", periods=15, freq="M")]
    df = pd.DataFrame({"期間": ym, "M1B-原始值": np.linspace(10000, 12000, 15),
                       "M2-原始值": np.linspace(40000, 44000, 15)})
    d = fm.parse_cbc(df)
    assert "m1b_yoy" in set(d["field"])


def test_parse_cbc_handles_fullwidth_column_names():
    """政府 CSV 常把欄名寫成全形 Ｍ１Ｂ，子字串比對會完全找不到。"""
    ym = [str(p) for p in pd.period_range("2022-01", periods=15, freq="M")]
    df = pd.DataFrame({"期間": ym, "Ｍ１Ｂ": np.linspace(10000, 12000, 15),
                       "Ｍ２": np.linspace(40000, 44000, 15)})
    d = fm.parse_cbc(df)
    assert "m1b_yoy" in set(d["field"])


def test_parse_cbc_crosscheck_catches_wrong_column():
    """
    三項存款加總必須對得上「存款貨幣-計」。
    把總計欄動手腳 → 必須丟錯，而不是安靜產出看似合理的 M1B。
    """
    df = _cbc_like(30)
    df["存款貨幣-計-原始值"] = df["存款貨幣-計-原始值"] * 2.0     # 故意不一致
    with pytest.raises(ValueError, match="對不上"):
        fm.parse_cbc(df)


def test_parse_cbc_reports_full_column_list_when_unparseable():
    """湊不齊組成項目時，錯誤訊息要把實際欄名列出來，方便對照修正。"""
    df = pd.DataFrame({"期間": ["2024/01"], "某個不相干的欄": [1.0]})
    with pytest.raises(ValueError, match="組成項目"):
        fm.parse_cbc(df)


def test_halfwidth_conversion():
    assert fm._halfwidth("Ｍ１Ｂ") == "M1B"
    assert fm._halfwidth("M1B") == "M1B"
    assert fm._halfwidth("存款貨幣-計") == "存款貨幣-計"


def test_datalist_hint_lists_valid_ids(monkeypatch):
    """datalist 有回東西 → 錯誤訊息要把合法的 data_id 列出來。"""
    monkeypatch.setattr(fm, "_http_get", lambda url, **kw: json.dumps(
        {"status": 200, "data": ["10Y", "2Y", "30Y"]}).encode())
    hint = fm.finmind_datalist_hint("GovernmentBondsYield", "tok")
    assert "10Y" in hint and "3 個" in hint


def test_datalist_hint_distinguishes_auth_failure(monkeypatch):
    """datalist 也空 → 不是 data_id 寫錯，要明講是 token / 付費層問題。"""
    monkeypatch.setattr(fm, "_http_get", lambda url, **kw: json.dumps(
        {"status": 200, "data": []}).encode())
    hint = fm.finmind_datalist_hint("GovernmentBondsYield", "tok")
    assert "token" in hint and "付費層" in hint


def test_datalist_hint_survives_network_error(monkeypatch):
    """診斷本身失敗也不能蓋掉原本的錯誤。"""
    def boom(url, **kw):
        raise OSError("no route")
    monkeypatch.setattr(fm, "_http_get", boom)
    assert "datalist 也失敗" in fm.finmind_datalist_hint("X", "tok")


# ---------------------------------------------------------------------------
# 實測第二輪暴露的問題
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("１１３０３", "2024-03"),        # 全形民國年
    ("２０２４／０３", "2024-03"),     # 全形西元 + 全形斜線
    ("２０２４０３", "2024-03"),
    ("2024M03", "2024-03"),        # 央行常見的 M 分隔
    ("2024m3", "2024-03"),
    ("１１３年３月", "2024-03"),
])
def test_to_ym_handles_fullwidth_and_m_separator(raw, want):
    """
    央行的 CSV 連數字都是全形。欄名的全形已經處理了，值也要處理——
    否則會出現「欄位找到了，但一列都認不出月份」這種很難診斷的情況。
    """
    assert fm.to_ym(raw) == want


def test_parse_cbc_reports_unparseable_period_values():
    """
    一列都沒解析成功時，錯誤訊息要印出實際的期間值。

    不加這道的話，下一行會丟 `KeyError: None of ['ym'] are in the columns`
    ——那個訊息完全看不出真正的問題是「月份格式沒認出來」。
    """
    ym = ["民國113年第3季"] * 20                    # 故意用認不出的寫法
    df = pd.DataFrame({"期間": ym,
                       "貨幣總計數-Ｍ１Ｂ-原始值": np.linspace(1e4, 1.2e4, 20),
                       "貨幣總計數-Ｍ２-原始值": np.linspace(4e4, 4.4e4, 20)})
    with pytest.raises(ValueError, match="認不出月份格式"):
        fm.parse_cbc(df)


def test_parse_cbc_rejects_too_few_months():
    """不足 13 個月就算不出 12 個月年增率，要明講而不是回空表。"""
    ym = [str(p) for p in pd.period_range("2024-01", periods=6, freq="M")]
    df = pd.DataFrame({"期間": ym, "M1B-原始值": np.linspace(1e4, 1.1e4, 6),
                       "M2-原始值": np.linspace(4e4, 4.2e4, 6)})
    with pytest.raises(ValueError, match="不足以算"):
        fm.parse_cbc(df)


def test_parse_cbc_handles_real_cbc_column_shape():
    """
    仿實際欄名：全形 Ｍ１Ｂ、欄名中間有空格、值也是全形民國年月。
    這是 --probe 第二輪實際遇到的組合。
    """
    n = 30
    ym = [str(p) for p in pd.period_range("2022-01", periods=n, freq="M")]
    fw = str.maketrans("0123456789/", "０１２３４５６７８９／")
    roc = [f"{int(m[:4]) - 1911}{m[5:7]}".translate(fw) for m in ym]
    df = pd.DataFrame({
        "期間": roc,
        "貨幣總計數 -Ｍ１Ｂ-原始值": np.linspace(20000, 24000, n),
        "貨幣總計數 -Ｍ１Ｂ-年增率": np.full(n, 5.0),
        "貨幣總計數 -Ｍ２-原始值": np.linspace(50000, 55000, n),
        "貨幣總計數 -Ｍ２-年增率": np.full(n, 4.0),
    })
    d = fm.parse_cbc(df)
    assert {"m1b_yoy", "m2_yoy", "m1b_minus_m2"} == set(d["field"])
    assert d["ym"].min() == "2023-01"      # 前 12 個月拿來當基期
    assert (d[d["field"] == "m1b_yoy"]["value"] > 0).all()


def test_fetch_yields_uses_full_country_data_id(monkeypatch):
    """
    FinMind 的 data_id 是 'United States 10-Year'，不是文件寫的 '10-Year'。
    這是用 datalist 端點查出來的。
    """
    seen = []

    def fake_get(url, **kw):
        import urllib.parse as up
        q = up.parse_qs(up.urlsplit(url).query)
        did = q["data_id"][0]
        seen.append(did)
        data = ([{"date": "2024-01-31", "value": 4.0}]
                if did.startswith("United States") else [])
        return json.dumps({"status": 200, "data": data}).encode()

    monkeypatch.setattr(fm, "_http_get", fake_get)
    d = fm.fetch_yields(token="dummy")
    assert seen[0] == "United States 10-Year", f"第一個試的應該是完整國名：{seen}"
    assert set(d["field"]) == {"us10y", "us2y", "us_curve"}


# ---------------------------------------------------------------------------
# 殖利率：回應格式的防禦
# ---------------------------------------------------------------------------

def _fm_reply(rows):
    return json.dumps({"status": 200, "msg": "success", "data": rows}).encode()


def _yield_rows(n=40, val=4.0, col="value", start="2022-01-31"):
    """
    漂移量刻意設成 val 的 1%——太大的話小數版（0.042）會漂到 0.4 以上，
    測試自己就把量級搞混了，反而測不出「有沒有重複除以 100」。
    """
    dates = pd.date_range(start, periods=n, freq="ME").strftime("%Y-%m-%d")
    step = abs(val) * 0.01
    return [{"date": d, "name": "x", col: val + i * step}
            for i, d in enumerate(dates)]


def test_fetch_yields_accepts_alternate_value_column(monkeypatch):
    """FinMind 的欄名若不是 `value`，要能自動認出唯一的數值欄。"""
    monkeypatch.setattr(fm, "_http_get",
                        lambda url, **kw: _fm_reply(_yield_rows(col="yield")))
    d = fm.fetch_yields(token="t")
    assert set(d["field"]) == {"us10y", "us2y", "us_curve"}


def test_fetch_yields_reports_unknown_columns(monkeypatch):
    """認不出欄位時要把實際欄名與前兩筆印出來，不要丟裸 KeyError。"""
    rows = [{"日期時間": "2022-01-31", "殖利率數值": 4.0} for _ in range(5)]
    monkeypatch.setattr(fm, "_http_get", lambda url, **kw: _fm_reply(rows))
    with pytest.raises(RuntimeError, match="認不出日期／數值欄"):
        fm.fetch_yields(token="t")


def test_fetch_yields_does_not_double_scale_decimals(monkeypatch):
    """
    殖利率通常以 % 給（4.2 = 4.2%），但若來源已經是小數（0.042），
    再除以 100 會變成 0.00042——不會報錯，只會讓曲線斜率縮成噪音。
    """
    monkeypatch.setattr(fm, "_http_get",
                        lambda url, **kw: _fm_reply(_yield_rows(val=0.042)))
    d = fm.fetch_yields(token="t")
    v = d[d["field"] == "us10y"]["value"]
    assert 0.01 < v.median() < 0.2, f"量級不對：{v.median()}"


def test_fetch_yields_scales_percentages(monkeypatch):
    monkeypatch.setattr(fm, "_http_get",
                        lambda url, **kw: _fm_reply(_yield_rows(val=4.2)))
    d = fm.fetch_yields(token="t")
    v = d[d["field"] == "us10y"]["value"]
    assert 0.01 < v.median() < 0.2, f"% 沒有轉成小數：{v.median()}"


def test_fetch_yields_errors_when_no_overlapping_months(monkeypatch):
    """兩個年期各自有資料但月份不重疊時，要說清楚各自的範圍。"""
    def fake(url, **kw):
        import urllib.parse as up
        did = up.parse_qs(up.urlsplit(url).query)["data_id"][0]
        start = "2022-01-31" if "10-Year" in did else "2015-01-31"
        return _fm_reply(_yield_rows(n=12, start=start))
    monkeypatch.setattr(fm, "_http_get", fake)
    with pytest.raises(RuntimeError, match="沒有共同的月份"):
        fm.fetch_yields(token="t")


def test_fetch_yields_retries_with_recent_start(monkeypatch):
    """
    免費層有時限制可回溯的歷史深度：要 2010 年起回空、近三年就給得出來。
    不重試的話會把「歷史太深」誤判成「沒有這個資料集」。
    """
    seen = []

    def fake(url, **kw):
        import urllib.parse as up
        q = up.parse_qs(up.urlsplit(url).query)
        st = q["start_date"][0]
        seen.append(st)
        # 只有「近期」起點才給資料
        data = _yield_rows(n=30, start="2023-01-31") if st >= "2020-01-01" else []
        return _fm_reply(data)

    monkeypatch.setattr(fm, "_http_get", fake)
    d = fm.fetch_yields(token="t", start="2010-01-01")
    assert seen[0] == "2010-01-01", "應該先試完整歷史"
    assert any(s >= "2020-01-01" for s in seen), "回空之後要改試近期起點"
    assert set(d["field"]) == {"us10y", "us2y", "us_curve"}


def test_usage_hint_says_quota_full(monkeypatch):
    """額度用完 → 要說「等重置就好，資料集沒問題」。"""
    monkeypatch.setattr(fm, "_http_get", lambda url, **kw: json.dumps(
        {"user_count": 600, "api_request_limit": 600, "level": 0}).encode())
    assert "額度已滿" in fm.finmind_usage_hint("t")


def test_usage_hint_says_tier_when_quota_remains(monkeypatch):
    """額度還有剩卻取不到 → 就是資料集需要付費層，要建議先跳過。"""
    monkeypatch.setattr(fm, "_http_get", lambda url, **kw: json.dumps(
        {"user_count": 12, "api_request_limit": 600, "level": 0}).encode())
    hint = fm.finmind_usage_hint("t")
    assert "付費層" in hint and "先跳過" in hint


def test_datalist_hint_chains_into_usage(monkeypatch):
    """
    datalist 認得 data_id、資料端點卻回空——這個組合要直接接到用量診斷，
    而不是重複建議「改 data_id」（那已經被排除了）。
    """
    def fake(url, **kw):
        if "datalist" in url:
            return json.dumps({"data": ["United States 10-Year"]}).encode()
        return json.dumps({"user_count": 5, "api_request_limit": 600,
                           "level": 0}).encode()
    monkeypatch.setattr(fm, "_http_get", fake)
    hint = fm.finmind_datalist_hint("GovernmentBondsYield", "t")
    assert "問題不在寫法" in hint and "付費層" in hint
