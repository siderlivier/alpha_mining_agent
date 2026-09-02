"""
受限 DSL 引擎（規格書第 5 章）。

職責：
  1. 解析因子公式字串（Python ast，白名單校驗，絕不使用 eval）
  2. 複雜度硬限制（深度、欄位數、窗口白名單、禁嵌套 if_else、禁魔術常數）
  3. 向量化執行（monthly 面板：index=ym、columns=stock_id 的寬表）
  4. AST 正規化 + 等價哈希（交換律排序、雙重否定消除、less→greater 改寫）

資料模型：
  data:      dict[field_name -> pd.DataFrame]，所有 DataFrame 共用相同的
             index（月份，已排序）與 columns（stock_id）
  group_map: dict[stock_id -> 產業名]，截面運算子預設在產業內計算

所有時間序列運算子僅向後看（rolling/shift 皆為正向落後），前瞻偏差在
語言層面封鎖——這是本模組最重要的性質，由 tests/test_dsl_no_lookahead.py 保證。
"""
from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 預設限制（可由 config.yaml 覆寫，經參數傳入）
# ---------------------------------------------------------------------------
WINDOW_WHITELIST = frozenset({3, 6, 12, 24})
MAX_DEPTH = 3
MAX_FIELDS = 3
MIN_PERIODS_FRAC = 2 / 3   # 滾動窗口最低有效樣本比例
SAFE_DIV_EPS = 1e-12
# clip_std 的截尾倍數。寫死而非開放參數——提案規則禁止引入新常數參數，
# 比照 cs_z 的 1%/99% winsorize 也是寫死的做法。
CLIP_STD_K = 3.0

# 運算子分類（名稱 -> (參數型別串, 回傳型別)）
#   n=數值運算元, w=窗口整數, b=布林運算元
_SIGNATURES = {
    # 時間序列
    "ts_mean":  ("nw", "n"), "ts_std": ("nw", "n"), "ts_max": ("nw", "n"),
    "ts_min":   ("nw", "n"), "ts_med": ("nw", "n"), "ts_rank": ("nw", "n"),
    "delay":    ("nw", "n"), "delta":  ("nw", "n"),
    "ts_slope": ("nw", "n"), "ts_rsq": ("nw", "n"), "ts_resi": ("nw", "n"),
    "ts_corr":  ("nnw", "n"),
    # 人工核准的提案（2026-08-19）：streak 證據 A-0537/A-0694/A-0765
    "streak":   ("nw", "n"),
    # 人工核准的提案（2026-08-27）：
    #   streak_true 證據 A-0913/A-0915/A-0918（模型三次用 streak(布林,n) 撞型別錯）
    #   clip_std    證據 A-0522/A-0823/A-0902
    "streak_true": ("bw", "n"),
    "clip_std":    ("nw", "n"),
    # 截面
    "cs_rank": ("n", "n"), "cs_rank_all": ("n", "n"), "cs_z": ("n", "n"),
    # 人工核准的提案（2026-08-19）：
    #   rank_nz         證據 A-0116/A-0143/A-0281/A-0353/A-0361/A-0433
    #   industry_demean 證據 A-0180/A-0181/A-0257/A-0615/A-0652
    "rank_nz": ("n", "n"), "industry_demean": ("n", "n"),
    # 算術
    "add": ("nn", "n"), "sub": ("nn", "n"), "mul": ("nn", "n"), "sdiv": ("nn", "n"),
    "log1p_abs": ("n", "n"), "abs": ("n", "n"), "sign": ("n", "n"), "neg": ("n", "n"),
    # 邏輯
    "greater": ("nn", "b"), "less": ("nn", "b"),
    "and_": ("bb", "b"), "or_": ("bb", "b"),
    "if_else": ("bnn", "n"),
}
_COMMUTATIVE = frozenset({"add", "mul", "and_", "or_"})
_TS_OPS = frozenset(k for k, v in _SIGNATURES.items() if "w" in v[0])


class DSLError(ValueError):
    """所有 DSL 層拒絕以此例外表達；code 進 attempt 紀錄的 rejected_syntax 明細。"""

    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


# ---------------------------------------------------------------------------
# 內部節點表示：("call", name, [args...]) / ("field", name) / ("const", value)
# 窗口參數以 ("window", int) 表示
# ---------------------------------------------------------------------------

def _convert(node: ast.AST):
    """Python ast -> 內部節點。任何白名單以外的語法一律拒絕。"""
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise DSLError("bad_call", "僅允許直接函式呼叫（禁止屬性/下標存取）")
        if node.keywords:
            raise DSLError("bad_call", "禁止關鍵字參數")
        name = node.func.id
        if name not in _SIGNATURES:
            raise DSLError("unknown_op", f"未知運算子: {name}")
        return ("call", name, [_convert(a) for a in node.args])
    if isinstance(node, ast.Name):
        return ("field", node.id)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise DSLError("bad_const", f"禁止常數: {node.value!r}")
        return ("const", node.value)
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}
        if type(node.op) not in ops:
            raise DSLError("bad_op", f"禁止運算符 {type(node.op).__name__}（除法請用 sdiv）")
        return ("call", ops[type(node.op)], [_convert(node.left), _convert(node.right)])
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return ("call", "neg", [_convert(node.operand)])
        raise DSLError("bad_op", f"禁止一元運算符 {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise DSLError("bad_op", "禁止鏈式比較")
        cmp = {ast.Gt: "greater", ast.Lt: "less"}
        if type(node.ops[0]) not in cmp:
            raise DSLError("bad_op", "比較僅允許 > 與 <")
        return ("call", cmp[type(node.ops[0])],
                [_convert(node.left), _convert(node.comparators[0])])
    raise DSLError("bad_syntax", f"禁止語法節點 {type(node).__name__}")


# ---------------------------------------------------------------------------
# 校驗：型別、窗口、常數、複雜度
# ---------------------------------------------------------------------------

def _typecheck(node, window_whitelist) -> str:
    """回傳節點型別 'n'（數值）或 'b'（布林）；同時校驗窗口與常數。"""
    kind = node[0]
    if kind == "field":
        return "n"
    if kind == "const":
        # 規格 5.3：一般位置的常數僅允許 0（比較閾值/分支值）
        if node[1] != 0:
            raise DSLError("magic_const",
                           f"常數僅允許 0（閾值請改用 ts_med 等結構性比較）: {node[1]}")
        return "n"
    _, name, args = node
    sig, ret = _SIGNATURES[name]
    if len(args) != len(sig):
        raise DSLError("bad_arity", f"{name} 需要 {len(sig)} 個參數，收到 {len(args)}")
    for a, expect in zip(args, sig):
        if expect == "w":
            if a[0] != "const" or not isinstance(a[1], int) or isinstance(a[1], bool):
                raise DSLError("bad_window", f"{name} 的窗口參數必須是整數字面值")
            if a[1] not in window_whitelist:
                raise DSLError("bad_window",
                               f"窗口 {a[1]} 不在白名單 {sorted(window_whitelist)}")
            a_type = "w"
        else:
            a_type = _typecheck(a, window_whitelist)
            if a_type != expect:
                raise DSLError("type_error",
                               f"{name} 的參數型別錯誤（需要 {expect}，得到 {a_type}）")
    return ret


def _depth(node) -> int:
    if node[0] != "call":
        return 0
    _, name, args = node
    sig = _SIGNATURES[name][0]
    child = [(_depth(a)) for a, e in zip(args, sig) if e != "w"]
    return 1 + (max(child) if child else 0)


def _fields(node, out: set):
    if node[0] == "field":
        out.add(node[1])
    elif node[0] == "call":
        for a in node[2]:
            _fields(a, out)
    return out


def _check_no_nested_if(node, inside=False):
    if node[0] != "call":
        return
    is_if = node[1] == "if_else"
    if is_if and inside:
        raise DSLError("nested_if", "if_else 不得嵌套 if_else")
    for a in node[2]:
        _check_no_nested_if(a, inside or is_if)


# ---------------------------------------------------------------------------
# 正規化與等價哈希
# ---------------------------------------------------------------------------

def _canon(node) -> str:
    kind = node[0]
    if kind == "field":
        return node[1]
    if kind == "const":
        return repr(node[1])
    _, name, args = node
    # 雙重否定消除
    if name == "neg" and args[0][0] == "call" and args[0][1] == "neg":
        return _canon(args[0][2][0])
    # less(x,y) -> greater(y,x)
    if name == "less":
        return _canon(("call", "greater", [args[1], args[0]]))
    parts = [_canon(a) for a in args]
    if name in _COMMUTATIVE:
        parts = sorted(parts)
    return f"{name}({','.join(parts)})"


# ---------------------------------------------------------------------------
# 公開：解析結果
# ---------------------------------------------------------------------------

@dataclass
class ParsedFactor:
    expr: str
    tree: tuple = field(repr=False)
    depth: int = 0
    fields: frozenset = frozenset()
    canonical: str = ""
    fhash: str = ""


def parse(expr: str,
          allowed_fields=None,
          window_whitelist=WINDOW_WHITELIST,
          max_depth=MAX_DEPTH,
          max_fields=MAX_FIELDS) -> ParsedFactor:
    """解析 + 全套校驗。任何違規 raise DSLError（含 code 供 attempt 歸檔）。"""
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("empty", "空公式")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise DSLError("bad_syntax", f"無法解析: {e}") from None
    node = _convert(tree.body)

    ret = _typecheck(node, frozenset(window_whitelist))
    if ret != "n":
        raise DSLError("type_error", "頂層表達式必須是數值型（布林請包進 if_else）")
    _check_no_nested_if(node)

    d = _depth(node)
    if d > max_depth:
        raise DSLError("too_deep", f"運算子深度 {d} 超過上限 {max_depth}")
    if d == 0:
        raise DSLError("trivial", "裸欄位不構成因子（至少一個運算子）")

    flds = frozenset(_fields(node, set()))
    if len(flds) > max_fields:
        raise DSLError("too_many_fields", f"使用 {len(flds)} 個欄位超過上限 {max_fields}")
    if allowed_fields is not None:
        unknown = flds - set(allowed_fields)
        if unknown:
            raise DSLError("unknown_field", f"未知欄位: {sorted(unknown)}")

    canon = _canon(node)
    fh = hashlib.sha1(canon.encode()).hexdigest()[:16]
    return ParsedFactor(expr=expr, tree=node, depth=d, fields=flds,
                        canonical=canon, fhash=fh)


# ---------------------------------------------------------------------------
# 執行引擎
# ---------------------------------------------------------------------------

def _mp(n: int) -> int:
    return max(2, math.ceil(n * MIN_PERIODS_FRAC))


def _roll_apply(x: pd.DataFrame, n: int, fn) -> pd.DataFrame:
    return x.rolling(n, min_periods=_mp(n)).apply(fn, raw=True)


def _f_ts_rank(w: np.ndarray) -> float:
    v = w[-1]
    if np.isnan(v):
        return np.nan
    arr = w[~np.isnan(w)]
    return float((arr <= v).sum()) / len(arr)


def _f_streak(w: np.ndarray) -> float:
    """
    w 是長度 n 的「單期變化符號」序列，w[-1] 是當期。
    回傳「從當期往回數，與當期同號的連續期數」（1..n）。

    ⛔ 只讀 w 內的資料，而 rolling 視窗的右端就是當期 → 結構上不可能前瞻。
    當期無變化（0）或缺值時回 0：方向未定義，不該被當成趨勢的一部分。
    """
    last = w[-1]
    if not np.isfinite(last) or last == 0:
        return 0.0
    c = 0
    for v in w[::-1]:
        if not np.isfinite(v) or v != last:
            break
        c += 1
    return float(c)


def _f_streak_true(w: np.ndarray) -> float:
    """
    w 是長度 n 的布林序列（1.0/0.0/NaN），w[-1] 是當期。
    回傳「從當期往回數，連續為真的期數」（0..n）；當期為假或缺值即 0。

    與 _f_streak 的差別：這個數的是「條件連續成立」，_f_streak 數的是
    「變化方向連續同號」。模型要的通常是前者（連續 N 期 ROE>0）。
    """
    if not np.isfinite(w[-1]) or w[-1] <= 0:
        return 0.0
    c = 0
    for v in w[::-1]:
        if not np.isfinite(v) or v <= 0:
            break
        c += 1
    return float(c)


def _linreg(w: np.ndarray):
    """回傳 (slope, rsq, resi_last)；樣本不足回 NaN 三元組。"""
    idx = np.arange(len(w), dtype=float)
    m = ~np.isnan(w)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    t, y = idx[m], w[m]
    tb, yb = t.mean(), y.mean()
    sxx = ((t - tb) ** 2).sum()
    if sxx == 0:
        return np.nan, np.nan, np.nan
    b = ((t - tb) * (y - yb)).sum() / sxx
    syy = ((y - yb) ** 2).sum()
    rsq = 0.0 if syy == 0 else (b * b * sxx) / syy
    resi = w[-1] - (yb + b * (idx[-1] - tb)) if not np.isnan(w[-1]) else np.nan
    return b, rsq, resi


class Engine:
    """在寬表面板上執行已解析的因子。"""

    def __init__(self, data: dict, group_map: dict):
        if not data:
            raise DSLError("no_data", "空資料")
        ref = next(iter(data.values()))
        for k, df in data.items():
            if not df.index.equals(ref.index) or not df.columns.equals(ref.columns):
                raise DSLError("data_misaligned", f"欄位 {k} 的 index/columns 不一致")
        self.data = data
        self.index, self.columns = ref.index, ref.columns
        # 產業 -> 該產業的股票欄位（不在 group_map 的股票不參與產業內截面運算）
        self._group_cols = {}
        for sid, g in group_map.items():
            if sid in self.columns:
                self._group_cols.setdefault(g, []).append(sid)

    # -- 截面輔助 ----------------------------------------------------------
    def _per_group(self, x: pd.DataFrame, fn) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for g, cols in self._group_cols.items():
            out[cols] = fn(x[cols])
        return out

    @staticmethod
    def _winsor_z(sub: pd.DataFrame) -> pd.DataFrame:
        lo = sub.quantile(0.01, axis=1)
        hi = sub.quantile(0.99, axis=1)
        w = sub.clip(lo, hi, axis=0)
        mu, sd = w.mean(axis=1), w.std(axis=1)
        return w.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

    # -- 主遞迴 ------------------------------------------------------------
    def eval(self, node) -> pd.DataFrame:
        kind = node[0]
        if kind == "field":
            if node[1] not in self.data:
                raise DSLError("unknown_field", f"資料中無欄位 {node[1]}")
            return self.data[node[1]]
        if kind == "const":
            return pd.DataFrame(float(node[1]), index=self.index, columns=self.columns)

        _, name, args = node
        a = args  # 縮寫

        if name in _TS_OPS:
            n = a[-1][1]
            if name == "ts_corr":
                x, y = self.eval(a[0]), self.eval(a[1])
                return x.rolling(n, min_periods=_mp(n)).corr(y)
            x = self.eval(a[0])
            if name == "ts_mean":
                return x.rolling(n, min_periods=_mp(n)).mean()
            if name == "ts_std":
                return x.rolling(n, min_periods=_mp(n)).std()
            if name == "ts_max":
                return x.rolling(n, min_periods=_mp(n)).max()
            if name == "ts_min":
                return x.rolling(n, min_periods=_mp(n)).min()
            if name == "ts_med":
                return x.rolling(n, min_periods=_mp(n)).median()
            if name == "ts_rank":
                return _roll_apply(x, n, _f_ts_rank)
            if name == "delay":
                return x.shift(n)
            if name == "delta":
                return x - x.shift(n)
            if name == "ts_slope":
                return _roll_apply(x, n, lambda w: _linreg(w)[0])
            if name == "ts_rsq":
                return _roll_apply(x, n, lambda w: _linreg(w)[1])
            if name == "ts_resi":
                return _roll_apply(x, n, lambda w: _linreg(w)[2])
            if name == "streak":
                # 先取單期變化的符號，再數結尾連續同號期數
                return _roll_apply(np.sign(x - x.shift(1)), n, _f_streak)
            if name == "streak_true":
                # x 已是布林（1.0/0.0/NaN，見 greater/less/and_/or_ 的實作）
                return _roll_apply(x, n, _f_streak_true)
            if name == "clip_std":
                # 截尾至「前 n 期（不含當期）」的均值 ±3σ，抑制時序離群值。
                #
                # ⚠️ 刻意偏離提案的偽代碼（它用 ts_mean(x,n)/ts_std(x,n)，窗口含
                #    當期）：那樣寫的話離群值會把自己的上界一起撐高而截不到。
                #    實測 [10]*15 + [1000]：含當期的上界是 949.9（1000 毫髮無傷），
                #    不含當期是 10.68（正確截尾）。shift(1) 只會用到更舊的資料，
                #    前瞻風險反而更低。
                prev = x.shift(1)
                mu = prev.rolling(n, min_periods=_mp(n)).mean()
                sd = prev.rolling(n, min_periods=_mp(n)).std()
                # 歷史窗口零變異時不截尾——否則會把序列硬壓成常數
                # （比照 sdiv 的零分母保護與 cs_z 的 sd.replace(0, nan)）
                sd = sd.mask(sd.abs() < SAFE_DIV_EPS)
                # 暖機期 mu/sd 為 NaN，pandas 的 clip 遇 NaN 邊界會保持原值
                return x.clip(mu - CLIP_STD_K * sd, mu + CLIP_STD_K * sd)

        if name == "cs_rank":
            return self._per_group(self.eval(a[0]), lambda s: s.rank(axis=1, pct=True))
        if name == "cs_rank_all":
            return self.eval(a[0]).rank(axis=1, pct=True)
        if name == "cs_z":
            return self._per_group(self.eval(a[0]), self._winsor_z)
        if name == "rank_nz":
            # 產業內百分位排名，但把 0 與缺值排除在排名之外（結果為 NaN）。
            # 解決 if_else(cond, X, 0) 在 cs_rank 下大量同分並列稀釋 IC 的問題
            # （對應 learnings.md [promoted] 全域規則第 3 條）。
            v = self.eval(a[0])
            return self._per_group(v.where(v.notna() & (v != 0)),
                                   lambda s: s.rank(axis=1, pct=True))
        if name == "industry_demean":
            # 產業內去均值：保留量級（cs_rank 會把量級資訊丟掉），
            # 可表達「去除產業共同 driver 後的殘差」。
            return self._per_group(self.eval(a[0]),
                                   lambda s: s.sub(s.mean(axis=1), axis=0))

        if name == "add":
            return self.eval(a[0]) + self.eval(a[1])
        if name == "sub":
            return self.eval(a[0]) - self.eval(a[1])
        if name == "mul":
            return self.eval(a[0]) * self.eval(a[1])
        if name == "sdiv":
            x, y = self.eval(a[0]), self.eval(a[1])
            return x / y.mask(y.abs() < SAFE_DIV_EPS)
        if name == "log1p_abs":
            x = self.eval(a[0])
            return np.sign(x) * np.log1p(x.abs())
        if name == "abs":
            return self.eval(a[0]).abs()
        if name == "sign":
            return np.sign(self.eval(a[0]))
        if name == "neg":
            return -self.eval(a[0])

        if name in ("greater", "less"):
            x, y = self.eval(a[0]), self.eval(a[1])
            res = (x > y) if name == "greater" else (x < y)
            return res.astype(float).mask(x.isna() | y.isna())
        if name in ("and_", "or_"):
            x, y = self.eval(a[0]), self.eval(a[1])
            if name == "and_":
                res = ((x == 1) & (y == 1)).astype(float)
            else:
                res = ((x == 1) | (y == 1)).astype(float)
            return res.mask(x.isna() | y.isna())
        if name == "if_else":
            c, x, y = self.eval(a[0]), self.eval(a[1]), self.eval(a[2])
            return x.where(c == 1, y).mask(c.isna())

        raise DSLError("unknown_op", f"未實作運算子 {name}")  # 理論上到不了


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def compute(expr_or_parsed, data: dict, group_map: dict, **parse_kw) -> pd.DataFrame:
    """解析（如尚未解析）並計算因子值矩陣。"""
    pf = (expr_or_parsed if isinstance(expr_or_parsed, ParsedFactor)
          else parse(expr_or_parsed, **parse_kw))
    return Engine(data, group_map).eval(pf.tree)


def operators_card() -> str:
    """產生運算子清單（供 Generate prompt 的 DSL 規格卡使用，M3）。"""
    lines = []
    for name, (sig, ret) in _SIGNATURES.items():
        arg = sig.replace("n", "x,").replace("b", "cond,").replace("w", "n,").rstrip(",")
        lines.append(f"{name}({arg}) -> {'bool' if ret == 'b' else 'num'}")
    return "\n".join(lines)
