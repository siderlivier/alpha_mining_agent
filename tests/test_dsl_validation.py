"""
語法/複雜度防線測試（規格書 5.3）與等價哈希測試（規格書 5.5）。
"""
import pytest

import dsl
from dsl import DSLError


def _code(expr, **kw):
    with pytest.raises(DSLError) as ei:
        dsl.parse(expr, **kw)
    return ei.value.code


# ---- 複雜度硬限制 ----------------------------------------------------------

def test_depth_limit():
    # 深度 4：cs_rank(ts_mean(delta(log1p_abs(f1),3),6))
    assert _code("cs_rank(ts_mean(delta(log1p_abs(f1), 3), 6))") == "too_deep"


def test_depth_3_ok():
    pf = dsl.parse("cs_rank(ts_mean(delta(f1, 3), 6))")
    assert pf.depth == 3


def test_window_whitelist():
    assert _code("ts_mean(f1, 17)") == "bad_window"
    assert _code("ts_mean(f1, 23)") == "bad_window"


def test_window_must_be_literal():
    assert _code("ts_mean(f1, f2)") == "bad_window"


def test_magic_constant_rejected():
    assert _code("if_else(f1 > 13.7, f2, f3)") == "magic_const"
    assert _code("f1 * 2") == "magic_const"


def test_zero_constant_allowed():
    dsl.parse("if_else(f1 > 0, f2, f3)")  # 不應 raise


def test_nested_if_rejected():
    assert _code("if_else(f1 > 0, if_else(f2 > 0, f1, f2), f3)") == "nested_if"


def test_field_count_limit():
    assert _code("f1 + f2 + f3 + f4") == "too_many_fields"


def test_unknown_field():
    assert _code("ts_mean(bad_field, 6)",
                 allowed_fields={"f1", "f2"}) == "unknown_field"


def test_bare_field_rejected():
    assert _code("f1") == "trivial"


# ---- 注入與危險語法 --------------------------------------------------------

def test_injection_rejected():
    assert _code("__import__('os').system('x')") in ("bad_call", "unknown_op", "bad_const")
    assert _code("f1.shift(-1)") == "bad_call"
    assert _code("(lambda: 1)()") == "bad_call"
    assert _code("f1[0]") == "bad_syntax"
    assert _code("f1 / f2") == "bad_op"          # 除法必須用 sdiv
    assert _code("f1 ** 2") == "bad_op"
    assert _code("ts_mean(f1, n=6)") == "bad_call"


def test_type_rules():
    assert _code("if_else(f1, f2, f3)") == "type_error"       # cond 必須是布林
    assert _code("and_(f1, f2)") == "type_error"              # and_ 需要布林參數
    assert _code("f1 > 0") == "type_error"                    # 頂層必須是數值
    assert _code("cs_rank(f1 > 0)") == "type_error"


# ---- 等價哈希 --------------------------------------------------------------

def test_hash_commutative():
    assert dsl.parse("f1 * f2").fhash == dsl.parse("f2 * f1").fhash
    assert dsl.parse("ts_mean(f1, 3) + cs_rank(f2)").fhash == \
           dsl.parse("cs_rank(f2) + ts_mean(f1, 3)").fhash


def test_hash_less_greater_rewrite():
    assert dsl.parse("if_else(f1 < f2, f1, f2)").fhash == \
           dsl.parse("if_else(f2 > f1, f1, f2)").fhash


def test_hash_double_negation():
    assert dsl.parse("neg(neg(f1)) * f2").fhash == dsl.parse("f1 * f2").fhash


def test_hash_distinguishes():
    assert dsl.parse("f1 - f2").fhash != dsl.parse("f2 - f1").fhash
    assert dsl.parse("ts_mean(f1, 3)").fhash != dsl.parse("ts_mean(f1, 6)").fhash


def test_parsed_metadata():
    pf = dsl.parse("ts_rank(f1, 12) * sign(f2)")
    assert pf.depth == 2
    assert pf.fields == frozenset({"f1", "f2"})
