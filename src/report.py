"""
M4：人類專用報表——因子庫總覽 + 挖礦統計。

⚠️ 本報表包含密封的 test 期指標，**只供人類閱讀決策**，
   內容嚴禁複製進任何 prompt 或 learnings.md。

執行：python src/report.py               # 終端版（逐因子區塊，cmd 友善）
     python src/report.py --html       # 產生 memory/report.html 並自動開啟（推薦）
     python src/report.py --save       # 另存 memory/report.md（markdown 表格）
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import sys
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import Memory


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def _collect(mem: Memory) -> dict:
    lib = mem.own_library()   # 參考因子不列入成果報表
    attempts = mem.load_attempts()
    vc = Counter(a.get("verdict", "?") for a in attempts)
    by_cat = defaultdict(lambda: [0, 0])
    for a in attempts:
        by_cat[a.get("category", "?")][0] += 1
        if a.get("verdict") == "passed":
            by_cat[a.get("category", "?")][1] += 1
    fc = Counter(a.get("failure_type") for a in attempts
                 if a.get("failure_type") not in (None, "", "none"))
    rounds = sorted({a.get("round") for a in attempts if a.get("round")})
    bp = mem.root / "budget.json"
    budget = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {}
    return {"lib": lib, "attempts": attempts, "vc": vc, "by_cat": by_cat,
            "fc": fc, "rounds": rounds, "budget": budget}


FUNNEL_ORDER = ["rejected_syntax", "rejected_duplicate", "rejected_stage1",
                "rejected_stage2", "rejected_stage3", "rejected_stage4"]
STAGE_LABEL = {"rejected_syntax": "Stage0 語法", "rejected_duplicate": "公式去重",
               "rejected_stage1": "Stage1 快篩", "rejected_stage2": "Stage2 庫去相關",
               "rejected_stage3": "Stage3 批內去重", "rejected_stage4": "Stage4 完整驗證"}


# ---------------------------------------------------------------------------
# 終端版（逐因子區塊，避免 cmd 表格對齊問題）
# ---------------------------------------------------------------------------

def build_text(d: dict) -> str:
    L = ["=" * 72,
         "因子挖掘報表（人類專用，含密封 test 指標，勿餵 agent）",
         f"產出時間：{datetime.now().isoformat(timespec='seconds')}",
         "=" * 72]

    L.append(f"\n──── 因子庫（{len(d['lib'])} 個）────")
    for fid, m in d["lib"].items():
        st, va = m.get("sub_train") or {}, m.get("validation") or {}
        se = m.get("test_metrics_sealed") or {}
        scope = m.get("industry_scope")
        L.append(f"\n{fid}【{m.get('name_zh','')}】 "
                 f"{m.get('aspect','')}"
                 f"{'｜⭐' + scope + '限定' if scope else ''}"
                 f"｜{m.get('category','')}｜"
                 f"第{m.get('round','?')}輪｜深度{m.get('depth','?')}")
        L.append(f"  公式  {m.get('formula','')}")
        L.append(f"  說明  {m.get('desc_zh','')}")
        L.append(f"  ICIR  train {fmt(st.get('icir'))} → "
                 f"valid {fmt(va.get('icir'))}（衰減 {fmt(va.get('decay_pct'),0)}%）→ "
                 f"⚠test {fmt(se.get('icir'))}"
                 f"（衰減 {fmt(se.get('decay_vs_subtrain_pct'),0)}%）")
        L.append(f"  其他  覆蓋率 {fmt(m.get('coverage'))}｜"
                 f"月換手 {fmt(m.get('turnover_m'))}")
    if not d["lib"]:
        L.append("（尚無入庫因子）")

    n = len(d["attempts"])
    L.append(f"\n──── 漏斗統計（{n} 筆 attempts）────\n")
    if n:
        alive = n
        for s in FUNNEL_ORDER:
            died = d["vc"].get(s, 0)
            bar = "█" * died
            L.append(f"  {STAGE_LABEL[s]:<12s} 淘汰 {died:>3d}  "
                     f"{alive:>3d} → {alive - died:<3d} {bar}")
            alive -= died
        L.append(f"  {'通過 passed':<12s}      {d['vc'].get('passed',0):>3d} 個"
                 f"（總通過率 {d['vc'].get('passed',0)/n:.0%}）")

        if d["fc"]:
            L.append("\n  失因分類  " + "｜".join(
                f"{k} {v} 筆" for k, v in d["fc"].most_common()))

        cats = sorted(d["by_cat"].items(), key=lambda x: (-x[1][1], -x[1][0]))
        passed_cats = [f"{k}({p}/{t})" for k, (t, p) in cats if p]
        multi = [f"{k}({p}/{t})" for k, (t, p) in cats if t > 1 and not p]
        L.append(f"\n  已探索 category 共 {len(d['by_cat'])} 個")
        if passed_cats:
            L.append("  有產出  " + "、".join(passed_cats))
        if multi:
            L.append("  重複嘗試未過  " + "、".join(multi))
        if d["rounds"]:
            L.append(f"\n  已完成輪次  {d['rounds'][0]} ~ {d['rounds'][-1]}")

    b = d["budget"]
    if b:
        L.append(f"\n──── 預算 ────\n\n  本週（{b.get('week_of','?')}）"
                 f"已用 {b.get('rounds_used','?')} 輪、"
                 f"約 {b.get('est_tokens_used',0):,} tokens｜"
                 f"歷史總輪數 {b.get('total_rounds','?')}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTML 版（推薦閱讀方式）
# ---------------------------------------------------------------------------

def _decay_color(v):
    if v is None:
        return "#888"
    return "#c0392b" if v > 50 else "#e67e22" if v > 30 else "#27ae60"


def build_html(d: dict) -> str:
    e = html_mod.escape
    rows = []
    for fid, m in d["lib"].items():
        st, va = m.get("sub_train") or {}, m.get("validation") or {}
        se = m.get("test_metrics_sealed") or {}
        vd, td = va.get("decay_pct"), se.get("decay_vs_subtrain_pct")
        rows.append(f"""<tr>
<td><b>{fid}</b></td><td>{e(str(m.get('name_zh','')))}</td>
<td>{e(str(m.get('aspect','')))}{('<br><b>⭐' + e(str(m.get('industry_scope'))) + '限定</b>') if m.get('industry_scope') else ''}</td><td class="cat">{e(str(m.get('category','')))}</td>
<td><code>{e(str(m.get('formula','')))}</code></td><td>{m.get('depth','')}</td>
<td>{fmt(st.get('icir'))}</td><td>{fmt(va.get('icir'))}</td>
<td style="color:{_decay_color(vd)}">{fmt(vd,0)}%</td>
<td class="sealed">{fmt(se.get('icir'))}</td>
<td class="sealed" style="color:{_decay_color(td)}">{fmt(td,0)}%</td>
<td>{fmt(m.get('coverage'))}</td><td>{fmt(m.get('turnover_m'))}</td>
<td>{m.get('round','')}</td></tr>
<tr class="desc"><td></td><td colspan="13">💡 {e(str(m.get('desc_zh','')))}</td></tr>""")

    n = len(d["attempts"])
    funnel = []
    alive = n
    for s in FUNNEL_ORDER:
        died = d["vc"].get(s, 0)
        pct = died / n * 100 if n else 0
        funnel.append(f"""<tr><td>{STAGE_LABEL[s]}</td><td>{died}</td>
<td>{alive} → {alive - died}</td>
<td><div class="bar" style="width:{pct * 3:.0f}px"></div></td></tr>""")
        alive -= died

    cats = sorted(d["by_cat"].items(), key=lambda x: (-x[1][1], -x[1][0]))
    cat_rows = "".join(f"<tr><td>{e(k)}</td><td>{t}</td><td>{p}</td></tr>"
                       for k, (t, p) in cats)
    b = d["budget"]
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>因子挖掘報表</title><style>
body{{font-family:'Microsoft JhengHei',system-ui,sans-serif;margin:2em auto;
     max-width:1250px;color:#222;background:#fafafa}}
h1{{font-size:1.4em}} h2{{font-size:1.15em;margin-top:1.6em;
    border-left:4px solid #2563eb;padding-left:.5em}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:.9em}}
th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left}}
th{{background:#f1f5f9;white-space:nowrap}}
code{{background:#f6f8fa;padding:1px 5px;border-radius:4px;font-size:.92em}}
.sealed{{background:#fff7ed}} .cat{{font-size:.82em;color:#555}}
.desc td{{border-top:none;color:#666;font-size:.85em;background:#fcfcfc}}
.bar{{height:12px;background:#ef4444;border-radius:3px}}
.warn{{background:#fef2f2;border:1px solid #fecaca;padding:.7em 1em;
      border-radius:6px;color:#991b1b}}
.small{{color:#666;font-size:.85em}}</style></head><body>
<h1>因子挖掘報表</h1>
<div class="warn">⚠️ 本報表含密封 test 期指標（橙色底欄位），只供人類閱讀，
嚴禁把任何內容複製進 prompt 或 learnings.md。</div>
<p class="small">產出時間 {datetime.now().isoformat(timespec='seconds')}</p>

<h2>因子庫（{len(d['lib'])} 個）</h2>
<table><tr><th>id</th><th>名稱</th><th>面向</th><th>category</th><th>公式</th>
<th>深度</th><th>train ICIR</th><th>valid ICIR</th><th>valid 衰減</th>
<th>⚠test ICIR</th><th>⚠test 衰減</th><th>覆蓋</th><th>換手</th><th>輪</th></tr>
{''.join(rows) if rows else '<tr><td colspan="14">（尚無入庫因子）</td></tr>'}</table>

<h2>漏斗（{n} 筆 attempts，通過 {d['vc'].get('passed',0)} 個
＝ {d['vc'].get('passed',0)/n:.0%}）</h2>
<table><tr><th>關卡</th><th>淘汰</th><th>存活</th><th></th></tr>
{''.join(funnel)}</table>
<p>失因分類：{'、'.join(f"{e(str(k))} {v} 筆" for k, v in d['fc'].most_common()) or '—'}</p>

<h2>已探索 category（{len(d['by_cat'])} 個）</h2>
<table><tr><th>category</th><th>嘗試</th><th>通過</th></tr>{cat_rows}</table>

<h2>預算</h2>
<p>本週（{b.get('week_of','?')}）已用 {b.get('rounds_used','?')} 輪、
約 {b.get('est_tokens_used',0):,} tokens｜歷史總輪數 {b.get('total_rounds','?')}
｜已完成輪次 {d['rounds'][0] if d['rounds'] else '—'} ~
{d['rounds'][-1] if d['rounds'] else '—'}</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Markdown 版（--save，維持原格式供存檔）
# ---------------------------------------------------------------------------

def build_md(d: dict) -> str:
    L = ["# 因子挖掘報表（人類專用，含密封 test 指標，勿餵 agent）",
         f"\n產出時間：{datetime.now().isoformat(timespec='seconds')}\n",
         f"## 因子庫（{len(d['lib'])} 個）\n"]
    if d["lib"]:
        L.append("| id | 名稱 | 面向 | category | 公式 | 深度 | train | valid "
                 "| 衰減% | ⚠test | ⚠test衰減% | 覆蓋 | 換手 | 輪 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for fid, m in d["lib"].items():
            st, va = m.get("sub_train") or {}, m.get("validation") or {}
            se = m.get("test_metrics_sealed") or {}
            L.append(f"| {fid} | {m.get('name_zh','')} | {m.get('aspect','')} "
                     f"| {m.get('category','')} | `{m.get('formula','')}` "
                     f"| {m.get('depth','')} | {fmt(st.get('icir'))} "
                     f"| {fmt(va.get('icir'))} | {fmt(va.get('decay_pct'),0)} "
                     f"| {fmt(se.get('icir'))} "
                     f"| {fmt(se.get('decay_vs_subtrain_pct'),0)} "
                     f"| {fmt(m.get('coverage'))} | {fmt(m.get('turnover_m'))} "
                     f"| {m.get('round','')} |")
        for fid, m in d["lib"].items():
            L.append(f"- **{fid}【{m.get('name_zh','')}】**：{m.get('desc_zh','')}")
    L.append(build_text(d).split("──── 漏斗統計")[1].join(["\n## 漏斗統計", ""])
             if d["attempts"] else "")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--html", action="store_true")
    a = ap.parse_args()
    mem = Memory()
    d = _collect(mem)
    if a.html:
        out = mem.root / "report.html"
        out.write_text(build_html(d), encoding="utf-8")
        print(f"✅ 已存 {out}，正在開啟瀏覽器…")
        webbrowser.open(out.as_uri())
        return
    print(build_text(d))
    if a.save:
        out = mem.root / "report.md"
        out.write_text(build_md(d), encoding="utf-8")
        print(f"\n✅ 已存 {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
