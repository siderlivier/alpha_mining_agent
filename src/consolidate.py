"""
M4：整理回合（規格書 7.4 + 運算子提案流程）。

流程：聚合審計 → 組 consolidate prompt（經驗庫全文 + 近期 attempts 摘要
     + audit + DSL 受限訊號）→ LLM 重寫 learnings.md → 驗證與備份後落盤
     → 運算子提案經機械驗證後寫入 operator_proposals.json（pending_review）

安全機制：
  - 重寫前自動備份至 memory/learnings_history/learnings_r{N}.md
  - 新文本必須含所有必要小節，否則拒絕採用（保留原文）
  - 運算子提案三道機械驗證（名稱格式/純函數簽名/證據引用存在且 ≥2）
    ——通過也只是 pending_review，人類批准前絕不實作

執行：python src/consolidate.py           # 真實 LLM
     python src/consolidate.py --mock    # 測試
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import audit_text
from memory import Memory
from mining_loop import (BUD, CFG, PROMPTS, Meter, budget_report, call_llm,
                         commit_usage, load_budget, log, save_budget)

REQUIRED_SECTIONS = ("## 全域規則", "## 禁忌方向", "## 待驗證假設佇列")
VALID_SIGNATURES = ("(x)", "(x, n)", "(x,n)", "(x, y, n)", "(x,y,n)")
# 整理回合看得到的 attempt 範圍。
#   ATTEMPTS_DETAIL：最近幾筆逐筆列出（模型需要具體案例才能寫出有內容的規則）
#   ATTEMPTS_WINDOW：往前涵蓋到哪，超出 DETAIL 的部分只給聚合統計
# 為什麼要分兩層：整理必須看到「距離上次整理」的完整區間才有資格砍規則，
# 但 150 筆全文會讓 prompt 多出 ~16,000 字元，把呼叫推過 timeout（實測 7/29
# 那次整理已經花了 534s / 上限 600s）。聚合只要 ~1,500 字元就能保住證據基礎。
ATTEMPTS_DETAIL = 45
ATTEMPTS_WINDOW = max(ATTEMPTS_DETAIL, BUD["consolidation_every_n_rounds"]
                      * BUD["candidates_per_round"])
ATTEMPTS_TAIL = ATTEMPTS_DETAIL   # 向後相容（舊名稱）

_CTRL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t",
                 "\b": "\\b", "\f": "\\f"}


def escape_raw_controls(s: str) -> str:
    """
    把 JSON 字串字面值「內部」的裸控制字元轉成合法跳脫序列。

    整理回合要求 LLM 把一份 ~8000 字元的 markdown 塞進 learnings_md 欄位，
    等於要它把 200+ 個換行全部寫成 \\n。漏掉任何一個，json.loads 就會噴
    「Invalid control character」。那份輸出其實是好的，只是跳脫沒做乾淨——
    在本地修掉即可，不必再花一次 LLM 呼叫。
    """
    out, in_str, esc = [], False, False
    for ch in s:
        if esc:                      # 前一個字元是反斜線 → 原樣保留
            out.append(ch)
            esc = False
        elif ch == "\\":
            out.append(ch)
            esc = True
        elif ch == '"':
            in_str = not in_str
            out.append(ch)
        elif in_str and ch in _CTRL_ESCAPES:
            out.append(_CTRL_ESCAPES[ch])
        elif in_str and ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("找不到 JSON 物件")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = text[start:i + 1]
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        # 最常見失因：markdown 換行沒跳脫。就地修一次再試。
                        return json.loads(escape_raw_controls(raw))
    raise ValueError("JSON 物件未閉合")


def _aggregate_block(recs: list[dict]) -> str:
    """把較早的 attempts 壓成聚合統計——保住證據基礎，但不吃 prompt 額度。"""
    def tally(key, top=10):
        c = Counter(str(r.get(key) or "?").strip() or "?" for r in recs)
        return "；".join(f"{k} {v}" for k, v in c.most_common(top))

    out = [f"- verdict 分布：{tally('verdict')}",
           f"- failure_type 分布：{tally('failure_type')}",
           f"- category 分布：{tally('category', 12)}"]
    # 反覆出現的 lesson = 值得升級成 [promoted] 的候選
    lc = Counter(str(r.get("lesson") or "").strip()[:45]
                 for r in recs if str(r.get("lesson") or "").strip())
    rep = [f"    ({v}×) {k}" for k, v in lc.most_common(12) if v >= 2]
    if rep:
        out.append("- 重複出現的 lesson（出現次數 × 前 45 字，升級 [promoted] 的候選）：")
        out += rep
    return "\n".join(out)


def attempts_summary(mem: Memory, detail=ATTEMPTS_DETAIL,
                     window=ATTEMPTS_WINDOW) -> str:
    recs = mem.load_attempts()
    if not recs:
        return "（無）"
    win = recs[-window:]
    det = win[-detail:]
    older = win[:-detail] if len(win) > detail else []

    lines = []
    if older:
        lines += [f"## 較早的 {len(older)} 筆（{older[0]['id']}~{older[-1]['id']}，"
                  f"聚合統計，不逐筆列出）",
                  _aggregate_block(older), ""]
    lines.append(f"## 最近 {len(det)} 筆（逐筆：id | verdict | category | 公式 | "
                 f"失因 | lesson）")
    for r in det:
        lines.append(f"{r['id']} | {r.get('verdict','?')} | {r.get('category','?')} "
                     f"| {r.get('formula','')[:45]} | {r.get('failure_type','')} "
                     f"| {str(r.get('lesson',''))[:70]}")
    return "\n".join(lines)


def load_limitations(mem: Memory) -> str:
    p = mem.root / "dsl_limitations.jsonl"
    if not p.exists():
        return "（無）"
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for l in lines[-30:]:
        try:
            d = json.loads(l)
            out.append(f"- [{d.get('attempt','?')}] {d.get('note','')}")
        except json.JSONDecodeError:
            continue
    return "\n".join(out) if out else "（無）"


def validate_proposal(p: dict, mem: Memory) -> str | None:
    """回傳 None = 通過；否則回傳拒絕原因。三道機械驗證。"""
    name = str(p.get("name", ""))
    if not re.fullmatch(r"[a-z_][a-z0-9_]{2,20}", name):
        return f"名稱不合法: {name!r}"
    sig = str(p.get("signature", "")).replace(" ", "")
    if sig not in [s.replace(" ", "") for s in VALID_SIGNATURES]:
        return f"簽名 {p.get('signature')!r} 不在純函數白名單 {VALID_SIGNATURES}"
    ev = p.get("evidence") or []
    if len(ev) < 2:
        return f"證據引用不足（{len(ev)} < 2）"
    known = {r["id"] for r in mem.load_attempts()}
    missing = [e for e in ev if e not in known]
    if missing:
        return f"引用了不存在的 attempt: {missing}"
    if not p.get("semantics") or not p.get("why"):
        return "缺少 semantics 或 why"
    return None


def run(mem: Memory | None = None, mock_fn=None, meter: Meter | None = None) -> dict:
    """
    整理回合。meter 由呼叫端傳入並在 call_llm 之後立刻累加——這樣即使後面
    解析失敗拋例外，呼叫端仍然握有「這通已經花了多少」，可以正確記帳。
    run() 本身不寫 budget.json，由呼叫端決定何時落帳。
    """
    if mock_fn and mem is None:
        import tempfile
        mem = Memory(Path(tempfile.mkdtemp(prefix="alpha-consolidate-mock-")))
    mem = mem or Memory()
    if mock_fn and mem.root.resolve() == Memory().root.resolve():
        raise ValueError("mock cannot write production memory")
    mem.ensure()
    emit = print if mock_fn else log
    meter = meter if meter is not None else Meter()
    cap = BUD["learnings_token_cap"]

    tpl = (PROMPTS / "consolidate.md").read_text(encoding="utf-8")
    prompt = tpl.format(
        token_cap=cap, char_cap=cap * 2,
        learnings=mem.prompt_learnings(),
        attempts_summary=attempts_summary(mem),
        audit=audit_text(mem),
        limitations=load_limitations(mem),
    )
    # 整理回合的輸出遠比挖礦長（要吐一整份 markdown），用專屬的較長 timeout。
    # meter 直接傳給 call_llm：成功或逾時都會計帳。
    timeout = float(CFG["llm"].get("consolidate_timeout_sec",
                                   CFG["llm"].get("timeout_sec", 1800)))
    emit(f"  整理 prompt {len(prompt):,} 字元，timeout {timeout:.0f}s，開始呼叫...")
    r = call_llm(prompt, mock_fn, meter=meter, timeout=timeout)
    out = r.text
    emit(f"  整理回應 {len(out):,} 字元，耗時 {r.elapsed:.0f}s")

    # 原始輸出一律先落盤。這通呼叫的 token 在 call_llm 回來的當下就已經花掉了，
    # 解析失敗時如果什麼都不留，才是真的白花；留著就能手動救回或事後檢查。
    raw_dir = mem.root / "consolidate_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    raw_path.write_text(out, encoding="utf-8")

    try:
        res = parse_json_object(out)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"整理輸出無法解析（{e}）。原始輸出已存於 {raw_path}——"
            f"可從中取出 learnings_md 手動貼進 memory/learnings.md，"
            f"或直接重跑 consolidate。learnings.md 本身未被更動。") from e

    # ---- learnings 重寫（驗證 + 備份）----
    new_text = res.get("learnings_md", "")
    ok = isinstance(new_text, str) and all(s in new_text for s in REQUIRED_SECTIONS)
    if ok:
        hist = mem.root / "learnings_history"
        hist.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (hist / f"learnings_{stamp}.md").write_text(
            mem.read_learnings(), encoding="utf-8")
        mem.write_learnings(new_text)
        emit(f"  整理：learnings.md 已重寫（備份 learnings_{stamp}.md），"
            f"{len(new_text)} 字元")
    else:
        emit("  ⚠️ 整理輸出缺必要小節，保留原 learnings.md 不動")

    # ---- 運算子提案 ----
    accepted, rejected = [], []
    for p in res.get("operator_proposals") or []:
        if not isinstance(p, dict):
            continue
        reason = validate_proposal(p, mem)
        if reason:
            rejected.append({"proposal": p, "rejected": reason})
        else:
            p["status"] = "pending_review"
            p["proposed_at"] = datetime.now().isoformat(timespec="seconds")
            accepted.append(p)
    if accepted or rejected:
        pp = mem.root / "operator_proposals.json"
        existing = (json.loads(pp.read_text(encoding="utf-8"))
                    if pp.exists() else {})
        for k in ("pending", "implemented", "rejected_log"):
            existing.setdefault(k, [])
        # 同名提案不重複累積。⚠️ 三個清單都要查：
        #   pending     還沒審
        #   implemented 已實作（會出現在 operators_card，理論上不會再提，但保險）
        #   rejected_log 已被人工否決——只查 pending 的話，被否決的提案會在
        #                下一次整理回合原封不動地再被提一次
        known = {x["name"] for x in existing["pending"]}
        known |= {x["name"] for x in existing["implemented"]}
        known |= {x.get("proposal", {}).get("name")
                  for x in existing["rejected_log"]}
        fresh = [p for p in accepted if p["name"] not in known]
        dup = len(accepted) - len(fresh)
        existing["pending"] += fresh
        existing["rejected_log"] += rejected
        pp.write_text(json.dumps(existing, ensure_ascii=False, indent=1),
                      encoding="utf-8")
        # 報「實際新增」而非「通過驗證」——舊版在全部重複時仍會印通過數，
        # 讓人以為有新提案待審（8/19 那次就是 3 個全重複卻印「3 通過」）
        emit(f"  運算子提案：新增 {len(fresh)} 個待審"
            + (f"（{dup} 個與既有/已實作/已否決同名，略過）" if dup else "")
            + (f"、{len(rejected)} 個未通過機械驗證" if rejected else "")
            + f"。目前待審共 {len(existing['pending'])} 個，"
              f"看 memory/operator_proposals.json。")

    return {"learnings_updated": ok, "proposals": len(accepted),
            "notes": res.get("notes", ""),
            # tokens 是 CLI 回報的真實用量（output_format=json 時）。
            # est_tokens 是舊欄位名，僅為相容而保留，數值相同——新程式碼請用 tokens。
            "tokens": r.tokens, "est_tokens": r.tokens,
            "cost_usd": r.cost_usd, "measured": r.measured}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-budget", action="store_true",
                    help="不要把這次的用量計入 budget.json")
    a = ap.parse_args()
    mock_fn = None
    if a.mock:
        def mock_fn(prompt):
            return json.dumps({
                "learnings_md": "# 因子挖掘經驗庫\n\n## 全域規則\n\n- [MOCK] 整理測試\n\n"
                                "## 禁忌方向\n\n（尚無）\n\n## 待驗證假設佇列\n\n（尚無）\n",
                "operator_proposals": [], "notes": "mock 整理"},
                ensure_ascii=False)
    meter = Meter()
    try:
        res = run(mock_fn=mock_fn, meter=meter)
        print(json.dumps(res, ensure_ascii=False))
    finally:
        # 獨立執行也要計帳——舊版完全不碰 budget.json，
        # 導致手動跑的整理在週用量裡完全看不到。
        if meter.calls and not (a.mock or a.no_budget):
            b = load_budget()
            commit_usage(b, meter)
            save_budget(b)
            print(f"\n本次整理用量：{meter}", file=sys.stderr)
            print(budget_report(b), file=sys.stderr)


if __name__ == "__main__":
    main()
