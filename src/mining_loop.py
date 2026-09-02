"""
M3：挖礦迴圈 orchestrator（規格書第 8 章）。

一輪 = 預算檢查 → 組 Generate prompt → claude -p → 解析候選
     → 本地四階段漏斗 → 組 Distill prompt → claude -p → 解析歸因
     → attempts 落盤 → passed 入庫（密封 test）→ lesson 追加 learnings.md
     → 更新 budget.json

模式：
  python src/mining_loop.py                  # 跑 1 輪（真實 LLM）
  python src/mining_loop.py --rounds 5       # 連跑 5 輪（額度內）
  python src/mining_loop.py --dry-run        # 只印 Generate prompt，不呼叫 LLM
  python src/mining_loop.py --mock           # 用內建假 LLM 跑通管線（測試用）

⛔ 洩漏紀律：本模組組裝的任何 prompt 只允許引用
   learnings.md / library_summary() / 診斷 JSON——三者皆已通過洩漏檢查。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsl
import eval_candidates as ec
from admit import admit_from_results
from memory import Memory

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
BUD = CFG["budget"]
PROMPTS = ROOT / CFG["paths"]["prompts_dir"]
BUDGET_PATH = ROOT / CFG["paths"]["memory_dir"] / "budget.json"
LOG_PATH = ROOT / "mining.log"


def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# 預算（規格 8.4）
# ---------------------------------------------------------------------------

def week_start() -> str:
    d = date.today()
    return str(d - timedelta(days=d.weekday()))


EMPTY_BREAKDOWN = {"input": 0, "output": 0,
                   "cache_creation": 0, "cache_read": 0}


def _blank_budget(total_rounds: int, last_consolidate_round: int = 0) -> dict:
    return {"week_of": week_start(),
            "rounds_used": 0,
            "total_rounds": total_rounds,
            # 全期狀態，不隨週重置：用來擋住整理回合的重試風暴
            "last_consolidate_round": last_consolidate_round,
            "tokens_used": 0,            # 真實 token（含 cache），週一重置
            "cost_usd": 0.0,             # CLI 回報的估算成本，週一重置
            "tokens_breakdown": dict(EMPTY_BREAKDOWN),
            "llm_calls": 0,
            "estimated_calls": 0,        # 沒拿到真實 usage、只能用字數估算的通數
            "failed_calls": 0}           # 失敗但仍然計費的通數（累積在本週內）


def load_budget() -> dict:
    """讀取本週預算；跨週自動重置（total_rounds 是全期累計，不重置）。"""
    old = {}
    if BUDGET_PATH.exists():
        try:
            old = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("⚠️ budget.json 損毀，本週用量從 0 起算（total_rounds 一併歸零）")
            old = {}
    b = _blank_budget(old.get("total_rounds", 0),
                      old.get("last_consolidate_round", 0))
    if old.get("week_of") == week_start():
        b.update(old)
        # 舊版欄位遷移：est_tokens_used（字數估算）→ tokens_used
        if "tokens_used" not in old and "est_tokens_used" in old:
            b["tokens_used"] = old["est_tokens_used"]
        b.pop("est_tokens_used", None)
        b.setdefault("tokens_breakdown", dict(EMPTY_BREAKDOWN))
    return b


def save_budget(b: dict):
    BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_PATH.write_text(json.dumps(b, ensure_ascii=False, indent=1),
                           encoding="utf-8")


def commit_usage(b: dict, meter: "Meter") -> dict:
    """把一個 Meter 的用量併進預算。⚠️ 呼叫成功或失敗都要走這裡。"""
    b["tokens_used"] = b.get("tokens_used", 0) + meter.tokens
    b["cost_usd"] = round(b.get("cost_usd", 0.0) + meter.cost, 6)
    bd = b.setdefault("tokens_breakdown", dict(EMPTY_BREAKDOWN))
    for k, v in meter.breakdown.items():
        bd[k] = bd.get(k, 0) + v
    b["llm_calls"] = b.get("llm_calls", 0) + meter.calls
    b["estimated_calls"] = b.get("estimated_calls", 0) + meter.estimated_calls
    return b


def budget_stop_reason(b: dict) -> str | None:
    """回傳不該再開新輪的理由；None = 還有額度。上限設 0 代表不限制。"""
    if b["rounds_used"] >= BUD["weekly_round_budget"]:
        return (f"週輪次預算已用完（{b['rounds_used']}/{BUD['weekly_round_budget']} 輪）")
    tcap = BUD.get("weekly_token_budget") or 0
    if tcap and b.get("tokens_used", 0) >= tcap:
        return f"週 token 預算已用完（{b.get('tokens_used', 0):,}/{tcap:,}）"
    ccap = BUD.get("weekly_cost_budget_usd") or 0
    if ccap and b.get("cost_usd", 0.0) >= ccap:
        return f"週成本預算已用完（${b.get('cost_usd', 0.0):.2f}/${ccap:.2f}）"
    return None


def budget_report(b: dict) -> str:
    bd = b.get("tokens_breakdown") or EMPTY_BREAKDOWN
    tcap = BUD.get("weekly_token_budget") or 0
    ccap = BUD.get("weekly_cost_budget_usd") or 0
    lines = [
        f"本週（{b['week_of']} 起）用量：",
        f"  輪次    {b['rounds_used']}/{BUD['weekly_round_budget']}",
        f"  tokens  {b.get('tokens_used', 0):,}"
        + (f"/{tcap:,}" if tcap else "（未設上限）"),
        f"  成本    ${b.get('cost_usd', 0.0):.4f}"
        + (f"/${ccap:.2f}" if ccap else "（未設上限）"),
        f"  明細    in {bd.get('input', 0):,} / out {bd.get('output', 0):,} / "
        f"cache 建立 {bd.get('cache_creation', 0):,}、讀取 {bd.get('cache_read', 0):,}",
        f"  呼叫    {b.get('llm_calls', 0)} 通"
        + (f"（其中 {b['failed_calls']} 通失敗仍計費）"
           if b.get("failed_calls") else "")
        + (f"（其中 {b['estimated_calls']} 通為字數估算）"
           if b.get("estimated_calls") else ""),
        f"  全期累計輪次 {b.get('total_rounds', 0)}",
    ]
    stop = budget_stop_reason(b)
    lines.append(f"  ⛔ {stop}" if stop else "  ✅ 額度內")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM 呼叫
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    """一通 LLM 呼叫的結果與用量。"""
    text: str
    tokens: int = 0
    cost_usd: float = 0.0
    breakdown: dict = field(default_factory=lambda: dict(EMPTY_BREAKDOWN))
    measured: bool = False       # True = CLI 回報的真實 usage；False = 字數估算
    elapsed: float = 0.0         # 牆鐘秒數，用來判斷離 timeout 還有多少餘裕
    timed_out: bool = False


# claude -p --output-format json 的 usage 欄位 → 我們的 breakdown 鍵名
_USAGE_KEYS = {"input_tokens": "input",
               "output_tokens": "output",
               "cache_creation_input_tokens": "cache_creation",
               "cache_read_input_tokens": "cache_read"}


def estimate_tokens(prompt: str, out: str) -> int:
    """
    拿不到真實 usage 時的退路。

    舊版用 //3，那是英文的比例（約 4 字元/token）。這些 prompt 是中英混雜且
    中文佔多數（經驗庫、假設、歸因全是中文），中文約 1~1.5 字元/token，
    //3 會低估 2~3 倍。改用 //2 折衷——但這仍然只是估算，
    真正準確的數字請讓 llm.output_format = json 生效。
    """
    return (len(prompt) + len(out)) // 2


def dump_raw(kind: str, text: str) -> Path:
    """把一段原始輸出落盤，回傳路徑。用於逾時/解析失敗的事後搶救。"""
    d = ROOT / CFG["paths"]["memory_dir"] / "llm_raw"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kind}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _envelope_error(stdout: str) -> str:
    """--output-format json 失敗時，錯誤訊息被包在信封的 result 欄位裡。"""
    try:
        env = json.loads(stdout)
        if isinstance(env, dict):
            return str(env.get("result") or env.get("error") or stdout)
    except json.JSONDecodeError:
        pass
    return stdout


def _parse_envelope(stdout: str, prompt: str) -> LLMResult:
    """解析 --output-format json 的外層信封；解析不了就退回純文字 + 估算。"""
    try:
        env = json.loads(stdout)
    except json.JSONDecodeError:
        env = None
    if not isinstance(env, dict) or "result" not in env:
        # CLI 版本太舊或格式改變——不要因此讓整輪掛掉，退回舊行為就好
        return LLMResult(text=stdout, tokens=estimate_tokens(prompt, stdout))
    text = str(env.get("result") or "")
    usage = env.get("usage") or {}
    bd = {v: int(usage.get(k) or 0) for k, v in _USAGE_KEYS.items()}
    total = sum(bd.values())
    cost = float(env.get("total_cost_usd") or 0.0)
    if total <= 0:
        return LLMResult(text=text, tokens=estimate_tokens(prompt, text),
                         cost_usd=cost, breakdown=bd)
    return LLMResult(text=text, tokens=total, cost_usd=cost,
                     breakdown=bd, measured=True)


def call_llm(prompt: str, mock_fn=None, meter: "Meter | None" = None,
             timeout: float | None = None) -> LLMResult:
    """
    呼叫 CLI。meter 傳進來的話，用量會在「成功或逾時」兩種情況下都被記入
    ——逾時代表請求已經送出、token 已經計費，只是拿不到 usage 回報。
    """
    if timeout is None:
        timeout = float(CFG["llm"].get("timeout_sec", 1200))
    if mock_fn:
        out = mock_fn(prompt)
        res = LLMResult(text=out, tokens=estimate_tokens(prompt, out))
        return meter.add(res) if meter is not None else res
    cmd = CFG["llm"]["command"].split()
    fmt = str(CFG["llm"].get("output_format", "json")).lower()
    if fmt == "json" and "--output-format" not in cmd:
        cmd += ["--output-format", "json"]
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(
            f"找不到 {cmd[0]} CLI。請確認 Claude Code 已安裝且在 PATH 中"
            f"（cmd 執行 `claude --version` 檢查）。")
    # Windows：npm 安裝的 CLI 是 .cmd 批次檔，CreateProcess 無法直接執行
    if exe.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", exe] + cmd[1:]
    else:
        cmd = [exe] + cmd[1:]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        el = time.monotonic() - t0
        # 逾時 ≠ 沒花錢：請求早就送到伺服器了，只是我們沒等到回應。
        # 拿不到真實 usage，用估算值記帳，總比記 0 誠實。
        if meter is not None:
            meter.add(LLMResult(text="", tokens=estimate_tokens(prompt, ""),
                                elapsed=el, timed_out=True))
        # 搶救：subprocess 會把「被砍掉之前已寫出的 stdout」放進例外。
        # claude -p --output-format json 是結尾才一次吐出信封，所以通常是空的，
        # 但萬一 CLI 有先寫東西，留著總比丟掉好。
        partial = e.stdout or b""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        saved = dump_raw("timeout", partial) if partial.strip() else None
        raise RuntimeError(
            f"LLM 呼叫逾時（{timeout:.0f}s，prompt {len(prompt):,} 字元）。"
            f"這通的 token 已經花掉了（以估算值計入預算）。"
            + (f"逾時前收到的片段已存於 {saved}。" if saved
               else "逾時前沒有收到任何輸出（CLI 是結尾才一次回傳）。")
            + "請調高 config.yaml 的 llm.timeout_sec / "
              "llm.consolidate_timeout_sec，或縮短 prompt。") from e
    el = time.monotonic() - t0
    if r.returncode != 0:
        msg = (r.stderr or "").strip() or _envelope_error(r.stdout)
        raise RuntimeError(f"LLM 呼叫失敗: {msg[:500]}")
    if fmt == "json":
        res = _parse_envelope(r.stdout, prompt)
    else:
        res = LLMResult(text=r.stdout, tokens=estimate_tokens(prompt, r.stdout))
    res.elapsed = el
    if el > timeout * 0.8:
        log(f"  ⚠️ 本通耗時 {el:.0f}s，已達 timeout（{timeout:.0f}s）的 "
            f"{el / timeout:.0%}——建議調高上限或縮短 prompt")
    return meter.add(res) if meter is not None else res


class Meter:
    """
    一段工作期間的用量計數器。

    關鍵設計：Meter 由呼叫端建立、傳進去用。就算中途拋例外，已經花掉的量
    仍然留在呼叫端手上的這個物件裡——這就是「失敗的呼叫也要計帳」的作法。
    """

    def __init__(self):
        self.tokens = 0
        self.cost = 0.0
        self.calls = 0
        self.estimated_calls = 0
        self.timeouts = 0
        self.elapsed = 0.0
        self.breakdown = dict(EMPTY_BREAKDOWN)

    def add(self, r: LLMResult) -> LLMResult:
        self.tokens += r.tokens
        self.cost += r.cost_usd
        self.calls += 1
        self.elapsed += r.elapsed
        if r.timed_out:
            self.timeouts += 1
        if not r.measured:
            self.estimated_calls += 1
        for k, v in (r.breakdown or {}).items():
            self.breakdown[k] = self.breakdown.get(k, 0) + v
        return r

    def __str__(self):
        b = self.breakdown
        tail = f"，{self.estimated_calls} 通為估算" if self.estimated_calls else ""
        tail += f"，{self.timeouts} 通逾時" if self.timeouts else ""
        return (f"{self.tokens:,} tokens（in {b['input']:,} / out {b['output']:,}"
                f" / cache 建立 {b['cache_creation']:,}、讀取 {b['cache_read']:,}）"
                f"，${self.cost:.4f}，{self.calls} 通呼叫、耗時 {self.elapsed:.0f}s{tail}")


def parse_json_array(text: str) -> list:
    """從 LLM 輸出中抽取第一個 JSON 陣列（容忍圍欄與前後雜訊）。"""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("[")
    if start < 0:
        raise ValueError("輸出中找不到 JSON 陣列")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 陣列未閉合")


def llm_json(prompt: str, meter: Meter, mock_fn=None, label="LLM") -> list:
    """
    呼叫 + 解析，失敗重試一次（附錯誤訊息）。

    meter.add() 刻意放在解析「之前」——重試的那一通、以及最後放棄的那一通，
    token 都已經花掉了，一律要計入。

    呼叫前後都會 log：這幾通要跑 3~8 分鐘，中間完全沒有輸出的話終端機看起來
    像當掉了。有進度行才分得出「還在跑」和「真的卡住」。
    """
    to = float(CFG["llm"].get("timeout_sec", 1200))
    for attempt in range(1 + CFG["llm"]["retry_limit"]):
        log(f"  {label}: prompt {len(prompt):,} 字元，timeout {to:.0f}s，呼叫中"
            + (f"（第 {attempt + 1} 次）" if attempt else "") + "...")
        r = call_llm(prompt, mock_fn, meter=meter)
        log(f"  {label}: 回應 {len(r.text):,} 字元，耗時 {r.elapsed:.0f}s，"
            f"{r.tokens:,} tokens / ${r.cost_usd:.4f}")
        try:
            return parse_json_array(r.text)
        except (ValueError, json.JSONDecodeError) as e:
            log(f"  ⚠️ JSON 解析失敗（第 {attempt + 1} 次）: {e}")
            prompt = (prompt + "\n\n上次輸出無法解析為 JSON 陣列"
                      f"（錯誤：{e}）。請只輸出合法 JSON 陣列，不要任何其他文字。")
    raise RuntimeError("JSON 解析重試後仍失敗，本輪中止")


# ---------------------------------------------------------------------------
# Prompt 組裝
# ---------------------------------------------------------------------------

def fields_card() -> str:
    fy = yaml.safe_load((ROOT / "fields.yaml").read_text(encoding="utf-8"))
    lines = []
    for grp, flds in fy.items():
        for name, meta in flds.items():
            lines.append(f"- {name}: {meta['desc']}")
    return "\n".join(lines)


def build_generate_prompt(mem: Memory) -> str:
    tpl = (PROMPTS / "generate.md").read_text(encoding="utf-8")
    n = BUD["candidates_per_round"]
    return tpl.format(
        n=n,
        n_explore=max(3, n // 5),
        dsl_card=dsl.operators_card(),
        max_depth=CFG["dsl"]["max_depth"],
        max_fields=CFG["dsl"]["max_fields"],
        windows=CFG["dsl"]["window_whitelist"],
        fields_card=fields_card(),
        learnings=mem.read_learnings(),
        library=mem.library_summary(),
    )


def build_distill_prompt(cands: list[dict], diags: list[dict]) -> str:
    tpl = (PROMPTS / "distill.md").read_text(encoding="utf-8")
    dmap = {d["id"]: d for d in diags}
    pairs = []
    for c in cands:
        pairs.append({
            "candidate": {k: c.get(k) for k in
                          ("id", "category", "direction", "hypothesis",
                           "prediction", "formula", "name_zh")},
            "diagnostic": dmap.get(c["id"], {"verdict": "missing"}),
        })
    return tpl.format(pairs=json.dumps(pairs, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# 一輪
# ---------------------------------------------------------------------------

REQUIRED_CAND_KEYS = ("id", "category", "direction", "hypothesis",
                      "prediction", "formula", "name_zh", "desc_zh")

# learnings.md 只允許這三個小節。Distill 的 LLM 可能回傳任意 category 名當
# section（prompts/distill.md 原本就寫著「或 category名」），而
# memory.append_learning() 找不到標題就會在檔尾新建一個——長期會長出一堆
# 只有一兩條的孤兒小節。非白名單一律歸入「全域規則」。
ALLOWED_SECTIONS = ("全域規則", "禁忌方向", "待驗證假設佇列")
MAX_LEARNINGS_PER_ROUND = 3
MAX_QUEUE_PER_ROUND = 3      # 佇列原本完全沒有上限，是經驗庫膨脹的主因
QID_PATTERN = re.compile(r"Q-\d{3,}")   # Generate 回報的 from_queue 格式


def resolve_section(sec: str, existing: set) -> str | None:
    """
    把 LLM 給的 section 名對應到可用的小節；None = 不允許（歸入全域規則）。

    允許兩類：白名單三節，以及 learnings.md 裡「已經存在」的小節——整理回合
    會自行重組出像「紅海地圖」「聚合審計」這種有用的小節，不該把它們擋掉；
    真正要擋的是憑空生出一個只有一兩條的孤兒小節。
    標題可能帶後綴（「待驗證假設佇列（優先級由高到低）」），故用前綴互相比對。
    """
    sec = (sec or "").strip()
    if not sec:
        return None
    if sec in ALLOWED_SECTIONS:
        return sec
    for e in existing:
        if e.startswith(sec) or sec.startswith(e):
            return sec       # append_learning 用子字串比對，傳原名即可命中
    return None


def run_round(ctx: ec.Context, mem: Memory, round_id: int,
              meter: Meter, mock_fn=None) -> dict:
    # 1) Generate
    gen_prompt = build_generate_prompt(mem)
    raw_cands = llm_json(gen_prompt, meter, mock_fn, label="Generate")
    cands = []
    for c in raw_cands:
        if all(c.get(k) for k in REQUIRED_CAND_KEYS):
            cands.append(c)
        else:
            log(f"  ⚠️ 候選欄位不全，略過: {str(c)[:80]}")
    log(f"  Generate: {len(cands)} 個有效候選")
    if not cands:
        raise RuntimeError("本輪無有效候選")

    # 2) Evaluate（本地）
    diags = ec.evaluate_batch(cands, ctx)
    verdicts = {}
    for d in diags:
        verdicts[d["verdict"]] = verdicts.get(d["verdict"], 0) + 1
    log(f"  Evaluate: {verdicts}")

    # 3) Distill
    # ⚠️ Distill 失敗不該賠掉整輪。
    #    走到這裡，Generate 的 token 已經花掉、本地漏斗（Stage 1~4）也跑完了，
    #    而步驟 4（attempts 落盤）與步驟 5（入庫）都「不需要」Distill 的輸出——
    #    步驟 4 本來就用 dist_map.get(id, {}) 容錯，入庫只看 cands + diags。
    #    真正會缺席的只有歸因/lesson/佇列/DSL 訊號。
    #    所以這裡改成降級繼續，而不是把通過漏斗的因子一起丟掉。
    dist_prompt = build_distill_prompt(cands, diags)
    distill_ok = True
    try:
        dist = llm_json(dist_prompt, meter, mock_fn, label="Distill")
        dist = [d for d in dist if isinstance(d, dict) and d.get("id")]
    except Exception as e:
        distill_ok, dist = False, []
        log(f"  ⚠️ Distill 失敗（{e}）")
        log(f"     → 降級繼續：{len(cands)} 筆 attempts 與入庫照常，"
            f"但本輪沒有歸因、lesson 與佇列建議")
    dist_map = {d["id"]: d for d in dist}

    # 4) attempts 落盤（每個候選一筆，含歸因）
    dmap = {d["id"]: d for d in diags}
    cand_attempt = {}
    for c in cands:
        d = dmap[c["id"]]
        z = dist_map.get(c["id"], {})
        try:
            pf = dsl.parse(c["formula"], allowed_fields=ctx.fields)
            cx = {"depth": pf.depth, "fields": sorted(pf.fields)}
        except dsl.DSLError:
            cx = {"depth": None, "fields": None}
        aid = mem.write_attempt({
            "round": round_id,
            "category": c["category"],
            "hypothesis": c["hypothesis"],
            "prediction": c["prediction"],
            "formula": c["formula"],
            "name_zh": c.get("name_zh"),
            "direction": c.get("direction"),
            "complexity": cx,
            "result": {k: v for k, v in d.items() if k != "id"},
            "diagnosis": z.get("diagnosis", ""),
            "failure_type": z.get("failure_type", ""),
            "lesson": z.get("lesson", ""),
            "lesson_confidence": z.get("lesson_confidence", "single_case"),
            "verdict": d["verdict"],
            # 這個候選是來消化哪一條佇列項目（Generate 回報），供消費機制使用
            **({"from_queue": c["from_queue"]} if c.get("from_queue") else {}),
            # 標記出來，整理回合與事後查帳才知道這筆為何沒有歸因
            **({} if distill_ok else {"distill_failed": True}),
        })
        cand_attempt[c["id"]] = aid

    # 4b) 佇列消費：Generate 回報 from_queue 的項目，不論成敗都算「已試過」
    #     ——佇列是待辦清單不是願望清單，試過就該移出，否則每輪 Generate
    #     都要重讀一次已經試過的東西。全文與結果落到 queue_consumed.jsonl，
    #     資訊不遺失（比照 attempts 只進不出的原則）。
    known_q = {q for q, _ in mem.queue_items()}
    consumed = []
    for c in cands:
        qid = str(c.get("from_queue") or "").strip().upper()
        if not qid or not QID_PATTERN.fullmatch(qid) or qid not in known_q:
            continue
        if mem.consume_queue_item(qid, {
                "round": round_id,
                "attempt": cand_attempt[c["id"]],
                "formula": c["formula"],
                "verdict": dmap[c["id"]]["verdict"]}):
            consumed.append(qid)
            known_q.discard(qid)
    if consumed:
        log(f"  佇列消費：{len(consumed)} 條已試過並移出（{', '.join(consumed)}）")

    # 5) 入庫
    for c in cands:
        c["attempt_id"] = cand_attempt[c["id"]]
    admitted = admit_from_results(cands, diags, ctx=ctx, mem=mem,
                                  round_id=round_id)
    # 入庫後刷新 ctx 的因子庫快取（下一輪 Stage 2 要看得到）
    if admitted:
        ctx.reload_library()

    # 6) learnings 追加（全輪上限：經驗 3 條、佇列 3 條）
    #    容錯：LLM 可能輸出字串而非 {section, line} 物件，一律接住不炸輪
    added = 0
    queued = 0
    queue_full = False
    dropped_sections = set()
    existing_sections = {m.group(1).strip() for m in
                         re.finditer(r"(?m)^## (.+)$", mem.read_learnings())}
    for z in dist:
        for item in z.get("learnings_additions") or []:
            if added >= MAX_LEARNINGS_PER_ROUND:
                break
            if isinstance(item, dict):
                sec = item.get("section") or "全域規則"
                line = str(item.get("line") or "").strip()
            else:
                sec, line = "全域規則", str(item).strip()
            resolved = resolve_section(sec, existing_sections)
            if resolved is None:
                dropped_sections.add(sec)
                sec = "全域規則"
            else:
                sec = resolved
            if not line.startswith("-"):
                line = f"- {line}" if line else ""
            if line:
                aid = cand_attempt.get(z["id"], "")
                if aid and aid not in line:
                    line = f"{line}（attempt: {aid}）"
                mem.append_learning(sec, line)
                added += 1
        for q in z.get("queue_suggestions") or []:
            if queued >= MAX_QUEUE_PER_ROUND:
                break
            q = str(q if not isinstance(q, dict) else q.get("line", "")).strip()
            if q:
                if mem.append_learning("待驗證假設佇列",
                                       q if q.startswith("-") else f"- {q}"):
                    queued += 1
                else:
                    queue_full = True
    if dropped_sections:
        log(f"  ⚠️ 未知小節已歸入「全域規則」: {sorted(dropped_sections)}")
    if queue_full:
        log("  ⚠️ 待驗證假設佇列已達上限，本輪建議未寫入（等整理回合精簡）")

    # 7) DSL 表達力受限訊號（整理回合的運算子提案素材）
    # 用 mem.root 而不是 ROOT/memory_dir：consolidate.load_limitations() 是從
    # mem.root 讀的，寫、讀兩邊必須指向同一個檔案（傳入自訂 root 的 Memory 時
    # 舊寫法會寫到別的地方，讀出來永遠是空的）。順帶確保目錄存在。
    lim_path = mem.root / "dsl_limitations.jsonl"
    lim_path.parent.mkdir(parents=True, exist_ok=True)
    with lim_path.open("a", encoding="utf-8") as f:
        for z in dist:
            note = str(z.get("dsl_limitation") or "").strip()
            if note and note.lower() not in ("null", "none", "省略"):
                f.write(json.dumps(
                    {"attempt": cand_attempt.get(z["id"], ""), "note": note},
                    ensure_ascii=False) + "\n")

    log(f"  Distill{'' if distill_ok else '（失敗，降級）'}: "
        f"attempts {len(cands)} 筆、入庫 {len(admitted)}、"
        f"lesson +{added}、queue +{queued}")
    return {"n_cands": len(cands), "verdicts": verdicts, "admitted": admitted,
            "distill_ok": distill_ok}


# ---------------------------------------------------------------------------
# Mock LLM（測試管線用；固定劇本，不需 claude CLI）
# ---------------------------------------------------------------------------

def make_mock():
    state = {"calls": 0}

    def mock(prompt: str) -> str:
        state["calls"] += 1
        if "本輪候選與診斷" in prompt:   # Distill
            seg = prompt.split("# 本輪候選與診斷", 1)[1]
            pairs = parse_json_array(seg)
            out = []
            for p in pairs:
                cid = p["candidate"]["id"]
                v = p["diagnostic"].get("verdict", "?")
                ft = "none" if v == "passed" else (
                    "duplicate" if "stage2" in v or "duplicate" in v
                    else "expression_bad" if v in ("rejected_syntax", "rejected_stage3")
                    else "hypothesis_wrong")
                out.append({"id": cid,
                            "diagnosis": f"mock 歸因（verdict={v}）",
                            "failure_type": ft,
                            "lesson": f"mock lesson for {cid}",
                            "lesson_confidence": "single_case",
                            "learnings_additions":
                                ([{"section": "全域規則",
                                   "line": f"- [MOCK|single_case] {cid} 測試經驗。"}]
                                 if cid == "C-1" else []),
                            "queue_suggestions": []})
            return json.dumps(out, ensure_ascii=False)
        # Generate：混合合法/違規/重複的劇本
        cands = [
            {"id": "C-1", "category": "revenue_momentum", "direction": "pos",
             "hypothesis": "營收資訊擴散慢", "prediction": "ICIR>0.4",
             "formula": "cs_rank(rev_yoy_ma3 + rev_yoy)",
             "name_zh": "營收動能複合", "desc_zh": "營收年增與其均線之和的產業內排名"},
            {"id": "C-2", "category": "quality_trend", "direction": "pos",
             "hypothesis": "毛利率改善反映定價權，市場低估其持續性",
             "prediction": "ICIR>0.35",
             "formula": "cs_rank(delta(gross_margin, 12))",
             "name_zh": "毛利率動能", "desc_zh": "毛利率年變化的產業內排名"},
            {"id": "C-3", "category": "bad_syntax", "direction": "pos",
             "hypothesis": "違規測試", "prediction": "-",
             "formula": "ts_mean(rev_yoy, 17)",
             "name_zh": "違規窗口", "desc_zh": "應死於語法檢查"},
            {"id": "C-4", "category": "value", "direction": "pos",
             "hypothesis": "高盈餘殖利率的均值回歸", "prediction": "ICIR>0.3",
             "formula": "cs_rank(ep)",
             "name_zh": "盈餘殖利率", "desc_zh": "EPS/股價的產業內排名"},
        ]
        return json.dumps(cands, ensure_ascii=False)

    return mock


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--budget", action="store_true",
                    help="只印出本週預算用量後結束，不呼叫 LLM")
    a = ap.parse_args()

    if a.budget:
        print(budget_report(load_budget()))
        return

    mem = Memory()
    mem.ensure()

    if a.dry_run:
        print(build_generate_prompt(mem))
        return

    mock_fn = make_mock() if a.mock else None
    log(f"===== mining_loop 啟動（rounds={a.rounds}, mock={a.mock}）=====")
    ctx = ec.Context()

    for _ in range(a.rounds):
        b = load_budget()
        # 預算檢查在「開新輪之前」——已經在跑的那一輪一定會跑完並存檔，
        # 不會半途中斷讓資料處於不一致狀態。
        stop = budget_stop_reason(b)
        if stop and not a.mock:
            log(f"⛔ {stop}，下週一自動重置。")
            break
        round_id = b["total_rounds"] + 1
        log(f"--- Round {round_id}（本週第 {b['rounds_used'] + 1} 輪）---")

        meter = Meter()
        ok, res = True, {}
        try:
            res = run_round(ctx, mem, round_id, meter, mock_fn)
        except Exception as e:
            ok = False
            log(f"  ❌ 本輪失敗: {e}")

        # ⚠️ 記帳與成敗無關：token 在 call_llm 回傳的當下就已經花掉了。
        #    失敗的輪次不佔 rounds_used（維持原設計），但用量一定要進帳。
        b = load_budget()
        commit_usage(b, meter)
        if ok:
            b["rounds_used"] += 1
            b["total_rounds"] = round_id
        else:
            b["failed_calls"] = b.get("failed_calls", 0) + meter.calls
        save_budget(b)
        state = ("完成" if res.get("distill_ok", True) else "完成（Distill 降級）") \
            if ok else "中止（用量已計入）"
        log(f"  {state}。本輪 {meter}")
        log(f"  本週累計 {b['tokens_used']:,} tokens / ${b['cost_usd']:.4f}"
            f"（輪次 {b['rounds_used']}/{BUD['weekly_round_budget']}）")
        if not ok:
            break

        # 整理觸發：固定輪次，或經驗庫已經超過 token cap 的兩倍。
        # ⚠️ 這個檢查必須放在「一輪完整結束後」，不能塞進 memory.read_learnings()：
        #    consolidate.run() 和 append_learning() 內部都會呼叫 read_learnings()，
        #    在那裡觸發整理會無限遞迴。
        by_round = round_id % BUD["consolidation_every_n_rounds"] == 0
        learnings_chars = len(mem.read_learnings())
        # 超長觸發只是安全網，門檻要遠高於整理回合的產出目標
        # （整理後約 5,500 字元、每輪增長約 1,000 字元；若門檻設在 8,000，
        #   等於整理完 3 輪就又觸發一次，8/19 實測就是這樣連燒三輪）
        hard_cap = BUD.get("learnings_hard_cap_chars") \
            or BUD["learnings_token_cap"] * 6
        by_size = learnings_chars > hard_cap
        # 整理失敗時 by_size 條件不會消失，會每輪重試——加最小間隔擋住重試風暴
        gap = BUD.get("consolidation_min_gap_rounds", 3)
        since = round_id - b.get("last_consolidate_round", 0)
        if by_size and not by_round and since < gap:
            log(f"  （經驗庫 {learnings_chars} 字元已超過 {hard_cap}，"
                f"但距上次整理僅 {since} 輪 < {gap}，本輪不重試）")
            by_size = False
        if by_round or by_size:
            why = "輪次" if by_round else f"經驗庫過長({learnings_chars}字元)"
            log(f"  📋 觸發整理回合（{why}，round={round_id}）...")
            b = load_budget()
            b["last_consolidate_round"] = round_id   # 不論成敗都記，避免重試風暴
            save_budget(b)
            cmeter = Meter()
            try:
                import consolidate
                cres = consolidate.run(mem=mem, meter=cmeter, mock_fn=mock_fn)
                # 整理會全文重寫，新增的佇列條目通常沒有 [Q-xxx]，補回去
                # 否則下一輪的消費機制認不得它們
                if (added := mem.normalize_queue_ids()):
                    log(f"  佇列補編號：{added} 條")
                log(f"  整理完成：{cres['notes']}")
            except Exception as e:
                log(f"  ⚠️ 整理回合失敗（不影響挖礦）: {e}")
                b = load_budget()
                b["failed_calls"] = b.get("failed_calls", 0) + cmeter.calls
                save_budget(b)
            finally:
                # 整理成功或失敗都要計帳——這裡是舊版漏記的地方
                b = load_budget()
                commit_usage(b, cmeter)
                save_budget(b)
                if cmeter.calls:
                    log(f"  整理用量 {cmeter}；本週累計 "
                        f"{b['tokens_used']:,} tokens / ${b['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
