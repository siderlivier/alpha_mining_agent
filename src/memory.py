"""
M2：記憶層（規格書第 7 章）。

兩層結構：
  memory/attempts/A-XXXX.json   逐筆嘗試紀錄，只進不出、永不刪改
  memory/library.json           入庫因子中繼資料（含密封 test 指標）
  memory/factor_values.parquet  入庫因子月頻值（Stage 2 去相關用，long 格式）
  memory/learnings.md           蒸餾經驗（唯一注入 prompt 的記憶）

因子入庫欄位（使用者要求）：
  id / created_at / round 之外，必含 name_zh（中文名稱）、desc_zh（一句話解釋）、
  aspect（技術面/基本面/籌碼面/混合，由公式用到的欄位自動推導）、category。

⛔ 洩漏紀律：test_metrics_sealed 只寫入 library.json 供人類查閱，
   任何回傳給 LLM 的摘要（library_summary）都不得包含它。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# fields.yaml 群組 → 面向（新增資料源時在此擴充，例如籌碼面欄位群）
ASPECT_MAP = {
    "price_volume": "技術面",
    "quality": "基本面",
    "growth": "基本面",
    "value": "基本面",
    "chips": "籌碼面",
}

LEARNINGS_SKELETON = """\
# 因子挖掘經驗庫

（本檔由 agent 於 Distill / 整理回合維護，人類可隨時修訂。
 每條經驗須附證據編號（attempt id）與置信等級
 [single_case|corroborated|promoted]，被推翻的規則標記 [失效] 而非刪除。）

## 全域規則

（尚無）

## 禁忌方向

（尚無）

## 待驗證假設佇列

（尚無）
"""

ATTEMPT_REQUIRED = ("category", "hypothesis", "prediction", "formula", "verdict")

# 「待驗證假設佇列」的硬上限（安全網，見 append_learning 的說明）。
# 小節標題可能帶後綴（例如「待驗證假設佇列（優先級由高到低）」），故用 startswith 比對。
# 參考因子（前置專案匯入的已知因子）的 id 前綴。
# 它們與 agent 自己挖的 F-xxx 同住 library.json，靠這個前綴與 reference 旗標區分。
REFERENCE_PREFIX = "R-"

QUEUE_SECTION = "待驗證假設佇列"
QUEUE_MAX_ITEMS = 50

# 清單項目：'- x'、'* x'、'+ x'、'1. x'、'2) x' 都算一條
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
# 同上但不要求後面有內容，用來取「項目符號結束的位置」以便插入 [Q-xxx]
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# 佇列條目的編號標記 [Q-042]
QID_RE = re.compile(r"\[Q-(\d{3,})\]")


def field_aspects() -> dict:
    """欄位名 -> 面向。"""
    fy = yaml.safe_load((ROOT / "fields.yaml").read_text(encoding="utf-8"))
    out = {}
    for grp, flds in fy.items():
        for f in flds:
            out[f] = ASPECT_MAP.get(grp, "其他")
    return out


def derive_aspect(fields_used, fmap=None) -> str:
    """由公式用到的欄位自動推導面向；跨面向 → 混合。"""
    fmap = fmap or field_aspects()
    aspects = {fmap.get(f, "其他") for f in fields_used}
    return aspects.pop() if len(aspects) == 1 else "混合"


class Memory:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else ROOT / "memory"
        self.attempts_dir = self.root / "attempts"
        self.lib_path = self.root / "library.json"
        self.values_path = self.root / "factor_values.parquet"
        self.learnings_path = self.root / "learnings.md"

    # ---- 初始化 -----------------------------------------------------------
    def ensure(self):
        """
        建立必要檔案，並確保佇列條目都有 [Q-xxx] 編號。

        補編號放在這裡而不是只在整理之後：整理回合會全文重寫 learnings.md，
        重寫出來的佇列沒有編號；人工編輯 learnings.md 時也不會自己加。
        沒有編號的條目對消費機制是隱形的——Generate 無從回報 from_queue，
        於是那條永遠不會被移出，每輪都重讀。
        （實測：33 條佇列中有 23 條因此沒有編號。）
        ensure() 是所有進入點（mining_loop / consolidate / unblock / report）
        都會呼叫的地方，而 normalize_queue_ids() 是冪等的，放這裡最保險。
        """
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        if not self.lib_path.exists():
            self.lib_path.write_text("{}", encoding="utf-8")
        if not self.learnings_path.exists():
            self.learnings_path.write_text(LEARNINGS_SKELETON, encoding="utf-8")
        self.normalize_queue_ids()

    # ---- 嘗試紀錄（只進不出）---------------------------------------------
    def next_attempt_id(self) -> str:
        nums = [int(m.group(1)) for p in self.attempts_dir.glob("A-*.json")
                if (m := re.match(r"A-(\d+)$", p.stem))]
        return f"A-{(max(nums) + 1 if nums else 1):04d}"

    def write_attempt(self, rec: dict) -> str:
        missing = [k for k in ATTEMPT_REQUIRED if k not in rec]
        if missing:
            raise ValueError(f"attempt 缺少必要欄位: {missing}")
        rec.setdefault("id", self.next_attempt_id())
        rec.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        rec.setdefault("lesson_confidence", "single_case")
        path = self.attempts_dir / f"{rec['id']}.json"
        if path.exists():
            raise FileExistsError(f"{rec['id']} 已存在（attempts 只進不出，禁止覆寫）")
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return rec["id"]

    def load_attempts(self) -> list[dict]:
        out = []
        for p in sorted(self.attempts_dir.glob("A-*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec.get("voided") or "id" not in rec:
                continue   # 人工作廢的紀錄（檔案保留佔位，id 不重用）
            out.append(rec)
        return out

    # ---- 因子庫 -----------------------------------------------------------
    def library(self) -> dict:
        """全部因子，含前置專案匯入的參考因子（R-xxx）。"""
        if not self.lib_path.exists():
            return {}
        return json.loads(self.lib_path.read_text(encoding="utf-8"))

    def own_library(self) -> dict:
        """
        只有 agent 自己挖出來的因子（F-xxx）。

        report / audit / 統計一律用這個——參考因子是「已知的既有因子」，
        算進 agent 的成果會虛灌數字，也會讓過擬合審計的樣本被污染。
        """
        return {k: v for k, v in self.library().items()
                if not v.get("reference") and not k.startswith(REFERENCE_PREFIX)}

    def reference_library(self) -> dict:
        return {k: v for k, v in self.library().items()
                if v.get("reference") or k.startswith(REFERENCE_PREFIX)}

    def _save_library(self, lib: dict):
        self.lib_path.write_text(json.dumps(lib, ensure_ascii=False, indent=1),
                                 encoding="utf-8")

    def next_factor_id(self) -> str:
        nums = [int(m.group(1)) for k in self.library()
                if (m := re.match(r"F-(\d+)$", k))]
        return f"F-{(max(nums) + 1 if nums else 1):03d}"

    def admit(self, cand: dict, diag: dict, parsed, fac: pd.DataFrame,
              test_metrics_sealed: dict | None = None,
              round_id: int | None = None,
              industry_scope: str | None = None) -> str:
        """
        入庫一個 passed / passed_industry 因子。
        cand 必含 name_zh / desc_zh（使用者要求：中文名稱與簡單解釋）。
        面向由公式欄位自動推導；industry_scope 非 None 表示產業限定因子。
        """
        for k in ("name_zh", "desc_zh"):
            if not cand.get(k):
                raise ValueError(f"入庫需要 {k}（中文名稱與解釋）")
        if diag.get("verdict") not in ("passed", "passed_industry"):
            raise ValueError(f"只有 passed 候選可入庫（{cand['id']}: {diag.get('verdict')}）")

        lib = self.library()
        if any(m.get("fhash") == parsed.fhash for m in lib.values()):
            raise ValueError(f"等價因子已在庫中（fhash={parsed.fhash}）")

        fid = self.next_factor_id()
        lib[fid] = {
            "name_zh": cand["name_zh"],
            "desc_zh": cand["desc_zh"],
            "aspect": derive_aspect(parsed.fields),
            "category": cand.get("category", ""),
            "formula": cand["formula"],
            "fhash": parsed.fhash,
            "fields": sorted(parsed.fields),
            "depth": parsed.depth,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "round": round_id,
            "industry_scope": industry_scope,      # None = 全池
            "industry_metrics": diag.get("industry_metrics"),
            "source_attempt": cand.get("attempt_id"),
            "sub_train": diag.get("sub_train"),
            "validation": diag.get("validation"),
            "coverage": diag.get("coverage"),
            "turnover_m": diag.get("turnover_m"),
            # ⛔ 密封欄位：僅供人類查閱，任何 prompt 組裝不得引用
            "test_metrics_sealed": test_metrics_sealed,
        }
        self._save_library(lib)

        # 因子值入庫（long 格式，Stage 2 去相關的資料源）
        long = (fac.stack().rename("value").reset_index())
        long.columns = ["ym", "stock_id", "value"]
        long.insert(0, "factor_id", fid)
        if self.values_path.exists():
            old = pd.read_parquet(self.values_path)
            if len(old):
                long = pd.concat([old, long], ignore_index=True)
        long.to_parquet(self.values_path, index=False)
        return fid

    def library_summary(self) -> str:
        """
        注入 Generate prompt 的因子庫一覽。
        規格 8.3：只給 id/中文名/面向/category/公式/一句話解釋，
        「不含」任何績效數字（防趨附）、「絕不含」密封 test 指標。
        """
        own, ref = self.own_library(), self.reference_library()
        if not own and not ref:
            return "（因子庫目前為空）"
        lines = []
        if own:
            for fid, m in own.items():
                scope = m.get("industry_scope")
                tag = f"/{scope}限定" if scope else ""
                lines.append(
                    f"- {fid}【{m['name_zh']}】({m['aspect']}/{m['category']}{tag}) "
                    f"`{m['formula']}` — {m['desc_zh']}")
        if ref:
            # ⚠️ 參考因子刻意「不給公式」：它們是前置專案用原始財報欄位算的
            #    （例如 gp_to_px = 毛利/股價），本 DSL 根本沒有那些欄位，
            #    給了只會誘使模型拼出無效公式（rejected_syntax）。
            #    列出來的目的是「別再提相近變體」，不是讓它照抄。
            lines.append("")
            lines.append("【已知因子｜前置專案已挖出，提相近變體會死於 Stage 2 相關性檢查】")
            lines.append("（這些不是本 DSL 的公式，無法也不需要直接重現；"
                         "只需避開同一個訊號來源）")
            for fid, m in ref.items():
                lines.append(f"- {fid}【{m['name_zh']}】({m['aspect']}) — {m['desc_zh']}")
        return "\n".join(lines)

    # ---- 經驗檔 -----------------------------------------------------------
    def read_learnings(self) -> str:
        return (self.learnings_path.read_text(encoding="utf-8")
                if self.learnings_path.exists() else LEARNINGS_SKELETON)

    def write_learnings(self, text: str):
        self.learnings_path.write_text(text, encoding="utf-8")

    # ---- 待驗證假設佇列：編號、列舉、消費 ---------------------------------
    def _queue_header(self, text: str | None = None) -> str | None:
        """回傳檔案裡實際的佇列標題（可能帶後綴），沒有則 None。"""
        text = self.read_learnings() if text is None else text
        for m in re.finditer(r"(?m)^## (.+)$", text):
            if m.group(1).strip().startswith(QUEUE_SECTION):
                return m.group(1).strip()
        return None

    def _ensure_qid(self, line: str) -> str:
        """條目沒有 [Q-xxx] 就補一個（取目前最大號 +1）。"""
        if QID_RE.search(line):
            return line
        nums = [int(m.group(1)) for m in QID_RE.finditer(self.read_learnings())]
        qid = f"Q-{(max(nums) + 1 if nums else 1):03d}"
        body = LIST_PREFIX_RE.match(line)
        if body:
            return f"{line[:body.end()]}[{qid}] {line[body.end():]}"
        return f"- [{qid}] {line.lstrip('- ').strip()}"

    def queue_items(self) -> list[tuple[str, str]]:
        """回傳佇列中的 [(Q-xxx, 條目全文), ...]；沒編號的不列入。"""
        hdr = self._queue_header()
        if hdr is None:
            return []
        seg = self.read_learnings().partition(f"## {hdr}")[2].partition("\n## ")[0]
        out = []
        for l in seg.splitlines():
            m = QID_RE.search(l)
            if m and LIST_ITEM_RE.match(l):
                out.append((f"Q-{m.group(1)}", l.strip()))
        return out

    def normalize_queue_ids(self) -> int:
        """
        給佇列中所有還沒有 [Q-xxx] 的條目補上編號，回傳補了幾條。
        整理回合會全文重寫 learnings.md，新增的條目常常沒有編號——
        跑完整理後呼叫這個，佇列的消費機制才不會漏掉它們。
        """
        text = self.read_learnings()
        hdr = self._queue_header(text)
        if hdr is None:
            return 0
        head, _, tail = text.partition(f"## {hdr}")
        seg, nl, rest = tail.partition("\n## ")
        nums = [int(m.group(1)) for m in QID_RE.finditer(text)]
        nxt = (max(nums) + 1) if nums else 1
        lines, added = [], 0
        for l in seg.splitlines():
            m = LIST_PREFIX_RE.match(l)
            if m and l[m.end():].strip() and not QID_RE.search(l):
                l = f"{l[:m.end()]}[Q-{nxt:03d}] {l[m.end():]}"
                nxt += 1
                added += 1
            lines.append(l)
        if added:
            self.write_learnings(head + f"## {hdr}" + "\n".join(lines)
                                 + ("\n## " + rest if nl else "\n"))
        return added

    def consume_queue_item(self, qid: str, record: dict) -> bool:
        """
        把某條佇列項目標記為已消費：從 learnings.md 移除，並把全文＋結果
        追加到 memory/queue_consumed.jsonl（只進不出，資訊不遺失）。
        回傳 True = 有找到並移除。
        """
        text = self.read_learnings()
        hdr = self._queue_header(text)
        if hdr is None:
            return False
        head, _, tail = text.partition(f"## {hdr}")
        seg, nl, rest = tail.partition("\n## ")
        kept, hit = [], None
        for l in seg.splitlines():
            m = QID_RE.search(l)
            if m and f"Q-{m.group(1)}" == qid and LIST_ITEM_RE.match(l):
                hit = l.strip()
            else:
                kept.append(l)
        if hit is None:
            return False
        body = "\n".join(kept).rstrip() + "\n"
        if not any(LIST_ITEM_RE.match(l) for l in kept):
            body = body.rstrip() + "\n\n（尚無）\n"
        self.write_learnings(head + f"## {hdr}" + body
                             + ("\n## " + rest if nl else "\n"))
        log = self.root / "queue_consumed.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"qid": qid, "text": hit,
                                "ts": datetime.now().isoformat(timespec="seconds"),
                                **record}, ensure_ascii=False) + "\n")
        return True

    def count_section_items(self, section: str) -> int:
        """
        數某小節底下的條目數。

        ⚠️ 必須同時認得 '- ' 和 '1. ' 兩種清單：Distill 追加的是 '-'，但整理
        回合的 LLM 重寫時常改成編號清單。只認 '-' 的話，整理過後計數會變成 0，
        佇列上限就形同失效（實測 8/19 整理後就是這樣）。
        """
        text = self.read_learnings()
        header = f"## {section}"
        if header not in text:
            return 0
        seg = text.partition(header)[2].partition("\n## ")[0]
        return sum(1 for l in seg.splitlines() if LIST_ITEM_RE.match(l))

    def append_learning(self, section: str, line: str) -> bool:
        """
        在指定小節（## section）下追加一條；小節不存在則新建於檔尾。
        回傳 True = 已寫入，False = 因總量上限被擋下。

        「待驗證假設佇列」有硬上限（QUEUE_MAX_ITEMS）：它是唯一會被無限追加、
        且沒有任何「消費後移除」機制的小節。上限是安全網——正常情況下每輪
        上限 3 條 + 每 10 輪整理一次重新排序精簡，不會碰到；只有在整理連續
        失敗時才會生效，防止再次長到 156 條那種狀態。
        滿了就拒收新項（而不是丟掉舊的），因為佇列頂端是整理回合排過序的
        高優先項，丟舊的等於丟掉最有價值的部分。

        寫進佇列的條目會自動補上 [Q-xxx] 編號，供 Generate 回報 from_queue
        與後續的消費機制使用。
        """
        if section.startswith(QUEUE_SECTION):
            if self.count_section_items(section) >= QUEUE_MAX_ITEMS:
                return False
            line = self._ensure_qid(line)
        text = self.read_learnings()
        header = f"## {section}"
        if header in text:
            head, _, tail = text.partition(header)
            # 移除該節的「（尚無）」占位
            seg, nl, rest = tail.partition("\n## ")
            seg = seg.replace("（尚無）\n", "").rstrip() + f"\n{line}\n"
            text = head + header + seg + ("\n## " + rest if nl else "\n")
        else:
            text = text.rstrip() + f"\n\n{header}\n\n{line}\n"
        self.write_learnings(text)
        return True
