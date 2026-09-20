"""Versioned factor-scope checks shared by CLI and local UI. No test-based selection.

The caller supplies an already rebuilt monthly_base. This tool does not fetch data
or silently relabel stocks. Current upstream classifications must agree with it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

import dsl
import eval_candidates as ec
from memory import Memory, approved_groups

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def ancestors(group, registry, visiting=None):
    visiting = set() if visiting is None else visiting
    if group not in registry or group in visiting:
        raise ValueError(f"產業血緣未知或有循環：{group}")
    parents = registry[group]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError(f"產業血緣必須為字串清單：{group}")
    result = {group}
    for parent in parents:
        result |= ancestors(parent, registry, visiting | {group})
    return result


def validate_panel(base, universe, policy):
    required = {"stock_id", "ym", "group", "fwd_ret_1m"}
    if not required <= set(base) or base.empty:
        raise ValueError("月頻面板缺欄位或為空")
    if not {"stock_id", "group"} <= set(universe) or universe.empty:
        raise ValueError("上游產業分類來源缺欄位或為空")
    if base[list(required - {"fwd_ret_1m"})].isna().any().any():
        raise ValueError("股票／月份／產業不能缺值")
    if base.duplicated(["stock_id", "ym"]).any():
        raise ValueError("面板 stock_id/ym 重複")
    if not base.ym.astype(str).str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("月份必須是 YYYY-MM")
    registry = policy.get("groups", {})
    if not registry:
        raise ValueError("scope_control.groups 尚未設定")
    for group in registry:
        ancestors(group, registry)
    unknown = set(base.group) - set(registry)
    if unknown:
        raise ValueError(f"未知產業，請先定義分類及血緣：{sorted(unknown)}")
    if (base.groupby("stock_id").group.nunique() > 1).any():
        raise ValueError("同股票有歷史分類變動；本版需先建立明確 PIT 分類政策，不取最後值覆蓋")
    source = universe[["stock_id", "group"]].copy()
    if source.isna().any().any():
        raise ValueError("上游股票分類缺值")
    source.stock_id = source.stock_id.astype(str)
    if source.isna().any().any() or (source.groupby("stock_id").group.nunique() > 1).any():
        raise ValueError("上游股票分類缺值或互相衝突")
    check = base[["stock_id", "group"]].drop_duplicates().merge(
        source.drop_duplicates(), on="stock_id", how="left", suffixes=("", "_source"), validate="one_to_one")
    wrong = check[check.group != check.group_source]
    if len(wrong):
        raise ValueError(f"面板與上游分類不一致或來源缺股票：{wrong.stock_id.head(10).tolist()}")
    numeric = base.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("面板含無限大數值")
    return {"rows": len(base), "stocks": int(base.stock_id.nunique()),
            "groups": sorted(base.group.unique()), "classification_conflicts": 0}


def bootstrap(fid, meta, policy):
    result = copy.deepcopy(meta)
    if "groups_at_admission" not in result:
        if fid not in policy.get("legacy_factor_ids", []):
            raise ValueError(f"{fid} 缺少入庫產業來源，不可推測")
        result["groups_at_admission"] = list(policy["legacy_groups"])
        result["scope_provenance"] = "verified_legacy_four_groups"
    birth = set(result["groups_at_admission"])
    scope = approved_groups(result)
    result.setdefault("approved_groups", sorted(birth if scope is None else scope))
    result.setdefault("excluded_at_admission", sorted(birth - set(result["approved_groups"])))
    result.setdefault("groups_seen", sorted(birth))
    for key in ("groups_at_admission", "approved_groups", "excluded_at_admission", "groups_seen"):
        value = result[key]
        if not isinstance(value, list) or not all(isinstance(g, str) for g in value):
            raise ValueError(f"{fid}.{key} 必須是產業清單")
    if set(result["approved_groups"]) & set(result["excluded_at_admission"]):
        raise ValueError(f"{fid} 核准與排除產業衝突")
    if not birth <= set(result["groups_seen"]):
        raise ValueError(f"{fid} groups_seen 遺失入庫產業")
    for group in set().union(*(set(result[k]) for k in (
            "groups_at_admission", "approved_groups", "excluded_at_admission", "groups_seen"))):
        ancestors(group, policy["groups"])
    return result


def candidate_groups(meta, current, registry):
    seen = set(meta["groups_seen"])
    excluded = set(meta["excluded_at_admission"])
    # Renaming/splitting/merging an old group cannot manufacture an unseen group.
    return sorted(g for g in set(current) - seen
                  if not (ancestors(g, registry) & (seen | excluded)))


def context_for(base, cfg):
    months = sorted(base.ym.unique())
    stocks = sorted(base.stock_id.unique())
    fields = set(base) - {"stock_id", "ym", "group", "fwd_ret_1m"}
    def pivot(c):
        return base.pivot(index="ym", columns="stock_id", values=c).reindex(index=months, columns=stocks)
    return SimpleNamespace(months=months, fields=fields,
        data={f: pivot(f) for f in fields}, fwd=pivot("fwd_ret_1m"),
        group_map=base.groupby("stock_id").group.last().to_dict(),
        mask=lambda span: [m for m in months if cfg["split"][span][0] <= m <= cfg["split"][span][1]])


def assess_group(ctx, fac, group, use, policy):
    months = ctx.mask("sub_train") + ctx.mask("validation")
    cols = ec.group_cols(ctx, fac).get(group, [])
    sub = fac[cols]
    series = ec.industry_ic_series(ctx, sub, months).get(group, pd.Series(dtype=float))
    tr = series.reindex(ctx.mask("sub_train")).dropna()
    va = series.reindex(ctx.mask("validation")).dropna()
    record = {"group": group, "train_months": len(tr), "valid_months": len(va), "reasons": []}
    limits = ec.FUN["stage4b_industry"]
    if len(tr) < limits["min_months_train"] or len(va) < limits["min_months_valid"]:
        return dict(record, status="insufficient_data", reasons=["有效月份不足，等待更多資料"])
    if tr.std() <= 0 or va.std() <= 0:
        return dict(record, status="rejected", reasons=["IC 變異為零，ICIR 無法判斷"])
    _, train = ec.icir(tr)
    _, valid = ec.icir(va)
    decay = (1 - valid / train) * 100 if train and train > 0 else None
    legs, long_turn = ec.leg_stats(ctx, sub, sorted(set(months)), cols=cols, min_n=10)
    coverage = ec.coverage(ctx, sub, months, cols=cols)
    record.update(train_icir=train, valid_icir=valid, decay_pct=decay,
                  coverage=coverage, legs=legs, long_turnover=long_turn,
                  short_turnover=legs.get("short_turnover_m"), required_use=use)
    # Same qualification function as candidate mining, with additional fixed-use and turnover gates.
    if not ec.industry_qualify(ctx, sub, {group: series}):
        record["reasons"].append("未通過既有 Stage 4b ICIR／衰減／覆蓋／交易腿門檻")
    sides = {"long_only": ["long"], "short_only": ["short"], "long_short": ["long", "short"]}
    if use not in sides:
        return dict(record, status="blocked", reasons=["未知交易用途"])
    for side in sides[use]:
        excess = legs.get(side + "_excess_ann")
        turnover = long_turn if side == "long" else legs.get("short_turnover_m")
        if excess is None or not np.isfinite(excess) or excess <= 0:
            record["reasons"].append(f"{side} 腿超額非正或缺值")
        if turnover is None or not np.isfinite(turnover) or turnover > policy["max_monthly_turnover"]:
            record["reasons"].append(f"{side} 換手缺值或超過門檻")
    return dict(record, status="rejected" if record["reasons"] else "passed")


def input_paths(root, cfg):
    return {"base": root / cfg["paths"]["monthly_base"],
            "library": root / cfg["paths"]["memory_dir"] / "library.json",
            "values": root / cfg["paths"]["memory_dir"] / "factor_values.parquet",
            "classification": (root / cfg["scope_control"]["classification_source"]).resolve(),
            "config": root / "config.yaml", "fields": root / "fields.yaml",
            **{p.stem: p for p in [Path(__file__), Path(ec.__file__), Path(dsl.__file__),
                                   Path(__file__).with_name("memory.py")]}}


def prepare(root=ROOT, progress=print):
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if cfg["funnel"] != ec.FUN:
        raise ValueError("門檻已變動，請重新啟動工作程序後執行")
    if not cfg["funnel"]["stage4b_industry"].get("enabled"):
        raise ValueError("Stage 4b 已停用，不能自動作產業資格判定")
    turn_limit = cfg["scope_control"]["max_monthly_turnover"]
    if not isinstance(turn_limit, (int, float)) or not np.isfinite(turn_limit) or not 0 <= turn_limit <= 1:
        raise ValueError("產業換手門檻須介於 0 與 1")
    paths = input_paths(root, cfg)
    hashes = {k: digest(p) for k, p in paths.items()}
    memory = Memory(paths["library"].parent)
    lib, old_values = memory.snapshot()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    folder = memory.root / "scope_runs" / run_id
    folder.mkdir(parents=True)
    report = {"run_id": run_id, "status": "building", "inputs": hashes, "factors": [],
              "warnings": ["參考池未換版；本工具不修正既有標籤、股數及回測問題。", "上游分類一致性檢查不是對企業實際業務歸屬的人工審定。"]}
    write_json(folder / "report.json", report)
    try:
        base = pd.read_parquet(paths["base"])
        if not {"stock_id", "ym", "group", "fwd_ret_1m"} <= set(base):
            raise ValueError("月頻面板缺少必要欄位")
        if base[["stock_id", "ym", "group"]].isna().any().any():
            raise ValueError("股票／月份／產業不能缺值")
        base.stock_id = base.stock_id.astype(str)
        base.ym = base.ym.astype(str)
        universe = pd.read_parquet(paths["classification"])
        report["data"] = validate_panel(base, universe, cfg["scope_control"])
        state_path = memory.root / "scope_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if state.get("active_hashes") == hashes:
            report["status"] = "no_change"
            write_json(folder / "report.json", report)
            return report
        ctx = context_for(base, cfg)
        engine = dsl.Engine(ctx.data, ctx.group_map)
        updated = copy.deepcopy(lib)
        frames = [old_values[old_values.factor_id.map(lambda f: f not in lib or lib[f].get("reference") or f.startswith("R-"))]]
        own = {f: v for f, v in lib.items() if not v.get("reference") and not f.startswith("R-")}
        if not own:
            raise ValueError("沒有自有因子可檢查")
        for i, (fid, original) in enumerate(sorted(own.items()), 1):
            progress(f"[{i}/{len(own)}] {fid} {original.get('name_zh', '')}")
            meta = bootstrap(fid, original, cfg["scope_control"])
            candidates = candidate_groups(meta, report["data"]["groups"], cfg["scope_control"]["groups"])
            fac = ec.orient_factor(engine.eval(dsl.parse(meta["formula"], allowed_fields=ctx.fields).tree), meta)
            before = sorted(meta["approved_groups"])
            assessments = []
            for group in candidates:
                assessment = assess_group(ctx, fac, group, meta.get("trading_use", "long_only"), cfg["scope_control"])
                if assessment["status"] == "blocked":
                    raise ValueError(f"{fid}: {assessment['reasons']}")
                assessments.append(assessment)
                if assessment["status"] != "insufficient_data":
                    meta["groups_seen"] = sorted(set(meta["groups_seen"]) | {group})
                if assessment["status"] == "passed":
                    meta["approved_groups"] = sorted(set(meta["approved_groups"]) | {group})
            if assessments:
                meta.setdefault("scope_history", []).append({"run_id": run_id,
                    "data_hash": hashes["base"], "config_hash": hashes["config"],
                    "split": {k: cfg["split"][k] for k in ("sub_train", "validation")},
                    "before": before, "after": meta["approved_groups"], "assessments": assessments})
            meta["values_version"] = run_id
            # Historical industry_scope/metrics stay immutable; approved_groups governs current use.
            updated[fid] = meta
            cols = [s for s, g in ctx.group_map.items() if g in meta["approved_groups"]]
            values = fac[cols].stack().rename("value").reset_index()
            values.columns = ["ym", "stock_id", "value"]
            values = values.merge(base[["ym", "stock_id"]], on=["ym", "stock_id"], validate="one_to_one")
            values.insert(0, "factor_id", fid)
            frames.append(values)
            report["factors"].append({"id": fid, "name": meta.get("name_zh", fid),
                "use": meta.get("trading_use", "long_only"), "before": before,
                "after": meta["approved_groups"], "excluded": meta["excluded_at_admission"],
                "assessments": assessments, "value_rows": len(values)})
        values = pd.concat(frames, ignore_index=True)
        if values.duplicated(["factor_id", "ym", "stock_id"]).any() or not np.isfinite(values.value).all():
            raise ValueError("輸出因子值重複或含非有限值")
        write_json(folder / "library.new.json", updated)
        values.to_parquet(folder / "values.new.parquet", index=False)
        if hashes != {k: digest(p) for k, p in paths.items()}:
            raise ValueError("計算期間輸入改變，請重跑；未套用")
        report.update(status="ready", staged_hashes={"library": digest(folder / "library.new.json"),
                      "values": digest(folder / "values.new.parquet")},
                      summary={"factors": len(own), "expanded": sum(f["before"] != f["after"] for f in report["factors"]),
                               "pending": sum(a["status"] == "insufficient_data" for f in report["factors"] for a in f["assessments"])})
    except Exception as exc:
        report.update(status="blocked", error=str(exc))
    write_json(folder / "report.json", report)
    return report


def run_folder(memory, run_id):
    if not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", run_id):
        raise ValueError("不合法 run_id")
    return memory.root / "scope_runs" / run_id


def apply(run_id, root=ROOT):
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    paths = input_paths(root, cfg)
    memory = Memory(paths["library"].parent)
    folder = run_folder(memory, run_id)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "ready":
        raise ValueError("只有 ready 工作可套用；已完成工作不重複提交")
    with memory.locked():
        if report["inputs"] != {k: digest(p) for k, p in paths.items()}:
            raise ValueError("資料／設定／因子庫／程式已改變，請重新檢查")
        for key, name in (("library", "library.new.json"), ("values", "values.new.parquet")):
            if digest(folder / name) != report["staged_hashes"][key]:
                raise ValueError("暫存內容已被修改")
            shutil.copy2(paths[key], folder / (key + ".before"))
            shutil.copy2(folder / name, folder / (key + ".ready"))
        pending = memory.root / ".scope_pending.json"
        write_json(pending, {"run_id": run_id})
        try:
            os.replace(folder / "values.ready", paths["values"])
            os.replace(folder / "library.ready", paths["library"])
            report["status"] = "committed"
            write_json(folder / "report.json", report)
            write_json(memory.root / "scope_state.json", {"run_id": run_id,
                "active_hashes": {k: digest(p) for k, p in paths.items()}})
            pending.unlink()
        except Exception:
            for key in ("values", "library"):
                shutil.copy2(folder / (key + ".before"), folder / (key + ".restore"))
                os.replace(folder / (key + ".restore"), paths[key])
            report["status"] = "rolled_back"
            write_json(folder / "report.json", report)
            pending.unlink(missing_ok=True)
            raise
    return report


def recover(root=ROOT):
    """Recover an interrupted scope commit. Never deletes a live process lock."""
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    memory = Memory(root / cfg["paths"]["memory_dir"])
    lock = memory.root / ".library.lock"
    if lock.exists():
        pid = int(lock.read_text())
        # Windows os.kill(pid, 0) is unsafe; inspect liveness via OpenProcess.
        if os.name == "nt":
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle(handle)
                raise RuntimeError("工作程序仍存在，不能回復")
            if ctypes.get_last_error() != 87:
                raise RuntimeError("無法確認工作程序已結束，保留鎖")
        else:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("工作程序仍存在，不能回復")
        lock.unlink()
    pending = memory.root / ".scope_pending.json"
    if not pending.exists():
        with memory.locked():
            pass  # Also recovers ordinary admission/reference transactions.
        return {"status": "nothing_to_recover"}
    run_id = json.loads(pending.read_text(encoding="utf-8"))["run_id"]
    folder = run_folder(memory, run_id)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    # Recovery owns the same lock but intentionally bypasses the pending guard.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        for key, target in (("values", memory.values_path), ("library", memory.lib_path)):
            if digest(folder / (key + ".before")) != report["inputs"][key]:
                raise ValueError("備份 hash 不符，停止回復")
        for key, target in (("values", memory.values_path), ("library", memory.lib_path)):
            shutil.copy2(folder / (key + ".before"), folder / (key + ".restore"))
            os.replace(folder / (key + ".restore"), target)
        report["status"] = "rolled_back"
        write_json(folder / "report.json", report)
        pending.unlink()
    finally:
        lock.unlink(missing_ok=True)
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--auto-apply", action="store_true")
    mode.add_argument("--apply", metavar="RUN_ID")
    mode.add_argument("--recover", action="store_true")
    ap.add_argument("--result-file", type=Path, help="工作結果 JSON（供本機 UI 使用）")
    args = ap.parse_args()
    try:
        result = recover() if args.recover else apply(args.apply) if args.apply else prepare()
        if args.auto_apply and result["status"] == "ready":
            result = apply(result["run_id"])
        if args.result_file:
            write_json(args.result_file, result)
            print(json.dumps({k: result[k] for k in ("run_id", "status", "summary", "error") if k in result}, ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in ("ready", "committed", "no_change", "rolled_back", "nothing_to_recover") else 1
    except Exception as exc:
        failure = {"status": "blocked", "error": str(exc)}
        if args.result_file:
            write_json(args.result_file, failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
