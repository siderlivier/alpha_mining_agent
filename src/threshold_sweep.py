"""
門檻掃描：同時報 Sharpe 與 IR，看「換裁判會不會換結論」。

    python src/threshold_sweep.py

這支的存在理由：--riskadj 只在預設門檻 0.5 上比較，而 0.5 是個沒調過的
預設值。若結論只在 0.5 成立、換個門檻就翻盤，那它就不是結論而是巧合。
掃過整段門檻可以分辨這兩者——而且比「挑一個最好的門檻」誠實，
**挑最好的那個就是在測試集上選模型**。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm_regime as hr
import factor_lab as fl
import backtest as bt

proc, feats, defensive = hr._factor_setup()
print(f"防禦組（--apply 口徑，前 len/6 = {len(defensive)} 個）：{defensive}\n")

s_all = fl.walk_forward(proc, feats, "ridge")
s_def = fl.walk_forward(proc, defensive, "ridge")

base = hr._full_metrics(hr._returns_for(proc, s_all), "靜態")
perfect_bull = set((proc.groupby("ym")["fwd_ret_1m"].mean() > 0)
                   .pipe(lambda s: s[s].index))
pf = hr._full_metrics(hr._returns_for(
    proc, pd.Series(np.where(proc["ym"].isin(perfect_bull), s_all, s_def),
                    index=proc.index)), "完美")

print(f"{'':<28}{'看多月數':>9}{'Sharpe':>9}{'IR':>8}{'beta':>8}"
      f"{'CAGR':>9}{'MaxDD':>9}")
print(f"{'靜態：不切換':<28}{'—':>9}{base['Sharpe']:>9.2f}{base['IR']:>8.2f}"
      f"{base['beta']:>8.3f}{base['CAGR']:>9.2%}{base['MaxDD']:>9.2%}")
print("-" * 80)

for fset, tag in [(("ret", "vol"), "純價格"),
                  (("ret", "vol", "m1b_minus_m2"), "＋M1B−M2")]:
    d = hr.build_series(fset)
    st = hr.walk_forward_states(d, fset, hr.N_STATES, seed=hr.SEED)
    for th in (0.1, 0.2, 0.3, 0.5, 0.7):
        bull = set(st[st["p_bull"] >= th].index)
        sw = pd.Series(np.where(proc["ym"].isin(bull), s_all, s_def),
                       index=proc.index)
        m = hr._full_metrics(hr._returns_for(proc, sw), f"{tag} @{th}")
        flag = "  ✅" if m["Sharpe"] > base["Sharpe"] else ""
        print(f"{m['label']:<28}{len(bull & set(proc['ym'].unique())):>9}"
              f"{m['Sharpe']:>9.2f}{m['IR']:>8.2f}{m['beta']:>8.3f}"
              f"{m['CAGR']:>9.2%}{m['MaxDD']:>9.2%}{flag}")
    print()

print("-" * 80)
print(f"{'完美預知（上限）':<28}{len(perfect_bull):>9}{pf['Sharpe']:>9.2f}"
      f"{pf['IR']:>8.2f}{pf['beta']:>8.3f}{pf['CAGR']:>9.2%}{pf['MaxDD']:>9.2%}")
