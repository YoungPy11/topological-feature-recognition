#!/usr/bin/env python3
"""阶段 3 第 3 层续 · 稳健性检验：增量 IC 的 top_k 敏感性与置换检验。

为什么必须做
-----------
主检验里"拓扑(收益选特征)对 32 因子残差 IC = +0.0331，t=2.78，p=0.013"是
在**4 个变体 × 18 个月**里挑出来的最好那个，直接报出来会高估显著性。本脚本做两件事：

1. **top_k 敏感性**：k = 5 / 10 / 20 / 40 / 80 都跑一遍。若结论只在某个 k 上成立 → 脆弱。
2. **置换检验（负对照）**：每月内随机打乱拓扑因子值（200 次），
   得到"均值残差 IC"的零分布，看真实值落在什么分位。
   —— 这是唯一能说明"p 值不是多重比较挑出来的"的办法。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from experiment_3d_increment import build_topo_factor, residualize  # noqa: E402

OUT = ROOT / "results" / "exp3d_increment"


def mean_ic(use, months, xcol, fcols_use, shuffle_seed=None):
    """逐月算 [拓扑 对因子回归] 残差的 Rank IC，返回月度序列。"""
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    F_all = use[fcols_use].to_numpy(dtype=float)
    ret = use["ret"].to_numpy(dtype=float)
    x_all = use[xcol].to_numpy(dtype=float)
    out = []
    for m in months:
        sel = (use.month == m).to_numpy()
        y = ret[sel]
        F = F_all[sel]
        x = x_all[sel].copy()
        ok = np.isfinite(F).all(axis=1) & np.isfinite(y) & np.isfinite(x)
        if ok.sum() < max(30, F.shape[1] + 5):
            continue
        if rng is not None:                   # 月内置换（负对照）
            x[ok] = rng.permutation(x[ok])
        r = residualize(x, F)
        if r is None:
            continue
        mm = np.isfinite(r) & np.isfinite(y)
        if mm.sum() < 10 or np.std(r[mm]) < 1e-12:
            continue
        v = spearmanr(r[mm], y[mm]).correlation
        if np.isfinite(v):
            out.append(float(v))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n300_win120_s20")
    ap.add_argument("--qr-root", default="/home/young/quant-research")
    ap.add_argument("--topks", default="5,10,20,40,80")
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--target", default="y_ret", choices=["y_ret", "y_rv"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("阶段3 第3层续 · 稳健性检验（top_k 敏感性 + 置换检验）")
    print("=" * 88)

    sys.path.insert(0, args.qr_root)
    from src.etl.panel_guard import load_panel                       # noqa: E402
    from src.factors.compute import compute_factors, factor_columns  # noqa: E402

    print("\n[1/3] 载入既有因子面板 …")
    panel = load_panel()
    panel, kept = compute_factors(panel)
    fcols = factor_columns(kept)
    base = panel[["Stkcd", "month", "ret"] + fcols].copy()
    base["Stkcd"] = base["Stkcd"].astype(str).str.zfill(6)
    print(f"  既有因子 {len(fcols)} 个")

    print(f"\n[2/3] top_k 敏感性（target={args.target}）…")
    topks = [int(x) for x in args.topks.split(",")]
    scan, use_ref, fcols_ref, months_ref, xcol_ref = [], None, None, None, None
    for k in topks:
        df, _ = build_topo_factor(args.tag, top_k=k, target=args.target)
        u = base.merge(df.rename(columns={"topo": "topo"}),
                       on=["Stkcd", "month"], how="inner")
        cov = u[fcols].notna().mean()
        fu = [c for c in fcols if cov[c] >= 0.60]
        months = sorted(u.month.unique())
        ics = mean_ic(u, months, "topo", fu)
        scan.append(dict(top_k=k, n_months=int(len(ics)),
                         ic=float(ics.mean()) if len(ics) else np.nan,
                         t=float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))))
                         if len(ics) > 1 and ics.std(ddof=1) > 0 else np.nan,
                         pos=float(np.mean(ics > 0)) if len(ics) else np.nan))
        print(f"  top_k={k:<4} 月份={len(ics):<3} 残差 IC={scan[-1]['ic']:+.4f}  "
              f"t={scan[-1]['t']:+.2f}  IC>0={scan[-1]['pos']:.0%}")
        if k == 20:
            use_ref, fcols_ref, months_ref, xcol_ref = u, fu, months, "topo"

    print(f"\n[3/3] 置换检验（负对照，{args.perm} 次月内打乱）…")
    obs = mean_ic(use_ref, months_ref, xcol_ref, fcols_ref)
    obs_mean = float(obs.mean())
    null = np.array([mean_ic(use_ref, months_ref, xcol_ref, fcols_ref,
                             shuffle_seed=s).mean() for s in range(args.perm)])
    null = null[np.isfinite(null)]
    p_perm = float((np.abs(null) >= abs(obs_mean)).mean())
    print(f"  实测月度残差 IC 均值 = {obs_mean:+.4f}（{len(obs)} 个月，"
          f"IC>0 {(obs>0).sum()}/{len(obs)}）")
    print(f"  零分布：均值={null.mean():+.4f} 标准差={null.std(ddof=1):.4f}  "
          f"5%/{95}% 分位=[{np.percentile(null,5):+.4f}, {np.percentile(null,95):+.4f}]")
    print(f"  → 置换 p = {p_perm:.3f}（{args.perm} 次）")

    with open(OUT / f"robust_{args.tag}_{args.target}.json", "w", encoding="utf-8") as f:
        json.dump(dict(tag=args.tag, target=args.target, topk_scan=scan,
                       perm_n=args.perm, obs_mean_ic=obs_mean,
                       obs_n_months=int(len(obs)),
                       obs_pos_ratio=float((obs > 0).mean()),
                       null_mean=float(null.mean()), null_std=float(null.std(ddof=1)),
                       null_q05=float(np.percentile(null, 5)),
                       null_q95=float(np.percentile(null, 95)),
                       p_perm=p_perm,
                       note="置换=每月内随机打乱拓扑因子值，重算残差 IC 均值"),
                  f, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {OUT / f'robust_{args.tag}_{args.target}.json'}")


if __name__ == "__main__":
    main()
