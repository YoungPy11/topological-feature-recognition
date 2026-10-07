#!/usr/bin/env python3
"""阶段 3 第 3 层续：拓扑因子相对**已有 33 因子**的增量 IC。

为什么必须做这一步
----------------
第 2/3 层的"强基线"是**窗口原始矩**（过去波动的水平），不是量化里真正的对照组。
"拓扑有没有增量"的最终判据只能是：**在已有 33 因子之外，它还能不能解释收益**。

协议（严格无前视）
----------------
1. **拓扑因子构造**：用第 2 层a 保存的滚动 PH 特征，按 (股票, 月) 去重；
   用**扩张窗口**（只用 $t$ 月之前的历史）选 top-k 特征再加权合成 → `topo_score`。
2. **已有 33 因子**：直接调 `~/quant-research` 的 `compute_factors()`
   （已做按月横截面缩尾 + zscore + 方向统一）。
3. **增量检验**（每月一个截面）：
   * 横截面 OLS 把 `topo_score` 对 33 因子回归 → 取**残差**
   * 比较 `IC(topo)` 与 `IC(残差)`（对下月收益的 Spearman）
   * 对月度 IC 序列做**配对检验**（配对 t + Wilcoxon + 胜出月数）
4. 同时报告"已有因子等权合成"的 IC 作为参照。

⚠️ 样本量很小：HF 数据 2023-10 起，重叠月份约 20–30 个，股票仅 300 只（且是数据最完整的一批，
偏大盘）。**这是本检验最大的局限**，结论必须带这个前提读。

用法
----
    python src/experiment_3d_increment.py --tag n300_win120_s20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
OUT3B = ROOT / "results" / "exp3b_csmar_hf"
OUT = ROOT / "results" / "exp3d_increment"


# --------------------------------------------------------------------------
def build_topo_factor(tag: str, top_k: int = 20, min_months: int = 6,
                      target: str = "y_rv"):
    """扩张窗口构造拓扑因子：(股票, 月) -> topo_score。

    target: 'y_rv'（未来 20 日已实现波动）或 'y_ret'（未来 20 日收益）—— 选特征用的标签。
    严格无前视：$t$ 月的合成权重只用 $t$ 月之前的历史月份估计。
    """
    z = np.load(OUT3B / f"features_{tag}.npz", allow_pickle=True)
    idx = pd.read_csv(OUT3B / f"index_{tag}.csv")
    idx["date"] = pd.to_datetime(idx["date"])
    idx["month"] = idx["date"].dt.to_period("M").astype(str)
    XT = z["XT"]

    # (股票, 月) 去重：同月常有两个窗口，保留末日最新的那个
    idx["_i"] = np.arange(len(idx))
    kept = np.sort(idx.sort_values("date")
                      .groupby(["Stkcd", "month"], as_index=False)
                      .tail(1)["_i"].to_numpy())
    idx = idx.iloc[kept].reset_index(drop=True)
    XT = XT[kept]

    y = z["y_rv"] if target == "y_rv" else z["y_ret"]
    y = y[kept]
    idx["_y"] = y

    months = sorted(idx["month"].unique())
    out = []
    for mi, m in enumerate(months):
        sel = (idx["month"] == m).to_numpy()
        past = idx["month"].isin(months[:mi]).to_numpy()
        if mi < min_months or past.sum() < 200 or sel.sum() < 10:
            continue
        yv = idx["_y"].to_numpy()
        okp = past & np.isfinite(yv)
        cors = np.zeros(XT.shape[1])
        for j in range(XT.shape[1]):
            col = XT[:, j]
            if np.std(col[okp]) < 1e-12:
                continue
            r = spearmanr(col[okp], yv[okp]).correlation
            cors[j] = 0.0 if not np.isfinite(r) else r
        top = np.argsort(-np.abs(cors))[:top_k]
        w = cors[top]
        if np.abs(w).sum() < 1e-12:
            continue
        score = XT[sel][:, top] @ w
        for stk, sc in zip(idx.loc[sel, "Stkcd"].to_numpy(), score):
            out.append(dict(Stkcd=str(stk).zfill(6), month=m, topo=float(sc)))
    df = pd.DataFrame(out)
    return df, dict(n_months=df.month.nunique() if len(df) else 0,
                    n_rows=len(df), target=target, top_k=top_k)


# --------------------------------------------------------------------------
def rank_ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.std(x[m]) < 1e-12:
        return np.nan
    r = spearmanr(x[m], y[m]).correlation
    return float(r) if np.isfinite(r) else np.nan


def residualize(x, F):
    """横截面 OLS：把 x 对因子矩阵 F（含截距）回归，返回残差。"""
    m = np.isfinite(x) & np.isfinite(F).all(axis=1)
    if m.sum() < F.shape[1] + 5:
        return None
    X = np.c_[np.ones(m.sum()), F[m]]
    xv = x[m]
    try:
        beta, *_ = np.linalg.lstsq(X, xv, rcond=None)
    except np.linalg.LinAlgError:
        return None
    res = np.full(len(x), np.nan)
    res[m] = xv - X @ beta
    return res


def summarize(name, ics):
    v = np.array([x for x in ics if np.isfinite(x)])
    if len(v) < 3:
        return dict(name=name, n=len(v), ic=np.nan, icir=np.nan, t=np.nan,
                    pos=np.nan, ic_series=[])
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    icir = mean / sd if sd > 0 else np.nan
    t = mean / (sd / np.sqrt(len(v))) if sd > 0 else np.nan
    return dict(name=name, n=int(len(v)), ic=mean, icir=float(icir),
                t=float(t), pos=float(np.mean(v > 0)),
                ic_series=[float(x) for x in v])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n300_win120_s20")
    ap.add_argument("--qr-root", default="/home/young/quant-research")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-months", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("阶段3 第3层续：拓扑因子相对既有 33 因子的增量 IC")
    print("=" * 88)

    # ---------- 1. 已有因子面板 ----------
    sys.path.insert(0, args.qr_root)
    print(f"\n[1/4] 载入 ~/quant-research 面板并计算既有因子（{args.qr_root}）…")
    from src.etl.panel_guard import load_panel            # noqa: E402
    from src.factors.compute import compute_factors, factor_columns  # noqa: E402
    panel = load_panel()
    print(f"  面板 {len(panel):,} 行 / {panel.ts_code.nunique()} 只 / "
          f"{panel.month.nunique()} 月  {panel.month.min()} → {panel.month.max()}")
    panel, kept = compute_factors(panel)
    fcols = factor_columns(kept)
    print(f"  既有因子 {len(kept)} 个：{', '.join(f.name for f in kept)}")

    # ---------- 2. 拓扑因子 ----------
    print("\n[2/4] 构造拓扑因子（扩张窗口，严格无前视）…")
    tfs = {}
    for tgt in ("y_rv", "y_ret"):
        df, info = build_topo_factor(args.tag, top_k=args.top_k,
                                     min_months=args.min_months, target=tgt)
        tfs[tgt] = df
        print(f"  target={tgt:<7} {info['n_rows']} 行 / {info['n_months']} 月")

    # ---------- 3. 对齐 ----------
    print("\n[3/4] 与既有面板对齐 …")
    base = panel[["Stkcd", "ts_code", "month", "ret"] + fcols].copy()
    base["Stkcd"] = base["Stkcd"].astype(str).str.zfill(6)
    for tgt, df in tfs.items():
        if len(df) == 0:
            continue
        base = base.merge(df.rename(columns={"topo": f"topo_{tgt}"}),
                          on=["Stkcd", "month"], how="left")
    have = [c for c in base.columns if c.startswith("topo_")]
    use = base[base[have].notna().any(axis=1)].copy()
    print(f"  重叠样本 {len(use):,} 行 / {use.Stkcd.nunique()} 只 / "
          f"{use.month.nunique()} 月  {use.month.min()} → {use.month.max()}")
    if use.month.nunique() < 6:
        print("[中止] 重叠月份太少，无法做横截面检验")
        return

    # ⚠️ 关键：财务类因子在近期月份大量缺失，要求「33 个全非空」会只剩个位数月份。
    #    改为「挑覆盖率够的因子子集 + 该子集上的完整样本」。
    cov = use[fcols].notna().mean()
    fcols_use = [c for c in fcols if cov[c] >= 0.60]
    print(f"  因子覆盖率：{len(fcols_use)}/{len(fcols)} 个 ≥60%"
          f"（被剔除：{', '.join(c[2:] for c in fcols if cov[c] < 0.60) or '无'}）")
    if len(fcols_use) < 5:
        print("[中止] 可用因子太少")
        return

    # ---------- 4. 逐月增量检验 ----------
    print("\n[4/4] 逐月横截面：把拓扑因子对既有因子回归 → 残差 IC")
    months = sorted(use.month.unique())
    F_all = use[fcols_use].to_numpy(dtype=float)
    ret = use["ret"].to_numpy(dtype=float)

    series = {k: [] for k in ("factor_composite", "raw_rv", "resid_rv",
                              "raw_ret", "resid_ret",
                              "resid_rv_comp", "resid_ret_comp")}
    detail = []
    for m in months:
        sel = (use.month == m).to_numpy()
        y = ret[sel]
        F = F_all[sel]
        ok_f = np.isfinite(F).all(axis=1) & np.isfinite(y)
        if ok_f.sum() < max(30, F.shape[1] + 5):
            continue
        # 既有因子等权合成（方向已统一为越大越好）—— 作为稳健的单控制变量
        comp = np.full(sel.sum(), np.nan)
        comp[ok_f] = np.nanmean(F[ok_f], axis=1)
        series["factor_composite"].append(rank_ic(comp, y))
        row = dict(month=m, n=int(ok_f.sum()))
        for tgt in ("y_rv", "y_ret"):
            c = f"topo_{tgt}"
            if c not in use.columns:
                continue
            key = "rv" if tgt == "y_rv" else "ret"
            x = use[c].to_numpy(dtype=float)[sel]
            raw = rank_ic(x, y)
            series[f"raw_{key}"].append(raw)
            # 控制 1：对全部可用因子回归
            r = residualize(x, F)
            series[f"resid_{key}"].append(rank_ic(r, y) if r is not None else np.nan)
            # 控制 2：只对既有因子的等权合成回归（对缺失更稳健）
            r2 = residualize(x, comp[:, None])
            series[f"resid_{key}_comp"].append(
                rank_ic(r2, y) if r2 is not None else np.nan)
            row[f"ic_raw_{key}"] = raw
            row[f"n_factors"] = int(F.shape[1])
        detail.append(row)

    rows = []
    print(f"\n{'组合':<22}{'月数':>5}{'均值 IC':>10}{'ICIR':>8}{'t':>8}{'IC>0':>8}")
    print("-" * 62)
    for k, ic in series.items():
        s = summarize(k, ic)
        rows.append(s)
        if s["n"]:
            print(f"{k:<22}{s['n']:>5}{s['ic']:>10.4f}{s['icir']:>8.3f}"
                  f"{s['t']:>8.2f}{s['pos']:>8.1%}")

    # ---- 配对检验：残差 IC 是否显著不为 0；残差 vs 原始 ----
    print("\n配对检验（逐月，同一批月份）")
    print(f"{'比较':<34}{'ΔIC':>9}{'胜出':>8}{'配对 t':>9}{'p':>8}")
    print("-" * 70)
    paired = {}
    for a, b in [("resid_rv", "factor_composite"), ("raw_rv", "factor_composite"),
                 ("resid_ret", "factor_composite"), ("raw_ret", "factor_composite"),
                 ("resid_rv_comp", "factor_composite"),
                 ("resid_ret_comp", "factor_composite"),
                 ("resid_rv", "raw_rv"), ("resid_ret", "raw_ret"),
                 ("resid_rv_comp", "raw_rv"), ("resid_ret_comp", "raw_ret")]:
        va = np.array(series[a], dtype=float)
        vb = np.array(series[b], dtype=float)
        m = np.isfinite(va) & np.isfinite(vb)
        if m.sum() < 4:
            continue
        va, vb = va[m], vb[m]
        d = va - vb
        t, p = ttest_rel(va, vb)
        w = wilcoxon(va, vb).pvalue if np.any(d != 0) else np.nan
        # 残差是否显著不为 0
        t0 = va.mean() / (va.std(ddof=1) / np.sqrt(len(va))) if va.std(ddof=1) > 0 else np.nan
        p0 = float(2 * (1 - stats.t.cdf(abs(t0), len(va) - 1))) if np.isfinite(t0) else np.nan
        paired[f"{a}_vs_{b}"] = dict(d_ic=float(d.mean()),
                                     win=int((d > 0).sum()), n=int(len(d)),
                                     paired_t=float(t), paired_p=float(p),
                                     wilcoxon_p=float(w) if np.isfinite(w) else None,
                                     a_ic_mean=float(va.mean()),
                                     a_t_vs_zero=float(t0), a_p_vs_zero=float(p0))
        print(f"{a + ' - ' + b:<34}{d.mean():>+9.4f}{int((d>0).sum())}/{len(d):<5}"
              f"{t:>9.2f}{p:>8.3f}")
    print("\n（残差 IC 显著不为 0 才算真增量）")
    for a in ("resid_rv", "resid_ret", "resid_rv_comp", "resid_ret_comp"):
        s = summarize(a, series[a])
        if s["n"] < 3:
            continue
        v = np.array([x for x in series[a] if np.isfinite(x)])
        t0 = s["ic"] / (v.std(ddof=1) / np.sqrt(len(v))) if v.std(ddof=1) > 0 else np.nan
        p0 = float(2 * (1 - stats.t.cdf(abs(t0), len(v) - 1))) if np.isfinite(t0) else np.nan
        print(f"  {a:<16} 均值 IC={s['ic']:+.4f}  对 0 的 t={t0:+.2f}  p={p0:.3f}  n={s['n']}")

    # ---- 落盘 ----
    pd.DataFrame(detail).to_csv(OUT / f"monthly_{args.tag}.csv",
                                index=False, encoding="utf-8")
    with open(OUT / f"result_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(dict(tag=args.tag, n_months=len(months),
                       overlap_rows=int(len(use)),
                       overlap_stocks=int(use.Stkcd.nunique()),
                       n_factors=len(kept),
                       factor_names=[f.name for f in kept],
                       summary=rows, paired=paired,
                       note="拓扑因子用扩张窗口构造，严格只用 t 月之前的历史；"
                            "样本仅 2023-10 起的 HF 区间，月份数与股票数都很少"),
                      f, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {OUT / f'monthly_{args.tag}.csv'}")
    print(f"[已保存] {OUT / f'result_{args.tag}.json'}")


if __name__ == "__main__":
    main()
