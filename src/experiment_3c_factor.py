#!/usr/bin/env python3
"""阶段 3 第 3 层：把拓扑特征当横截面因子来评估（标准因子考核协议）。

为什么要和第 2 层分开
--------------------
第 2 层算的是**时序池化 IC**（把 300 只股票的窗口混在一起算 Spearman），
回答"这个合成分数能否预测该股票的未来波动"。
第 3 层算的是**横截面 Rank IC**（每个月内对股票排序），
回答"这个因子能否在截面上挑出未来波动最高/最低的股票"——这才是量化里标准的因子考核，
也是与 `~/quant-research` 那套 33 因子评估可直接比较的口径。

协议（严格无前视）
----------------
* 横截面：把窗口末日按自然月分桶，每月至少 `--min-names` 只股票才构成一个截面
* 因子选择：**扩张窗口** —— 只用 t 月之前的月份选 top-k 特征，再加权合成 t 月的因子值
* 标签：未来 h 日已实现波动（`fwd_rv20`）/ 未来 h 日收益（`fwd_ret20`）
* 报告：Rank IC 均值 / ICIR / Newey-West t / IC>0 占比

用法
----
    python src/experiment_3c_factor.py --tag n300_win120_s20
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
OUT3B = ROOT / "results" / "exp3b_csmar_hf"
OUT = ROOT / "results" / "exp3c_factor"


def load(tag):
    z = np.load(OUT3B / f"features_{tag}.npz", allow_pickle=True)
    idx = pd.read_csv(OUT3B / f"index_{tag}.csv")
    idx["date"] = pd.to_datetime(idx["date"])
    idx["month"] = idx["date"].dt.to_period("M").astype(str)
    return dict(XT=z["XT"], XC=z["XC"], XR=z["XR"], ring=z["ring"],
                y_rv=z["y_rv"], y_ret=z["y_ret"], idx=idx,
                meta=json.loads(str(z["meta"])))


def nw_tstat(ic):
    """Newey-West 调整的 t（滞后 = floor(4*(T/100)^(2/9))）。"""
    ic = np.asarray([v for v in ic if np.isfinite(v)], dtype=float)
    T = len(ic)
    if T < 4:
        return float("nan"), float("nan"), float("nan")
    mean = ic.mean()
    e = ic - mean
    L = int(np.floor(4 * (T / 100.0) ** (2 / 9)))
    gamma0 = float(e @ e) / T
    s = gamma0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)
        s += 2 * w * float(e[l:] @ e[:-l]) / T
    se = np.sqrt(max(s, 1e-18) / T)
    return mean, float(mean / se) if se > 0 else float("nan"), float(np.std(ic, ddof=1))


def factor_series(feats, names, groups, min_names=20, top_k=20,
                  expanding_min_months=6):
    """扩张窗口因子选择 + 月度横截面 Rank IC。

    返回 dict(group -> ic 序列) 与每个月的截面规模。
    """
    idx = names
    months = sorted(idx["month"].unique())
    X = {g: feats[g] for g in groups}

    ics = {g: [] for g in groups}
    n_used = []
    detail = []
    for mi, m in enumerate(months):
        sel = (idx["month"] == m).to_numpy()
        if sel.sum() < min_names:
            continue
        past = idx["month"].isin(months[:mi]).to_numpy()
        if mi < expanding_min_months or past.sum() < 200:
            continue
        n_used.append(int(sel.sum()))
        for g in groups:
            Xg = X[g]
            if g == "ALL":
                Xg = np.hstack([X[k] for k in ("TOPO", "CLASSIC", "RAW")])
            # 在历史月份上按 |Spearman IC| 选 top-k 特征（严格只用过去）
            yv = names["y_rv"].to_numpy()
            oky = past & np.isfinite(yv)
            if oky.sum() < 100:
                continue
            cors = np.zeros(Xg.shape[1])
            for j in range(Xg.shape[1]):
                col = Xg[:, j]
                if np.std(col[oky]) < 1e-12:
                    continue
                r = spearmanr(col[oky], yv[oky]).correlation
                cors[j] = 0.0 if not np.isfinite(r) else r
            top = np.argsort(-np.abs(cors))[:top_k]
            w = np.sign(cors[top]) * np.abs(cors[top])
            if np.abs(w).sum() < 1e-12:
                continue
            score = Xg[:, top] @ w
            # 截面 Rank IC：当月内 score 与标签的 Spearman
            yv_now = names["y_rv"].to_numpy()
            m_ok = sel & np.isfinite(yv_now) & np.isfinite(score)
            if m_ok.sum() < min_names or np.std(score[m_ok]) < 1e-12:
                continue
            r = spearmanr(score[m_ok], yv_now[m_ok]).correlation
            ic = float(r) if np.isfinite(r) else np.nan
            ics[g].append(ic)
            if g == groups[0]:
                detail.append(dict(month=m, n=int(m_ok.sum()), ic_rv=ic))
    return ics, n_used, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n300_win120_s20")
    ap.add_argument("--min-names", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 84)
    print(f"阶段3 第3层：横截面 Rank IC 因子评估   tag={args.tag}")
    print("=" * 84)

    d = load(args.tag)
    idx = d["idx"].copy()
    idx["y_rv"] = d["y_rv"]
    idx["y_ret"] = d["y_ret"]
    feats = dict(TOPO=d["XT"], CLASSIC=d["XC"], RAW=d["XR"],
                 ALL=np.hstack([d["XT"], d["XC"], d["XR"]]))

    # ⚠️ 去重：步长 20 天 ≈ 1 个月，同一个月内常出现 2 个窗口
    #    （不去重的话截面会被同一只股票灌水，月股票数能到 600 > 300）
    idx["_i"] = np.arange(len(idx))
    kept = (idx.sort_values("date")
               .groupby(["Stkcd", "month"], as_index=False)
               .tail(1)["_i"].to_numpy())
    kept = np.sort(kept)
    print(f"[去重] {len(idx)} → {len(kept)} 行"
          f"（每股票每月只保留末日最新窗口）")
    feats = {k: v[kept] for k, v in feats.items()}
    idx = idx.iloc[kept].reset_index(drop=True)

    print(f"窗口 {len(idx)}  股票 {idx.Stkcd.nunique()}  月份 {idx.month.nunique()}")
    print(f"维度：TOPO={d['XT'].shape[1]}  CLASSIC={d['XC'].shape[1]}  RAW={d['XR'].shape[1]}")
    print(f"协议：月度横截面（≥{args.min_names} 只）· 扩张窗口选 top-{args.top_k} 特征 · 严格只用过去")

    groups = ["TOPO", "CLASSIC", "RAW", "ALL"]
    ics, n_used, detail = factor_series(feats, idx, groups,
                                        min_names=args.min_names, top_k=args.top_k)
    print(f"\n有效横截面 {len(n_used)} 个月，每月股票数 "
          f"min={min(n_used) if n_used else '-'} "
          f"median={int(np.median(n_used)) if n_used else '-'} "
          f"max={max(n_used) if n_used else '-'}")
    print()
    print(f"{'因子组':<10} {'RankIC均值':>10} {'IC标准差':>10} {'ICIR':>8} "
          f"{'NW t':>8} {'IC>0占比':>9} {'月数':>5}")
    print("-" * 72)
    rows = []
    for g in groups:
        v = np.array([x for x in ics[g] if np.isfinite(x)])
        if len(v) == 0:
            continue
        mean, t, sd = nw_tstat(v)
        icir = mean / sd if sd > 0 else np.nan
        pos = float(np.mean(v > 0))
        print(f"{g:<10} {mean:>10.4f} {sd:>10.4f} {icir:>8.3f} {t:>8.2f} {pos:>9.1%} {len(v):>5}")
        rows.append(dict(group=g, rank_ic=mean, ic_std=sd, icir=icir, nw_t=t,
                         positive_ratio=pos, n_months=int(len(v)),
                         ic_series=[float(x) for x in v]))

    payload = dict(tag=args.tag, top_k=args.top_k, min_names=args.min_names,
                   n_months=int(len(n_used)), groups=rows, detail=detail,
                   note="Rank IC 为月度横截面 Spearman；特征选择严格只用 t 月之前的历史")
    with open(OUT / f"factor_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).drop(columns=["ic_series"]).to_csv(
        OUT / f"factor_{args.tag}.csv", index=False, encoding="utf-8")
    print(f"\n[已保存] {OUT / f'factor_{args.tag}.json'} / .csv")


if __name__ == "__main__":
    main()
