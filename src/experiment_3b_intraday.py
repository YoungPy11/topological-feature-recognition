#!/usr/bin/env python3
"""阶段 3 第 2 层b：日内 5 分钟序列的拓扑指纹（baostock）。

与第 2 层a 的关系
----------------
2a 用的是 CSMAR **日频**微观结构序列（窗口 = 120 个交易日，样本 = 股票）；
2b 用的是 baostock **日内 5 分钟**序列（窗口 = 一个交易日 48 个 bar，样本 = 股票×日）。
2b 才能真正看到"下单节律"——日频看不到订单流模式。

样本与通道
----------
* 样本：(股票, 交易日)，每个样本一条 48 点的日内序列
* 通道 4 条：`ret`（bar 收益）、`logvol`（log 成交量）、`rng`（振幅）、`lamt`（log 成交额）
* **日内 U 型季节调整（严格 PIT）**：减去过去 `--season-win` 天**同一 bar 序号**的均值。
  不做这一步的话每天长得都一样，PH 只会检测到 U 型本身。

向量化用**固定 range (0, 2)**
--------------------------
`clean_cloud(unit_sphere)` 保证点云最大半径 = 1 → 任意两点距离 ≤ 2，
所以 H1 的 birth/death 必落在 [0,2]，直接用固定网格即可，**不需要数据驱动 fit**（也就不需要两遍扫描）。

标签与任务
----------
* T1 涨停日分类：涨停日 vs 正常日（阈值按板块：创业板/科创板 20%，其余 10%，取 98%）
* T2 purged walk-forward IC：预测未来 5 日已实现波动（日收益标准差）

用法
----
    python src/experiment_3b_intraday.py --n-stocks 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tda3_common import (FixedVectorizer, baseline_features, clean_cloud,   # noqa: E402
                         crit_pers_entropy, crit_ring_count, ph_diagrams,
                         purged_walk_forward, takens_embed)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "intraday"
OUT = ROOT / "results" / "exp3b_intraday"

M_EMB = 3
TAU_EMB = 2
N_BAR = 48
CHANNELS = ["ret", "logvol", "rng", "lamt"]
N_CH = len(CHANNELS)

# 点云经 unit_sphere 归一化后直径 ≤ 2 → 固定 range
VEC = FixedVectorizer(n_grid=12, n_landscape=2,
                      birth_range=(0.0, 2.0), death_range=(0.0, 2.0))


def ph_safe(x):
    try:
        cl = clean_cloud(takens_embed(x, m=M_EMB, tau=TAU_EMB),
                         normalize="unit_sphere")
        h = ph_diagrams(cl, maxdim=1)[1]
        return h[np.isfinite(h[:, 1])] if len(h) else np.zeros((0, 2))
    except Exception:      # noqa: BLE001
        return np.zeros((0, 2))


def bars_to_channels(g: pd.DataFrame):
    """一只股票的 5 分钟 bar → (dates, arr[n_days, 48, C])，缺 bar 的日子丢弃。"""
    g = g.copy()
    g["bar"] = g["time"].astype(str).str[8:12]
    # 48 个合法 bar 序号（09:35..11:30, 13:05..15:00）
    g = g.sort_values(["date", "time"])
    days, arrs, closes = [], [], []
    for d, sub in g.groupby("date", sort=True):
        if len(sub) != N_BAR:
            continue
        c = sub["close"].to_numpy(dtype=float)
        if not np.isfinite(c).all() or np.any(c <= 0):
            continue
        v = sub["volume"].to_numpy(dtype=float)
        h = sub["high"].to_numpy(dtype=float)
        lo = sub["low"].to_numpy(dtype=float)
        a = sub["amount"].to_numpy(dtype=float)
        cprev = np.r_[c[0], c[:-1]]
        ret = np.log(np.maximum(c, 1e-9) / np.maximum(cprev, 1e-9))
        ch = np.column_stack([
            ret,
            np.log1p(np.maximum(v, 0.0)),
            (h - lo) / np.maximum(c, 1e-9),
            np.log1p(np.maximum(a, 0.0)),
        ])
        if not np.isfinite(ch).all():
            continue
        days.append(str(d))
        arrs.append(ch)
        closes.append(float(c[-1]))
    if not days:
        return None, None, None
    return np.array(days), np.stack(arrs), np.array(closes)


def seasonal_adjust(arr, win=60):
    """PIT 日内季节调整：减去过去 win 天同一 bar 序号的均值。

    arr: (n_days, 48, C) → 同形状的调整后数组（前 win 天用可得的更短历史）。
    """
    n, B, C = arr.shape
    out = np.empty_like(arr)
    csum = np.cumsum(arr, axis=0)                    # (n, B, C)
    for i in range(n):
        lo = max(0, i - win)
        m = csum[i] - (csum[lo - 1] if lo > 0 else 0.0)
        m = m / (i - lo + 1)
        out[i] = arr[i] - m
    return out


def process_stock(stkcd):
    p = CACHE / f"{stkcd}.parquet"
    if not p.exists():
        return None
    g = pd.read_parquet(p)
    if len(g) == 0:
        return None
    days, arr, closes = bars_to_channels(g)
    if days is None or len(days) < 120:
        return None

    adj = seasonal_adjust(arr, win=60)

    n = len(days)
    topo, classic, raw, ring = [], [], [], []
    for i in range(n):
        w = adj[i]                                   # (48, C)
        t_vec, c_vec, r_vec = [], [], []
        rgs = []
        for c in range(N_CH):
            x = w[:, c]
            if np.std(x) < 1e-12:
                t_vec.append(VEC.transform_one(np.zeros((0, 2))))
                c_vec.append(np.zeros(8))
                r_vec.extend([0.0, 0.0])
                rgs.append(0.0)
                continue
            h = ph_safe(x)
            t_vec.append(VEC.transform_one(h))
            c_vec.append(baseline_features(x))
            r_vec.extend([float(np.mean(x)), float(np.std(x))])
            rgs.append(crit_ring_count(h, 0.3))
        topo.append(np.concatenate(t_vec))
        classic.append(np.concatenate(c_vec))
        raw.append(np.array(r_vec))
        ring.append(float(np.max(rgs)))
    return dict(stkcd=str(stkcd), days=days, closes=closes,
                topo=np.array(topo), classic=np.array(classic),
                raw=np.array(raw), ring=np.array(ring))


def limit_threshold(stkcd):
    s = str(stkcd).zfill(6)
    if s.startswith("3") or s.startswith("688"):
        return 0.20
    return 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stocks", type=int, default=120)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--season-win", type=int, default=60)
    ap.add_argument("--h-rv", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p.stem for p in CACHE.glob("*.parquet"))
    if not files:
        print(f"[中止] {CACHE} 里没有 5 分钟数据，先跑 src/fetch_intraday.py")
        return
    stocks = files[:args.n_stocks]
    tag = args.tag or f"n{len(stocks)}_5min"

    print("=" * 84)
    print(f"阶段3 第2层b：日内 5 分钟拓扑指纹   tag={tag}")
    print(f"  股票 {len(stocks)} 只（缓存里有 {len(files)} 只）")
    print(f"  样本 = (股票, 交易日)  ·  窗口 = 48 个 5 分钟 bar  ·  通道 {CHANNELS}")
    print(f"  嵌入 m={M_EMB} tau={TAU_EMB}  ·  季节调整窗口 = {args.season_win} 天（PIT）")
    print(f"  向量化 range 固定 (0,2) —— unit_sphere 后直径 ≤ 2")
    print(f"  workers={args.workers}")
    print("=" * 84)

    t0 = time.time()
    print("\n[1/4] 逐股票构造特征 …")
    res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_stock, stocks, chunksize=1)):
            if r is not None:
                res.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(stocks):
                el = time.time() - t0
                print(f"    [{i+1}/{len(stocks)}] {el:.1f}s "
                      f"({el/(i+1):.2f}s/股票)  有效 {len(res)}", flush=True)
    print(f"  完成，用时 {time.time()-t0:.0f}s")

    print("\n[2/4] 拼接面板与标签 …")
    rows = []
    XT, XC, XR, RING = [], [], [], []
    for r in res:
        n = len(r["days"])
        c = r["closes"]
        # ⚠️ 涨停判定必须用**简单收益率**：+10% 涨停的对数收益只有 9.53%，
        #    用 log 收益配 9.8% 阈值会一个都检不出来（踩过）。
        sret = np.r_[np.nan, c[1:] / c[:-1] - 1.0]
        lret = np.r_[np.nan, np.log(c[1:] / c[:-1])]
        lim = limit_threshold(r["stkcd"])
        for i in range(n):
            # 未来 h 日已实现波动（严格无前视）；用对数收益算波动
            fwd = lret[i + 1:i + 1 + args.h_rv]
            fwd_rv = float(np.nanstd(fwd, ddof=1)) if np.isfinite(fwd).sum() >= 3 else np.nan
            is_limit = int(np.isfinite(sret[i]) and sret[i] > 0.98 * lim)
            rows.append(dict(stkcd=r["stkcd"], date=r["days"][i],
                             ret=lret[i], sret=sret[i],
                             fwd_rv=fwd_rv, is_limit=is_limit))
            XT.append(r["topo"][i])
            XC.append(r["classic"][i])
            XR.append(r["raw"][i])
            RING.append(r["ring"][i])
    XT = np.array(XT); XC = np.array(XC); XR = np.array(XR); RING = np.array(RING)
    idx = pd.DataFrame(rows)
    idx["date"] = pd.to_datetime(idx["date"])
    print(f"  样本 {len(idx):,}  股票 {idx.stkcd.nunique()}  "
          f"日期 {idx.date.min().date()} → {idx.date.max().date()}")
    print(f"  维度：TOPO={XT.shape[1]}  CLASSIC={XC.shape[1]}  RAW={XR.shape[1]}")
    print(f"  涨停日 {idx.is_limit.sum():,} ({idx.is_limit.mean():.3%})  "
          f"fwd_rv 缺失 {idx.fwd_rv.isna().mean():.2%}")

    print("\n[3/4] T1：涨停日 vs 正常日 分类")
    print("    协议：分层 5 折；**每折只用训练集**做单变量筛选取 top-30，再拟合 RF")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = idx.is_limit.to_numpy()

    def topk_by_auc(Xtr, ytr, k=30):
        aucs = np.full(Xtr.shape[1], 0.5)
        pos = ytr == 1
        if pos.sum() == 0 or (~pos).sum() == 0:
            return np.arange(min(k, Xtr.shape[1]))
        for j in range(Xtr.shape[1]):
            col = Xtr[:, j]
            if np.std(col) < 1e-12:
                continue
            from sklearn.metrics import roc_auc_score as _auc
            try:
                aucs[j] = _auc(ytr, col)
            except Exception:      # noqa: BLE001
                pass
        return np.argsort(-np.abs(aucs - 0.5))[:k]

    t1 = {}
    if y.sum() < 20 or y.sum() == len(y):
        print(f"    [跳过] 涨停日样本仅 {int(y.sum())} 个，不足以训练")
    else:
        for name, X in [("TOPO", XT), ("CLASSIC", XC), ("RAW", XR),
                        ("TOPO+RAW", np.hstack([XT, XR])),
                        ("TOPO+CLASSIC+RAW", np.hstack([XT, XC, XR])),
                        ("RING_only", RING[:, None])]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            oof = np.zeros(len(y))
            ok_folds = True
            for tr, te in skf.split(X, y):
                if len(np.unique(y[tr])) < 2:
                    ok_folds = False
                    break
                idxk = topk_by_auc(X[tr], y[tr], k=30)
                clf = make_pipeline(StandardScaler(),
                                    RandomForestClassifier(n_estimators=100,
                                                           min_samples_leaf=10,
                                                           n_jobs=1,
                                                           random_state=0))
                clf.fit(X[tr][:, idxk], y[tr])
                oof[te] = clf.predict_proba(X[te][:, idxk])[:, 1]
            if not ok_folds:
                continue
            t1[name] = float(roc_auc_score(y, oof))
            print(f"    {name:<20} AUC={t1[name]:.4f}")

    print(f"\n[4/4] T2：purged walk-forward IC（预测未来 {args.h_rv} 日 RV）")

    from scipy.stats import spearmanr

    def ic_report(X, yv, name):
        """X/yv 已是过滤后的数组；折必须按过滤后的长度生成。"""
        n = len(yv)
        folds_local = purged_walk_forward(n, n_splits=5,
                                          label_horizon=args.h_rv, embargo=5)
        ics = []
        for tr, te in folds_local:
            if len(te) < 100:
                continue
            cors = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                col = X[tr, j]
                if np.std(col) < 1e-12:
                    continue
                r = spearmanr(col, yv[tr]).correlation
                cors[j] = 0.0 if not np.isfinite(r) else r
            top = np.argsort(-np.abs(cors))[:20]
            pred = X[te][:, top] @ cors[top]
            if np.std(pred) < 1e-12 or np.std(yv[te]) < 1e-12:
                continue
            r = spearmanr(pred, yv[te]).correlation
            if np.isfinite(r):
                ics.append(float(r))
        ics = np.array(ics)
        if len(ics) == 0:
            return dict(name=name, ic=float("nan"), t=float("nan"), n=0)
        sd = ics.std(ddof=1) if len(ics) > 1 else np.nan
        return dict(name=f"{name}->fwd_rv{args.h_rv}", ic=float(ics.mean()),
                    ic_std=float(sd),
                    t=float(ics.mean() / (sd / np.sqrt(len(ics)))) if sd and sd > 0 else float("nan"),
                    n=int(len(ics)), ic_series=[float(v) for v in ics])

    ok = np.isfinite(idx.fwd_rv.to_numpy())
    yv = idx.fwd_rv.to_numpy()
    print(f"    有效标签 {int(ok.sum()):,} / {len(ok):,}；"
          f"purged walk-forward 5 折（label_horizon={args.h_rv}, embargo=5）")
    ic_res = {}
    for name, X in [("TOPO", XT), ("CLASSIC", XC), ("RAW", XR),
                    ("TOPO+RAW", np.hstack([XT, XR])),
                    ("TOPO+CLASSIC", np.hstack([XT, XC])),
                    ("ALL", np.hstack([XT, XC, XR]))]:
        r = ic_report(X[ok], yv[ok], name)
        ic_res[r["name"]] = r
        print(f"    {r['name']:<22} IC={r['ic']:+.4f}  t={r['t']:+.2f}  folds={r['n']}")

    print("\n    判别力上限参考（RING_only 单标量 AUC，见 T1）")

    np.savez_compressed(OUT / f"features_{tag}.npz",
                        XT=XT, XC=XC, XR=XR, RING=RING,
                        y_limit=idx.is_limit.to_numpy(),
                        y_fwd_rv=idx.fwd_rv.to_numpy(),
                        y_ret=idx.ret.to_numpy(),
                        meta=json.dumps(dict(channels=CHANNELS, m=M_EMB, tau=TAU_EMB,
                                             n_bar=N_BAR, season_win=args.season_win,
                                             h_rv=args.h_rv, n_stocks=len(stocks))))
    idx.to_csv(OUT / f"index_{tag}.csv", index=False)
    with open(OUT / f"result_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(dict(tag=tag, t1_limit=t1, ic=ic_res,
                       n_samples=int(len(idx)),
                       n_stocks=int(idx.stkcd.nunique()),
                       limit_rate=float(idx.is_limit.mean()),
                       args=vars(args)), f, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {OUT / f'features_{tag}.npz'}")
    print(f"[已保存] {OUT / f'index_{tag}.csv'}   {OUT / f'result_{tag}.json'}")


if __name__ == "__main__":
    main()
