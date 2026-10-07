#!/usr/bin/env python3
"""阶段 3 第 2 层a：CSMAR 日频高频指标序列的拓扑指纹。

数据实测结论（2026-10-06）
-------------------------
CSMAR 全量清单（198 库 / 5001 表）**没有任何日内/逐笔表**；HF 系列全部是**日频**：
  HF_BSImbalance  3,940,390 行 × 42 列 / 5,458 股 / 722 日 / 2023-10-09–2026-09-28
  HF_Spread       3,940,390 行 × 19 列
  HF_StockRealized 3,964,339 行 × 7 列
因此本层不做逐笔，做的是**日频微观结构状态序列**的滚动延迟嵌入 + PH。

流程
----
1. 载入三张 HF 表 → 合并 → 派生 8 个通道（买卖不平衡 / 价差 / 已实现波动）
2. 每只股票按时间轴 → 滚动窗口（默认 120 日，步长 20 日）
3. 每个窗口：逐通道延迟嵌入 → PH → 固定 range 向量化；同窗口经典基线
4. 标签（严格无前视）：未来 h 日平均 RV、未来 h 日收益（来自 csmar_dalyr_full）
5. 时序 purged walk-forward 初检 + 描述统计
6. 特征矩阵落盘，供第 3 层做横截面因子评估

用法
----
    python src/experiment_3b_csmar_hf.py --quick
    python src/experiment_3b_csmar_hf.py --n-stocks 300
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
                         crit_ring_count, crit_pers_entropy, purged_walk_forward,
                         takens_embed)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp3b_csmar_hf"

DEFAULT_DATA = "/mnt/c/Users/LENOVO/.dsh-tools/csmar-data/model-data"
DEFAULT_DALYR = "/mnt/c/Users/LENOVO/.dsh-tools/csmar-data/csmar_dalyr_full.parquet"

M_EMB = 3
TAU_EMB = 5

BS_COLS = ["Stkcd", "Trddt", "B_Volume", "S_Volume", "B_Num", "S_Num",
           "B_Order", "S_Order", "B_Volume_L", "S_Volume_L",
           "B_Volume_S", "S_Volume_S"]
SP_COLS = ["Stkcd", "Trddt", "Qsp_equal", "Esp_equal"]
RV_COLS = ["Stkcd", "Trddt", "RV", "RSkew"]

CHANNELS_3B = ["bs_vol_imb", "bs_num_imb", "bs_order_imb", "large_vol_imb",
               "small_vol_imb", "spread_q", "spread_e", "rv"]
N_CH3 = len(CHANNELS_3B)


# --------------------------------------------------------------------------
# 数据装配
# --------------------------------------------------------------------------
def load_panel(data_root: str, n_stocks: int, start: str | None,
               min_days: int = 600, min_cover: float = 0.90,
               seed: int = 0) -> pd.DataFrame:
    """选样本股 → 读三张 HF 表 → 派生 8 个通道。

    ⚠️ CSMAR 的 parquet 把数值列存成 large_string，
    且**缺失不是空值而是字面字符串 `'nan'`**，必须 `pd.to_numeric(errors="coerce")`。

    ⚠️ 样本选择必须做**双条件筛选**（这是踩过的坑）：
    只按"交易日数最多"取前 N 只，会优先选中 `B_Order`/`S_Order` **整体缺失**的
    349 只股票（含一批 920xxx 北交所代码），导致 `bs_order_imb` 通道 100% 为空。
    正确做法：先要求 `min_days` 个交易日 **且** 关键列可用率 ≥ `min_cover`，再按完整度取前 N。
    """
    bs = pd.read_parquet(os.path.join(data_root, "HF_BSImbalance.parquet"),
                         columns=BS_COLS)
    bs["Trddt"] = pd.to_datetime(bs["Trddt"])
    bs["Stkcd"] = bs["Stkcd"].astype(str)
    if start:
        bs = bs[bs.Trddt >= pd.Timestamp(start)]

    # 数值化（'nan' 字符串 → NaN）
    for c in BS_COLS:
        if c not in ("Stkcd", "Trddt"):
            bs[c] = pd.to_numeric(bs[c], errors="coerce")

    # 股票可用性：交易日数 + 关键列可用率
    g = bs.groupby("Stkcd").agg(days=("Trddt", "nunique"),
                                order_ok=("B_Order", "count"),
                                sorder_ok=("S_Order", "count"),
                                bvol_ok=("B_Volume", "count"))
    g["days"] = g["days"].astype(float)
    for c in ("order_ok", "sorder_ok", "bvol_ok"):
        g[c] = g[c] / g["days"].clip(lower=1)
    elig = g[(g.days >= min_days) & (g.order_ok >= min_cover) &
             (g.sorder_ok >= min_cover) & (g.bvol_ok >= min_cover)]
    print(f"  可选股票池 {len(elig)} / {len(g)} 只"
          f"（条件：交易日 ≥{min_days} 且关键列可用率 ≥{min_cover:.0%}）"
          f"；被剔除 {len(g) - len(elig)} 只")

    if len(elig) < n_stocks:
        print(f"  [warn] 可用股票不足 {n_stocks} 只，放宽到 {len(elig)} 只")
    # 取完整度最高的 n_stocks 只；同分按 Stkcd 排序，保证可复现
    keep = set(elig.sort_values(["days", "Stkcd"], ascending=[False, True])
               .index[:n_stocks])
    bs = bs[bs.Stkcd.isin(keep)]
    print(f"  样本股 {len(keep)} 只，BSImbalance {len(bs):,} 行，"
          f"平均可用率 B_Order={bs.groupby('Stkcd').B_Order.count().mean()/bs.groupby('Stkcd').Trddt.nunique().mean():.1%}")

    sp = pd.read_parquet(os.path.join(data_root, "HF_Spread.parquet"), columns=SP_COLS)
    sp["Trddt"] = pd.to_datetime(sp["Trddt"])
    sp["Stkcd"] = sp["Stkcd"].astype(str)
    sp = sp[sp.Stkcd.isin(keep)]
    for c in ("Qsp_equal", "Esp_equal"):
        sp[c] = pd.to_numeric(sp[c], errors="coerce")

    rv = pd.read_parquet(os.path.join(data_root, "HF_StockRealized.parquet"),
                         columns=RV_COLS)
    rv["Trddt"] = pd.to_datetime(rv["Trddt"])
    rv["Stkcd"] = rv["Stkcd"].astype(str)
    rv = rv[rv.Stkcd.isin(keep)]
    for c in ("RV", "RSkew"):
        rv[c] = pd.to_numeric(rv[c], errors="coerce")

    df = bs.merge(sp, on=["Stkcd", "Trddt"], how="inner") \
           .merge(rv, on=["Stkcd", "Trddt"], how="inner")
    df = df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)

    eps = 1e-9
    out = pd.DataFrame({
        "Stkcd": df.Stkcd, "Trddt": df.Trddt,
        "bs_vol_imb": (df.B_Volume - df.S_Volume) / (df.B_Volume + df.S_Volume + eps),
        "bs_num_imb": (df.B_Num - df.S_Num) / (df.B_Num + df.S_Num + eps),
        "bs_order_imb": (df.B_Order - df.S_Order) / (df.B_Order + df.S_Order + eps),
        "large_vol_imb": (df.B_Volume_L - df.S_Volume_L) /
                         (df.B_Volume_L + df.S_Volume_L + eps),
        "small_vol_imb": (df.B_Volume_S - df.S_Volume_S) /
                         (df.B_Volume_S + df.S_Volume_S + eps),
        "spread_q": df.Qsp_equal.astype(float),
        "spread_e": df.Esp_equal.astype(float),
        "rv": df.RV.astype(float),
        "rskew": df.RSkew.astype(float),
    })
    return out.replace([np.inf, -np.inf], np.nan)


def attach_returns(panel: pd.DataFrame, dalyr_path: str) -> pd.DataFrame:
    """从 csmar_dalyr_full 取 CSMAR 官方日收益与涨跌停标记。

    用 `Dretwd`（考虑现金红利再投资的日收益率）而非自算收益 —— 避免复权口径问题。
    `LimitUp` / `LimitDown` 用于第 2 层的弱标签（封板日 vs 正常日）。

    ⚠️ 该表 18,933,381 行，**必须分批读**（`iter_batches` + 早过滤），
    否则一次性 read_parquet 会把 WSL 的 7.5 GB 打爆。
    """
    import pyarrow.parquet as pq

    want = ["Stkcd", "Trddt", "Dretwd", "LimitUp", "LimitDown", "Trdsta"]
    try:
        pf = pq.ParquetFile(dalyr_path)
        names = pf.schema_arrow.names
        cols = [c for c in want if c in names]
        missing = [c for c in want if c not in names]
        if missing:
            print(f"  [warn] dalyr 缺少列 {missing}，将跳过")
    except Exception as e:      # noqa: BLE001
        print(f"  [warn] 无法读取 {dalyr_path} ({type(e).__name__}: {e})，收益标签不可用")
        return panel

    keep = set(panel["Stkcd"].astype(str).unique())
    d_lo = panel["Trddt"].min().strftime("%Y-%m-%d")
    d_hi = panel["Trddt"].max().strftime("%Y-%m-%d")

    frames, n_read = [], 0
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
        d = batch.to_pandas()
        n_read += len(d)
        d["Stkcd"] = d["Stkcd"].astype(str)
        d = d[d["Stkcd"].isin(keep)]
        if d.empty:
            continue
        # Trddt 是 "YYYY-MM-DD" 字符串，字典序即时间序
        d = d[(d["Trddt"] >= d_lo) & (d["Trddt"] <= d_hi)]
        if len(d):
            frames.append(d)
    if not frames:
        print(f"  [warn] dalyr 扫描 {n_read:,} 行未匹配到样本股/日期范围，收益标签不可用")
        return panel
    d = pd.concat(frames, ignore_index=True)
    print(f"  dalyr 扫描 {n_read:,} 行 → 命中 {len(d):,} 行")

    d["Trddt"] = pd.to_datetime(d["Trddt"])
    d = d.rename(columns={"Dretwd": "ret"})
    if "ret" in d.columns:
        d["ret"] = pd.to_numeric(d["ret"], errors="coerce")
    for c in ("LimitUp", "LimitDown"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    return panel.merge(d, on=["Stkcd", "Trddt"], how="left")


# --------------------------------------------------------------------------
# 单窗口特征
# --------------------------------------------------------------------------
def _prep_series(x, min_finite=0.8):
    """把一条窗口序列整成可嵌入的形式。

    返回 None 表示该通道在窗口内不可用（有效值太少 / 常量）。
    ⚠️ 不要用 `np.nan_to_num` 填 NaN —— 会把缺失变成 0，制造假跳变。
    """
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.mean() < min_finite:
        return None
    if not ok.all():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[ok], x[ok])
    if np.std(x) < 1e-12:
        return None
    return x


def window_features(args):
    """(Stkcd, t_end, t_date, win_matrix, fwd_rv, fwd_ret) -> dict"""
    stk, t_end, t_date, mat, fwd_rv, fwd_ret = args
    mat = np.asarray(mat, dtype=np.float64)

    diags, rings, ent, valid = [], [], [], []
    for c in range(mat.shape[1]):
        x = _prep_series(mat[:, c])
        if x is None:
            diags.append(np.zeros((0, 2)))
            rings.append(0.0)
            ent.append(0.0)
            valid.append(0)
            continue
        try:
            cl = clean_cloud(takens_embed(x, m=M_EMB, tau=TAU_EMB),
                             normalize="unit_sphere")
            h = ph_diagrams_safe(cl)
        except Exception:      # noqa: BLE001  退化窗口（去重后点数不足）
            h = np.zeros((0, 2))
        diags.append(h)
        rings.append(crit_ring_count(h, 0.3))
        ent.append(crit_pers_entropy(h))
        valid.append(1)

    # 经典基线 / 原始统计也只在有效通道上算，用中位数补退化位置保持定长
    bl_parts, raw_parts = [], []
    for c in range(mat.shape[1]):
        x = _prep_series(mat[:, c], min_finite=0.5)
        if x is None:
            bl_parts.append(np.zeros(8))
            raw_parts.append(np.array([np.nan, np.nan]))
        else:
            bl_parts.append(baseline_features(x))
            raw_parts.append(np.array([float(np.mean(x)), float(np.std(x))]))
    bl = np.concatenate(bl_parts)
    raw = np.concatenate(raw_parts)

    return dict(stk=stk, t_end=int(t_end), t_date=str(t_date), diagrams=diags,
                ring=float(np.max(rings)), ring_mean=float(np.mean(rings)),
                ent=float(np.mean(ent)), ring_ch=rings, n_valid=int(sum(valid)),
                baseline=bl, raw=raw, fwd_rv=fwd_rv, fwd_ret=fwd_ret)


def ph_diagrams_safe(cloud):
    from tda3_common import ph_diagrams
    h = ph_diagrams(cloud, maxdim=1)[1]
    return h[np.isfinite(h[:, 1])] if len(h) else np.zeros((0, 2))


def build_windows(panel, win, stride, h_rv, h_ret):
    """严格无前视：窗口 = [s, e)，标签 = [e, e+h)。"""
    jobs = []
    for stk, g in panel.groupby("Stkcd", sort=False):
        g = g.reset_index(drop=True)
        mat = g[CHANNELS_3B].to_numpy(dtype=np.float64)
        rvv = g["rv"].to_numpy(dtype=np.float64)
        retv = (g["ret"].to_numpy(dtype=np.float64)
                if "ret" in g.columns else None)
        T = len(g)
        if T < win + max(h_rv, h_ret) + 5:
            continue
        for s in range(0, T - win + 1, stride):
            e = s + win
            if e + h_rv > T:
                break
            fwd_rv = float(np.nanmean(rvv[e:e + h_rv]))
            fwd_ret = np.nan
            if retv is not None:
                r = retv[e:e + h_ret]
                if np.isfinite(r).sum() >= max(2, h_ret // 2):
                    fwd_ret = float(np.nansum(r))
            jobs.append((str(stk), e - 1, str(g["Trddt"].iloc[e - 1].date()),
                         mat[s:e], fwd_rv, fwd_ret))
    return jobs


def run_jobs(jobs, workers):
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(window_features, jobs, chunksize=8)):
            res.append(r)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"    [{i+1}/{len(jobs)}] {el:6.1f}s ({el/(i+1)*1000:.1f}ms/窗口)",
                      flush=True)
    return res


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--dalyr", default=DEFAULT_DALYR)
    ap.add_argument("--n-stocks", type=int, default=300)
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--h-rv", type=int, default=20)
    ap.add_argument("--h-ret", type=int, default=20)
    ap.add_argument("--start", default=None)
    ap.add_argument("--min-days", type=int, default=600,
                    help="样本股最少交易日数")
    ap.add_argument("--min-cover", type=float, default=0.90,
                    help="关键列最低可用率（B_Order/S_Order/B_Volume）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.quick:
        args.n_stocks = 40
        args.stride = 40
        args.workers = min(args.workers, 6)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"n{args.n_stocks}_win{args.win}_s{args.stride}"

    print("=" * 84)
    print(f"阶段3 第2层a：CSMAR 日频 HF 序列的拓扑指纹  tag={tag}")
    print(f"  n_stocks={args.n_stocks}  win={args.win}  stride={args.stride}  "
          f"h_rv={args.h_rv}  h_ret={args.h_ret}")
    print(f"  嵌入 m={M_EMB} tau={TAU_EMB}  通道({N_CH3})={CHANNELS_3B}")
    print("=" * 84)

    t0 = time.time()
    print("\n[1/5] 载入并合并 HF 表 …")
    panel = load_panel(args.data_root, args.n_stocks, args.start,
                       min_days=args.min_days, min_cover=args.min_cover)
    print(f"  面板 shape={panel.shape}  股票数={panel.Stkcd.nunique()}  "
          f"日期 {panel.Trddt.min().date()} → {panel.Trddt.max().date()}")
    print(f"  缺失率：" + " ".join(
        f"{c}={panel[c].isna().mean():.1%}" for c in CHANNELS_3B))

    print("\n[2/5] 附加收益（无前视）…")
    panel = attach_returns(panel, args.dalyr)
    has_ret = "ret" in panel.columns
    print(f"  收益可用={has_ret}")

    print("\n[3/5] 构造滚动窗口 …")
    jobs = build_windows(panel, args.win, args.stride, args.h_rv, args.h_ret)
    print(f"  {len(jobs)} 个窗口（{len(jobs)/max(panel.Stkcd.nunique(),1):.1f} 个/股）")
    if not jobs:
        print("[中止] 没有可用窗口，检查数据或参数")
        return

    print("\n[4/5] 逐窗口计算 PH + 经典基线 …")
    res = run_jobs(jobs, args.workers)
    print(f"  完成，用时 {time.time()-t0:.0f}s")

    # ---- 固定 range 向量化（维度不漂移）----
    vec = FixedVectorizer(n_grid=12, n_landscape=2)
    vec.fit_range([d for r in res for d in r["diagrams"]])
    XT = np.array([np.concatenate([vec.transform_one(d) for d in r["diagrams"]])
                   for r in res])
    XC = np.array([r["baseline"] for r in res])
    XR = np.array([r["raw"] for r in res])
    # RAW 里有退化通道留下的 NaN：用列中位数补，避免 IC 直接变成 nan
    if not np.isfinite(XR).all():
        med = np.nanmedian(XR, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        XR = np.where(np.isfinite(XR), XR, med)
    ring = np.array([r["ring"] for r in res])
    ring_mean = np.array([r["ring_mean"] for r in res])
    y_rv = np.array([r["fwd_rv"] for r in res])
    y_ret = np.array([r["fwd_ret"] for r in res])
    idx = pd.DataFrame({"Stkcd": [r["stk"] for r in res],
                        "date": [r["t_date"] for r in res]})
    idx["date"] = pd.to_datetime(idx["date"])
    print(f"  [维度] 拓扑={XT.shape[1]}  经典={XC.shape[1]}  原始统计={XR.shape[1]}")

    # ---- 描述统计：拓扑判据在真实数据上的分布 ----
    print("\n[5/5] 描述统计与初检")
    desc = dict(
        n_windows=int(len(res)), n_stocks=int(idx.Stkcd.nunique()),
        ring_max_mean=float(np.mean(ring)), ring_max_std=float(np.std(ring)),
        ring_max_p10=float(np.percentile(ring, 10)),
        ring_max_p50=float(np.percentile(ring, 50)),
        ring_max_p90=float(np.percentile(ring, 90)),
        ring_mean_mean=float(np.mean(ring_mean)),
        ring_channel_mean={CHANNELS_3B[i]: float(np.mean([r["ring_ch"][i] for r in res]))
                           for i in range(N_CH3)},
    )
    print(f"  窗口数={desc['n_windows']}  股票数={desc['n_stocks']}")
    print(f"  逐通道环数(>30%max) 最大值的分布："
          f"p10={desc['ring_max_p10']:.1f} p50={desc['ring_max_p50']:.1f} "
          f"p90={desc['ring_max_p90']:.1f}")
    print("  各通道平均环数：" + "  ".join(
        f"{k}={v:.1f}" for k, v in desc["ring_channel_mean"].items()))

    # ---- 时序 purged walk-forward：特征组 vs 前视 RV / 收益 ----
    def ic_report(X, y, name, folds):
        from scipy.stats import spearmanr
        ics = []
        for tr, te in folds:
            if len(te) < 20:
                continue
            # 用训练集选与 y 相关性最强的 top-20 特征，再在测试集算 IC
            with np.errstate(invalid="ignore"):
                cors = np.array([spearmanr(X[tr, j], y[tr]).correlation
                                 if np.std(X[tr, j]) > 1e-12 else 0.0
                                 for j in range(X.shape[1])])
            cors = np.nan_to_num(cors)
            top = np.argsort(-np.abs(cors))[:20]
            pred = X[te][:, top] @ cors[top]
            if np.std(pred) < 1e-12 or np.std(y[te]) < 1e-12:
                continue
            ics.append(float(spearmanr(pred, y[te]).correlation))
        ics = np.array([v for v in ics if np.isfinite(v)])
        if len(ics) == 0:
            return dict(name=name, n_folds=0, ic_mean=float("nan"),
                        ic_std=float("nan"), icir=float("nan"), t_nw=float("nan"))
        mean, sd = float(ics.mean()), float(ics.std(ddof=1)) if len(ics) > 1 else float("nan")
        icir = mean / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        t_nw = mean / (sd / np.sqrt(len(ics))) if sd and np.isfinite(sd) and sd > 0 else float("nan")
        return dict(name=name, n_folds=int(len(ics)), ic_mean=mean, ic_std=sd,
                    icir=icir, t_nw=t_nw,
                    ic_series=[float(v) for v in ics])

    folds = purged_walk_forward(len(res), n_splits=5,
                                label_horizon=args.h_rv, embargo=5)
    print(f"  purged walk-forward: {len(folds)} 折 "
          f"(label_horizon={args.h_rv}, embargo=5)")
    ic_res = {}
    for tname, y in [("fwd_rv20", y_rv)] + ([("fwd_ret20", y_ret)] if has_ret else []):
        ok = np.isfinite(y)
        # ⚠️ RAW（窗口内原始均值/标准差）必须作为基准：
        #    波动率有极强自相关，单靠"过去 RV 的水平"就能拿到很高的 IC。
        #    "拓扑有没有增量"要看 TOPO+RAW 是否显著优于 RAW，而不是看 TOPO 单独的 IC。
        for fname, X in [("TOPO", XT), ("CLASSIC", XC), ("RAW", XR),
                         ("TOPO+CLASSIC", np.hstack([XT, XC])),
                         ("TOPO+RAW", np.hstack([XT, XR])),
                         ("CLASSIC+RAW", np.hstack([XC, XR])),
                         ("ALL", np.hstack([XT, XC, XR]))]:
            r = ic_report(X[ok], y[ok], f"{fname}->{tname}",
                          purged_walk_forward(int(ok.sum()), 5, args.h_rv, 5))
            ic_res[r["name"]] = r
            print(f"    {r['name']:<24} IC={r['ic_mean']:+.4f}  "
                  f"ICIR={r['icir']:+.3f}  t={r['t_nw']:+.2f}  folds={r['n_folds']}")

    # ---- 落盘 ----
    np.savez_compressed(OUT / f"features_{tag}.npz",
                        XT=XT, XC=XC, XR=XR, ring=ring, ring_mean=ring_mean,
                        y_rv=y_rv, y_ret=y_ret,
                        meta=json.dumps(dict(channels=CHANNELS_3B, m=M_EMB,
                                             tau=TAU_EMB, win=args.win,
                                             stride=args.stride,
                                             h_rv=args.h_rv, h_ret=args.h_ret,
                                             n_stocks=args.n_stocks)))
    idx.to_csv(OUT / f"index_{tag}.csv", index=False)
    with open(OUT / f"result_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(dict(tag=tag, desc=desc, ic=ic_res,
                       args=vars(args)), f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"\n[已保存] {OUT / f'features_{tag}.npz'}")
    print(f"[已保存] {OUT / f'index_{tag}.csv'}   {OUT / f'result_{tag}.json'}")


if __name__ == "__main__":
    main()
