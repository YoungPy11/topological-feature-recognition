#!/usr/bin/env python3
"""阶段 3 第 1 层：合成订单流双路线实验（v2，含周期抖动轴）。

实验
----
E1 SNR 扫描（jitter=0） : 经典周期图必然强，作为参照系
E2 jitter 扫描（SNR 固定）: **核心假设**——抖动让谱峰弥散但回路形状保留，
                            PH 的相对优势应随 jitter 上升
T2 六分类（固定 SNR/jitter）
T0 对抗性零模型 S0(AR) vs S4(Hawkes)

路线
----
A 相空间重建 : FNN 残留 + 相空间 kNN 一步预测 R2（无训练 -> 直接对单标量算 AUC）
B 形态指纹   : joint（多通道拼一个点云）与 per-channel（逐通道各自嵌入算 PH）两条
C 经典基线   : 8 个标量 x 7 通道
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ofgen import CHANNELS, N_CH, STRUCTURES, generate                # noqa: E402
from tda3_common import (FixedVectorizer, baseline_features,          # noqa: E402
                         clean_cloud, crit_ring_count, embedding_prediction_r2,
                         fnn_fraction, multichannel_cloud, ph_diagrams,
                         takens_embed)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp3a_synthetic"

STRUCTURED = ["S1_iceberg", "S2_twap", "S3_spoof", "S5_periodic"]

AUC_KEYS = ["A_r2", "A_fnn", "B_ring", "B_ringmax", "B_ph_rf", "B_perch_rf",
            "C_classic_rf", "BC_rf", "PC_C_rf"]

M_EMB = 3
TAU_EMB = 8
USE_CHANNELS = list(range(N_CH))
N_GRID = 24
N_LAND = 4


# --------------------------------------------------------------------------
def det_seed(*parts) -> int:
    """确定性种子。

    ⚠️ **不要用内置 `hash()` 派生随机种子**：字符串/hash 每个进程都加盐
    （PYTHONHASHSEED 随机），同一个 (结构, SNR, 重复号) 在两次运行里会得到
    不同的样本 —— 结果不可复现（本项目踩过：SNR=0.4 两次跑出 0.899 / 0.941）。
    """
    s = "|".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _one_sample(args):
    """一个样本的全部特征（子进程执行）。"""
    structure, snr, rep, mode, T, m, tau, jitter = args
    seed = det_seed(structure, round(snr, 6), round(jitter, 6), rep) % (2 ** 31)
    obs, meta = generate(structure, snr=snr, T=T, seed=seed, mode=mode,
                         jitter=jitter)

    # 路线 A：逐通道诊断取极值（现实协议：不知道信号在哪个通道）
    r2s, fnns = [], []
    diagrams_ch = []
    for c in range(obs.shape[1]):
        x = obs[:, c]
        if np.std(x) < 1e-12:
            diagrams_ch.append(np.zeros((0, 2)))
            continue
        r2s.append(embedding_prediction_r2(x, m=m, tau=tau, k=5))
        fnns.append(fnn_fraction(x, m=m, tau=tau))
        cl = clean_cloud(takens_embed(x, m=m, tau=tau), normalize="unit_sphere")
        h = ph_diagrams(cl, maxdim=1)[1]
        h = h[np.isfinite(h[:, 1])] if len(h) else np.zeros((0, 2))
        diagrams_ch.append(h)
    a_r2 = float(np.nanmax(r2s)) if r2s else float("nan")
    a_fnn = float(np.nanmin(fnns)) if fnns else float("nan")

    # 路线 B（joint）：多通道各自嵌入后拼成一个点云 -> PH
    cloud = clean_cloud(multichannel_cloud(obs, channels=USE_CHANNELS, m=m, tau=tau),
                        normalize="unit_sphere")
    dgms = ph_diagrams(cloud, maxdim=1)
    h1 = dgms[1]
    h1 = h1[np.isfinite(h1[:, 1])] if len(h1) else np.zeros((0, 2))

    # C：经典基线
    bl = np.concatenate([baseline_features(obs[:, c]) for c in range(obs.shape[1])])

    rings_ch = [crit_ring_count(d, 0.3) for d in diagrams_ch]

    return dict(structure=structure, snr=float(snr), jitter=float(jitter), rep=rep,
                a_r2=a_r2, a_fnn=a_fnn, ring03=crit_ring_count(h1, 0.3),
                ring_max=float(np.max(rings_ch)) if rings_ch else 0.0,
                n_ring=int(len(h1)), diagram=h1, diagrams_ch=diagrams_ch,
                baseline=bl, channel=meta["channel"] or "-")


def run_grid(jobs, workers, label=""):
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_one_sample, jobs, chunksize=4)):
            out.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"    {label}[{i+1}/{len(jobs)}] {el:6.1f}s "
                      f"({el/(i+1):.3f}s/sample)", flush=True)
    return out


def cv_auc(X, y, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(n_estimators=300, n_jobs=1,
                                                   random_state=seed))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def cv_predict(X, y, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(n_estimators=300, n_jobs=1,
                                                   random_state=seed))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return oof


def scalar_auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    m = np.isfinite(s)
    if m.sum() < 6 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def cell_auc(res, is_pos):
    """一个实验单元内的方法对比。

    A 系（相空间重建，无训练）: A_r2 / A_fnn
    B 系（拓扑形态）          : B_ring（joint 环数）/ B_ringmax（逐通道取最大）
                                B_ph_rf（joint PH 特征 + RF）
                                B_perch_rf（逐通道 PH 特征 + RF）
    C 系（经典基线）          : C_classic_rf
    """
    a_r2 = np.array([r["a_r2"] for r in res])
    a_fnn = np.array([r["a_fnn"] for r in res])
    ring = np.array([r["ring03"] for r in res])
    ringmax = np.array([r["ring_max"] for r in res])

    vec = FixedVectorizer(n_grid=N_GRID, n_landscape=N_LAND)
    vec.fit_range([r["diagram"] for r in res])
    XB = vec.transform([r["diagram"] for r in res])

    # 逐通道 PH 特征：向量化器在所有通道的所有持久图上统一 fit -> 维度不漂移
    vec_pc = FixedVectorizer(n_grid=12, n_landscape=2)
    vec_pc.fit_range([d for r in res for d in r["diagrams_ch"]])
    XPC = np.array([np.concatenate([vec_pc.transform_one(d) for d in r["diagrams_ch"]])
                    for r in res])

    XC = np.array([r["baseline"] for r in res])
    y = is_pos.astype(int)

    out = dict(
        A_r2=scalar_auc(a_r2[is_pos], a_r2[~is_pos]),
        A_fnn=scalar_auc(-a_fnn[is_pos], -a_fnn[~is_pos]),
        B_ring=scalar_auc(ring[is_pos], ring[~is_pos]),
        B_ringmax=scalar_auc(ringmax[is_pos], ringmax[~is_pos]),
        B_ph_rf=cv_auc(XB, y),
        B_perch_rf=cv_auc(XPC, y),
        C_classic_rf=cv_auc(XC, y),
        BC_rf=cv_auc(np.hstack([XB, XC]), y),
        PC_C_rf=cv_auc(np.hstack([XPC, XC]), y),
        ring_pos=float(np.mean(ring[is_pos])), ring_neg=float(np.mean(ring[~is_pos])),
        ringmax_pos=float(np.mean(ringmax[is_pos])),
        ringmax_neg=float(np.mean(ringmax[~is_pos])),
        r2_pos=float(np.nanmean(a_r2[is_pos])), r2_neg=float(np.nanmean(a_r2[~is_pos])),
        fnn_pos=float(np.nanmean(a_fnn[is_pos])), fnn_neg=float(np.nanmean(a_fnn[~is_pos])),
    )
    return out, XB.shape[1] + XPC.shape[1], XC.shape[1]


def make_cell_jobs(structs, snr, jitter, n, mode, T):
    jobs = []
    for st in structs:
        for r in range(n):
            jobs.append((st, snr, r, mode, T, M_EMB, TAU_EMB, jitter))
    return jobs


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=40)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--mode", default="add", choices=["add", "mult", "none"])
    ap.add_argument("--snrs", default="0.05,0.1,0.2,0.4,0.8,1.6,3.2")
    ap.add_argument("--jitters", default="0,0.1,0.25,0.5")
    ap.add_argument("--jitter-snr", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.quick:
        args.n_per_cell = 8
        args.snrs = "0.2,1.0"
        args.jitters = "0,0.5"
        args.workers = min(args.workers, 4)

    snrs = [float(s) for s in args.snrs.split(",")]
    jitters = [float(s) for s in args.jitters.split(",")]
    N = args.n_per_cell
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"m{M_EMB}_tau{TAU_EMB}_{args.mode}"
    rows = []

    print("=" * 84)
    print(f"stage3 layer1 v2: synthetic order flow, dual route   tag={tag}")
    print(f"  N/cell={N}  T={args.T}  mode={args.mode}")
    print(f"  SNR={snrs}   jitter={jitters}  (jitter axis at SNR={args.jitter_snr})")
    print(f"  m={M_EMB} tau={TAU_EMB}  workers={args.workers}")
    print("=" * 84)

    # ------------------ E1: SNR 扫描（jitter=0） ------------------
    print("\n[E1] SNR scan (jitter=0)")
    e1 = {}
    for snr in snrs:
        jobs = make_cell_jobs(STRUCTURED, snr, 0.0, N, args.mode, args.T) + \
               make_cell_jobs(["S0_noise"], snr, 0.0, 4 * N, args.mode, args.T)
        res = run_grid(jobs, args.workers, f"snr={snr} ")
        is_pos = np.array([r["structure"] != "S0_noise" for r in res])
        d, nb, nc = cell_auc(res, is_pos)
        e1[snr] = d
        print(f"  SNR={snr:<5} " + "  ".join(f"{k}={d[k]:.3f}" for k in AUC_KEYS))
        for k, v in d.items():
            if isinstance(v, float):
                rows.append(dict(task="E1", method=k, axis="snr", value_axis=snr,
                                 metric="auc_or_stat", value=v))
    print(f"  [dims] topo={nb}  classic={nc}")

    # ------------------ E2: jitter 扫描（SNR 固定） ------------------
    print(f"\n[E2] jitter scan (SNR={args.jitter_snr}) -- core hypothesis")
    e2 = {}
    for jit in jitters:
        jobs = make_cell_jobs(STRUCTURED, args.jitter_snr, jit, N, args.mode, args.T) + \
               make_cell_jobs(["S0_noise"], args.jitter_snr, jit, 4 * N, args.mode, args.T)
        res = run_grid(jobs, args.workers, f"jitter={jit} ")
        is_pos = np.array([r["structure"] != "S0_noise" for r in res])
        d, _, _ = cell_auc(res, is_pos)
        e2[jit] = d
        print(f"  jitter={jit:<5} " + "  ".join(f"{k}={d[k]:.3f}" for k in AUC_KEYS))
        for k, v in d.items():
            if isinstance(v, float):
                rows.append(dict(task="E2", method=k, axis="jitter", value_axis=jit,
                                 metric="auc_or_stat", value=v))

    # ------------------ T2: 六分类 ------------------
    t2_snr = 1.0
    print(f"\n[T2] 6-class (SNR={t2_snr}, jitter=0)")
    jobs2 = []
    for st in STRUCTURES:
        jobs2 += make_cell_jobs([st], t2_snr, 0.0, N, args.mode, args.T)
    res2 = run_grid(jobs2, args.workers, "T2 ")
    y2 = np.array([STRUCTURES.index(r["structure"]) for r in res2])
    vec2 = FixedVectorizer(n_grid=N_GRID, n_landscape=N_LAND)
    vec2.fit_range([r["diagram"] for r in res2])
    XB2 = vec2.transform([r["diagram"] for r in res2])
    vec_pc2 = FixedVectorizer(n_grid=12, n_landscape=2)
    vec_pc2.fit_range([d for r in res2 for d in r["diagrams_ch"]])
    XPC2 = np.array([np.concatenate([vec_pc2.transform_one(d) for d in r["diagrams_ch"]])
                     for r in res2])
    XC2 = np.array([r["baseline"] for r in res2])
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    t2 = {}
    for name, X in [("B_ph_rf", XB2), ("B_perch_rf", XPC2), ("C_classic_rf", XC2),
                    ("BC_rf", np.hstack([XB2, XC2])),
                    ("PC_C_rf", np.hstack([XPC2, XC2]))]:
        pred = cv_predict(X, y2)
        t2[name] = dict(acc=float(accuracy_score(y2, pred)),
                        f1=float(f1_score(y2, pred, average="macro")),
                        cm=confusion_matrix(y2, pred).tolist())
        rows.append(dict(task="T2", method=name, axis="snr", value_axis=t2_snr,
                         metric="acc", value=t2[name]["acc"]))
        rows.append(dict(task="T2", method=name, axis="snr", value_axis=t2_snr,
                         metric="f1_macro", value=t2[name]["f1"]))
        print(f"  {name:<16} acc={t2[name]['acc']:.3f} macroF1={t2[name]['f1']:.3f}")

    # ------------------ T0: 对抗性零模型 ------------------
    print("\n[T0] adversarial null: S0(AR) vs S4(Hawkes)")
    jobs3 = make_cell_jobs(["S0_noise", "S4_hawkes"], 1.0, 0.0, 4 * N, args.mode, args.T)
    res3 = run_grid(jobs3, args.workers, "T0 ")
    y3 = np.array([0 if r["structure"] == "S0_noise" else 1 for r in res3])
    vec3 = FixedVectorizer(n_grid=N_GRID, n_landscape=N_LAND)
    vec3.fit_range([r["diagram"] for r in res3])
    XB3 = vec3.transform([r["diagram"] for r in res3])
    XC3 = np.array([r["baseline"] for r in res3])
    a_r2_3 = np.array([r["a_r2"] for r in res3])
    a_fnn_3 = np.array([r["a_fnn"] for r in res3])
    ring3 = np.array([r["ring03"] for r in res3])
    ringmax3 = np.array([r["ring_max"] for r in res3])
    vec_pc3 = FixedVectorizer(n_grid=12, n_landscape=2)
    vec_pc3.fit_range([d for r in res3 for d in r["diagrams_ch"]])
    XPC3 = np.array([np.concatenate([vec_pc3.transform_one(d) for d in r["diagrams_ch"]])
                     for r in res3])
    t0 = dict(
        A_r2=scalar_auc(a_r2_3[y3 == 1], a_r2_3[y3 == 0]),
        A_fnn=scalar_auc(-a_fnn_3[y3 == 1], -a_fnn_3[y3 == 0]),
        B_ring=scalar_auc(ring3[y3 == 1], ring3[y3 == 0]),
        B_ringmax=scalar_auc(ringmax3[y3 == 1], ringmax3[y3 == 0]),
        B_ph_rf=cv_auc(XB3, y3), B_perch_rf=cv_auc(XPC3, y3),
        C_classic_rf=cv_auc(XC3, y3),
        BC_rf=cv_auc(np.hstack([XB3, XC3]), y3),
        ring_ar=float(np.mean(ring3[y3 == 0])), ring_hawkes=float(np.mean(ring3[y3 == 1])),
        ringmax_ar=float(np.mean(ringmax3[y3 == 0])),
        ringmax_hawkes=float(np.mean(ringmax3[y3 == 1])),
        n=int(len(y3)),
    )
    for k, v in t0.items():
        if isinstance(v, float):
            rows.append(dict(task="T0", method=k, axis="none", value_axis=0.0,
                             metric="auc_or_stat", value=v))
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in t0.items()))

    # ------------------ 落盘 ------------------
    with open(OUT / f"summary_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "method", "axis", "value_axis",
                                          "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    payload = dict(tag=tag, mode=args.mode, T=args.T, n_per_cell=N, snrs=snrs,
                   jitters=jitters, jitter_snr=args.jitter_snr,
                   m=M_EMB, tau=TAU_EMB, channels=CHANNELS,
                   E1_snr_scan={str(k): v for k, v in e1.items()},
                   E2_jitter_scan={str(k): v for k, v in e2.items()},
                   T2=t2, T2_snr=t2_snr, T0=t0)
    with open(OUT / f"result_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {OUT / f'summary_{tag}.csv'}")
    print(f"[saved] {OUT / f'result_{tag}.json'}")


if __name__ == "__main__":
    main()
