#!/usr/bin/env python3
"""阶段 3 第 1 层出图：AUC-SNR 曲线、AUC-jitter 曲线、T0 对照、T2 混淆矩阵。

用法
----
    python src/plot_3a.py --tags m3tau8_full lowsnr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp3a_synthetic"

METHODS = [
    ("A_r2", "A: 相空间 kNN 预测 R^2", "#1f77b4", "-"),
    ("A_fnn", "A: FNN 残留", "#aec7e8", "--"),
    ("B_ringmax", "B: 逐通道环数(取最大)", "#ff7f0e", ":"),
    ("B_ph_rf", "B: joint PH + RF", "#ffbb78", "--"),
    ("B_perch_rf", "B: 逐通道 PH + RF", "#d62728", "-"),
    ("C_classic_rf", "C: 经典基线(8 标量 x 7 通道)", "#2ca02c", "-"),
    ("PC_C_rf", "PH(逐通道) + 经典", "#9467bd", "-"),
]


def setup_font():
    """尽量用中文字体；找不到就退回英文标签。"""
    from matplotlib import font_manager
    want = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei",
            "Microsoft YaHei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for w in want:
        if w in have:
            plt.rcParams["font.sans-serif"] = [w]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 使用 {w}")
            return True
    print("[font] 未找到中文字体，图内标签改用英文")
    return False


def load(tag):
    p = OUT / f"result_{tag}.json"
    if not p.exists():
        print(f"[skip] {p} 不存在")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig_curve(a, b, axis_key, xlabel, title, outname, en):
    """两张图对比：同一坐标轴上画两个 tag 的曲线。"""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    styles = ["-", "--"]
    for k, (data, tag) in enumerate([(a, "full"), (b, "lowsnr")]):
        if data is None:
            continue
        block = data.get(axis_key, {})
        if not block:
            continue
        xs = sorted(float(k2) for k2 in block)
        for meth, label, color, ls in METHODS:
            ys = [block[str(x)].get(meth, np.nan) for x in xs]
            if all(not np.isfinite(v) for v in ys):
                continue
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=color,
                    ls=(ls if k == 0 else ":"),
                    alpha=(1.0 if k == 0 else 0.55),
                    label=f"{label}" + ("" if k == 0 else f" [{tag}]"))
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", label="chance (AUC=0.5)")
    ax.set_xlabel(xlabel if en else xlabel)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.35, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    if axis_key == "E1_snr_scan":
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(OUT / outname)
    plt.close(fig)
    print(f"[saved] {OUT / outname}")


def fig_t0(data, tag, en):
    if data is None or not data.get("T0"):
        return
    t0 = data["T0"]
    keys = ["A_r2", "B_ringmax", "B_ph_rf", "B_perch_rf", "C_classic_rf"]
    vals = [t0.get(k, np.nan) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.bar(range(len(keys)), vals, color="#4c72b0")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"T0 adversarial null: S0(AR) vs S4(Hawkes)  [{tag}]")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fn = f"fig_t0_{tag}.png"
    fig.savefig(OUT / fn)
    plt.close(fig)
    print(f"[saved] {OUT / fn}")


def fig_t2(data, tag):
    if data is None or not data.get("T2"):
        return
    t2 = data["T2"]
    names = [k for k in ("B_ph_rf", "B_perch_rf", "C_classic_rf", "PC_C_rf")
             if k in t2 and "cm" in t2[k]]
    if not names:
        return
    from ofgen import STRUCTURES
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2), dpi=140)
    if len(names) == 1:
        axes = [axes]
    for ax, nm in zip(axes, names):
        cm = np.array(t2[nm]["cm"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(STRUCTURES)))
        ax.set_xticklabels([s.split("_")[0] for s in STRUCTURES], fontsize=7)
        ax.set_yticks(range(len(STRUCTURES)))
        ax.set_yticklabels([s.split("_")[0] for s in STRUCTURES], fontsize=7)
        ax.set_title(f"{nm}\nacc={t2[nm]['acc']:.3f} F1={t2[nm]['f1']:.3f}",
                     fontsize=9)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                if cmn[i, j] > 0.01:
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"T2 六分类混淆矩阵（行归一化） [{tag}]", fontsize=11)
    fig.tight_layout()
    fn = f"fig_t2_confusion_{tag}.png"
    fig.savefig(OUT / fn)
    plt.close(fig)
    print(f"[saved] {OUT / fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["m3tau8_full", "lowsnr"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    en = not setup_font()
    main_tag = args.tags[0]
    a = load(main_tag)
    b = load(args.tags[1]) if len(args.tags) > 1 else None

    fig_curve(a, b, "E1_snr_scan",
              "SNR (= 结构幅度 / 背景标准差)" if not en else "SNR",
              "第1层 E1：判别力 vs 信噪比（jitter=0）" if not en else
              "E1: AUC vs SNR (jitter=0)",
              "fig_auc_snr.png", en)
    fig_curve(a, b, "E2_jitter_scan",
              "jitter（周期抖动的相对标准差）" if not en else "jitter",
              "第1层 E2：判别力 vs 周期抖动（核心假设）" if not en else
              "E2: AUC vs period jitter",
              "fig_auc_jitter.png", en)
    fig_t0(a, main_tag, en)
    fig_t2(a, main_tag)
    if b is not None:
        fig_t0(b, args.tags[1], en)
        fig_t2(b, args.tags[1])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
