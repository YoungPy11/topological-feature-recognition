"""阶段 3 共用库：嵌入 / PH / 判据 / 基线特征 / 滚动窗口 / purged 时序划分。

设计约定（与项目既有技能一致）
--------------------------------
* 所有点云入口统一 float64、去重、去 NaN、归一化（`clean_cloud`）
* 向量化一律传**固定 birth/death range**，否则维度随样本漂移、无法堆叠
* PH 用 ripser（**绝不用 gudhi**，本机策略拦截其 DLL）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ripser import ripser

from ofgen import CHANNELS, N_CH

# --------------------------------------------------------------------------
# 点云清洗 / 归一化
# --------------------------------------------------------------------------
def clean_cloud(X, normalize="unit_sphere", dedup=True):
    """X: (n, d) -> float64 (m, d)，已清洗归一化。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expect 2D (n,d), got {X.shape}")
    if not np.isfinite(X).all():
        X = X[np.isfinite(X).all(axis=1)]
    if dedup:
        X = np.unique(X, axis=0)
    if len(X) < 3:
        raise ValueError("too few points after cleaning")
    if normalize == "unit_sphere":
        X = X - X.mean(axis=0, keepdims=True)
        r = np.linalg.norm(X, axis=1).max()
        if r > 0:
            X = X / r
    elif normalize == "zscore":
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
    elif normalize == "per_channel_unit":
        # 多通道拼接专用：先各通道标准化，再整体缩放到单位球
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
        r = np.linalg.norm(X, axis=1).max()
        if r > 0:
            X = X / r
    elif normalize is None or normalize == "none":
        pass
    else:
        raise ValueError(normalize)
    return np.ascontiguousarray(X)


# --------------------------------------------------------------------------
# 嵌入
# --------------------------------------------------------------------------
def takens_embed(x, m=3, tau=1):
    """Takens 延迟嵌入：x(T,) -> (T-(m-1)tau, m)。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    T = len(x)
    n = T - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"series too short: T={T}, m={m}, tau={tau}")
    idx = np.arange(n)[:, None] + np.arange(m)[None, :] * tau
    return x[idx]


def multichannel_cloud(X, channels=None, m=3, tau=1, standardize=True):
    """路线 B 核心：多通道各自延迟嵌入后各通道标准化、再纵向拼成点云。

    X: (T, N_CH) -> (n, len(channels)*m)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if channels is None:
        channels = list(range(X.shape[1]))
    blocks = []
    for c in channels:
        E = takens_embed(X[:, c], m=m, tau=tau)
        if standardize:
            sd = E.std(axis=0)
            sd[sd == 0] = 1.0
            E = (E - E.mean(axis=0)) / sd
        blocks.append(E)
    return np.hstack(blocks)


# --------------------------------------------------------------------------
# PH
# --------------------------------------------------------------------------
def ph_diagrams(X, maxdim=1):
    """ripser 持久图。返回 list of (n_i, 2)。"""
    return ripser(np.ascontiguousarray(X, dtype=np.float64), maxdim=maxdim)["dgms"]


def finite(dgm):
    if len(dgm) == 0:
        return np.zeros((0, 2))
    return dgm[np.isfinite(dgm[:, 1])]


def betti_at(dgms, t):
    return [int(np.sum((d[:, 0] <= t) & (d[:, 1] > t))) for d in dgms]


# --------------------------------------------------------------------------
# 判据（沿用 2A 的「环占比」并扩展）
# --------------------------------------------------------------------------
def crit_ring_count(dgm, ratio=0.3):
    """2A 最优判据：寿命 > max*ratio 的环数。"""
    d = finite(dgm)
    if len(d) == 0:
        return 0.0
    p = d[:, 1] - d[:, 0]
    mx = p.max()
    return float(np.sum(p > ratio * mx)) if mx > 0 else 0.0


def crit_topk_pers(dgm, k=1):
    d = finite(dgm)
    if len(d) == 0:
        return 0.0
    p = np.sort(d[:, 1] - d[:, 0])[::-1]
    return float(p[:k].sum())


def crit_top1_over_mean(dgm):
    d = finite(dgm)
    if len(d) == 0:
        return 0.0
    p = d[:, 1] - d[:, 0]
    m = p.mean()
    return float(p.max() / m) if m > 1e-12 else 0.0


def crit_pers_entropy(dgm):
    d = finite(dgm)
    if len(d) == 0:
        return 0.0
    p = d[:, 1] - d[:, 0]
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    q = p / p.sum()
    return float(-(q * np.log(q)).sum())


# --------------------------------------------------------------------------
# 固定 range 向量化（维度不漂移）
# --------------------------------------------------------------------------
class FixedVectorizer:
    """Betti 曲线 + 持久景观 + 统计量，全部落在固定网格上。

    必须先 `fit_range(all_diagrams)` 确定 (birth_range, death_range)，再 transform。
    """

    def __init__(self, n_grid=24, n_landscape=4, birth_range=(0.0, 2.0),
                 death_range=(0.0, 2.0)):
        self.n_grid = n_grid
        self.n_landscape = n_landscape
        self.birth_range = birth_range
        self.death_range = death_range
        self.grid = np.linspace(birth_range[0], death_range[1], n_grid)

    def fit_range(self, diagram_list, q=0.995):
        b = np.concatenate([d[:, 0] for d in diagram_list if len(d)]) if any(len(d) for d in diagram_list) else np.array([0.0])
        dd = np.concatenate([d[:, 1] for d in diagram_list if len(d)]) if any(len(d) for d in diagram_list) else np.array([1.0])
        lo = float(min(b.min(), dd.min()))
        hi = float(max(np.quantile(b, q), np.quantile(dd, q), lo + 1e-6))
        self.birth_range = (lo, hi)
        self.death_range = (lo, hi)
        self.grid = np.linspace(lo, hi, self.n_grid)
        return self

    def _betti(self, dgm):
        if len(dgm) == 0:
            return np.zeros(self.n_grid)
        return np.array([np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t)) for t in self.grid], dtype=float)

    def _landscape(self, dgm):
        """前 k 条持久景观在固定网格上的取值。"""
        out = np.zeros((self.n_landscape, self.n_grid))
        if len(dgm) == 0:
            return out.ravel()
        L = np.zeros((len(dgm), self.n_grid))
        t = self.grid
        for i, (b, d) in enumerate(dgm):
            L[i] = np.maximum(0.0, np.minimum(t - b, d - t))
        L = np.sort(L, axis=0)[::-1]
        k = min(self.n_landscape, L.shape[0])
        out[:k] = L[:k]
        return out.ravel()

    def transform_one(self, dgm):
        dgm = finite(dgm)
        return np.concatenate([
            self._betti(dgm),
            self._landscape(dgm),
            np.array([crit_ring_count(dgm, 0.1),
                      crit_ring_count(dgm, 0.3),
                      crit_ring_count(dgm, 0.5),
                      crit_topk_pers(dgm, 1),
                      crit_topk_pers(dgm, 3),
                      crit_top1_over_mean(dgm),
                      crit_pers_entropy(dgm),
                      float(len(dgm))]),
        ])

    @property
    def dim(self):
        return self.n_grid + self.n_landscape * self.n_grid + 8

    def transform(self, diagram_list):
        return np.array([self.transform_one(d) for d in diagram_list])


# --------------------------------------------------------------------------
# 路线 A：嵌入重建质量诊断
# --------------------------------------------------------------------------
def fnn_fraction(x, m=3, tau=1, rtol=10.0, seed=0):
    """假近邻残留比例（简化版：只与 m+1 维比较最近邻距离膨胀比）。

    确定性系统在足够大的 m 下应 -> 0；噪声则维持高位。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    Em = takens_embed(x, m=m, tau=tau)
    if len(Em) < 20:
        return float("nan")
    Em1 = takens_embed(x, m=m + 1, tau=tau)
    n = min(len(Em), len(Em1))
    Em, Em1 = Em[:n], Em1[:n]
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=2).fit(Em)
    d, idx = nn.kneighbors(Em)
    d, idx = d[:, 1], idx[:, 1]
    extra = np.abs(Em1[:, -1] - Em1[idx, -1])
    denom = d + 1e-12
    ratio = np.sqrt((d ** 2 + extra ** 2)) / denom
    return float(np.mean(ratio > rtol))


def embedding_prediction_r2(x, m=3, tau=1, k=5, exclude=None):
    """相空间 kNN 一步预测的 R² —— 直接度量「嵌入是否重建出确定性动力学」。

    排除时间上相邻的邻居（|i-j| <= exclude），否则是平凡自预测。
    exclude 默认取 m（同一条轨迹上的近邻）。
    """
    if exclude is None:
        exclude = m
    x = np.asarray(x, dtype=np.float64).ravel()
    E = takens_embed(x, m=m, tau=tau)
    n = len(E)
    if n < 30:
        return float("nan")
    target = x[m * tau: m * tau + n]        # 对应 E[i] 的下一步观测
    L = min(len(E), len(target))
    E, target = E[:L], target[:L]
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=min(k + exclude + 2, L)).fit(E)
    _, idx = nn.kneighbors(E)
    pred = np.empty(L)
    for i in range(L):
        cand = [j for j in idx[i] if abs(int(j) - i) > exclude][:k]
        if not cand:
            cand = [int(idx[i][1])]
        pred[i] = target[cand].mean()
    ss_res = float(np.mean((target - pred) ** 2))
    ss_tot = float(np.var(target))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


# --------------------------------------------------------------------------
# 经典基线特征（必须有，否则无法宣称 PH 有用）
# --------------------------------------------------------------------------
def _ssa_first_ratio(x, L=None):
    x = np.asarray(x, dtype=np.float64).ravel()
    T = len(x)
    L = L or max(2, T // 8)
    K = T - L + 1
    if K < 2:
        return 0.0
    X = np.lib.stride_tricks.sliding_window_view(x, L)          # (K, L)
    C = (X.T @ X) / K
    w = np.linalg.eigvalsh(C)[::-1]
    w = np.clip(w, 0.0, None)
    return float(w[0] / (w.sum() + 1e-12))


def _sample_entropy(x, m=2, r_factor=0.2):
    x = np.asarray(x, dtype=np.float64).ravel()
    T = len(x)
    r = r_factor * x.std()
    if r <= 0 or T < 50:
        return 0.0
    # 取子样本控制 O(N^2)（512 点可全算）
    def phi(mm):
        if T - mm + 1 < 2:
            return 0.0
        V = np.lib.stride_tricks.sliding_window_view(x, mm)
        n = len(V)
        cnt = 0
        for i in range(n):
            d = np.max(np.abs(V - V[i]), axis=1)
            cnt += int(np.sum(d <= r)) - 1
        return cnt / (n * (n - 1)) if n > 1 else 0.0
    a, b = phi(m + 1), phi(m)
    if b <= 0 or a <= 0:
        return 0.0
    return float(-np.log(a / b))


def _hurst_rs(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    T = len(x)
    if T < 32:
        return 0.5
    sizes = [s for s in (8, 16, 32, 64, 128) if s <= T // 2]
    if len(sizes) < 2:
        return 0.5
    rs = []
    for s in sizes:
        vals = []
        for start in range(0, T - s + 1, s):
            seg = x[start:start + s]
            y = np.cumsum(seg - seg.mean())
            R = y.max() - y.min()
            S = seg.std()
            if S > 1e-12:
                vals.append(R / S)
        if vals:
            rs.append((s, float(np.mean(vals))))
    if len(rs) < 2:
        return 0.5
    logs = np.log([r[0] for r in rs])
    logr = np.log([max(r[1], 1e-9) for r in rs])
    return float(np.polyfit(logs, logr, 1)[0])


def baseline_features(x):
    """单通道经典基线：8 个标量。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - x.mean()
    sd = x.std()
    if sd < 1e-12:
        return np.zeros(8)
    xn = x / sd
    T = len(xn)

    # 自相关首过零
    ac = np.correlate(xn, xn, mode="full")[T - 1:]
    ac /= ac[0]
    tau0 = 1
    for k in range(1, min(T // 3, 200)):
        if ac[k] <= 0:
            tau0 = k
            break
    ac1 = float(ac[1]) if T > 1 else 0.0

    # 周期图峰值比（去掉 DC）；用中位数做分母 —— 对 AR(1) 的低频滚降更稳健
    F = np.abs(np.fft.rfft(xn)) ** 2
    F = F[1:]
    peak = float(F.max() / (np.median(F) + 1e-12)) if len(F) else 0.0
    # 峰值对应频率的归一化周期
    if len(F):
        k_peak = int(np.argmax(F)) + 1
        period = float(T / k_peak)
    else:
        period = 0.0

    return np.array([
        float(tau0), ac1, peak, period,
        _ssa_first_ratio(xn), _sample_entropy(xn),
        _hurst_rs(xn), float(np.std(x)),
    ])


# --------------------------------------------------------------------------
# 滚动窗口 / purged 时序划分
# --------------------------------------------------------------------------
def rolling_windows(x, win, stride):
    """(T, ...) -> 生成器，产出 (start, window)。"""
    x = np.asarray(x)
    T = len(x)
    for s in range(0, T - win + 1, stride):
        yield s, x[s:s + win]


def purged_walk_forward(n_samples, n_splits=5, label_horizon=20, embargo=5):
    """purged walk-forward + embargo，产出 (train_idx, test_idx) 列表。

    严格遵守：训练段右端与测试段左端之间挖掉 label_horizon+embargo，
    且训练段中标签会跨入测试段的样本被剔除（purge）。
    """
    fold = n_samples // (n_splits + 1)
    if fold < 2:
        return [(np.arange(max(1, n_samples - 1)), np.array([n_samples - 1]))]
    out = []
    for k in range(1, n_splits + 1):
        tr_end = fold * k
        te_start = tr_end + label_horizon + embargo
        te_end = min(n_samples, te_start + fold)
        if te_start >= n_samples:
            break
        tr = np.arange(0, max(1, tr_end - label_horizon))
        te = np.arange(te_start, te_end)
        if len(tr) < 5 or len(te) < 1:
            continue
        out.append((tr, te))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # ================= 自检 1：已知形状 =================
    # 复用仓库既有的 ph_pipeline 生成器。
    # ⚠️ 点数必须控制：ripser 的 maxdim=2 复杂度随 n 爆炸，2000 点会把 WSL 的 VM
    #    内存打满并杀掉整个 WSL 服务（已实测）。
    # 判定准则用「Betti 平台」而不是「显著条数」：稀疏采样下 Rips 复形的
    # H2 窗口很窄，且 H0 的有限条在连通前必然存在，不是有效的判据。
    from ph_pipeline import (compute_persistence_diagrams, generate_sphere,
                             generate_torus)

    def betti_plateau(dg, lo, hi, n=48):
        ts = np.linspace(lo, hi, n)
        return [(float(t), tuple(betti_at(dg, t))) for t in ts]

    def report_shape(name, X, eps_range, want, maxdim=2):
        diam = float(np.linalg.norm(X - X.mean(0), axis=1).max()) * 2
        dg = compute_persistence_diagrams(X, maxdim=maxdim)
        seq = betti_plateau(dg, eps_range[0], eps_range[1])
        hit_t = [t for t, b in seq if b == want]
        runs = []
        for t, b in seq:
            if runs and runs[-1][0] == b:
                runs[-1][1] = t
            else:
                runs.append([b, t])
        print(f"\n{name}: n={len(X)}  直径≈{diam:.2f}")
        for b, t1 in runs:
            print(f"    Betti={b}  ε∈[{t1:.2f}, ]")
        ok = len(hit_t) > 0
        print(f"  期望 Betti {want} 出现于 {len(hit_t)}/{len(seq)} 个 ε "
              f"（ε∈[{min(hit_t):.2f},{max(hit_t):.2f}]）" if ok
              else f"  期望 Betti {want} 未出现 ❌")
        return ok

    print("=== 自检 1：已知形状（复用仓库 ph_pipeline 生成器）===")
    # 环面用规则网格采样（随机稀疏采样下 H2 窗口会消失）；球面用随机采样
    _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, 30, endpoint=False),
                         np.linspace(0, 2 * np.pi, 30, endpoint=False))
    _R, _r = 2.0, 1.0
    Xt_grid = np.c_[(_R + _r * np.cos(_v.ravel())) * np.cos(_u.ravel()),
                    (_R + _r * np.cos(_v.ravel())) * np.sin(_u.ravel()),
                    _r * np.sin(_v.ravel())]
    ok_a = report_shape("torus(30x30 网格)", Xt_grid, (0.10, 1.20), (1, 2, 1))
    ok_b = report_shape("sphere(随机 500)", generate_sphere(n_points=500, noise=0.02),
                        (0.30, 1.60), (1, 0, 1))

    # ================= 自检 2：2A 回归 =================
    # 直接复用 2A 的实际模块与判据协议，保证与 results/exp2a_criterion 可比。
    print("\n=== 自检 2：2A 回归（直接调用 2A 的生成器与判据）===")
    try:
        from experiment_2a_embedding import time_delay_embedding as _tde2a
        from experiment_2a_time_series import GENERATORS

        M_FIXED, TAU_FIXED, N_SAMPLES, NOISE = 4, 4, 10, 0.05

        def crit_2a(dgm, ratio=0.3):
            if len(dgm) == 0:
                return 0
            pers = dgm[:, 1] - dgm[:, 0]
            mx = pers.max()
            return 0 if mx <= 0 else int(np.sum(pers > ratio * mx))

        print(f"  协议与 experiment_2a_criterion.py 完全一致 "
              f"(m={M_FIXED}, tau={TAU_FIXED}, n={N_SAMPLES}, noise={NOISE})")
        res2a = {}
        for sig_type, gen_fn in GENERATORS.items():
            base = gen_fn()
            base = (base - base.min()) / (base.max() - base.min() + 1e-9)
            vals = []
            for rep in range(N_SAMPLES):
                sg = base + NOISE * np.random.RandomState(rep).randn(len(base))
                dg = compute_persistence_diagrams(
                    _tde2a(sg, m=M_FIXED, tau=TAU_FIXED), maxdim=1)
                h1 = dg[1]
                h1 = h1[~np.isinf(h1[:, 1])]
                vals.append(crit_2a(h1, 0.3))
            res2a[sig_type] = (float(np.mean(vals)), float(np.std(vals)))
            print(f"    {sig_type:<14} 环数 = {res2a[sig_type][0]:7.2f} "
                  f"± {res2a[sig_type][1]:5.2f}")
        if {"periodic", "noise_white"} <= set(res2a):
            sep = (res2a["noise_white"][0] - res2a["periodic"][0]) / (
                (res2a["periodic"][1] + res2a["noise_white"][1]) / 2 + 1e-9)
            print(f"  分离度（噪声−周期)/std = {sep:.2f}   "
                  f"[2A 报告的最佳判据分离度 = 9.12]")
            print("  -> 与本项目 2A 结论一致" if sep > 1.0 else "  -> 与 2A 不一致，需排查")
    except Exception as e:      # noqa: BLE001
        print(f"  [跳过] 无法导入 2A 模块：{type(e).__name__}: {e}")

    # ================= 自检 3：purged 时序划分 =================
    folds = purged_walk_forward(200, n_splits=4, label_horizon=20, embargo=5)
    print(f"\n=== 自检 3：purged walk-forward + embargo ===")
    print(f"  {len(folds)} 折（要求 gap ≥ label_horizon+embargo = 25）")
    for tr, te in folds:
        gap = te[0] - tr[-1]
        print(f"  train[{tr[0]}..{tr[-1]}] test[{te[0]}..{te[-1]}] gap={gap} "
              f"{'✅' if gap >= 25 else '❌'}")
