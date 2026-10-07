"""阶段 3 第 1 层：合成订单流生成器（可控 ground truth）。v2

设计修正（v1 → v2）
-------------------
v1 用 LOB 状态机做背景，结果**背景本身自相关 0.96、谱峰 75.9**，
S0 根本不像"无结构"——注入的结构信号被淹没，实验无法解释。
v2 换成**替代零模型（surrogate null）**：逐通道 AR(1) + 可选 Hawkes 自激强度调制。
这样零模型是"有短记忆但无周期"的平稳过程，正是经典基线（自相关/周期图）看的东西，
竞争关系才公平。

新增关键实验轴：**周期抖动 jitter**
------------------------------
若结构信号是严格周期的，周期图必然碾压拓扑方法，实验没有意义。
现实中算法拆单的间隔是抖动的。抖动让谱峰弥散，但相空间里的**回路形状仍然存在**
——这正是 PH 可能胜出的机制。jitter 因此是第 1 层的核心自变量之一。

观测模型
--------
    x(t) = background(t) + A * e_c * s(t)         (mode=add)
    x(t) = background(t) * (1 + A * s(t))         (mode=mult)
    SNR := A / std(background_c)

结构清单
--------
S0 noise    : 纯 AR(1) 背景（无结构）
S1 iceberg  : 锯齿波（吃完即补）      → 注入 bimb
S2 twap     : 脉冲串（等间隔拆单）    → 注入 tvol
S3 spoof    : 方波（挂单→撤单）      → 注入 far_ask
S4 hawkes   : 背景换成自激簇发（对抗性零模型，无周期）
S5 periodic : 正弦                    → 注入 ofi
"""
from __future__ import annotations

import numpy as np

CHANNELS = ["ofi", "tvol", "bimb", "spread", "mid_ret", "far_bid", "far_ask"]
N_CH = len(CHANNELS)

STRUCTURES = ["S0_noise", "S1_iceberg", "S2_twap", "S3_spoof", "S4_hawkes", "S5_periodic"]

# 各通道 AR(1) 系数与尺度（盘口状态量记忆长、收益/价差记忆短）
_CH_PHI = np.array([0.35, 0.20, 0.45, 0.25, 0.00, 0.50, 0.50])
_CH_SCALE = np.array([1.00, 1.40, 1.00, 0.60, 1.00, 1.20, 1.20])

_INJECT_CH = {
    "S1_iceberg": "bimb",
    "S2_twap": "tvol",
    "S3_spoof": "far_ask",
    "S5_periodic": "ofi",
}

_BASE_PERIOD = {
    "S1_iceberg": 21,
    "S2_twap": 26,
    "S3_spoof": 40,
    "S5_periodic": 32,
}


# --------------------------------------------------------------------------
# 背景（替代零模型）
# --------------------------------------------------------------------------
def _hawkes_intensity(T: int, mu: float, alpha: float, beta: float, rng) -> np.ndarray:
    """Hawkes(1) 自激过程的分箱强度（Ogata thinning）。"""
    lam_bar = mu / max(1e-9, 1.0 - alpha / beta)
    if not np.isfinite(lam_bar) or lam_bar <= 0:
        lam_bar = mu * 10.0
    lam_bar = max(lam_bar, mu * 1.05)
    times, t = [], 0.0
    while t < T:
        t += rng.exponential(1.0 / lam_bar)
        if t >= T:
            break
        lam_t = mu + alpha * sum(np.exp(-beta * (t - ti)) for ti in times[-300:] if ti < t)
        if rng.uniform() <= lam_t / lam_bar:
            times.append(t)
    inten = np.zeros(T)
    for ti in times:
        b = int(ti)
        if 0 <= b < T:
            inten[b] += 1.0
    # 平滑一下，避免逐 bin 过于稀疏
    k = np.ones(5) / 5.0
    inten = np.convolve(inten, k, mode="same")
    return inten


def gen_background(T: int = 512, seed: int = 0, hawkes: bool = False,
                   phi_scale: float = 1.0) -> np.ndarray:
    """生成 (T, N_CH) 的无周期背景。

    hawkes=True 时，创新项方差被 Hawkes 强度调制 → 簇发但无周期（对抗性零模型）。
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(T, N_CH))

    if hawkes:
        inten = _hawkes_intensity(T, mu=0.10, alpha=1.5, beta=2.0, rng=rng)
        inten = inten / (inten.mean() + 1e-12)
        eps = eps * np.sqrt(np.clip(inten, 0.05, 20.0))[:, None]

    x = np.zeros((T, N_CH))
    phi = np.clip(_CH_PHI * phi_scale, 0.0, 0.95)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + eps[t]
    x[:, 0] += 0.0                       # ofi 已零均值
    x[:, 3] = np.abs(x[:, 3]) + 1.2      # spread 必须为正
    return x * _CH_SCALE[None, :]


# --------------------------------------------------------------------------
# 结构信号 s(t)：由抖动相位驱动
# --------------------------------------------------------------------------
def _phase(T: int, period: float, jitter: float, rng) -> np.ndarray:
    """累积相位 φ(t) ∈ [0,1)。

    jitter=0  → 严格周期
    jitter>0  → 每步周期 P*(1+jitter*N(0,1))，谱峰弥散但回路形状保留
    """
    if jitter <= 0:
        return (np.arange(T) / period) % 1.0
    steps = 1.0 / (period * np.clip(1.0 + jitter * rng.normal(size=T), 0.25, 4.0))
    return np.cumsum(steps) % 1.0


def struct_signal(kind: str, T: int, seed: int = 0, jitter: float = 0.0,
                  period: float | None = None) -> np.ndarray:
    """返回归一化到单位 std 的结构信号 s(t)，形状 (T,)。"""
    rng = np.random.default_rng(seed + 991)
    P = float(period or _BASE_PERIOD[kind])
    ph = _phase(T, P, jitter, rng)

    if kind == "S1_iceberg":
        s = ph                                   # 锯齿波：吃完再补
    elif kind == "S2_twap":
        s = np.where(ph < 0.12, 1.0, 0.0)        # 脉冲串
    elif kind == "S3_spoof":
        s = np.where(ph < 0.5, 1.0, -1.0)        # 方波
    elif kind == "S5_periodic":
        s = np.sin(2 * np.pi * ph)               # 正弦
    else:
        raise ValueError(f"no structural signal for {kind}")

    s = s - s.mean()
    sd = s.std()
    return s / sd if sd > 1e-12 else s


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def generate(structure: str, snr: float = 1.0, T: int = 512, seed: int = 0,
             mode: str = "add", jitter: float = 0.0, scale: float = 1.0):
    """生成一个样本。返回 (obs (T,N_CH), meta)。"""
    hawkes = structure == "S4_hawkes"
    bg = gen_background(T=T, seed=seed, hawkes=hawkes)
    meta = dict(structure=structure, snr=float(snr), mode=mode,
                jitter=float(jitter), channel=None)

    if structure in ("S0_noise", "S4_hawkes"):
        return bg, meta                # 无结构：S0=AR背景, S4=Hawkes背景

    ch = _INJECT_CH[structure]
    c = CHANNELS.index(ch)
    meta["channel"] = ch
    s = struct_signal(structure, T, seed=seed, jitter=jitter)
    A = snr * np.std(bg[:, c]) * scale

    if mode == "add":
        obs = bg.copy()
        obs[:, c] = obs[:, c] + A * s
    elif mode == "mult":
        obs = bg.copy()
        obs[:, c] = obs[:, c] * (1.0 + snr * 0.5 * s)
    elif mode == "none":
        obs = bg.copy()
    else:
        raise ValueError(mode)
    return obs, meta


def make_dataset(structures, snrs, n_per_cell=40, T=512, mode="add",
                 jitter=0.0, seed0=0):
    X, y, metas = [], [], []
    for si, st in enumerate(structures):
        for ai, snr in enumerate(snrs):
            for r in range(n_per_cell):
                seed = seed0 + 100003 * si + 1009 * ai + r
                obs, meta = generate(st, snr=snr, T=T, seed=seed, mode=mode,
                                     jitter=jitter)
                X.append(obs)
                y.append(si)
                metas.append(meta)
    return np.array(X), np.array(y), metas


# --------------------------------------------------------------------------
def sanity_check():
    """自检：背景应为无周期（谱峰低）；结构信号在注入通道上应可辨。"""
    print("生成器自检 v2 (T=512, SNR=1.0, mode=add)")
    hdr = f"{'structure':<12} {'chan':>8} {'var_ratio':>10} {'ac1_bg':>8} {'ac1_obs':>8} " \
          f"{'spec_peak':>10} {'ring_proxy':>11}"
    print(hdr)
    print("-" * len(hdr))

    # 背景本身的无周期性检验（纯 S0，不看注入）
    bg = gen_background(T=512, seed=999)
    for ci, cname in enumerate(CHANNELS):
        x = bg[:, ci] - bg[:, ci].mean()
        ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        F = np.abs(np.fft.rfft(x)) ** 2
        peak = float(F[1:].max() / (F[1:].mean() + 1e-12))
        print(f"  [bg-only] {cname:<8} ac1={ac1:6.3f}  spec_peak={peak:7.2f}")

    print()
    for st in STRUCTURES:
        obs, meta = generate(st, snr=1.0, T=512, seed=12345)
        ch = meta["channel"]
        ci = CHANNELS.index(ch) if ch else int(np.argmax(obs.std(0)))
        bg = gen_background(T=512, seed=12345, hawkes=(st == "S4_hawkes"))
        vr = obs[:, ci].std() / (bg[:, ci].std() + 1e-12)
        x = obs[:, ci] - obs[:, ci].mean()
        ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        F = np.abs(np.fft.rfft(x)) ** 2
        peak = float(F[1:].max() / (F[1:].mean() + 1e-12))
        # 环的粗代理：自相关首次过零后是否有第二峰
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac = ac / ac[0]
        second = float(ac[len(ac) // 4:].max()) if len(ac) > 8 else 0.0
        print(f"{st:<12} {str(ch):>8} {vr:>10.2f} {'-':>8} {ac1:>8.3f} "
              f"{peak:>10.2f} {second:>11.3f}")


if __name__ == "__main__":
    sanity_check()
