#!/usr/bin/env python3
"""阶段 3 第 2 层b：baostock 5 分钟线下载器（可续跑）。

实测（2026-10-06）
----------------
* baostock 5 分钟线：**48 bar/日**，`time` 形如 `20231009093500000`（首 bar 收于 09:35）
* 5 分钟数据 **2020 年才有**（2015/2018 查询返回 0 行）
* 速度：1 年 ≈ 11,616 行 / 10.5 s → 单只 3 年 ≈ 32 s
* 用 `sz.000001` 格式；`6xxxxx`→`sh.`，`0/3xxxxx`→`sz.`，`8/4/9xxxxx`（北交所）baostock 不支持 → 跳过

用法
----
    python src/fetch_intraday.py --n-stocks 60 --workers 4
    python src/fetch_intraday.py --n-stocks 60 --workers 4 --check   # 只报告已有进度
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_LIST = ROOT / "results" / "exp3b_csmar_hf" / "index_n300_win120_s20.csv"
CACHE = ROOT / "data" / "intraday"

FIELDS = "date,time,code,open,high,low,close,volume,amount,adjustflag"
FREQ = "5"


def to_bs_code(stkcd: str):
    """CSMAR 6 位代码 → baostock 代码；北交所返回 None。"""
    s = str(stkcd).zfill(6)
    if s[0] == "6":
        return "sh." + s
    if s[0] in ("0", "3"):
        return "sz." + s
    return None      # 8/4/9 开头 = 北交所/新三板，baostock 不支持


def _worker_init():
    import baostock as bs
    bs.login()


def fetch_one(args):
    """下载一只股票的 5 分钟线 → data/intraday/<code>.parquet。可续跑。"""
    stkcd, start, end = args
    out = CACHE / f"{stkcd}.parquet"
    if out.exists() and out.stat().st_size > 0:
        try:
            n = len(pd.read_parquet(out, columns=["time"]))
            return stkcd, n, 0.0, "cached"
        except Exception:      # noqa: BLE001
            pass

    import baostock as bs
    code = to_bs_code(stkcd)
    if code is None:
        return stkcd, 0, 0.0, "skip_bj"

    t0 = time.time()
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=start, end_date=end,
                                      frequency=FREQ, adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return stkcd, 0, time.time() - t0, f"empty({rs.error_code})"
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(out, index=False)
    return stkcd, len(df), time.time() - t0, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stocks", type=int, default=60)
    ap.add_argument("--start", default="2023-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stock-list", default=str(STOCK_LIST))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)

    sl = Path(args.stock_list)
    if not sl.exists():
        print(f"[中止] 找不到股票清单 {sl}")
        print("       先跑 src/experiment_3b_csmar_hf.py 生成 index_*.csv")
        sys.exit(1)
    stocks = pd.read_csv(sl)["Stkcd"].astype(str).str.zfill(6).drop_duplicates()
    stocks = [s for s in stocks if to_bs_code(s) is not None][:args.n_stocks]
    print(f"股票清单 {sl.name}：取前 {len(stocks)} 只可下载的（沪/深主板+创业板）")
    print(f"区间 {args.start} → {args.end}  频率 {FREQ} 分钟  workers={args.workers}")

    done = [s for s in stocks if (CACHE / f"{s}.parquet").exists()]
    print(f"已有缓存 {len(done)}/{len(stocks)} 只  →  {CACHE}")
    if args.check:
        tot = sum(len(pd.read_parquet(CACHE / f"{s}.parquet", columns=["time"]))
                  for s in done)
        print(f"缓存总行数 {tot:,}")
        return

    todo = [(s, args.start, args.end) for s in stocks
            if not (CACHE / f"{s}.parquet").exists()]
    if not todo:
        print("全部已缓存，无需下载")
        return

    t0 = time.time()
    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init) as ex:
        for i, (stk, n, el, status) in enumerate(ex.map(fetch_one, todo, chunksize=1)):
            ok += (status == "ok")
            if (i + 1) % 5 == 0 or i + 1 == len(todo):
                et = time.time() - t0
                eta = et / (i + 1) * (len(todo) - i - 1)
                print(f"  [{i+1}/{len(todo)}] {stk} {status} {n}行 {el:.1f}s  "
                      f"累计 {et/60:.1f}min  预计剩余 {eta/60:.1f}min", flush=True)
    print(f"\n完成：新增 {ok} 只，用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"缓存目录 {CACHE}")


if __name__ == "__main__":
    main()
