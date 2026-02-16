"""Summarize rows, columns, date range, and duplicate headers in pollen_vk.csv."""

import re
from collections import Counter

import pandas as pd


def main() -> None:
    path = "data_vk/pollen_vk.csv"
    df = pd.read_csv(path)

    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print(f"date range: {df['Date'].min()} .. {df['Date'].max()}")

    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    ctr = Counter(cols_lower)
    dups = sorted([c for c, n in ctr.items() if n > 1])
    print(f"case-insensitive duplicates: {len(dups)}")
    if dups:
        print("examples:")
        for name in dups[:20]:
            originals = [c for c in cols if c.lower() == name]
            print("  ", originals)

    ctr_exact = Counter(cols)
    exact_dups = sorted([c for c, n in ctr_exact.items() if n > 1])
    print(f"exact duplicates: {len(exact_dups)}")

    normalized = {}
    for c in cols:
        key = re.sub(r"[^a-z]", "", c.lower())
        if key:
            normalized.setdefault(key, []).append(c)
    near = [v for v in normalized.values() if len(v) > 1]
    print(f"normalized duplicates: {len(near)}")
    if near:
        print("examples:")
        for group in near[:20]:
            print("  ", group)

    na_ratio = df.isna().mean().sort_values(ascending=False)
    print("top NaN ratios:")
    print(na_ratio.head(10).to_string())

    zero_cols = []
    for c in df.columns:
        if c == "Date":
            continue
        series = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if (series == 0).all():
            zero_cols.append(c)
    print(f"all-zero columns: {len(zero_cols)}")
    if zero_cols:
        print("examples:")
        print("  ", zero_cols[:20])


if __name__ == "__main__":
    main()
