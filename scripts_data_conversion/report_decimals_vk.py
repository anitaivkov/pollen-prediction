"""Check for decimal values in pollen_vk.csv."""

import pandas as pd


def main() -> None:
    path = "data_vk/pollen_vk.csv"
    df = pd.read_csv(path)

    decimal_cols = []
    for col in df.columns:
        if col == "Date":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        has_decimal = ((series.fillna(0) % 1) != 0).any()
        if has_decimal:
            decimal_cols.append(col)

    print(f"columns with decimal values: {len(decimal_cols)}")
    if decimal_cols:
        print("examples:")
        print("  ", decimal_cols[:20])

    if decimal_cols:
        sample_col = decimal_cols[0]
        series = pd.to_numeric(df[sample_col], errors="coerce")
        sample_vals = series[series.notna() & ((series % 1) != 0)].head(5).tolist()
        print(f"sample decimal values from {sample_col}:")
        print("  ", sample_vals)


if __name__ == "__main__":
    main()
