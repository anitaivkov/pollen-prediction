"""Convert multi-sheet pollen XLSX into a single CSV."""

import argparse
import re
from pathlib import Path

import pandas as pd


SUMMARY_KEYWORDS = {
    "suma",
    "sum",
    "ukupno",
    "total",
    "zbroj",
}


def _normalize_header(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _find_header_row(raw: pd.DataFrame) -> int:
    max_scan = min(len(raw), 10)
    for idx in range(max_scan):
        row = raw.iloc[idx]
        string_cells = 0
        for cell in row:
            if isinstance(cell, str) and cell.strip():
                string_cells += 1
        if string_cells >= 3:
            return idx
    return 0


def _detect_month_day_cols(header: list[str], data: pd.DataFrame) -> tuple[int, int]:
    if data.shape[1] >= 2:
        col0 = pd.to_numeric(data.iloc[:, 0], errors="coerce")
        col1 = pd.to_numeric(data.iloc[:, 1], errors="coerce")
        if col0.between(1, 12).mean() >= 0.7 and col1.between(1, 31).mean() >= 0.7:
            return 0, 1

    month_col = None
    day_col = None

    for idx, name in enumerate(header):
        name_norm = _normalize_header(name).lower()
        if month_col is None and (
            "mjesec" in name_norm or "mesec" in name_norm or name_norm == "mj" or "month" in name_norm
        ):
            month_col = idx
        if day_col is None and (
            "dan" in name_norm or "datum" in name_norm or "day" in name_norm
        ):
            day_col = idx

    if month_col is None or day_col is None:
        best_month = (-1.0, None)
        best_day = (-1.0, None)
        for col_idx in range(data.shape[1]):
            series = pd.to_numeric(data.iloc[:, col_idx], errors="coerce").dropna()
            if series.empty:
                continue
            month_ratio = series.between(1, 12).mean()
            day_ratio = series.between(1, 31).mean()
            max_val = series.max()
            if month_ratio > best_month[0] and max_val <= 12:
                best_month = (month_ratio, col_idx)
            if day_ratio > best_day[0] and max_val <= 31:
                best_day = (day_ratio, col_idx)
        if month_col is None and best_month[1] is not None and best_month[0] >= 0.7:
            month_col = best_month[1]
        if day_col is None and best_day[1] is not None and best_day[0] >= 0.7:
            day_col = best_day[1]

    if month_col is None:
        month_col = 0
    if day_col is None:
        day_col = 1 if month_col != 1 else 0

    return month_col, day_col


def _find_data_end_col(header: list[str], start_col: int) -> int:
    for idx in range(start_col, len(header)):
        if not _normalize_header(header[idx]):
            return idx
    return len(header)


def _extract_year(sheet_name: str) -> int:
    match = re.search(r"(19|20)\d{2}", str(sheet_name))
    if not match:
        raise ValueError(f"Cannot infer year from sheet name: {sheet_name}")
    return int(match.group(0))


def _collect_sheet_data(sheet_name: str, raw: pd.DataFrame) -> pd.DataFrame:
    header_row = _find_header_row(raw)
    header = [_normalize_header(val) for val in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1 :].reset_index(drop=True)

    month_col, day_col = _detect_month_day_cols(header, data)
    month_vals = pd.to_numeric(data.iloc[:, month_col], errors="coerce")
    day_vals = pd.to_numeric(data.iloc[:, day_col], errors="coerce")

    valid_rows = month_vals.between(1, 12) & day_vals.between(1, 31)
    data = data.loc[valid_rows].copy()
    month_vals = month_vals.loc[valid_rows].astype(int)
    day_vals = day_vals.loc[valid_rows].astype(int)

    year = _extract_year(sheet_name)
    date = pd.to_datetime(
        {"year": year, "month": month_vals, "day": day_vals},
        errors="coerce",
    )
    data = data.loc[date.notna()].copy()
    date = date.loc[date.notna()]

    data_start_col = max(month_col, day_col) + 1
    data_end_col = _find_data_end_col(header, data_start_col)

    plant_columns: dict[str, pd.Series] = {}
    for col_idx in range(data_start_col, data_end_col):
        name = header[col_idx]
        if col_idx in (month_col, day_col):
            continue
        if not name:
            continue
        if name.lower() in SUMMARY_KEYWORDS:
            continue
        if isinstance(raw.iloc[header_row, col_idx], (int, float)):
            continue

        series_raw = pd.to_numeric(data.iloc[:, col_idx], errors="coerce")
        numeric_ratio = series_raw.notna().mean() if len(series_raw) else 0
        if numeric_ratio < 0.7:
            continue

        cleaned_name = name.strip()
        series = series_raw.fillna(0).round(0).astype(int)
        if cleaned_name in plant_columns:
            plant_columns[cleaned_name] = plant_columns[cleaned_name].fillna(0) + series
        else:
            plant_columns[cleaned_name] = series

    result = pd.DataFrame(plant_columns)
    result.insert(0, "Date", date.dt.strftime("%Y-%m-%d"))
    return result


def _apply_template_order(df: pd.DataFrame, template_path: Path | None) -> pd.DataFrame:
    if template_path is None or not template_path.exists():
        return df
    template = pd.read_csv(template_path, nrows=0)
    template_cols = [c for c in template.columns if c != "Date"]
    current_cols = [c for c in df.columns if c != "Date"]
    ordered = [c for c in template_cols if c in current_cols]
    extras = sorted([c for c in current_cols if c not in ordered])
    return df[["Date", *ordered, *extras]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert multi-sheet pollen XLSX to a single CSV like data_lux/pollen.csv",
    )
    parser.add_argument("--input", required=True, help="Path to the .xlsx file (e.g. data_vk/raw/Pelud_VK-Karlo.xlsx)")
    parser.add_argument("--output", required=True, help="Path to the output .csv file")
    parser.add_argument(
        "--template-csv",
        default="data_lux/pollen.csv",
        help="Optional template CSV to enforce column order",
    )
    parser.add_argument(
        "--sheets",
        nargs="*",
        help="Optional list of sheet names to include (default: all)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    template_path = Path(args.template_csv) if args.template_csv else None

    xls = pd.ExcelFile(input_path)
    sheet_names = args.sheets if args.sheets else xls.sheet_names

    all_frames = []
    for sheet in sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        frame = _collect_sheet_data(sheet, raw)
        all_frames.append(frame)

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined = combined.fillna(0)
    combined = _apply_template_order(combined, template_path)
    combined = combined.sort_values("Date")
    for col in combined.columns:
        if col == "Date":
            continue
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).round(0).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
