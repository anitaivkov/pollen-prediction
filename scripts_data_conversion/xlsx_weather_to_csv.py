"""Convert VK weather XLSX yearly matrices into a combined CSV."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


STATS_KEYS = {
    "ZBROJ",
    "SRED",
    "STD",
    "MAKS",
    "MIN",
    "AMPL",
    "DAN",
}

ROMAN_TO_MONTH = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}

SHEET_TO_COLUMN = {
    "sred.temp": "temp_mean",
    "maks.temp": "temp_max",
    "min.temp": "temp_min",
    "oborina": "precip",
    "temp-tla1": "soil_temp_1",
    "temp-tla2": "soil_temp_2",
    "tlak": "pressure",
    "sij.sunca": "sunshine",
    "vjetar": "wind",
}


def _normalize_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _is_year_row(row: pd.Series) -> bool:
    year_val = pd.to_numeric(row.iloc[0], errors="coerce")
    if pd.isna(year_val):
        return False
    year_val = int(year_val)
    if year_val < 1900 or year_val > 2100:
        return False

    month_hits = 0
    for cell in row.iloc[1:14]:
        cell_norm = _normalize_cell(cell).upper()
        if cell_norm in ROMAN_TO_MONTH:
            month_hits += 1
        else:
            num = pd.to_numeric(cell, errors="coerce")
            if not pd.isna(num) and 1 <= int(num) <= 12:
                month_hits += 1
    return month_hits >= 6


def _extract_month_columns(row: pd.Series) -> Dict[int, int]:
    month_cols: Dict[int, int] = {}
    for col_idx, cell in enumerate(row.iloc[1:14], start=1):
        cell_norm = _normalize_cell(cell).upper()
        if cell_norm in ROMAN_TO_MONTH:
            month_cols[col_idx] = ROMAN_TO_MONTH[cell_norm]
            continue
        num = pd.to_numeric(cell, errors="coerce")
        if not pd.isna(num) and 1 <= int(num) <= 12:
            month_cols[col_idx] = int(num)
    return month_cols


def _iter_day_rows(
    raw: pd.DataFrame,
    start_row: int,
) -> Iterable[Tuple[int, pd.Series]]:
    row_idx = start_row
    while row_idx < len(raw):
        row = raw.iloc[row_idx]
        first = row.iloc[0]
        if _is_year_row(row):
            break
        if isinstance(first, str) and _normalize_cell(first).upper() in STATS_KEYS:
            break
        day_val = pd.to_numeric(first, errors="coerce")
        if pd.isna(day_val):
            row_idx += 1
            continue
        day_int = int(day_val)
        if day_int < 1 or day_int > 31:
            row_idx += 1
            continue
        yield day_int, row
        row_idx += 1


def _parse_sheet(sheet_name: str, raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    row_idx = 0
    while row_idx < len(raw):
        row = raw.iloc[row_idx]
        if not _is_year_row(row):
            row_idx += 1
            continue
        year = int(pd.to_numeric(row.iloc[0], errors="coerce"))
        month_cols = _extract_month_columns(row)
        if not month_cols:
            row_idx += 1
            continue

        for day, data_row in _iter_day_rows(raw, row_idx + 1):
            for col_idx, month in month_cols.items():
                value = pd.to_numeric(data_row.iloc[col_idx], errors="coerce")
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "Date": f"{year:04d}-{month:02d}-{day:02d}",
                        "Value": float(value),
                    }
                )

        row_idx += 1

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.groupby("Date", as_index=False)["Value"].mean()
    result = result.rename(columns={"Value": SHEET_TO_COLUMN.get(sheet_name, sheet_name)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VK weather XLSX with yearly matrices into a combined CSV",
    )
    parser.add_argument("--input", required=True, help="Path to VK-klima.xlsx (e.g. data_vk/raw/VK-klima.xlsx)")
    parser.add_argument("--output", required=True, help="Path to the output .csv file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    xls = pd.ExcelFile(input_path)
    frames: List[pd.DataFrame] = []

    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        frame = _parse_sheet(sheet, raw)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No data parsed from the workbook.")

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="Date", how="outer")

    combined = combined.sort_values("Date")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
