"""Preview the pollen XLSX layout for quick manual inspection."""

import pandas as pd

xls = pd.ExcelFile("data_vk/Pelud_VK-Karlo.xlsx")
print("sheets:", xls.sheet_names[:5], "...")

for sheet in xls.sheet_names[:2]:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=12)
    print("\nSheet", sheet)
    print(raw.head(12))
