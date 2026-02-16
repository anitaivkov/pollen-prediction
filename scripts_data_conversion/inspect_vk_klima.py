"""Preview the VK weather XLSX sheet structure for debugging."""

import pandas as pd


xlsx_path = "data_vk/VK-klima.xlsx"
xls = pd.ExcelFile(xlsx_path)
print("sheets:", xls.sheet_names)

for sheet in xls.sheet_names:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=60)
    print("\nSheet:", sheet)
    print(raw.head(60))
