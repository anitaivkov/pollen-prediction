# Data conversion scripts

## Files

- xlsx_to_csv.py
  - Convert multi-sheet pollen XLSX into a single CSV.
  - Example:
    - python xlsx_to_csv.py --input data_vk/Pelud_VK-Karlo.xlsx --output data_vk/pollen_vk.csv

- xlsx_weather_to_csv.py
  - Convert VK weather XLSX (yearly matrices) into a single CSV.
  - Example:
    - python xlsx_weather_to_csv.py --input data_vk/VK-klima.xlsx --output data_vk/weather_vk.csv

- report_pollen_vk.py
  - Quick summary (rows, columns, date range, duplicate headers) for data_vk/pollen_vk.csv.

- report_decimals_vk.py
  - Check for decimal values in data_vk/pollen_vk.csv.

- inspect_xlsx.py
  - Preview the first sheet of the pollen XLSX to confirm layout.

- inspect_vk_klima.py
  - Preview the weather XLSX sheet structure.
