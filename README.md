# Pollen LSTM - Kratke upute

Ovaj projekt koristi LSTM model za predviđanje koncentracije peludi na temelju meteoroloških varijabli i povijesnih podataka koncentracija peludi.

## Struktura projekta
- `data_lux/` - koncentracije peludi i meteorološki podaci za Luksemburg
- `data_vk/` - ulazni podaci (CSV datoteke za pelud i meteorologiju) i original .xslx datoteke u `data_vk/raw/`
- `models_vk/` - rezultati treniranja, organizirani po biljkama
- `scripts_data_conversion/` - python skripte za konverziju podataka iz .xslx  u .csv format
- `pollen_lstm_simple.py` - glavna skripta za treniranje modela

## Pokretanje treniranja
Primjer naredbe pokretanja skripte `pollen_lstm_simple.py`:
```
python pollen_lstm_simple.py --pollen data_vk/pollen_vk.csv --weather data_vk/weather_vk.csv --weather2 data_vk/weather_vk.csv --target Ambrosia --output models_vk
```

## Izlazne datoteke
Za svaku biljku rezultati se spremaju u podmapu, npr. `models_vk/ambrosia/`:
- `.keras` — spremljeni model
- `.joblib` — scaler i informacije o ulaznim značajkama
- `.txt` — osnovne metrike modela
- `.csv` — log treniranja po epohama

## Bilješke
- Model automatski dijeli podatke 70% za treniranje, 30% za testiranje.
- Za predikciju na novim podacima koristi se spremljeni model i scaler.
- Za svaku biljku koristi se 3 meteorološke varijable: temp_max, pressure, wind (i njihove forecast vrijednosti koje su za sada duplikati iz datoteke weather.csv, plan je implementirati prave weather forecast varijable).

## Autor
Anita Ivkov, 2026.

Završni rad s naslovom "Razvoj i evaluacija modela strojnog učenja za predviđanje koncentracije peluda".
