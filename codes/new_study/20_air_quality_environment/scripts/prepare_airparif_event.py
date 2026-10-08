"""Create a tidy Airparif event dataset and pollutant-availability audit."""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = _release_path('E:\\lunwen4\\2\\空气质量\\Paris_2022_event')
INVENTORY = ROOT / "processed" / "airparif_2022_station_inventory.csv"
OUT = ROOT / "processed"
START = pd.Timestamp("2022-07-16 00:00:00", tz="UTC")
END = pd.Timestamp("2022-07-20 00:00:00", tz="UTC")
START_TEXT = "2022-07-16 00:00:00Z"
END_TEXT = "2022-07-20 00:00:00Z"

# Explicit source-context groups.  They prevent roadside emission contrasts from
# being interpreted as ventilation effects.  Named road sites are traffic sites;
# EIFF3 is an elevated observation and temporary school data are kept separately.
TRAFFIC = {
    "A1", "AUT", "BASCH", "BONAP", "BP_EST", "CELES", "ELYS",
    "HAUS", "OPERA", "RD934", "RN2", "RN20", "RN6", "SOULT",
}
D075_BACKGROUND = {"PA01H", "PA07", "PA12", "PA13", "PA15L", "PA18"}
ELEVATED = {"EIFF3"}
TEMPORARY = {"ANATOLEF"}


def context(code: str) -> str:
    if code in TRAFFIC:
        return "traffic"
    if code in D075_BACKGROUND:
        return "D075_urban_background"
    if code in ELEVATED:
        return "elevated"
    if code in TEMPORARY:
        return "temporary"
    if code.startswith("RUR"):
        return "regional_rural_background"
    return "urban_or_suburban_background"


def read_station(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header_rows = [next(reader) for _ in range(6)]
        width = len(header_rows[4])
        selected_rows = [row for row in reader if row and START_TEXT <= row[0] <= END_TEXT]
    records = []
    for row in selected_rows:
        time = pd.to_datetime(row[0], utc=True)
        for column in range(1, min(width, len(row))):
            pollutant = header_rows[4][column].strip().upper()
            if pollutant not in {"NO", "NO2", "NOX", "O3", "PM10", "PM25"}:
                continue
            value = pd.to_numeric(row[column], errors="coerce")
            records.append(
                {
                    "time_utc": time,
                    "station_name": header_rows[1][column].strip(),
                    "station_code": header_rows[2][column].strip(),
                    "pollutant": pollutant,
                    "value_ugm3": value,
                    "unit_original": header_rows[5][column].strip(),
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    inventory = pd.read_csv(INVENTORY)
    frames = [read_station(Path(path)) for path in inventory["raw_path"]]
    long = pd.concat(frames, ignore_index=True)
    long["site_context"] = long["station_code"].map(context)
    long = long.sort_values(["station_code", "time_utc", "pollutant"])
    long.to_csv(OUT / "airparif_event_long.csv", index=False, encoding="utf-8-sig")

    wide = long.pivot_table(
        index=["time_utc", "station_code", "station_name", "site_context"],
        columns="pollutant",
        values="value_ugm3",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    # Molar concentration is suitable for Ox because O3 and NO2 have different
    # molar masses. ug m-3 divided by g mol-1 equals umol m-3.
    if {"O3", "NO2"}.issubset(wide):
        paired = wide["O3"].notna() & wide["NO2"].notna()
        wide["Ox_umol_m3"] = np.where(
            paired, wide["O3"] / 47.9982 + wide["NO2"] / 46.0055, np.nan
        )
        wide["Ox_O3eq_ugm3"] = np.where(
            paired, wide["O3"] + wide["NO2"] * 47.9982 / 46.0055, np.nan
        )
    wide.to_csv(OUT / "airparif_event_wide.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    expected = int((END - START) / pd.Timedelta(hours=1)) + 1
    for (code, name, group), frame in wide.groupby(
        ["station_code", "station_name", "site_context"], dropna=False
    ):
        row = {
            "station_code": code,
            "station_name": name,
            "site_context": group,
            "expected_hours": expected,
        }
        for pollutant in ("NO", "NO2", "NOX", "O3", "PM10", "PM25", "Ox_umol_m3"):
            row[f"n_{pollutant}"] = int(frame[pollutant].notna().sum()) if pollutant in frame else 0
            row[f"coverage_{pollutant}"] = row[f"n_{pollutant}"] / expected
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["site_context", "station_code"])
    summary.to_csv(OUT / "event_station_pollutant_coverage.csv", index=False, encoding="utf-8-sig")

    paired = summary.loc[summary["n_Ox_umol_m3"] > 0]
    print("event rows long", len(long), "wide", len(wide), "stations", len(summary))
    print("stations with NO2", int((summary.n_NO2 > 0).sum()))
    print("stations with O3", int((summary.n_O3 > 0).sum()))
    print("stations with paired Ox", len(paired))
    print(paired[["station_code", "station_name", "site_context", "n_Ox_umol_m3"]].to_string(index=False))


if __name__ == "__main__":
    main()
