"""Download official Airparif 2022 station CSVs and inventory their contents.

Airparif publishes one ArcGIS item per monitoring station and year.  The item
data endpoint contains a CSV with six metadata/header rows followed by hourly
UTC observations.  This script discovers the files from Airparif's DCAT feed,
keeps the raw files unchanged, and writes a compact station/pollutant inventory.
"""

from __future__ import annotations

from release_paths import resolve_path as _release_path

import csv
import json
import re
from pathlib import Path

import requests


OUT = _release_path('E:\\lunwen4\\2\\空气质量\\Paris_2022_event')
RAW = OUT / "raw" / "airparif_2022_station_csv"
PROCESSED = OUT / "processed"
DCAT_URL = "https://data-airparif-asso.opendata.arcgis.com/data.json"
ITEM_RE = re.compile(r"[?&]id=([0-9a-f]{32})", re.I)


def discover_items() -> list[dict[str, str]]:
    response = requests.get(DCAT_URL, timeout=120)
    response.raise_for_status()
    datasets = response.json()["dataset"]
    items: list[dict[str, str]] = []
    for dataset in datasets:
        title = str(dataset.get("title", "")).strip()
        description = re.sub(r"<[^>]+>", " ", str(dataset.get("description", "")))
        identifier = str(dataset.get("identifier", ""))
        match = ITEM_RE.search(identifier)
        if not (match and re.match(r"^2022(?:\s|_)", title, re.I)):
            continue
        if "station" not in description.lower():
            continue
        items.append(
            {
                "item_id": match.group(1),
                "title": title,
                "description": " ".join(description.split()),
                "identifier": identifier,
            }
        )
    unique = {item["item_id"]: item for item in items}
    return sorted(unique.values(), key=lambda item: item["title"])


def download_item(item: dict[str, str]) -> Path:
    item_id = item["item_id"]
    metadata_url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}?f=pjson"
    metadata_response = requests.get(metadata_url, timeout=60)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    if metadata.get("type") != "CSV":
        raise ValueError(f"{item_id} is {metadata.get('type')}, not CSV")
    station_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", metadata["title"]).strip("_")
    target = RAW / f"{station_slug}_{item_id}.csv"
    if not target.exists() or target.stat().st_size != int(metadata.get("size", -1)):
        data_url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data"
        data_response = requests.get(data_url, timeout=120)
        data_response.raise_for_status()
        target.write_bytes(data_response.content)
    item.update(
        {
            "arcgis_title": str(metadata.get("title", "")),
            "owner": str(metadata.get("owner", "")),
            "license": str(metadata.get("licenseInfo", "")),
            "size_bytes": str(target.stat().st_size),
            "raw_path": str(target),
        }
    )
    return target


def inspect_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        metadata = [next(rows) for _ in range(6)]
        dates = []
        nonempty = [0] * max(0, len(metadata[-1]) - 1)
        row_count = 0
        for row in rows:
            if not row:
                continue
            row_count += 1
            dates.append(row[0])
            for index, value in enumerate(row[1:]):
                if value.strip():
                    nonempty[index] += 1
    station_codes = metadata[2][1:]
    station_names = metadata[1][1:]
    pollutants = metadata[4][1:]
    units = metadata[5][1:]
    return {
        "station_codes": "|".join(dict.fromkeys(station_codes)),
        "station_names": "|".join(dict.fromkeys(station_names)),
        "pollutants": "|".join(pollutants),
        "units": "|".join(units),
        "nonempty_counts": "|".join(str(value) for value in nonempty),
        "rows": str(row_count),
        "start_utc": min(dates) if dates else "",
        "end_utc": max(dates) if dates else "",
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    items = discover_items()
    if not items:
        raise RuntimeError("No 2022 Airparif station CSV items were discovered")
    inventory = []
    for index, item in enumerate(items, start=1):
        path = download_item(item)
        record = dict(item)
        record.update(inspect_csv(path))
        inventory.append(record)
        print(f"[{index:02d}/{len(items):02d}] {record['arcgis_title']}: {record['pollutants']}", flush=True)
    fields = list(inventory[0])
    with (PROCESSED / "airparif_2022_station_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    provenance = {
        "publisher": "Airparif",
        "catalog": DCAT_URL,
        "license": sorted(set(row["license"] for row in inventory)),
        "downloaded_station_files": len(inventory),
        "raw_files_are_unmodified": True,
    }
    (PROCESSED / "download_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COMPLETE", len(inventory), "station files", OUT)


if __name__ == "__main__":
    main()
