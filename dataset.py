"""Fetch benchmark dataset from Roboflow API — no local download."""

import csv
import io
import os
import zipfile

import requests
from dotenv import load_dotenv

load_dotenv()


def load_records(taxonomy: dict) -> list:
    """Fetch and parse Roboflow export in memory; no files written to disc."""
    workspace = os.environ["RF_WORKSPACE"]
    project   = os.environ["RF_PROJECT"]
    version   = os.environ.get("RF_VERSION", "1")
    api_key   = os.environ["RF_API_KEY"]

    print("Fetching dataset from Roboflow...")

    # Request export to get a short-lived download link
    resp = requests.get(
        f"https://api.roboflow.com/{workspace}/{project}/{version}/multiclass",
        params={"api_key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    link = resp.json()["export"]["link"]

    # Pull the entire ZIP into memory
    zip_resp = requests.get(link, timeout=120)
    zip_resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(zip_resp.content))

    # Find _classes.csv inside the ZIP
    csv_name = next((n for n in zf.namelist() if n.endswith("_classes.csv")), None)
    if csv_name is None:
        raise FileNotFoundError("_classes.csv not found in Roboflow export ZIP")

    # Map bare filename → bytes for every image in the ZIP
    image_bytes = {
        os.path.basename(n): zf.read(n)
        for n in zf.namelist()
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    key_cls  = set(taxonomy["key"])
    base_cls = set(taxonomy["base"])
    records  = []

    reader = csv.DictReader(io.StringIO(zf.read(csv_name).decode("utf-8")))
    for row in reader:
        row = {
            k.lower().replace(" ", "_").replace("-", "_"): v
            for k, v in row.items()
        }
        img_col  = next(
            (k for k in row if k in ("filename", "file", "image", "name")),
            list(row.keys())[0],
        )
        filename = os.path.basename(row[img_col])

        if filename not in image_bytes:
            continue

        key_gt  = [c for c in key_cls  if int(row.get(c, 0) or 0)]
        base_gt = [c for c in base_cls if int(row.get(c, 0) or 0)]
        if not key_gt and not base_gt:
            continue

        records.append({
            "filename":  filename,
            "img_bytes": image_bytes[filename],
            "key":       key_gt,
            "base":      base_gt,
        })

    return records
