"""Download Octopus Agile half-hourly import prices for one region.

Usage (from the repo root):
    python src/data/download_prices.py --region C

Writes:
    data/raw/agile_<region>_raw.parquet   every price row from every Agile product version
    data/raw/agile_products.json          the product versions found and their availability

The product and tariff endpoints are public, so no API key is needed.
Cleaning (UTC alignment, gaps, overlapping versions) happens later in prepare.py;
this script only fetches and stores what the API returns, unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://api.octopus.energy/v1"

# Agile export tariffs share the "AGILE" prefix; we only want import prices.
EXCLUDE_MARKERS = ("OUTGOING", "EXPORT")

# Region letters used in tariff codes (GSP groups). C = London.
REGIONS = set("ABCDEFGHJKLMNP")

PAGE_SIZE = 1500  # largest page the unit-rate endpoint accepts


def get_json(session: requests.Session, url: str, params: dict | None = None,
             retries: int = 4) -> dict | None:
    """GET a JSON page with simple backoff. Returns None on 404 (tariff not offered)."""
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)  # back off on rate limiting or server errors
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return None


def iter_pages(session: requests.Session, url: str, params: dict | None = None):
    """Yield every result across a paginated endpoint by following the 'next' link."""
    data = get_json(session, url, params)
    while data is not None:
        yield from data.get("results", [])
        next_url = data.get("next")
        if not next_url:
            break
        data = get_json(session, next_url)  # 'next' already carries the query string


def is_agile_import(code: str) -> bool:
    return code.startswith("AGILE") and not any(m in code for m in EXCLUDE_MARKERS)


def discover_agile_products(session: requests.Session, start_year: int = 2018) -> dict[str, dict]:
    """Find every Agile import product version.

    The catalogue only lists products on sale at a given moment, so we take
    quarterly snapshots with `available_at` and keep every Agile code we see.
    """
    found: dict[str, dict] = {}
    now = datetime.now(timezone.utc)
    for year in range(start_year, now.year + 1):
        for month in (1, 4, 7, 10):
            at = datetime(year, month, 1, tzinfo=timezone.utc)
            if at > now:
                break
            params = {"available_at": at.strftime("%Y-%m-%dT%H:%M:%SZ")}
            for product in iter_pages(session, f"{API}/products/", params):
                code = product.get("code", "")
                if is_agile_import(code):
                    found[code] = {
                        "code": code,
                        "full_name": product.get("full_name"),
                        "available_from": product.get("available_from"),
                        "available_to": product.get("available_to"),
                    }
    return found


def fetch_unit_rates(session: requests.Session, product_code: str, region: str) -> pd.DataFrame:
    """All half-hourly unit rates for one product version in one region."""
    tariff = f"E-1R-{product_code}-{region}"
    url = f"{API}/products/{product_code}/electricity-tariffs/{tariff}/standard-unit-rates/"
    rows = list(iter_pages(session, url, {"page_size": PAGE_SIZE}))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["product_code"] = product_code
    df["tariff_code"] = tariff
    # The API returns ISO strings in UTC ("...Z"); keep them tz-aware in UTC.
    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True)
    df["valid_to"] = pd.to_datetime(df["valid_to"], utc=True)
    keep = ["product_code", "tariff_code", "valid_from", "valid_to",
            "value_exc_vat", "value_inc_vat"]
    return df[[c for c in keep if c in df.columns]]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One line per product version: rows, first and last half-hour, price range."""
    return (df.groupby("product_code")
              .agg(rows=("valid_from", "size"),
                   first=("valid_from", "min"),
                   last=("valid_from", "max"),
                   min_p=("value_inc_vat", "min"),
                   max_p=("value_inc_vat", "max"))
              .sort_values("first"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="C", help="GSP region letter (C = London)")
    parser.add_argument("--out-dir", default="data/raw", type=Path)
    parser.add_argument("--start-year", default=2018, type=int)
    args = parser.parse_args()

    region = args.region.upper()
    if region not in REGIONS:
        raise SystemExit(f"Unknown region '{region}'. Use one of {''.join(sorted(REGIONS))}.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        products = discover_agile_products(session, args.start_year)
        print(f"Found {len(products)} Agile import product versions")

        frames = []
        for code in sorted(products):
            df = fetch_unit_rates(session, code, region)
            print(f"  {code}: {len(df):>7,} rows")
            if not df.empty:
                frames.append(df)

    if not frames:
        raise SystemExit("No price rows returned; check the region and network access.")

    prices = (pd.concat(frames, ignore_index=True)
                .sort_values(["valid_from", "product_code"])
                .reset_index(drop=True))

    raw_path = args.out_dir / f"agile_{region}_raw.parquet"
    prices.to_parquet(raw_path, index=False)
    (args.out_dir / "agile_products.json").write_text(json.dumps(products, indent=2))

    print(f"\nSaved {len(prices):,} rows to {raw_path}")
    print(summarise(prices).to_string())


if __name__ == "__main__":
    main()
