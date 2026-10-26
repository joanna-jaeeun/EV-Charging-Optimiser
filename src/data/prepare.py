"""Clean the raw Agile download into one tidy half-hourly price series.

Usage (from the repo root):
    python src/data/prepare.py
    python src/data/prepare.py --product AGILE-23-12-06 --region C

Reads:
    data/raw/agile_<region>_raw.parquet   (from download_prices.py)
    data/raw/agile_products.json          (launch dates per product version)
Writes:
    data/processed/agile_<region>_prices.parquet

Cleaning decisions (each one is explained in the README):
    1. Keep ONE product version, so the series follows one pricing formula.
       Default AGILE-23-12-06: slot-for-slot identical to the version on sale
       today (AGILE-24-10-01) wherever both exist, and it starts a year earlier.
    2. Drop slots before the version's launch date (back-filled, never charged).
    3. Work in UTC; add Europe/London time only as a display column,
       so clock-change days (46 / 50 local slots) never break the arithmetic.
    4. Drop exact duplicate rows; stop if one slot has two different prices.
    5. Put every slot on a complete 30-minute grid; missing slots stay NaN
       and are flagged, never silently filled.
    6. Keep negative prices: they are real and matter most for charging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SLOT = pd.Timedelta(minutes=30)
LOCAL_TZ = "Europe/London"
VALID_LOCAL_DAY_SLOTS = {46, 48, 50}   # spring-forward, normal, fall-back
PRICE_BOUNDS = (-50.0, 100.0)          # p/kWh; outside this is suspicious, not removed


def load_launch(products_path: Path, product: str) -> pd.Timestamp | None:
    """Launch time (UTC) of a product version, from the downloader's product list."""
    if not products_path.exists():
        return None
    products = json.loads(products_path.read_text())
    info = products.get(product)
    if not info or not info.get("available_from"):
        return None
    ts = pd.Timestamp(info["available_from"])
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def clean(raw: pd.DataFrame, product: str,
          launch: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Return a complete UTC half-hourly series for one product plus a cleaning report."""
    df = raw.loc[raw["product_code"] == product,
                 ["valid_from", "valid_to", "value_exc_vat", "value_inc_vat"]].copy()
    if df.empty:
        raise ValueError(f"No rows for product {product!r}")

    report = {"product": product, "rows_in": len(df)}

    # 2. Slots before launch were back-filled, nobody paid them.
    if launch is not None:
        report["dropped_pre_launch"] = int((df["valid_from"] < launch).sum())
        df = df[df["valid_from"] >= launch]

    # 4. Exact duplicates are harmless; conflicting prices for one slot are not.
    before = len(df)
    df = df.drop_duplicates()
    report["dropped_exact_duplicates"] = before - len(df)
    conflicts = df["valid_from"].duplicated(keep=False)
    if conflicts.any():
        raise ValueError(f"{int(conflicts.sum())} rows give different prices for the same slot")

    bad_length = (df["valid_to"] - df["valid_from"]) != SLOT
    if bad_length.any():
        raise ValueError(f"{int(bad_length.sum())} slots are not 30 minutes long")

    # 5. Complete 30-minute grid in UTC; gaps become NaN and are flagged.
    df = df.set_index("valid_from").sort_index()
    grid = pd.date_range(df.index.min(), df.index.max(), freq=SLOT)
    df = df.reindex(grid)
    df.index.name = "ts_utc"

    out = (df.drop(columns="valid_to")
             .rename(columns={"value_inc_vat": "price_inc_vat",
                              "value_exc_vat": "price_exc_vat"})
             .reset_index())
    out["is_missing"] = out["price_inc_vat"].isna()
    # 3. Local time is for display and calendar features only.
    out["ts_local"] = out["ts_utc"].dt.tz_convert(LOCAL_TZ)
    out = out[["ts_utc", "ts_local", "price_inc_vat", "price_exc_vat", "is_missing"]]

    report.update({
        "rows_out": len(out),
        "missing_slots": int(out["is_missing"].sum()),
        "first_utc": str(out["ts_utc"].iloc[0]),
        "last_utc": str(out["ts_utc"].iloc[-1]),
        "negative_slots": int((out["price_inc_vat"] < 0).sum()),
    })
    return out, report


def validate(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Data checks (test layer 2). Errors stop the pipeline; warnings are reported."""
    errors, warnings = [], []

    if df["ts_utc"].duplicated().any():
        errors.append("duplicate timestamps")
    steps = df["ts_utc"].diff().dropna()
    if not (steps == SLOT).all():
        errors.append(f"{int((steps != SLOT).sum())} gaps between rows are not 30 minutes")

    # Every whole local day must have 46, 48 or 50 slots; first and last day may be partial.
    per_day = df.groupby(df["ts_local"].dt.date).size()
    inner = per_day.iloc[1:-1]
    odd = inner[~inner.isin(VALID_LOCAL_DAY_SLOTS)]
    if len(odd):
        errors.append(f"local days with unexpected slot counts: {odd.to_dict()}")

    lo, hi = PRICE_BOUNDS
    price = df["price_inc_vat"]
    outside = price[(price < lo) | (price > hi)]
    if len(outside):
        warnings.append(f"{len(outside)} prices outside {PRICE_BOUNDS} p/kWh")
    if df["is_missing"].any():
        warnings.append(f"{int(df['is_missing'].sum())} missing slots (left as NaN)")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="C")
    parser.add_argument("--product", default="AGILE-23-12-06")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--keep-pre-launch", action="store_true",
                        help="keep back-filled slots before the launch date")
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw_dir / f"agile_{args.region}_raw.parquet")
    launch = None if args.keep_pre_launch else load_launch(
        args.raw_dir / "agile_products.json", args.product)
    if launch is None and not args.keep_pre_launch:
        print("Warning: launch date not found; keeping all slots")

    prices, report = clean(raw, args.product, launch)
    errors, warnings = validate(prices)

    print("Cleaning report")
    for key, value in report.items():
        print(f"  {key:26} {value}")
    if launch is not None:
        print(f"  {'launch_utc':26} {launch}")
    for w in warnings:
        print(f"  WARNING  {w}")
    if errors:
        for e in errors:
            print(f"  ERROR    {e}")
        raise SystemExit("Validation failed; nothing written.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"agile_{args.region}_prices.parquet"
    prices.to_parquet(out_path, index=False)
    print(f"\nSaved {len(prices):,} rows to {out_path}")


if __name__ == "__main__":
    main()
