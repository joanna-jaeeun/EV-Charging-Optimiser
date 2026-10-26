"""Offline tests for prepare.py, using a small synthetic raw download."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import prepare  # noqa: E402

MAIN = "AGILE-24-10-01"


def make_raw(start="2025-10-24 00:00", end="2025-10-28 23:30", product=MAIN):
    """Half-hourly rows in UTC; window covers the 26 Oct 2025 clock change."""
    ts = pd.date_range(start, end, freq="30min", tz="UTC")
    price = [((i % 48) - 10) * 0.5 for i in range(len(ts))]  # includes negatives
    return pd.DataFrame({
        "product_code": product,
        "valid_from": ts,
        "valid_to": ts + pd.Timedelta(minutes=30),
        "value_exc_vat": price,
        "value_inc_vat": price,
    })


def test_keeps_one_product_and_drops_pre_launch():
    raw = pd.concat([make_raw(), make_raw(product="AGILE-18-02-21")])
    launch = pd.Timestamp("2025-10-25 00:00", tz="UTC")
    out, report = prepare.clean(raw, MAIN, launch)
    assert out["ts_utc"].min() == launch
    assert report["dropped_pre_launch"] == 48
    assert report["rows_in"] == len(make_raw())


def test_missing_slot_is_flagged_not_filled():
    raw = make_raw().drop(index=100)
    out, report = prepare.clean(raw, MAIN)
    assert report["missing_slots"] == 1
    assert out["price_inc_vat"].isna().sum() == 1
    assert out["is_missing"].sum() == 1


def test_exact_duplicate_dropped_conflict_raises():
    raw = make_raw()
    out, report = prepare.clean(pd.concat([raw, raw.iloc[[5]]]), MAIN)
    assert report["dropped_exact_duplicates"] == 1

    conflict = raw.iloc[[5]].copy()
    conflict["value_inc_vat"] += 1
    with pytest.raises(ValueError, match="different prices"):
        prepare.clean(pd.concat([raw, conflict]), MAIN)


def test_clock_change_day_has_50_local_slots_and_passes_validation():
    out, _ = prepare.clean(make_raw(), MAIN)
    per_day = out.groupby(out["ts_local"].dt.date).size()
    assert per_day[pd.Timestamp("2025-10-26").date()] == 50
    errors, _ = prepare.validate(out)
    assert errors == []


def test_negative_prices_are_kept():
    out, report = prepare.clean(make_raw(), MAIN)
    assert report["negative_slots"] > 0
    assert (out["price_inc_vat"] < 0).any()


def test_validate_catches_broken_spacing():
    out, _ = prepare.clean(make_raw(), MAIN)
    broken = out.drop(index=200)  # leaves a 60-minute step
    errors, _ = prepare.validate(broken)
    assert any("not 30 minutes" in e for e in errors)


def test_wrong_slot_length_raises():
    raw = make_raw()
    raw.loc[10, "valid_to"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="not 30 minutes"):
        prepare.clean(raw, MAIN)
