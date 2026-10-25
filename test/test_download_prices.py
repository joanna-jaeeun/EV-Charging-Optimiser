"""Offline tests for the price downloader: no network, a fake session serves fixed pages."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import download_prices as dp  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Maps a URL (ignoring query params) to a response; 'next' links are full URLs."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(404))


def rate(ts_from, ts_to, price):
    return {"valid_from": ts_from, "valid_to": ts_to,
            "value_exc_vat": price / 1.05, "value_inc_vat": price}


RATES_URL = (f"{dp.API}/products/AGILE-24-10-01/electricity-tariffs/"
             "E-1R-AGILE-24-10-01-C/standard-unit-rates/")
PAGE_2 = RATES_URL + "?page=2"


def test_iter_pages_follows_next_link():
    session = FakeSession({
        RATES_URL: FakeResponse(200, {"results": [1, 2], "next": PAGE_2}),
        PAGE_2: FakeResponse(200, {"results": [3], "next": None}),
    })
    assert list(dp.iter_pages(session, RATES_URL)) == [1, 2, 3]


def test_missing_tariff_returns_empty_frame():
    session = FakeSession({})  # every URL 404s
    assert dp.fetch_unit_rates(session, "AGILE-24-10-01", "C").empty


def test_unit_rates_are_utc_and_keep_negative_prices():
    session = FakeSession({
        RATES_URL: FakeResponse(200, {"results": [
            rate("2026-10-24T23:00:00Z", "2026-10-24T23:30:00Z", 12.5),
            rate("2026-10-24T23:30:00Z", "2026-10-25T00:00:00Z", -3.2),
        ], "next": None}),
    })
    df = dp.fetch_unit_rates(session, "AGILE-24-10-01", "C")
    assert len(df) == 2
    assert str(df["valid_from"].dt.tz) == "UTC"
    assert (df["value_inc_vat"] < 0).any()  # negative prices are real, never dropped
    assert set(df["tariff_code"]) == {"E-1R-AGILE-24-10-01-C"}


def test_export_tariffs_are_excluded():
    assert dp.is_agile_import("AGILE-24-10-01")
    assert dp.is_agile_import("AGILE-FLEX-22-11-25")
    assert not dp.is_agile_import("AGILE-OUTGOING-19-05-13")
    assert not dp.is_agile_import("GO-VAR-22-10-14")


def test_summarise_one_row_per_product():
    df = pd.DataFrame({
        "product_code": ["A", "A", "B"],
        "valid_from": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"], utc=True),
        "value_inc_vat": [10.0, -1.0, 20.0],
    })
    out = dp.summarise(df)
    assert list(out.index) == ["A", "B"]
    assert out.loc["A", "min_p"] == -1.0
