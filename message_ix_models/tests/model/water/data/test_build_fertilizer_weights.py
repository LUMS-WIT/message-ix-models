"""Tests for pre_processing/build_fertilizer_weights_IRB.py -- specifically
the shapefile polygon-area parser and the province-weight normalization it
feeds, per the crop/irrigation migration plan's Phase 4 requirement ("weights
per subasin sum to 1.0; single-province basins get weight exactly 1.0").

This script isn't part of the importable message_ix_models package (see its
own module docstring), so it's imported here the same way it expects to be
run directly: with its own directory on sys.path.
"""

import sys
from pathlib import Path

import pytest

_PRE_PROCESSING_DIR = (
    Path(__file__).resolve().parents[4]
    / "model"
    / "water"
    / "data"
    / "pre_processing"
)
sys.path.insert(0, str(_PRE_PROCESSING_DIR))

from _shapefile import read_dbf, read_shp_polygon_areas  # noqa: E402
from build_fertilizer_weights_IRB import (  # noqa: E402
    NETWORK_RAW_DIR,
    SOURCE_DIR,
    _match_pid_to_subasin,
    _province_weights,
    build,
)


@pytest.fixture(scope="module")
def basin_recs():
    return read_dbf(NETWORK_RAW_DIR / "Indus_bcu.dbf")


@pytest.fixture(scope="module")
def prov_recs():
    return read_dbf(SOURCE_DIR / "Indus_prov_bcu.dbf")


@pytest.fixture(scope="module")
def areas(prov_recs):
    areas = read_shp_polygon_areas(SOURCE_DIR / "Indus_prov_bcu.shp")
    assert len(areas) == len(prov_recs)
    return areas


def test_shp_polygon_areas_are_positive(areas):
    """Every real polygon record has a nonzero, positive area -- the
    shoelace formula's abs() is doing its job, and no record was
    misaligned with the .dbf (which would likely produce nonsensical
    near-zero or wildly duplicate areas)."""
    assert all(a > 0 for a in areas)
    assert len(set(round(a, 6) for a in areas)) > len(areas) / 2, (
        "expected mostly-distinct areas; too many duplicates suggests "
        "records aren't being read correctly"
    )


def test_pid_to_subasin_matches_all_basins(basin_recs, prov_recs):
    """Every basin in Indus_bcu.dbf matches some province-intersection
    SUBASIN by real outlet coordinates -- the whole point of coordinate
    matching rather than assuming PID and SUBASIN share a numbering."""
    mapping = _match_pid_to_subasin(basin_recs, prov_recs)
    assert len(mapping) == len(basin_recs)
    # The known, hand-verified permutation for Pakistan basins (see module
    # docstring) -- pinned here so a future data change that silently
    # breaks the coordinate match is caught immediately, not just via a
    # generic "did every PID get *some* match" check.
    assert mapping["PAK_1"] == "Indus|2"
    assert mapping["PAK_9"] == "Indus|1"
    assert mapping["PAK_12"] == "Indus|9"


def test_province_weights_sum_to_one(prov_recs, areas):
    """Every subasin's province weights sum to exactly 1.0 -- the
    normalization is a real partition, not an approximation."""
    weights = _province_weights(prov_recs, areas)
    assert weights  # non-empty
    for subasin, w in weights.items():
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9), subasin


def test_single_province_basin_gets_weight_one(prov_recs, areas):
    """A subasin with only one overlapping province gets weight exactly
    1.0 for it -- the real, hand-verified case here is Indus|4 and
    Indus|6 (PAK_3 and PAK_4's basins), each entirely within one
    province."""
    weights = _province_weights(prov_recs, areas)
    for subasin in ["Indus|4", "Indus|6"]:
        assert len(weights[subasin]) == 1
        assert next(iter(weights[subasin].values())) == pytest.approx(1.0)


def test_build_produces_real_data_for_every_pakistan_basin():
    """The end-to-end build covers all 13 real Pakistan basins with
    plausible (nonzero, varying) emission factors -- not just that the
    weighting math works in isolation."""
    df = build()
    assert not df.empty
    pak_rows = df[df["node"].str.endswith("|PAKISTAN")]
    assert pak_rows["node"].nunique() == 13
    assert (pak_rows["value"] > 0).all()
    # Real basins should show real regional variation, not one constant
    # value copy-pasted everywhere.
    wheat_irrigated = pak_rows[
        (pak_rows["crop"] == "wheat") & (pak_rows["irrigation"] == "irrigated")
    ]
    assert wheat_irrigated["value"].nunique() > 1
