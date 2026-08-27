"""Build ``data/water/crops/emission_factor_IRB.csv`` -- real, basin-level
fertilizer-emission factors per crop -- from province-level fertilizer
application rates, area-weighted onto basins via a hand-parsed shapefile
(no GIS library available in this environment).

This is Phase 4 of the crop/irrigation migration plan: :mod:`land`'s
``crop_<crop>`` technologies carry ``emission_factor = 0`` (an honest
placeholder, not a fabricated number) until this script's output exists and
``config.yaml``'s ``emission_factor_enabled`` flag is turned on.

The problem this solves
------------------------
The R model's own fertilizer-emissions computation
(``basin_msggdx_load_inputs.r``, ``fertilizer_by_crop.df`` /
``fertilizer_emissions.df``) needs a province -> basin area-weighted
mapping, normally computed via GIS raster extraction against a province
polygon layer -- not available here (no geopandas/shapely/rasterio/fiona).
Resolved the same way :mod:`build_basin_links_IRB` resolves the network
module's basin-adjacency data: hand-parse the real shapefile directly (see
:mod:`_shapefile`) rather than fetching a substitute or reprojecting.

Source data
-----------
Committed under ``data/water/crops/raw/`` (same convention as Phase 0's
network/crop data)::

    Indus_prov_bcu.shp / .dbf / .shx / .prj  -- province x basin
        intersection layer. .dbf fields: PID (compound id, unused here),
        ADMIN (province, e.g. "PAK|Punjab"), SUBASIN (e.g. "Indus|9"),
        OUTX/OUTY (outlet coordinates), DOWN, REGION. 41 records -- more
        than 24 because several basins straddle more than one province
        (e.g. Indus|9 splits 5 ways). .shp holds each record's polygon
        geometry (no area field in the .dbf), read via
        :func:`_shapefile.read_shp_polygon_areas`.
    indus_fertilizer_dat.csv -- real per-province, per-crop,
        per-irrigation-type (irrigated/rainfed) N/P2O5/K2O application
        rates (kg/ha), keyed by a `region` column matching Indus_prov_bcu's
        `ADMIN` format exactly (e.g. "PAK|Punjab") for Pakistan, but only
        the single value "IND" for all of India (no per-state breakdown).
    fertilizer_emissions.csv -- kg CO2-equivalent emitted per kg of N /
        P2O5 / K2O applied (JEC E3-database).

The SUBASIN numbering is NOT the same as this project's PID numbering
-----------------------------------------------------------------------
It would be a mistake to assume ``Indus_prov_bcu.dbf``'s ``SUBASIN`` field
(``"Indus|9"``) uses the same basin-numbering convention as
``Indus_bcu.dbf``'s ``PID`` field (``"PAK_9"``) just because both go up to
13 for Pakistan -- **they don't**. Checked before use, not assumed: cross-
matching every Pakistan basin's real outlet coordinates (``OUTX``/``OUTY``,
present in *both* files) shows the numbering is a genuine permutation, not
an off-by-one or a coincidence -- e.g. ``PAK_1``'s real outlet
(72.14375, 31.16875) exactly matches ``Indus|2``'s outlet, not ``Indus|1``'s.
The full confirmed mapping (used below, keyed by exact outlet-coordinate
match, not by number)::

    PAK_1 -> Indus|2    PAK_6  -> Indus|8     PAK_11 -> Indus|13
    PAK_2 -> Indus|3    PAK_7  -> Indus|10    PAK_12 -> Indus|9
    PAK_3 -> Indus|4    PAK_8  -> Indus|11    PAK_13 -> Indus|12
    PAK_4 -> Indus|6    PAK_9  -> Indus|1
    PAK_5 -> Indus|7    PAK_10 -> Indus|5

(The same real ``Indus_bcu.dbf`` used for the network module's PID
crosswalk provides the PID side of this coordinate match -- see
``data/water/network/raw/``.)

Usage
-----
Run this script directly -- no editing needed, source data is committed
in-repo. It overwrites ``data/water/crops/emission_factor_IRB.csv`` in
place::

    python build_fertilizer_weights_IRB.py
"""

from pathlib import Path

import pandas as pd

from _pid_crosswalk import CROSSWALK
from _shapefile import read_dbf, read_shp_polygon_areas

_CROPS_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops"
_NETWORK_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "network"

SOURCE_DIR = _CROPS_DATA_DIR / "raw"
NETWORK_RAW_DIR = _NETWORK_DATA_DIR / "raw"
OUTPUT_PATH = _CROPS_DATA_DIR / "emission_factor_IRB.csv"

#: Coordinate match tolerance (decimal degrees) -- outlet points are
#: recorded to ~6 decimal places in both files and should match exactly or
#: near enough that a small tolerance is just floating-point safety, not a
#: real ambiguity (basins are km-scale apart, this tolerance is ~1m).
_COORD_TOL = 1e-4


def _match_pid_to_subasin(basin_recs: list[dict], prov_recs: list[dict]) -> dict[str, str]:
    """Match each PID (from Indus_bcu.dbf) to its SUBASIN (from
    Indus_prov_bcu.dbf) by exact outlet-coordinate match. See module
    docstring -- the two files' basin numbering is a genuine permutation,
    not directly comparable by number.
    """
    mapping = {}
    unmatched = []
    for b in basin_recs:
        bx, by = float(b["OUTX"]), float(b["OUTY"])
        found = None
        for p in prov_recs:
            px, py = float(p["OUTX"]), float(p["OUTY"])
            if abs(px - bx) < _COORD_TOL and abs(py - by) < _COORD_TOL:
                found = p["SUBASIN"]
                break
        if found:
            mapping[b["PID"]] = found
        else:
            unmatched.append(b["PID"])
    if unmatched:
        raise AssertionError(
            f"Could not match {len(unmatched)} PID(s) to a SUBASIN by outlet "
            f"coordinate: {unmatched}. Check Indus_bcu.dbf / Indus_prov_bcu.dbf "
            "are the expected versions."
        )
    return mapping


def _province_weights(prov_recs: list[dict], areas: list[float]) -> dict[str, dict[str, float]]:
    """Return ``{SUBASIN: {ADMIN: weight}}``, weights normalized to sum to
    1.0 within each SUBASIN (its provinces' real relative polygon areas)."""
    by_subasin: dict[str, dict[str, float]] = {}
    for rec, area in zip(prov_recs, areas, strict=True):
        by_subasin.setdefault(rec["SUBASIN"], {})[rec["ADMIN"]] = area
    weights = {}
    for subasin, admin_areas in by_subasin.items():
        total = sum(admin_areas.values())
        weights[subasin] = {
            admin: (area / total if total > 0 else 0.0) for admin, area in admin_areas.items()
        }
    return weights


def _fertilizer_region(admin: str) -> str:
    """indus_fertilizer_dat.csv has per-province rates for Pakistan
    (matching ADMIN exactly, e.g. "PAK|Punjab") but only one combined "IND"
    row for all of India -- collapse any Indian ADMIN to the country code.
    Afghanistan and China provinces have no fertilizer data at all (not
    present in the source file) -- returned as-is, filtered out downstream.
    """
    if admin.startswith("IND|"):
        return "IND"
    return admin


def build() -> pd.DataFrame:
    basin_recs = read_dbf(NETWORK_RAW_DIR / "Indus_bcu.dbf")
    prov_recs = read_dbf(SOURCE_DIR / "Indus_prov_bcu.dbf")
    areas = read_shp_polygon_areas(SOURCE_DIR / "Indus_prov_bcu.shp")
    assert len(prov_recs) == len(areas)

    pid_to_subasin = _match_pid_to_subasin(basin_recs, prov_recs)
    weights = _province_weights(prov_recs, areas)

    fert_rate = pd.read_csv(SOURCE_DIR / "indus_fertilizer_dat.csv")
    fert_emis = pd.read_csv(SOURCE_DIR / "fertilizer_emissions.csv")
    co2eq = fert_emis[fert_emis["out"] == "CO2eq"].set_index("fertilizer")["value"]

    # kg CO2eq per kg of each fertilizer type applied, per (region, crop,
    # irrigation type): sum over N/P2O5/K2O of rate[kg/ha] * co2eq[kg/kg].
    fert_rate = fert_rate.copy()
    fert_rate["co2eq_per_ha"] = fert_rate.apply(
        lambda r: r["value"] * co2eq.get(r["fertilizer"], 0.0), axis=1
    )
    region_crop_emissions = (
        fert_rate.groupby(["region", "crop", "irrigation"])["co2eq_per_ha"].sum()
    )

    rows = []
    skipped_no_fert_data = []
    for pid, bcu_name in CROSSWALK.items():
        subasin = pid_to_subasin.get(pid)
        if subasin is None:
            continue  # basin has no province-intersection record at all
        prov_weights = weights.get(subasin, {})

        for crop in sorted(fert_rate["crop"].unique()):
            for irrigation in ["irrigated", "rainfed"]:
                total_weight = 0.0
                weighted_emission = 0.0
                for admin, w in prov_weights.items():
                    region = _fertilizer_region(admin)
                    key = (region, crop, irrigation)
                    if key not in region_crop_emissions.index:
                        continue  # no fertilizer data for this province (AFG/CHN)
                    weighted_emission += w * region_crop_emissions[key]
                    total_weight += w
                if total_weight <= 0:
                    skipped_no_fert_data.append((bcu_name, crop, irrigation))
                    continue
                # Renormalize over only the provinces with real fertilizer
                # data, rather than silently understating the rate for
                # basins that straddle a data-covered and a data-absent
                # province (e.g. partly Afghanistan).
                rows.append(
                    {
                        "node": bcu_name,
                        "crop": crop,
                        "irrigation": irrigation,
                        "value": weighted_emission / total_weight,
                        "unit": "kgCO2eq/ha",
                    }
                )

    if skipped_no_fert_data:
        print(
            f"NOTE: {len(skipped_no_fert_data)} (basin, crop, irrigation) combos "
            "have no fertilizer data in any overlapping province (e.g. purely "
            "Afghan/Chinese basins) -- skipped, not zero-filled:"
        )
        for bcu_name, crop, irrigation in skipped_no_fert_data[:10]:
            print(f"  {bcu_name} / {crop} / {irrigation}")
        if len(skipped_no_fert_data) > 10:
            print(f"  ... and {len(skipped_no_fert_data) - 10} more")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    print(f"Built {len(df)} rows covering {df['node'].nunique()} basins, "
          f"{df['crop'].nunique()} crops")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
