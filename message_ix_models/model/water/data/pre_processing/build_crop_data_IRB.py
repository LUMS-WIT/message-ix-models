"""Build ``data/water/crops/*.csv`` from real Indus crop/irrigation data.

One-off, standalone conversion script (not part of the importable
:mod:`message_ix_models` package, run manually whenever the source data
changes) -- the crop/irrigation-data counterpart of
:mod:`build_basin_links_IRB`, following the exact same convention: real
source data committed under ``data/water/crops/raw/`` (small enough for a
normal git object, no Git LFS), converted here into clean CSVs consumed by
:mod:`message_ix_models.model.water.data.crops`.

Source data
-----------
From the team's own ``indus_model/input`` folder (same parent directory the
network module's raw data came from), committed under
``data/water/crops/raw/``::

    crop_input_data.csv                    -- per-crop land area (2015,
                                               irrigated/rainfed) and yield,
                                               PID-keyed.
    crop_irrigation_water_calibrated.csv   -- crop water requirement
                                               (MCM/day per Mha), PID-keyed,
                                               by (scenario, model, crop,
                                               node, year, time).
    land_availability_map.csv              -- total available land (Mha),
                                               PID-keyed.
    basin_irrigation_transfers.csv         -- monthly irrigation transfer
                                               profile (MCM/day), PID-keyed,
                                               wide format (one column per
                                               month).
    irrigation_electricity_fraction.csv    -- fraction of irrigation pumping
                                               that is electric, PID-keyed.
    historical_irrigation_withdrawals_act.csv -- historical sw/gw diversion
                                               activity, PID-keyed.
    crop_tech_data.csv                     -- per-crop costs/crop
                                               coefficient. NOT basin-keyed
                                               (``node`` is the single
                                               literal ``"pak"`` for every
                                               row) -- a flat table applied
                                               to all Pakistan basins alike.
    irr_tech_data.csv                      -- per-irrigation-method costs,
                                               water efficiency, electricity
                                               intensity, lifetime. NOT
                                               basin-keyed (``node`` is the
                                               single literal ``"Indus"``
                                               for every row) -- a flat
                                               table applied to all basins.

These are the same exact files the legacy R+GAMS indus_ix model
(``NEST/MESSAGEix/basin_msggdx_load_inputs.r`` /
``basin_msggdx_technologies.r``) reads to build its three-tier
crop / rainfed / irrigation technology structure -- this script is the
Python-side conversion, targeting the
:mod:`message_ix_models.model.water.data.crops` package's CSV schema.

The PID -> BCU_name crosswalk
------------------------------
Shared with :mod:`build_basin_links_IRB` via :mod:`_pid_crosswalk` -- see
that module / :mod:`build_basin_links_IRB`'s docstring for the evidence
behind the mapping. It is basin identity only, domain-agnostic, so both
scripts use the identical dict rather than maintaining separate copies.

Climate scenario / GCM baseline
--------------------------------
``crop_irrigation_water_calibrated.csv`` and
``historical_irrigation_withdrawals_act.csv`` both carry 15
``(scenario, model)`` combinations (``historical``/``rcp26``/``rcp60`` x
``ensemble``/4 GCMs) -- matching the R model's ``climate_scenario`` /
``climate_model`` variables. This script filters to
``("historical", "ensemble")`` as the baseline, matching the scenario the
real ``LTS_nexus_debug`` scenario this project already solves against was
itself calibrated under. Change ``CLIMATE_SCENARIO`` / ``CLIMATE_MODEL``
below to build against a different climate projection.

Usage
-----
Run this script directly -- no editing needed, source data is committed
in-repo. It overwrites the CSVs under ``data/water/crops/`` in place::

    python build_crop_data_IRB.py
"""

from pathlib import Path

import pandas as pd

from _pid_crosswalk import CROSSWALK

_CROPS_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops"

SOURCE_DIR = _CROPS_DATA_DIR / "raw"
OUTPUT_DIR = _CROPS_DATA_DIR

#: Baseline climate scenario/GCM used to filter the multi-scenario raw
#: files. See module docstring.
CLIMATE_SCENARIO = "historical"
CLIMATE_MODEL = "ensemble"

#: Crops confirmed present (identically) in both crop_input_data.csv and
#: crop_irrigation_water_calibrated.csv.
EXPECTED_CROPS = {
    "cotton", "fodder", "fruit", "maize", "pulses", "rice", "sugarcane",
    "vegetables", "wheat",
}

#: Irrigation methods confirmed present in irr_tech_data.csv.
EXPECTED_IRR_METHODS = {
    "irr_canal_lining_flood", "irr_drip", "irr_drip_smart", "irr_flood",
    "irr_smart", "irr_sprinkler", "irr_sprinkler_smart",
}


def _crosswalk_node(df: pd.DataFrame, node_col: str = "node") -> pd.DataFrame:
    """Map a PID-keyed column to this project's BCU_name via CROSSWALK.

    Drops any row whose PID isn't in CROSSWALK (there should be none --
    CROSSWALK covers all 24 basins across all 4 countries in the source
    data -- but this stays defensive rather than assuming).
    """
    before = len(df)
    df = df[df[node_col].isin(CROSSWALK)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  NOTE: dropped {dropped} row(s) with a node not in CROSSWALK")
    df[node_col] = df[node_col].map(CROSSWALK)
    return df


def build_crop_input_data() -> pd.DataFrame:
    """Per-crop land area (2015) and yield, PID-keyed -> BCU_name-keyed."""
    df = pd.read_csv(SOURCE_DIR / "crop_input_data.csv")
    df = _crosswalk_node(df, "node")
    missing = EXPECTED_CROPS - set(df["crop"])
    assert not missing, f"crop_input_data.csv missing expected crops: {missing}"
    return df


def build_crop_irrigation_water() -> pd.DataFrame:
    """Crop water requirement (MCM/day/Mha), filtered to the baseline
    climate scenario/GCM, PID-keyed -> BCU_name-keyed."""
    df = pd.read_csv(SOURCE_DIR / "crop_irrigation_water_calibrated.csv")
    df = df[
        (df["scenario"] == CLIMATE_SCENARIO) & (df["model"] == CLIMATE_MODEL)
    ].drop(columns=["scenario", "model"])
    df = _crosswalk_node(df, "node")
    missing = EXPECTED_CROPS - set(df["crop"])
    assert not missing, (
        f"crop_irrigation_water_calibrated.csv missing expected crops: {missing}"
    )
    return df


def build_land_availability() -> pd.DataFrame:
    """Total available land (Mha), PID-keyed -> BCU_name-keyed."""
    df = pd.read_csv(SOURCE_DIR / "land_availability_map.csv")
    return _crosswalk_node(df, "node")


def build_irrigation_electricity_fraction() -> pd.DataFrame:
    """Fraction of irrigation pumping that is electric, PID-keyed."""
    df = pd.read_csv(SOURCE_DIR / "irrigation_electricity_fraction.csv")
    df = df.rename(columns={"PID": "node"})
    return _crosswalk_node(df, "node")


def build_basin_irrigation_transfers() -> pd.DataFrame:
    """Monthly irrigation transfer profile (MCM/day), PID-keyed, reshaped
    from wide (one column per month) to long (one row per basin-month)."""
    df = pd.read_csv(SOURCE_DIR / "basin_irrigation_transfers.csv")
    df = df.rename(columns={"PID": "node"})
    month_cols = [c for c in df.columns if c.startswith("X2010.")]
    assert len(month_cols) == 12, f"expected 12 monthly columns, got {month_cols}"
    long_df = df.melt(
        id_vars=["node", "units"], value_vars=month_cols,
        var_name="month", value_name="value",
    )
    long_df["month"] = long_df["month"].str.replace("X2010.", "", regex=False).astype(int)
    return _crosswalk_node(long_df, "node")


def build_historical_irrigation_withdrawals() -> pd.DataFrame:
    """Historical sw/gw diversion activity, filtered to the baseline
    climate scenario/GCM, PID-keyed -> BCU_name-keyed."""
    df = pd.read_csv(SOURCE_DIR / "historical_irrigation_withdrawals_act.csv")
    df = df[
        (df["scenario"] == CLIMATE_SCENARIO) & (df["model"] == CLIMATE_MODEL)
    ].drop(columns=["scenario", "model"])
    return _crosswalk_node(df, "node")


def build_crop_tech_data() -> pd.DataFrame:
    """Per-crop costs/crop coefficient. Flat table (node is always "pak"
    in the source) applied to every Pakistan basin alike -- not
    basin-keyed, no crosswalk needed, ``node`` column dropped."""
    df = pd.read_csv(SOURCE_DIR / "crop_tech_data.csv")
    assert set(df["node"]) == {"pak"}, "crop_tech_data.csv is no longer flat"
    df = df.drop(columns=["node"])
    missing = EXPECTED_CROPS - set(df["crop"])
    assert not missing, f"crop_tech_data.csv missing expected crops: {missing}"
    return df


def build_irr_tech_data() -> pd.DataFrame:
    """Per-irrigation-method costs/efficiency/lifetime. Flat table (node is
    always "Indus" in the source) applied to every basin alike -- not
    basin-keyed, no crosswalk needed, ``node`` column dropped."""
    df = pd.read_csv(SOURCE_DIR / "irr_tech_data.csv")
    assert set(df["node"]) == {"Indus"}, "irr_tech_data.csv is no longer flat"
    df = df.drop(columns=["node"])
    missing = EXPECTED_IRR_METHODS - set(df["irr_tech"])
    assert not missing, f"irr_tech_data.csv missing expected methods: {missing}"
    return df


#: (build function, output filename) pairs.
BUILDERS = [
    (build_crop_input_data, "crop_input_data_IRB.csv"),
    (build_crop_irrigation_water, "crop_irrigation_water_IRB.csv"),
    (build_land_availability, "land_availability_IRB.csv"),
    (build_irrigation_electricity_fraction, "irrigation_electricity_fraction_IRB.csv"),
    (build_basin_irrigation_transfers, "basin_irrigation_transfers_IRB.csv"),
    (build_historical_irrigation_withdrawals, "historical_irrigation_withdrawals_IRB.csv"),
    (build_crop_tech_data, "crop_tech_data_IRB.csv"),
    (build_irr_tech_data, "irr_tech_data_IRB.csv"),
]

#: The 13 Pakistan BCU_names -- every PID-keyed output should cover all of
#: these (sanity check, not enforced as a hard assertion since some raw
#: files may legitimately have sparser basin coverage).
PAK_BASINS = {v for k, v in CROSSWALK.items() if k.startswith("PAK_")}


if __name__ == "__main__":
    for build_fn, filename in BUILDERS:
        print(f"Building {filename} ...")
        df = build_fn()
        n_nan = df.isna().sum().sum()
        if n_nan:
            print(f"  NOTE: {n_nan} NaN value(s) in output")
        if "node" in df.columns:
            covered = PAK_BASINS & set(df["node"])
            missing = PAK_BASINS - covered
            if missing:
                print(f"  NOTE: missing PAK basins in this file: {sorted(missing)}")
        print(f"  {len(df)} rows")
        df.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"  Wrote {OUTPUT_DIR / filename}")
    print("\nDone.")
