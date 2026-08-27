"""Shared configuration and data readers for the crop/rainfed/irrigation
technology port (see :mod:`land`, :mod:`rainfed`, :mod:`irrigation_tech` for
the technologies themselves). Mirrors the role :mod:`network`'s top-of-file
``CONFIG``/``_basin_region_map`` play for that module -- one place that
knows how to load the archetype config and read the real, crosswalked CSVs
:mod:`message_ix_models.model.water.data.pre_processing.build_crop_data_IRB`
produces, so ``land.py``/``rainfed.py``/``irrigation_tech.py`` don't each
reimplement it.
"""

import logging

import pandas as pd
import yaml

from message_ix_models import Context
from message_ix_models.model.water.utils import basin_region_map
from message_ix_models.util import package_data_path

log = logging.getLogger(__name__)

# Load configuration (crop list, irrigation-method archetypes, constants)
# once at import time -- same convention as network.CONFIG.
_CONFIG_PATH = package_data_path("water", "crops", "config.yaml")
with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

#: The 9 modeled crops.
CROPS: list[str] = CONFIG["crops"]

#: The 7 irrigation methods -> per-method archetype dict (currently just
#: ``historical_capacity: bool``).
IRRIGATION_METHODS: dict[str, dict] = CONFIG["irrigation_methods"]

__all__ = [
    "CONFIG",
    "CROPS",
    "IRRIGATION_METHODS",
    "basin_region_map",
    "crop_land_commodity",
    "crop_tech_name",
    "crop_yield_commodity",
    "irr_tech_name",
    "rainfed_tech_name",
    "read_crop_input_data",
    "read_crop_irrigation_water",
    "read_crop_tech_data",
    "read_emission_factor",
    "read_irr_tech_data",
    "read_land_availability",
]


def crop_tech_name(crop: str) -> str:
    return f"crop_{crop}"


def rainfed_tech_name(crop: str) -> str:
    return f"rainfed_{crop}"


def irr_tech_name(method: str, crop: str) -> str:
    return f"{method}_{crop}"


def crop_land_commodity(crop: str) -> str:
    return f"{crop}_land"


def crop_yield_commodity(crop: str) -> str:
    return f"{crop}_yield"


def _read_crop_csv(filename: str) -> pd.DataFrame:
    """Read one of the clean CSVs built by ``build_crop_data_IRB.py``.

    Returns an empty frame (with a logged warning) if the file doesn't
    exist -- same safe-no-op convention as :func:`network.read_basin_links`.
    """
    path = package_data_path("water", "crops", filename)
    if not path.exists():
        log.warning(
            "No %s found; crop module adds no technologies from this file.",
            filename,
        )
        return pd.DataFrame()
    return pd.read_csv(path)


def basins_with_crop_data() -> set[str]:
    """Every BCU_name present in the real, committed crop data -- independent
    of any one context's ``valid_basins``.

    Used to decide whether this module applies to a given context at all
    (e.g. an R12/global-model context has its own, unrelated basin naming
    that will never overlap with this Indus-specific data): the crop CSVs
    are not region-suffixed like ``basin_links_<regions>.csv``, so without
    this check the module would otherwise appear "applicable" (non-empty
    ``valid_basins``) for any region and silently register 9 crop
    technologies with no real data behind them. See
    :func:`.land.crop_technology_names`.
    """
    df = _read_crop_csv("crop_input_data_IRB.csv")
    return set(df["node"]) if not df.empty else set()


def _filter_basins(df: pd.DataFrame, context: "Context") -> pd.DataFrame:
    """Filter a node-keyed frame to ``context.valid_basins``, if set."""
    if df.empty or "node" not in df.columns:
        return df
    valid_basins = getattr(context, "valid_basins", None)
    if not valid_basins:
        return df
    return df[df["node"].isin(valid_basins)].reset_index(drop=True)


def read_crop_input_data(context: "Context") -> pd.DataFrame:
    """Per-crop 2015 land area (irrigated/rainfed) and yield.

    Columns: crop, par, node (BCU_name), time, unit, value. ``par`` in
    {crop_irr_land_2015, crop_rainfed_land_2015, irrigation_yield,
    rain-fed_yield}.
    """
    return _filter_basins(_read_crop_csv("crop_input_data_IRB.csv"), context)


def read_crop_irrigation_water(context: "Context") -> pd.DataFrame:
    """Crop water requirement, MCM/day per Mha, baseline climate scenario.

    Columns: crop, node (BCU_name), year, time, value.
    """
    return _filter_basins(_read_crop_csv("crop_irrigation_water_IRB.csv"), context)


def read_crop_irrigation_water_annual(context: "Context") -> pd.DataFrame:
    """Crop water requirement, day-weighted from monthly MCM/day/Mha to an
    annual MCM/Mha total, per (crop, node, year).

    The source data is decadal (year in {2010, 2020, ..., 2060}) --
    :func:`.irrigation_tech.add_irrigation_techs` maps each model year to
    the nearest available year here rather than requiring an exact match.
    """
    return annualize_monthly(read_crop_irrigation_water(context), ["crop", "node", "year"])


def read_land_availability(context: "Context") -> pd.DataFrame:
    """Total available land, Mha. Columns: node (BCU_name), value, units."""
    return _filter_basins(_read_crop_csv("land_availability_IRB.csv"), context)


def read_crop_tech_data() -> pd.DataFrame:
    """Per-crop costs/crop coefficient. Flat table (no node dimension --
    applies to every Pakistan basin alike). Columns: crop, par, time, unit,
    value."""
    return _read_crop_csv("crop_tech_data_IRB.csv")


#: Days per month (non-leap approximation -- matches the level of precision
#: already used elsewhere in this port, e.g. the R model's own day-count
#: conventions). Used to day-weight monthly MCM/day-style raw data into a
#: real annual total, rather than a naive sum-of-12-values.
DAYS_IN_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def annualize_monthly(df: pd.DataFrame, group_cols: list[str], month_col: str = "time") -> pd.DataFrame:
    """Day-weight and sum a monthly ``value`` column (MCM/day-style) into a
    real annual total (MCM/year-style) per ``group_cols``.

    Both :func:`read_crop_irrigation_water` and the historical irrigation
    withdrawal data are monthly MCM/day rates -- this is shared rather than
    reimplemented per caller.
    """
    if df.empty:
        return df
    weighted = df.assign(_annual=df["value"] * df[month_col].map(DAYS_IN_MONTH))
    return weighted.groupby(group_cols, as_index=False)["_annual"].sum().rename(
        columns={"_annual": "value"}
    )


def read_historical_irrigation_withdrawals(context: "Context") -> pd.DataFrame:
    """Historical sw/gw irrigation diversion activity, day-weighted to an
    annual total per (node, tec).

    Columns: node (BCU_name), tec (irrigation_sw_diversion /
    irrigation_gw_diversion), value (MCM/year).
    """
    df = _filter_basins(_read_crop_csv("historical_irrigation_withdrawals_IRB.csv"), context)
    return annualize_monthly(df, ["node", "tec"])


def read_emission_factor(context: "Context") -> pd.DataFrame:
    """Real, basin-level fertilizer-emission factors per crop (Phase 4,
    ``pre_processing/build_fertilizer_weights_IRB.py``).

    Columns: node (BCU_name), crop, irrigation (irrigated/rainfed), value
    (kgCO2eq/ha), unit. Not every (basin, crop, irrigation) combination is
    present -- basins with no fertilizer data in any overlapping province
    (e.g. purely Afghan/Chinese basins) are absent, not zero-filled; see
    the build script's docstring.
    """
    return _filter_basins(_read_crop_csv("emission_factor_IRB.csv"), context)


def read_irr_tech_data() -> pd.DataFrame:
    """Per-irrigation-method costs/efficiency/lifetime. Flat table (no node
    dimension -- applies to every basin alike). Columns: irr_tech, par,
    time, unit, value."""
    return _read_crop_csv("irr_tech_data_IRB.csv")
