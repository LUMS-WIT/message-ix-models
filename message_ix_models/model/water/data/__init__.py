"""Generate input data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from message_ix_models import ScenarioInfo
from message_ix_models.util import add_par_data

from .demands import add_irrigation_demand, add_sectoral_demands, add_water_availability
from .infrastructure import add_desalination, add_infrastructure_techs
from .irrigation import add_irr_structure
from .water_for_ppl import cool_tech, non_cooling_tec
from .water_supply import add_e_flow, add_water_supply

if TYPE_CHECKING:
    from message_ix_models import Context

log = logging.getLogger(__name__)

#: Type alias for data generation functions
DataFunc = Callable[["Context"], dict[str, pd.DataFrame]]

DATA_FUNCTIONS: list[DataFunc] = [
    add_water_supply,
    cool_tech,  # Water & parasitic_electricity requirements for cooling technologies
    non_cooling_tec,
    add_sectoral_demands,
    add_water_availability,
    add_irrigation_demand,
    add_infrastructure_techs,
    add_desalination,
    add_e_flow,
    add_irr_structure,
]

DATA_FUNCTIONS_COUNTRY: list[DataFunc] = [
    add_water_supply,
    cool_tech,  # Water & parasitic_electricity requirements for cooling technologies
    non_cooling_tec,
    add_sectoral_demands,
    add_water_availability,
    # add_irrigation_demand, # not used and coming from GLOBIOM for the global region
    add_infrastructure_techs,
    add_desalination,
    add_e_flow,
    # add if statement: if irrigation: land component from external model
]


def _sanitize_node_columns(
    data: dict[str, pd.DataFrame], source: str
) -> dict[str, pd.DataFrame]:
    """Normalize and drop rows with unresolved node dimensions."""
    result: dict[str, pd.DataFrame] = {}

    for par_name, df in data.items():
        if not hasattr(df, "columns"):
            result[par_name] = df
            continue

        clean = df.copy()

        if "node" in clean.columns and "node_share" in clean.columns:
            missing_node = clean["node"].isna()
            if clean["node"].dtype == "object":
                missing_node |= clean["node"].astype(str).isin({"", "None", "nan"})
            clean.loc[missing_node, "node"] = clean.loc[missing_node, "node_share"]

        node_cols = [c for c in clean.columns if c == "node" or c.startswith("node_")]
        if node_cols:
            invalid = pd.Series(False, index=clean.index)

            for col in node_cols:
                invalid |= clean[col].isna()
                if clean[col].dtype == "object":
                    invalid |= clean[col].astype(str).isin({"", "None", "nan"})

            if invalid.any():
                sample_cols = node_cols + [
                    c
                    for c in (
                        "technology",
                        "commodity",
                        "level",
                        "mode",
                        "shares",
                        "year_vtg",
                        "year_act",
                        "value",
                    )
                    if c in clean.columns and c not in node_cols
                ]
                sample = clean.loc[invalid, sample_cols].head(3).to_dict("records")
                log.warning(
                    "Dropping %s row(s) with unresolved node labels in %s from %s(): %s",
                    int(invalid.sum()),
                    par_name,
                    source,
                    sample,
                )
                clean = clean.loc[~invalid].copy()

        result[par_name] = clean

    return result


def add_data(scenario, context: "Context", dry_run=False):
    """Populate `scenario` with MESSAGEix-Nexus data."""

    info = ScenarioInfo(scenario)
    context["water build info"] = info

    def map_tech(t):
        return str(t).split("__")[0]

    data_funcs: list[DataFunc] = (
        [add_water_supply, cool_tech, non_cooling_tec]
        if context.nexus_set == "cooling"
        else DATA_FUNCTIONS
        if context.type_reg == "global"
        else DATA_FUNCTIONS_COUNTRY
    )

    for func in data_funcs:
        log.info(f"from {func.__name__}()")
        try:
            data = func(context)
        except Exception as e:
            raise RuntimeError(
                f"Water build failed in {func.__name__}() before add_par_data: "
                f"{type(e).__name__}: {e}"
            ) from e

        # 🔥 FORCE mapping AFTER function output
        def process_df(df):
            if hasattr(df, "columns") and "technology" in df.columns:
                df = df.copy()
                df["technology"] = df["technology"].astype(str).apply(map_tech)
            return df

        if isinstance(data, dict):
            data = {k: process_df(v) for k, v in data.items()}
            data = _sanitize_node_columns(data, func.__name__)
        else:
            data = process_df(data)
            data = _sanitize_node_columns({"_": data}, func.__name__)["_"]

        # 🔥 FINAL SAFETY CHECK (PRINT DEBUG)
        if isinstance(data, dict):
            for k, df in data.items():
                if hasattr(df, "columns") and "technology" in df.columns:
                    if df["technology"].str.contains("__ot_").any():
                        raise ValueError("❌ Unmapped technology still exists!")
        else:
            if hasattr(data, "columns") and "technology" in data.columns:
                if data["technology"].str.contains("__ot_").any():
                    raise ValueError("❌ Unmapped technology still exists!")

        try:

            # # Show all units before adding parameters
            # used_units = set()

            # for par_name, df in data.items():
            #     if "unit" in df.columns:
            #         units = (
            #             df["unit"]
            #             .dropna()
            #             .astype(str)
            #             .str.strip()
            #             .unique()
            #         )
            #         print(f"\nParameter: {par_name}")
            #         print(units)
            #         used_units.update(units)

            # print("\n========== ALL UNITS ==========")
            # for u in sorted(used_units):
            #     print(u)

            add_par_data(scenario, data, dry_run=dry_run)
        except Exception as e:
            if isinstance(data, dict):
                detail = {
                    k: getattr(v, "shape", None)
                    for k, v in data.items()
                    if hasattr(v, "shape")
                }
            else:
                detail = getattr(data, "shape", None)
            raise RuntimeError(
                f"Water add_par_data failed in {func.__name__}() with data shapes "
                f"{detail}: {type(e).__name__}: {e}"
            ) from e

    log.info("done")
