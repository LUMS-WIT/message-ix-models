"""Prepare data for ``rainfed_<crop>`` technologies -- the second layer of
the three-tier crop/rainfed/irrigation port of the legacy R+GAMS indus_ix
model's crop submodel (``NEST/MESSAGEix/basin_msggdx_technologies.r``,
~line 12694-12800).

Data model
----------
One technology per crop (``rainfed_<crop>``), present only in basins with
real, positive ``rain-fed_yield`` data for that crop (unlike
:mod:`.land`'s ``crop_<crop>``, which exists unconditionally everywhere --
this layer only where the R model's own data says rainfed cultivation of
this crop is real).

Consumes ``<crop>_land`` (the land :mod:`.land`'s ``crop_<crop>``
committed) and produces:

- the **generic**, cross-crop commodity ``crop_land`` (level ``area``) --
  not ``<crop>_land`` again. This matches the R model exactly
  (``commodity = 'crop_land'``, not ``paste0(crop, '_land')``): it's a
  shared "total cultivated land" pool across every crop, distinct from the
  crop-specific intermediate. Nothing currently consumes it (no
  land-availability cap wired in yet), same "structurally present,
  currently dangling" status as ``<crop>_yield`` below.
- ``<crop>_yield`` (level ``crop_yield``, new here -- not assumed to
  already exist as a generic ``primary`` level) -- simplified relative to
  the R model's monthly growing-season disaggregation and raw/residue/
  ethanol byproduct split (annual, raw yield only). Nothing downstream
  consumes yield yet in this port, so the extra fidelity isn't
  load-bearing; can be revisited if a downstream consumer (e.g. a food
  demand) is added later.

Zero-cost, matching the R model exactly (``rainfed_<crop>``'s
inv_cost/fix_cost/var_cost are all literal 0 there) -- omitted from this
module's output entirely rather than written as explicit zero rows, since
MESSAGE defaults an unset technology-year cost to zero regardless.

Vintaging: same fix as :mod:`.land`'s ``crop_<crop>`` (see that module's
docstring, "Annual (lft=1) vintaging dropped") -- the full, natural,
unrestricted ``get_vintage_and_active_years()`` result (not the R
model's raw ``lft = 1``, and not a single filtered-down vintage either;
both were tried and found broken -- see :mod:`.land` and
:mod:`.irrigation_tech`'s module docstrings for the full story).
``historical_new_capacity`` is registered only at the one genuine
pre-model seed year, where it actually applies; new investment beyond
that remains possible at any other model-horizon vintage, each with its
own real, matching cost/output data.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model.water.utils import get_vintage_and_active_years
from message_ix_models.util import broadcast, same_node, same_time

from .common import (
    CONFIG,
    CROPS,
    basins_with_crop_data,
    crop_land_commodity,
    crop_yield_commodity,
    rainfed_tech_name,
    read_crop_input_data,
)

log = logging.getLogger(__name__)

#: The generic, cross-crop "total cultivated land" commodity -- distinct
#: from land.py's crop-specific ``<crop>_land``. See module docstring.
GENERIC_LAND_COMMODITY = "crop_land"


def rainfed_technology_names(context: "Context") -> list[str]:
    """Return every ``rainfed_<crop>`` technology name for ``context``.

    Registers all 9 names regardless of per-basin data coverage (a
    technology with no parameter data behind it in a given build is inert,
    not harmful -- matches :func:`.land.crop_technology_names`'s
    convention). Empty if this module doesn't apply to ``context`` at all
    (see :func:`.common.basins_with_crop_data`).
    """
    valid_basins = getattr(context, "valid_basins", None) or set()
    if not (basins_with_crop_data() & set(valid_basins)):
        return []
    return [rainfed_tech_name(c) for c in CROPS]


def rainfed_commodity_names() -> list[str]:
    """Return every commodity this module produces: the generic
    ``crop_land`` plus every ``<crop>_yield``."""
    return [GENERIC_LAND_COMMODITY] + [crop_yield_commodity(c) for c in CROPS]


def add_rainfed_techs(context: "Context") -> dict[str, pd.DataFrame]:
    """Build ``rainfed_<crop>`` technologies.

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'output'.
    """
    valid_basins = set(getattr(context, "valid_basins", None) or set()) & basins_with_crop_data()
    if not valid_basins:
        return {}

    scen = context.get_scenario()
    scenario_info = ScenarioInfo(scen)
    lifetime = CONFIG["crop_technical_lifetime"]
    firstyear = scen.firstmodelyear
    sub_time = pd.Series(context.time)

    # Full, natural get_vintage_and_active_years() -- same design, same
    # reason, as land.py's crop_<crop> (see that module's docstring
    # "Annual (lft=1) vintaging dropped" for the full story).
    yv_ya = get_vintage_and_active_years(scenario_info, lifetime)
    year_vtg_all = sorted(set(yv_ya["year_vtg"]))
    hist_years = [y for y in year_vtg_all if y < firstyear]
    hist_year_vtg = max(hist_years) if hist_years else None

    df_input = read_crop_input_data(context)
    df_yield = df_input[(df_input["par"] == "rain-fed_yield") & (df_input["value"] > 0)]
    df_cap = df_input[df_input["par"] == "crop_rainfed_land_2015"]

    inp_frames, out_frames = [], []
    tl_frames, cons_frames, cf_frames = [], [], []
    hist_cap_frames = []

    for crop in CROPS:
        crop_yield = df_yield[df_yield["crop"] == crop]
        nodes = sorted(set(crop_yield["node"]) & valid_basins)
        if not nodes:
            continue

        tech = rainfed_tech_name(crop)
        node_loc = pd.Series(["B" + n for n in nodes])
        yield_value = (
            crop_yield.set_index("node")["value"].reindex(nodes).astype(float).reset_index(drop=True)
        )

        inp_frames.append(
            make_df(
                "input",
                technology=tech,
                value=1.0,
                unit="Mha",
                level="crop",
                commodity=crop_land_commodity(crop),
                mode="M1",
                node_loc=node_loc,
            )
            .pipe(broadcast, yv_ya, time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )

        out_frames.append(
            make_df(
                "output",
                technology=tech,
                value=1.0,
                unit="Mha",
                level="area",
                commodity=GENERIC_LAND_COMMODITY,
                mode="M1",
                node_loc=node_loc,
            )
            .pipe(broadcast, yv_ya, time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )
        out_frames.append(
            make_df(
                "output",
                technology=tech,
                value=yield_value,
                unit="kt",
                level="crop_yield",
                commodity=crop_yield_commodity(crop),
                mode="M1",
                node_loc=node_loc,
            )
            .pipe(broadcast, yv_ya, time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )

        tl_frames.append(
            make_df(
                "technical_lifetime", technology=tech, value=lifetime, unit="y", node_loc=node_loc
            ).pipe(broadcast, year_vtg=year_vtg_all)
        )
        cons_frames.append(
            make_df(
                "construction_time", technology=tech, value=1, unit="y", node_loc=node_loc
            ).pipe(broadcast, year_vtg=year_vtg_all)
        )
        cf_frames.append(
            make_df(
                "capacity_factor", technology=tech, value=1.0, unit="%", node_loc=node_loc
            ).pipe(broadcast, yv_ya, time=sub_time)
        )

        crop_cap = df_cap[
            (df_cap["crop"] == crop) & (df_cap["value"] > 0) & (df_cap["node"].isin(nodes))
        ]
        if hist_year_vtg is not None and not crop_cap.empty:
            hist_cap_frames.append(
                make_df(
                    "historical_new_capacity",
                    technology=tech,
                    node_loc="B" + crop_cap["node"],
                    year_vtg=hist_year_vtg,
                    value=crop_cap["value"],
                    unit="Mha",
                )
            )

    def _concat(frames):
        return (
            pd.concat(frames).drop_duplicates().reset_index(drop=True)
            if frames
            else pd.DataFrame()
        )

    results = {
        "input": _concat(inp_frames),
        "output": _concat(out_frames),
        "technical_lifetime": _concat(tl_frames),
        "construction_time": _concat(cons_frames),
        "capacity_factor": _concat(cf_frames),
    }
    if hist_cap_frames:
        results["historical_new_capacity"] = _concat(hist_cap_frames)

    return {k: v for k, v in results.items() if not v.empty}
