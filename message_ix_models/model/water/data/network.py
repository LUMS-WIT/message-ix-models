"""Prepare data for inter-basin network technologies.

This module generates MESSAGEix technologies for physical links *between*
basin nodes -- inter-basin water transfer (river routing, engineered canals)
and basin-to-basin electricity transmission -- which the rest of the water
module does not represent (every other technology in :mod:`infrastructure`,
:mod:`water_supply`, :mod:`water_for_ppl` and :mod:`irrigation` is
basin-local: ``node_loc == node_dest``, enforced via
:func:`~message_ix_models.util.same_node`).

Data model
----------
One CSV, ``data/water/network/basin_links_<regions>.csv``, holds one row per
directed edge between two basin nodes::

    from_basin, to_basin, link_type, name, distance_km, existing_capacity,
    capacity_unit, bidirectional, source

``link_type`` selects a technology archetype -- commodity/level, conveyance
or transmission efficiency, and cost structure -- from
``data/water/network/config.yaml``. One MESSAGEix technology is generated
per row, named ``"{link_type}|{from_basin}|{to_basin}"`` (plus a mirrored
reverse-direction technology when ``bidirectional`` is true), following the
same ``<prefix>|<from>|<to>`` convention as the legacy R+GAMS indus_ix
model's ``river|``/``conv_canal|``/``trs_|`` technologies (see
``basin_msggdx_technologies.r`` in the sibling ``NEST/MESSAGEix`` repo).

A row with a blank ``to_basin`` is a sink: water leaves the modeled system
entirely (the Python analogue of indus_ix's ``interbasin_canal``) -- the
generated technology has an input but no output.

This intentionally does *not* reuse the hub-and-spoke pattern in
:mod:`message_ix_models.tools.inter_pipe` (export -> global commodity hub ->
import): that topology is for trading a fungible commodity through a market,
which is physically wrong for water -- it must move from one specific basin
to another, only where a real conveyance/transmission link exists.

Because technology names are derived from CSV rows rather than statically
enumerated in ``data/water/technology.yaml`` (as other nexus technologies
are), :func:`network_technology_names` is called from
:func:`message_ix_models.model.water.build.get_spec` to register them in the
scenario's ``technology`` set directly.

Data availability
------------------
Real basin-to-basin adjacency (from GIS: a basin shapefile with a downstream
"DOWN" field, OSM transmission-line data) is not yet available in this
repository -- see the indus_ix gap analysis. ``basin_links_IRB.csv`` is a
small, clearly-marked placeholder seeded with illustrative (not
GIS-verified) edges, sized to exercise every ``link_type`` end-to-end. It is
meant to be replaced by a real adjacency-derived CSV with the same schema;
no code here needs to change for that swap.
"""

import logging

import pandas as pd
import yaml
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model.water.utils import (
    basin_region_map,
    get_vintage_and_active_years,
)
from message_ix_models.util import (
    broadcast,
    package_data_path,
    same_node,
    same_time,
)

log = logging.getLogger(__name__)

# Load configuration (link-type archetypes, unit conversions, toggles) once
# at import time -- same convention as water_for_ppl.CONFIG.
_CONFIG_PATH = package_data_path("water", "network", "config.yaml")
with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

#: Columns expected in basin_links_<regions>.csv; used to build an empty
#: frame with the right shape when no file exists for a region set.
_LINK_COLUMNS = [
    "from_basin",
    "to_basin",
    "link_type",
    "name",
    "distance_km",
    "existing_capacity",
    "capacity_unit",
    "bidirectional",
    "source",
]

def _tech_name(link_type: str, from_basin: str, to_basin: str) -> str:
    """Build a technology name for one directed network link."""
    return f"{link_type}|{from_basin}|{to_basin}"


def _dedupe_technology_names(names: list[str]) -> list[str]:
    """Disambiguate duplicate technology names with a stable ``#2``, ``#3``, ...
    suffix on every occurrence after the first.

    A basin pair can have more than one real, physically distinct link of
    the same type -- e.g. three separate real canals connecting the same
    two basins (seen in the actual Indus canal data: Qadirabad-Bulloki,
    Marala-Ravi, and Chenab-Bambanwala all connect the same pair). The base
    ``"{link_type}|{from}|{to}"`` name alone can't tell them apart, so
    without this, two distinct real canals would silently collapse onto one
    technology and one of their capacities would be lost.
    """
    seen: dict[str, int] = {}
    result = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}#{seen[name]}")
    return result


def _mirror_technology_name(
    original_name: str, link_type: str, new_from: str, new_to: str
) -> str:
    """Rebuild a technology name for the reverse direction of a bidirectional
    link, preserving any ``#N`` disambiguation suffix from the original."""
    suffix = ""
    if "#" in original_name:
        suffix = "#" + original_name.rsplit("#", 1)[1]
    return _tech_name(link_type, new_from, new_to) + suffix


def _is_sink(to_basin) -> bool:
    """True if `to_basin` denotes a link with no destination (water leaves
    the modeled system, e.g. indus_ix's interbasin_canal)."""
    return not isinstance(to_basin, str) or to_basin.strip() == ""


def _is_true(value) -> bool:
    return str(value).strip().lower() == "true"


def read_basin_links(context: "Context") -> pd.DataFrame:
    """Read and validate the basin-to-basin link table for ``context.regions``.

    Returns an empty frame (with a logged warning) if no link file exists
    for the current region set -- callers for regions without a curated
    network (R11, R12, ZMB, ...) get zero rows rather than an error, so
    this module is a safe no-op everywhere except where the data exists.
    """
    path = package_data_path("water", "network", f"basin_links_{context.regions}.csv")
    if not path.exists():
        log.warning(
            "No basin_links_%s.csv found; network module adds no inter-basin "
            "technologies for this region set.",
            context.regions,
        )
        return pd.DataFrame(columns=_LINK_COLUMNS)

    df = pd.read_csv(path, dtype={"from_basin": str, "to_basin": str})

    valid_basins = getattr(context, "valid_basins", None)
    if valid_basins:
        before = len(df)
        df = df[
            df["from_basin"].isin(valid_basins)
            & (df["to_basin"].isin(valid_basins) | df["to_basin"].isna())
        ]
        dropped = before - len(df)
        if dropped:
            log.info(
                "Dropped %d network link row(s) referencing basins excluded "
                "by basin filtering.",
                dropped,
            )

    link_types = CONFIG["link_types"]
    enabled = CONFIG.get("enabled_link_types", {})

    unknown = sorted(set(df["link_type"]) - set(link_types))
    if unknown:
        log.warning(
            "Dropping network link row(s) with link_type %s not defined in "
            "network/config.yaml.",
            unknown,
        )
        df = df[df["link_type"].isin(link_types)]

    df = df[df["link_type"].map(lambda lt: enabled.get(lt, True))]

    if df.empty:
        return df

    df = df.copy()
    df["distance_km"] = df["distance_km"].fillna(CONFIG["default_distance_km"])
    df["bidirectional"] = df["bidirectional"].map(_is_true)
    base_names = [
        _tech_name(lt, fb, "SINK" if _is_sink(tb) else tb)
        for lt, fb, tb in zip(df["link_type"], df["from_basin"], df["to_basin"], strict=True)
    ]
    df["technology"] = _dedupe_technology_names(base_names)
    return df.reset_index(drop=True)


def _expand_bidirectional(df_links: pd.DataFrame) -> pd.DataFrame:
    """Mirror each bidirectional row into a second, reverse-direction row.

    Sinks (blank to_basin) are never mirrored -- a link with no destination
    cannot run in reverse.
    """
    rows = list(df_links.to_dict("records"))
    for row in df_links.to_dict("records"):
        if row["bidirectional"] and not _is_sink(row["to_basin"]):
            mirrored = dict(row)
            mirrored["from_basin"], mirrored["to_basin"] = (
                row["to_basin"],
                row["from_basin"],
            )
            mirrored["technology"] = _mirror_technology_name(
                row["technology"],
                row["link_type"],
                mirrored["from_basin"],
                mirrored["to_basin"],
            )
            rows.append(mirrored)
    return pd.DataFrame(rows).reset_index(drop=True)


def network_technology_names(context: "Context") -> list[str]:
    """Return every technology name this module will generate for ``context``.

    Called from :func:`message_ix_models.model.water.build.get_spec` to
    register these (data-driven, not statically enumerable) names in the
    scenario's ``technology`` set ahead of :func:`add_network_techs` adding
    parameter data that references them.
    """
    df_links = read_basin_links(context)
    if df_links.empty:
        return []
    return sorted(set(_expand_bidirectional(df_links)["technology"]))


def _elec_cost_unit(scen, par_name: str, fallback: str) -> str:
    """Borrow the ``unit`` string already used for `par_name` on ``elec_t_d``
    in the base scenario, so transmission technology costs are recorded in
    whatever convention this specific model already uses -- rather than
    this module guessing (mirrors the ``get_template()`` pattern in
    :mod:`message_ix_models.tools.inter_pipe`).
    """
    try:
        existing = scen.par(par_name, filters={"technology": "elec_t_d"})
        if not existing.empty:
            return str(existing["unit"].iloc[0])
    except Exception:  # pragma: no cover - defensive; falls through below
        pass
    return fallback


def add_network_techs(context: "Context") -> dict[str, pd.DataFrame]:
    """Build inter-basin water-transfer and electricity-transmission techs.

    One technology per directed link (see module docstring); returns an
    empty dict if no ``basin_links_<regions>.csv`` exists for
    ``context.regions``.

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'output'.
    """
    df_links = read_basin_links(context)
    if df_links.empty:
        return {}

    directed = _expand_bidirectional(df_links)

    info = context["water build info"]
    scen = context.get_scenario()
    scenario_info = ScenarioInfo(scen)
    sub_time = pd.Series(context.time)
    year_wat = (*range(2010, info.Y[0] + 1, 5), *info.Y)
    firstyear = scen.firstmodelyear

    df_region = basin_region_map(context)
    inv_unit = _elec_cost_unit(scen, "inv_cost", fallback="USD/kW")
    fix_unit = _elec_cost_unit(scen, "fix_cost", fallback="USD/kW")

    inp_frames, out_frames = [], []
    tl_frames, cons_frames, cf_frames = [], [], []
    inv_frames, fix_frames, var_frames = [], [], []
    hist_cap_frames = []

    for row in directed.to_dict("records"):
        spec = CONFIG["link_types"][row["link_type"]]
        tech = row["technology"]
        from_node = "B" + row["from_basin"]
        sink = _is_sink(row["to_basin"])
        to_node = None if sink else "B" + row["to_basin"]
        is_elec = spec["commodity"] == "electr"

        lifetime = spec["technical_lifetime_years"]
        yv_ya = get_vintage_and_active_years(scenario_info, lifetime)
        act_unit = "GWa" if is_elec else "MCM"

        # --- input: draws the transferred commodity from the origin basin
        inp_frames.append(
            make_df(
                "input",
                technology=tech,
                value=1.0,
                unit=act_unit,
                level=spec["level"],
                commodity=spec["commodity"],
                mode="M1",
                node_loc=from_node,
            )
            .pipe(broadcast, yv_ya, time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )

        # --- optional pumping-electricity input for engineered canals,
        # sourced cross-node from the origin basin's parent region (the
        # same pattern every other water technology in this module uses
        # for electricity, e.g. infrastructure.prepare_input_dataframe).
        if spec.get("electricity_input"):
            region = df_region.loc[row["from_basin"], "region"]
            inp_frames.append(
                make_df(
                    "input",
                    technology=tech,
                    value=spec["electricity_intensity_gwa_per_mcm"],
                    unit="GWa/MCM",
                    level="final",
                    commodity="electr",
                    mode="M1",
                    node_loc=from_node,
                    node_origin=region,
                    time_origin="year",
                )
                .pipe(broadcast, yv_ya, time=sub_time)
            )

        # --- output: delivers the loss-adjusted commodity to the
        # destination basin. Sinks (interbasin_canal) have no output --
        # the water leaves the modeled system, matching the R model.
        # Transmission efficiency is distance-scaled (R model: ~0.006%/km);
        # canal/river efficiency is flat, per link_type, as in the R model.
        if not sink:
            efficiency = spec["conveyance_efficiency"]
            if "transmission_loss_per_km" in spec:
                efficiency = max(
                    0.0, 1.0 - spec["transmission_loss_per_km"] * row["distance_km"]
                )
            out_frames.append(
                make_df(
                    "output",
                    technology=tech,
                    value=efficiency,
                    unit=act_unit,
                    level=spec["level"],
                    commodity=spec["commodity"],
                    mode="M1",
                    node_loc=from_node,
                    node_dest=to_node,
                )
                .pipe(broadcast, yv_ya, time=sub_time)
                .pipe(same_time)
            )

        # --- technical lifetime / construction time
        #
        # NOTE: node_loc is a plain Python str here (one basin per loop
        # iteration, not a vectorized Series like the rest of this water
        # module uses). It MUST be passed to make_df(), never to
        # broadcast(**kwargs) -- broadcast() treats each kwarg as an
        # iterable of labels to take the cartesian product over, and a
        # bare str is itself iterable character-by-character, silently
        # exploding "BX|TEST" into seven one-character "node_loc" rows
        # ("B", "X", "|", "T", "E", "S", "T"). Caught by a real
        # scen.add_par() + solve round trip (see validate_network_solve.py)
        # -- the DataFrame-shape unit tests alone did not catch this.
        tl_frames.append(
            make_df(
                "technical_lifetime",
                technology=tech,
                value=lifetime,
                unit="y",
                node_loc=from_node,
            ).pipe(broadcast, year_vtg=year_wat)
        )
        cons_frames.append(
            make_df(
                "construction_time",
                technology=tech,
                value=spec["construction_time_years"],
                unit="y",
                node_loc=from_node,
            ).pipe(broadcast, year_vtg=year_wat)
        )

        # --- capacity factor
        cf_frames.append(
            make_df(
                "capacity_factor",
                technology=tech,
                value=CONFIG["capacity_factor"],
                unit="%",
                node_loc=from_node,
            )
            .pipe(broadcast, yv_ya, time=sub_time)
        )

        # --- costs, distance-scaled where a per-km rate is defined
        distance = row["distance_km"]
        if is_elec:
            inv_value = spec["investment_usd_per_mw_km"] * distance
            fix_value = spec["fix_cost_usd_per_mw"]
            var_value = spec.get("var_cost_usd_per_mw", 0.0)
        else:
            inv_value = spec["investment_usd_per_mcm_km"] * distance
            fix_value = spec["fix_cost_usd_per_mcm"]
            var_value = spec.get("var_cost_usd_per_mcm", 0.0)

        inv_frames.append(
            make_df(
                "inv_cost",
                technology=tech,
                value=inv_value,
                unit=inv_unit if is_elec else "USD/MCM",
                node_loc=from_node,
            ).pipe(broadcast, year_vtg=year_wat)
        )
        fix_frames.append(
            make_df(
                "fix_cost",
                technology=tech,
                value=fix_value,
                unit=fix_unit if is_elec else "USD/MCM",
                node_loc=from_node,
            ).pipe(broadcast, yv_ya)
        )
        var_frames.append(
            make_df(
                "var_cost",
                technology=tech,
                value=var_value,
                # Same unit as fix_cost, not a distinct ".../y" -- matches
                # the convention used throughout infrastructure.py (e.g.
                # var_cost and fix_cost both plainly "USD/MCM"), rather
                # than inventing a unit string absent from the platform.
                unit=fix_unit if is_elec else "USD/MCM",
                mode="M1",
                node_loc=from_node,
            ).pipe(broadcast, yv_ya, time=sub_time)
        )

        # --- historical capacity, if the placeholder/real data supplies one
        if pd.notna(row.get("existing_capacity")):
            cap_value = row["existing_capacity"]
            if is_elec:
                cap_value = cap_value / 1000.0  # MW -> GW
                cap_unit = "GW"
            else:
                cap_unit = "MCM/year"
            hist_years = [y for y in year_wat if y < firstyear]
            hist_cap_frames.append(
                make_df(
                    "historical_new_capacity",
                    technology=tech,
                    node_loc=from_node,
                    year_vtg=max(hist_years) if hist_years else year_wat[0],
                    value=cap_value,
                    unit=cap_unit,
                )
            )

    def _concat(frames):
        return pd.concat(frames).drop_duplicates().reset_index(drop=True) if frames else pd.DataFrame()

    results = {
        "input": _concat(inp_frames),
        "output": _concat(out_frames),
        "technical_lifetime": _concat(tl_frames),
        "construction_time": _concat(cons_frames),
        "capacity_factor": _concat(cf_frames),
        "inv_cost": _concat(inv_frames),
        "fix_cost": _concat(fix_frames),
        "var_cost": _concat(var_frames),
    }
    if hist_cap_frames:
        results["historical_new_capacity"] = _concat(hist_cap_frames)

    return {k: v for k, v in results.items() if not v.empty}
