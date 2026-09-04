"""Diagnostic plots for a solved MESSAGEix-Nexus scenario.

A Python port of NEST's ``basin_msggdx_diagnostics.r`` (the legacy R+GAMS
``indus_ix`` model's post-solve diagnostics/plotting script), rewritten
against data actually present in this repo's Pakistan/R12 build rather than
the old Indus-model's own level/commodity naming, which does not match --
see ``PORTING_NOTES`` below.

Mirrors the R script's core recipe for every report: pull raw scenario
tables (``ACT``, ``CAP``, ``input``, ``output``, costs), join activity
against per-unit values to get real quantities, classify technology names
into human-readable categories via regex, aggregate, and plot.

PORTING_NOTES
-------------
Three of the original R script's eight reports are deliberately NOT ported
here, because the data they need doesn't exist in this repo (not because
they were hard):

- ``storage.pdf`` needs the ``STORAGE`` GAMS variable, which requires a
  storage/reservoir module this repo doesn't have yet.
- ``water_electricity_flow_maps.pdf`` needs real basin/ocean GIS shapefiles
  (the R script reads them from a ``P:/is-wel/...`` network drive not
  present here) -- the same missing-external-data problem as
  ``storage_capacity.r``'s dam data.
- The R model's "final energy by use" split (``urban_final``/
  ``rural_final``/``industry_final``/``agriculture_final`` levels) has no
  equivalent here: this build's ``urban_mw``/``rural_mw``/``industry_mw``
  commodities are *water* demand (unit ``MCM/year``), not electricity
  end-use categories, so there is nothing structurally equivalent to port.

Everything else here was verified against a real solved scenario
(``MESSAGEix-Pakistan 1`` / ``LTS_nexus_debug`` v7) before being written,
not translated blind from the R model's own (different) naming.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Must be set before pyplot is imported. This module only ever saves PDFs to
# disk -- it never needs a GUI window -- and running with the default
# interactive backend (Tk) in the same process as ixmp's embedded JVM caused
# a hard JVM crash (Tcl_AsyncDelete / topLevelExceptionFilter) at interpreter
# shutdown, confirmed while testing this module against a real scenario.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402

if TYPE_CHECKING:
    from message_ix import Scenario

log = logging.getLogger(__name__)

#: Technology-name regex -> human category, verified against the real
#: technology set of MESSAGEix-Pakistan 1 / LTS_nexus_debug (819 techs).
#: Order matters: more specific patterns are listed before general ones.
TECH_CATEGORIES: dict[str, str] = {
    "irrigation": r"^irr_",
    "crop land": r"^crop_|^rainfed_",
    "network transfer": r"^canal_conv\||^river\||^trs_",
    "desalination": r"^membrane$|^distillation$|^desal_",
    "water extraction": r"^extract_",
    "hydro": r"^hydro_",
    "coal": r"^coal_",
    "gas": r"^gas_|^LNG_",
    "nuclear": r"^nuc_",
    "solar": r"^solar_",
    "wind": r"^wind_",
    "geothermal": r"^geo_",
    "biomass": r"^bio_",
}

#: Technologies that draw electricity to move or treat water -- the
#: "energy for water" nexus link. Verified present in the real scenario.
WATER_ELECTRICITY_TECHS = (
    "extract_surfacewater",
    "extract_groundwater",
    "extract_gw_fossil",
    "extract_salinewater_basin",
    "extract_salinewater_cool",
    "membrane",
    "distillation",
)

#: Hydropower technologies -- the "water for energy" nexus link.
HYDRO_TECHS = ("hydro_hc", "hydro_lc")


def classify_technology(
    tech: pd.Series, categories: dict[str, str] = TECH_CATEGORIES
) -> pd.Series:
    """Map technology names to a human category via regex.

    Unmatched technologies are labelled ``"other"`` rather than dropped, so
    totals computed from the categorized frame still reconcile with the raw
    data.
    """
    result = pd.Series("other", index=tech.index)
    for name, pattern in categories.items():
        mask = tech.str.contains(pattern, regex=True, na=False) & (result == "other")
        result = result.where(~mask, name)
    return result


def _load(scen: "Scenario") -> dict[str, pd.DataFrame]:
    """Pull the raw scenario tables every report below is built from."""
    return {
        "ACT": scen.var("ACT"),
        "CAP": scen.var("CAP"),
        "CAP_NEW": scen.var("CAP_NEW"),
        "input": scen.par("input"),
        "output": scen.par("output"),
        "inv_cost": scen.par("inv_cost"),
        "fix_cost": scen.par("fix_cost"),
        "var_cost": scen.par("var_cost"),
        "PRICE_COMMODITY": scen.var("PRICE_COMMODITY"),
    }


def cost_by_technology(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Investment + fixed + variable cost per technology per year.

    Port of ``basin_msggdx_diagnostics.r``'s "COSTS" section (lines 69-163):
    ``CAP_NEW``/``CAP``/``ACT`` joined against ``inv_cost``/``fix_cost``/
    ``var_cost``, summed per (node, technology, year), classified into a
    human-readable category.
    """
    cap_new = data["CAP_NEW"].rename(columns={"lvl": "cap_new"})
    inv = data["inv_cost"].rename(columns={"value": "inv_unit"})[
        ["node_loc", "technology", "year_vtg", "inv_unit"]
    ]
    ic = cap_new.merge(inv, on=["node_loc", "technology", "year_vtg"], how="left")
    ic["invc"] = ic["cap_new"] * ic["inv_unit"].fillna(0)
    ic = (
        ic.groupby(["node_loc", "technology", "year_vtg"], as_index=False)["invc"]
        .sum()
        .rename(columns={"year_vtg": "year"})
    )

    cap = data["CAP"].rename(columns={"lvl": "cap"})
    fix = data["fix_cost"].rename(columns={"value": "fix_unit"})[
        ["node_loc", "technology", "year_vtg", "year_act", "fix_unit"]
    ]
    fc = cap.merge(
        fix, on=["node_loc", "technology", "year_vtg", "year_act"], how="left"
    )
    fc["fixc"] = fc["cap"] * fc["fix_unit"].fillna(0)
    fc = (
        fc.groupby(["node_loc", "technology", "year_act"], as_index=False)["fixc"]
        .sum()
        .rename(columns={"year_act": "year"})
    )

    act = data["ACT"].rename(columns={"lvl": "act"})
    var = data["var_cost"].rename(columns={"value": "var_unit"})[
        ["node_loc", "technology", "year_vtg", "year_act", "mode", "time", "var_unit"]
    ]
    vc = act.merge(
        var,
        on=["node_loc", "technology", "year_vtg", "year_act", "mode", "time"],
        how="left",
    )
    vc["varc"] = vc["act"] * vc["var_unit"].fillna(0)
    vc = (
        vc.groupby(["node_loc", "technology", "year_act"], as_index=False)["varc"]
        .sum()
        .rename(columns={"year_act": "year"})
    )

    result = ic.merge(fc, on=["node_loc", "technology", "year"], how="outer")
    result = result.merge(vc, on=["node_loc", "technology", "year"], how="outer")
    result[["invc", "fixc", "varc"]] = result[["invc", "fixc", "varc"]].fillna(0)
    result["total_cost"] = result["invc"] + result["fixc"] + result["varc"]
    result["category"] = classify_technology(result["technology"])
    return result[result["total_cost"] > 0].reset_index(drop=True)


def _flow(
    data: dict[str, pd.DataFrame],
    par: str,
    filt: dict,
    value_col: str = "flow",
) -> pd.DataFrame:
    """Join ``ACT`` against a filtered slice of ``input``/``output`` to get
    a real physical quantity (activity x per-unit value)."""
    df = data[par]
    for col, values in filt.items():
        df = df[df[col].isin(values)] if isinstance(values, (list, tuple)) else df[df[col] == values]
    if df.empty:
        return pd.DataFrame(columns=["node_loc", "technology", "year", value_col])

    merged = df.merge(
        data["ACT"],
        on=["node_loc", "technology", "year_vtg", "year_act", "mode", "time"],
        how="inner",
        suffixes=("_rate", "_act"),
    )
    merged[value_col] = merged["value"] * merged["lvl"]
    return (
        merged.groupby(["node_loc", "technology", "year_act"], as_index=False)[
            value_col
        ]
        .sum()
        .rename(columns={"year_act": "year"})
    )


def nexus_interactions(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The three headline nexus numbers, ported from
    ``basin_msggdx_diagnostics.r``'s "Nexus interactions" section
    (lines 246-334), re-targeted at this build's real technology set.

    Returns
    -------
    dict with keys ``energy_for_water``, ``water_for_energy``,
    ``water_for_irrigation`` -- each a long DataFrame of
    (node_loc, technology, year, value).

    Note on ``water_for_energy``: confirmed empty in this build, not a bug.
    ``hydro_hc``/``hydro_lc`` have zero ``input`` rows at all -- hydropower
    here is resource/capacity-constrained, not represented as consuming a
    water commodity, unlike the old R model. Kept in the return value (as an
    empty frame) so callers can see this explicitly rather than the key
    silently disappearing.
    """
    energy_for_water = _flow(
        data,
        "input",
        {"commodity": "electr", "technology": list(WATER_ELECTRICITY_TECHS)},
        value_col="electr_MWa",
    )

    water_for_energy = _flow(
        data,
        "input",
        {
            "commodity": ["freshwater", "freshwater_basin", "surfacewater_basin"],
            "technology": list(HYDRO_TECHS),
        },
        value_col="water_MCM",
    )

    water_for_irrigation = _flow(
        data,
        "input",
        {"commodity": "surfacewater_basin", "level": "water_avail_basin"},
        value_col="water_MCM",
    )
    water_for_irrigation = water_for_irrigation[
        water_for_irrigation["technology"].str.startswith("irr_")
    ]

    return {
        "energy_for_water": energy_for_water,
        "water_for_energy": water_for_energy,
        "water_for_irrigation": water_for_irrigation,
    }


def energy_mix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Installed capacity by source category, per node and year.

    Port of ``basin_msggdx_diagnostics.r``'s "COMMODITIES" / capacity
    section (lines 495-731), simplified to installed capacity (``CAP``)
    classified by source -- the R script's separate "new capacity" and
    generation-by-source variants follow the identical pattern with
    ``CAP_NEW``/``ACT`` in place of ``CAP``.
    """
    cap = data["CAP"].copy()
    cap["category"] = classify_technology(cap["technology"])
    fuel_categories = {"coal", "gas", "nuclear", "hydro", "solar", "wind", "geothermal", "biomass"}
    cap = cap[cap["category"].isin(fuel_categories)]
    return (
        cap.groupby(["node_loc", "category", "year_act"], as_index=False)["lvl"]
        .sum()
        .rename(columns={"year_act": "year", "lvl": "capacity_GW"})
    )


def crop_land(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Land area committed per crop, split rainfed vs. irrigated.

    Port of ``basin_msggdx_diagnostics.r``'s "CROP AREA and WATER" section
    (lines 830-877). Confirmed directly against real data: land-committing
    technologies output commodity ``crop_land`` at level ``area``.
    """
    out = data["output"]
    land = out[(out["commodity"] == "crop_land") & (out["level"] == "area")]
    merged = land.merge(
        data["ACT"],
        on=["node_loc", "technology", "year_vtg", "year_act", "mode", "time"],
        how="inner",
    )
    merged["area_Mha"] = merged["value"] * merged["lvl"]
    merged["method"] = merged["technology"].apply(
        lambda t: "rainfed" if t.startswith("rainfed_") else "irrigated"
    )
    merged["crop"] = merged["technology"].str.replace(
        r"^(rainfed_|irr_[a-z_]+_)", "", regex=True
    )
    return (
        merged.groupby(["node_loc", "crop", "method", "year_act"], as_index=False)[
            "area_Mha"
        ]
        .sum()
        .rename(columns={"year_act": "year"})
    )


def commodity_shadow_price(
    data: dict[str, pd.DataFrame], commodities: tuple[str, ...] = ("freshwater", "crop_land")
) -> pd.DataFrame:
    """Marginal (shadow) price of selected commodities over time.

    Port of ``basin_msggdx_diagnostics.r``'s "commodity_costs.pdf" section
    (lines 1660-1732), which reads the ``PRICE_COMMODITY`` dual-value
    variable -- confirmed present and populated in this build's solution.
    """
    pc = data["PRICE_COMMODITY"]
    pc = pc[pc["commodity"].isin(commodities)]
    return pc.rename(columns={"lvl": "shadow_price"})[
        ["node", "commodity", "level", "year", "time", "shadow_price"]
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _stacked_bar(
    df: pd.DataFrame, x: str, y: str, hue: str, title: str, ylabel: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.empty:
        ax.text(0.5, 0.5, "no data in this build", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    else:
        pivot = df.groupby([x, hue])[y].sum().unstack(fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cost_by_technology(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "cost_by_technology.pdf"
    agg = df.groupby(["year", "category"], as_index=False)["total_cost"].sum()
    _stacked_bar(
        agg, "year", "total_cost", "category",
        "Total cost by technology category", "USD", path,
    )
    return path


def plot_nexus_interactions(flows: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    path = out_dir / "nexus_interactions.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = {
        "energy_for_water": ("Energy for water", "MWa"),
        "water_for_energy": ("Water for energy (hydro)", "MCM"),
        "water_for_irrigation": ("Water for irrigation", "MCM"),
    }
    for ax, (key, (title, unit)) in zip(axes, labels.items()):
        df = flows[key]
        if df.empty:
            ax.text(0.5, 0.5, "no data in this build", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        value_col = [c for c in df.columns if c not in ("node_loc", "technology", "year")][0]
        agg = df.groupby("year", as_index=False)[value_col].sum()
        ax.bar(agg["year"].astype(str), agg[value_col], color="#1f7a8c")
        ax.set_title(title)
        ax.set_ylabel(unit)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_energy_mix(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "energy_mix.pdf"
    _stacked_bar(
        df, "year", "capacity_GW", "category",
        "Installed capacity by source", "GW", path,
    )
    return path


def plot_crop_land(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "crop_land.pdf"
    _stacked_bar(
        df, "year", "area_Mha", "crop",
        "Crop area by crop type", "Mha", path,
    )
    return path


def plot_commodity_shadow_price(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "commodity_shadow_price.pdf"
    fig, ax = plt.subplots(figsize=(8, 5))
    for commodity, sub in df.groupby("commodity"):
        agg = sub.groupby("year", as_index=False)["shadow_price"].mean()
        ax.plot(agg["year"], agg["shadow_price"], marker="o", label=commodity)
    ax.set_title("Average commodity shadow price")
    ax.set_ylabel("USD per unit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_diagnostics(scen: "Scenario", out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Run every ported report against a solved scenario, saving plots to
    ``out_dir`` and returning the underlying data for inspection/testing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading scenario data for diagnostics")
    data = _load(scen)

    results: dict[str, pd.DataFrame] = {}

    results["cost_by_technology"] = cost_by_technology(data)
    plot_cost_by_technology(results["cost_by_technology"], out_dir)

    flows = nexus_interactions(data)
    results.update(flows)
    plot_nexus_interactions(flows, out_dir)

    results["energy_mix"] = energy_mix(data)
    plot_energy_mix(results["energy_mix"], out_dir)

    results["crop_land"] = crop_land(data)
    plot_crop_land(results["crop_land"], out_dir)

    results["commodity_shadow_price"] = commodity_shadow_price(data)
    plot_commodity_shadow_price(results["commodity_shadow_price"], out_dir)

    log.info("Diagnostics written to %s", out_dir)
    return results
