import pandas as pd
import pytest

from message_ix_models.model.water.data.network import (
    add_network_techs,
    network_technology_names,
    read_basin_links,
)
from message_ix_models.tests.model.water.conftest import water_params


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_read_basin_links(water_context):
    """The real basin-links file loads, and every row's basins are valid."""
    df = read_basin_links(water_context)
    assert not df.empty
    assert set(df["link_type"]) <= {
        "river",
        "canal_conv",
        "canal_lined",
        "interbasin_canal",
        "transmission",
    }
    # Every from_basin (and non-blank to_basin) is a real IRB basin.
    assert df["from_basin"].isin(water_context.valid_basins).all()
    dest = df["to_basin"].dropna()
    assert dest.isin(water_context.valid_basins).all()


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "R12"}],
    indirect=True,
)
def test_read_basin_links_missing_region_is_empty(water_context):
    """Regions without a basin_links_<regions>.csv get zero rows, not an error."""
    df = read_basin_links(water_context)
    assert df.empty
    assert network_technology_names(water_context) == []


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_network_technology_names_include_mirrored_bidirectional(water_context):
    """Every bidirectional=true row's reverse-direction technology is present.

    Doesn't hardcode a specific basin pair -- derives the check from
    whatever bidirectional rows actually exist in the live data, so this
    keeps working as basin_links_IRB.csv is replaced with newer real data.
    """
    df = read_basin_links(water_context)
    bidi = df[df["bidirectional"]]
    assert not bidi.empty, "expected at least one bidirectional row in the live data"

    names = set(network_technology_names(water_context))

    def _present(base_name):
        # Tolerate a "#N" disambiguation suffix (see
        # _dedupe_technology_names) -- some real basin pairs have more than
        # one distinct link of the same type.
        return any(n == base_name or n.startswith(base_name + "#") for n in names)

    for row in bidi.to_dict("records"):
        forward = f"{row['link_type']}|{row['from_basin']}|{row['to_basin']}"
        reverse = f"{row['link_type']}|{row['to_basin']}|{row['from_basin']}"
        assert _present(forward)
        assert _present(reverse)


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_add_network_techs(
    water_context, water_scenario, assert_message_params, assert_input_output_structure
):
    """add_network_techs() returns well-formed MESSAGE parameter data."""
    result = add_network_techs(context=water_context)

    assert_message_params(result, expected_keys=["input", "output"])
    assert_input_output_structure(result)

    # Sink (interbasin_canal) technologies, if the live data has any, have
    # an input but must never appear in output -- the water leaves the
    # modeled system. Not asserted unconditionally: which link_types are
    # present depends on the live data file. See
    # test_sink_technology_has_no_output for guaranteed, data-independent
    # coverage of this behavior.
    sink_tech = [
        t for t in result["input"]["technology"].unique() if t.startswith("interbasin_canal|")
    ]
    assert not any(
        t in result["output"]["technology"].unique() for t in sink_tech
    )

    # Every non-sink technology's output node_dest differs from its
    # node_loc -- this is the entire point of the module (every other
    # water technology in this codebase is basin-local, node_loc == node_dest).
    out = result["output"]
    assert (out["node_loc"] != out["node_dest"]).all()

    # Canal technologies (which pump) carry a second, electricity input row
    # sourced cross-node from the parent region, distinct from the primary
    # same-node water input row.
    canal_rows = result["input"][
        result["input"]["technology"].str.startswith(("canal_conv|", "canal_lined|"))
    ]
    assert set(canal_rows["commodity"]) == {"surfacewater_basin", "electr"}
    elec_rows = canal_rows[canal_rows["commodity"] == "electr"]
    assert (elec_rows["node_origin"] != elec_rows["node_loc"]).all()


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "R12"}],
    indirect=True,
)
def test_add_network_techs_missing_region_returns_empty_dict(
    water_context, water_scenario
):
    assert add_network_techs(context=water_context) == {}


def test_add_network_techs_solves(test_context, tmp_path, monkeypatch):
    """End-to-end: the generated data actually solves under GAMS, and the
    solved activity matches what the conveyance-loss math predicts.

    This is what caught a real bug (node_loc passed to broadcast(**kwargs)
    instead of make_df(), silently exploding "BX|TEST" into one row per
    character) that the DataFrame-shape checks in the other tests above did
    not: those only inspect columns/dtypes, never round-trip through a real
    ixmp backend or GAMS. A canal moves water from a basin with 100 MCM/year
    of availability to a basin needing 40 MCM/year, through a technology
    with 0.75 conveyance efficiency (25% loss) -- the only way to meet that
    demand is ACT == 40 / 0.75 == 53.333...
    """
    import pandas as pd
    from message_ix import Scenario

    from message_ix_models import ScenarioInfo
    from message_ix_models.model.water.data import network as network_mod

    regions = "NETTEST"
    basin_x, basin_y = f"1|{regions}", f"2|{regions}"
    node_x, node_y, node_region = f"B{basin_x}", f"B{basin_y}", f"R_{regions}"
    year = 2025

    pd.DataFrame(
        {
            "NAME": ["Upstream", "Downstream"],
            "BASIN": ["Upstream", "Downstream"],
            "REGION": [f"R_{regions}", f"R_{regions}"],
            "BASIN_ID": [1, 2],
            "area": [1.0, 1.0],
            "area_km2": [100.0, 100.0],
            "BCU_name": [basin_x, basin_y],
        }
    ).to_csv(tmp_path / f"basins_by_region_simpl_{regions}.csv", index=False)

    pd.DataFrame(
        [
            {
                "from_basin": basin_x,
                "to_basin": basin_y,
                "link_type": "canal_conv",
                "name": "pytest solve check",
                "distance_km": 55,
                "existing_capacity": "",
                "capacity_unit": "MCM/year",
                "bidirectional": "false",
                "source": "pytest fixture",
            }
        ]
    ).to_csv(tmp_path / f"basin_links_{regions}.csv", index=False)

    real_package_data_path = network_mod.package_data_path

    def fake_package_data_path(*parts):
        if parts in (
            ("water", "network", f"basin_links_{regions}.csv"),
            ("water", "delineation", f"basins_by_region_simpl_{regions}.csv"),
        ):
            return tmp_path / parts[-1]
        return real_package_data_path(*parts)

    monkeypatch.setattr(network_mod, "package_data_path", fake_package_data_path)
    # basin_region_map() (called from add_network_techs()) lives in utils.py
    # and holds its own separately-imported package_data_path reference, so
    # the delineation-file redirect above needs patching there too.
    from message_ix_models.model.water import utils as water_utils_mod

    monkeypatch.setattr(water_utils_mod, "package_data_path", fake_package_data_path)

    mp = test_context.get_platform()
    scen = Scenario(
        mp, model="network-solve-test", scenario="canal_conv", version="new"
    )
    scen.add_horizon(year=[2010, 2015, 2020, year], firstmodelyear=year)
    scen.add_set("node", ["World", node_region, node_x, node_y])
    scen.add_set("commodity", ["surfacewater_basin", "electr"])
    scen.add_set("level", ["water_avail_basin", "final"])
    scen.add_set("mode", ["M1"])
    scen.add_par("interestrate", [year], value=0.05, unit="-")

    for unit in ["MCM", "GWa", "GWa/MCM", "-", "y", "%", "USD/MCM", "USD/MCM/y", "USD/kW"]:
        if unit not in mp.units():
            mp.add_unit(unit)

    test_context.regions = regions
    test_context.time = "year"
    test_context.type_reg = "global"
    test_context.valid_basins = {basin_x, basin_y}
    test_context.set_scenario(scen)
    test_context["water build info"] = ScenarioInfo(scen)

    tech_names = network_mod.network_technology_names(test_context)
    scen.add_set("technology", tech_names)

    scen.add_par(
        "demand",
        pd.DataFrame(
            {
                "node": [node_region],
                "commodity": ["electr"],
                "level": ["final"],
                "year": [year],
                "time": ["year"],
                "value": [-1.0e6],
                "unit": ["GWa"],
            }
        ),
    )
    scen.add_par(
        "demand",
        pd.DataFrame(
            {
                "node": [node_x, node_y],
                "commodity": ["surfacewater_basin", "surfacewater_basin"],
                "level": ["water_avail_basin", "water_avail_basin"],
                "year": [year, year],
                "time": ["year", "year"],
                "value": [-100.0, 40.0],
                "unit": ["MCM", "MCM"],
            }
        ),
    )

    # add_network_techs() calls context.get_scenario(), which reconstructs a
    # *fresh* message_ix.Scenario(platform, model, scenario, version) handle
    # on every call. Once `scen` is checked out for editing, the JDBC
    # backend refuses a second live handle on the same identity ("currently
    # locked by user ..."), even from the same platform connection -- so
    # reconstructing one here would collide with the very `scen` we're
    # editing. Point get_scenario() at the object we already hold instead
    # of round-tripping through the backend by identity. Context overrides
    # __setattr__ (dict-backed config storage), so patch the class, not
    # the instance, or the assignment silently goes nowhere.
    monkeypatch.setattr(type(test_context), "get_scenario", lambda self: scen)

    result = network_mod.add_network_techs(test_context)
    for par_name, df in result.items():
        scen.add_par(par_name, df)

    scen.commit("pytest network solve check: network technology data")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})

    act = scen.var("ACT", filters={"technology": tech_names})
    assert len(act) == 1
    assert act["lvl"].iloc[0] == pytest.approx(40.0 / 0.75, abs=1e-4)


def _write_synthetic_links(tmp_path, monkeypatch, regions, rows):
    """Write a small synthetic basin_links_<regions>.csv, and a matching
    basin delineation file, redirecting network.package_data_path() so
    read_basin_links()/network_technology_names() see them. Used for
    structural tests that must not depend on whatever real data happens to
    be in the live basin_links_IRB.csv (sink handling, disambiguation) --
    those behaviors need guaranteed coverage independent of live content.
    """
    from message_ix_models.model.water.data import network as network_mod

    basins = sorted({b for r in rows for b in (r["from_basin"], r["to_basin"]) if b})
    pd.DataFrame(
        {
            "NAME": basins,
            "BASIN": basins,
            "REGION": [f"R_{regions}"] * len(basins),
            "BASIN_ID": range(1, len(basins) + 1),
            "area": [1.0] * len(basins),
            "area_km2": [100.0] * len(basins),
            "BCU_name": basins,
        }
    ).to_csv(tmp_path / f"basins_by_region_simpl_{regions}.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / f"basin_links_{regions}.csv", index=False)

    real_package_data_path = network_mod.package_data_path

    def fake_package_data_path(*parts):
        if parts in (
            ("water", "network", f"basin_links_{regions}.csv"),
            ("water", "delineation", f"basins_by_region_simpl_{regions}.csv"),
        ):
            return tmp_path / parts[-1]
        return real_package_data_path(*parts)

    monkeypatch.setattr(network_mod, "package_data_path", fake_package_data_path)
    return network_mod


def test_sink_technology_has_no_output(tmp_path, monkeypatch, test_context):
    """A blank to_basin (a sink, e.g. real out-of-system transfers) never
    gets mirrored even if marked bidirectional, and add_network_techs()
    gives it an input but no output. Synthetic data, independent of
    whatever basin_links_IRB.csv currently contains.
    """
    regions = "SINKTEST"
    network_mod = _write_synthetic_links(
        tmp_path,
        monkeypatch,
        regions,
        [
            {
                "from_basin": "1|SINKTEST", "to_basin": "", "link_type": "interbasin_canal",
                "name": "synthetic sink", "distance_km": 10, "existing_capacity": "",
                "capacity_unit": "MCM/year", "bidirectional": "true",  # must be ignored for a sink
                "source": "pytest synthetic",
            }
        ],
    )

    test_context.regions = regions
    test_context.time = "year"
    test_context.type_reg = "global"
    test_context.valid_basins = {"1|SINKTEST"}

    names = network_mod.network_technology_names(test_context)
    assert names == ["interbasin_canal|1|SINKTEST|SINK"]


def test_duplicate_link_disambiguation(tmp_path, monkeypatch, test_context):
    """Two distinct real links of the same type between the same basin pair
    (seen in the actual Indus canal data -- three separate real canals
    connect the same two basins) get distinct, stable technology names
    instead of silently colliding onto one.
    """
    regions = "DUPTEST"
    row = {
        "from_basin": "1|DUPTEST", "to_basin": "2|DUPTEST", "link_type": "canal_conv",
        "distance_km": 10, "existing_capacity": "", "capacity_unit": "MCM/year",
        "bidirectional": "false", "source": "pytest synthetic",
    }
    network_mod = _write_synthetic_links(
        tmp_path,
        monkeypatch,
        regions,
        [
            {**row, "name": "canal A"},
            {**row, "name": "canal B"},
            {**row, "name": "canal C"},
        ],
    )

    test_context.regions = regions
    test_context.time = "year"
    test_context.type_reg = "global"
    test_context.valid_basins = {"1|DUPTEST", "2|DUPTEST"}

    names = network_mod.network_technology_names(test_context)
    assert sorted(names) == [
        "canal_conv|1|DUPTEST|2|DUPTEST",
        "canal_conv|1|DUPTEST|2|DUPTEST#2",
        "canal_conv|1|DUPTEST|2|DUPTEST#3",
    ]
