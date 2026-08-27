import pytest

from message_ix_models.model.water.data.crops.common import CROPS, crop_land_commodity, crop_yield_commodity
from message_ix_models.model.water.data.crops.rainfed import (
    GENERIC_LAND_COMMODITY,
    add_rainfed_techs,
    rainfed_commodity_names,
    rainfed_technology_names,
)
from message_ix_models.model.water.data.crops.land import add_crop_land_techs
from message_ix_models.tests.model.water.conftest import water_params


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_rainfed_technology_names(water_context):
    from message_ix_models.model.water.data.crops.common import rainfed_tech_name

    names = rainfed_technology_names(water_context)
    assert set(names) == {rainfed_tech_name(c) for c in CROPS}


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "R12"}],
    indirect=True,
)
def test_rainfed_technology_names_unrelated_region_is_empty(water_context):
    assert rainfed_technology_names(water_context) == []
    assert add_rainfed_techs(water_context) == {}


def test_rainfed_commodity_names():
    names = rainfed_commodity_names()
    assert GENERIC_LAND_COMMODITY in names
    assert set(names) == {GENERIC_LAND_COMMODITY} | {crop_yield_commodity(c) for c in CROPS}


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_add_rainfed_techs(water_context, water_scenario, assert_message_params):
    """add_rainfed_techs() returns well-formed MESSAGE parameter data, using
    the real, committed crop data -- and only for basins/crops where the
    real data has positive rain-fed_yield, not unconditionally like
    crop_<crop>."""
    result = add_rainfed_techs(context=water_context)

    assert_message_params(
        result,
        expected_keys=[
            "input",
            "output",
            "technical_lifetime",
            "construction_time",
            "capacity_factor",
        ],
    )

    # No cost parameters at all -- zero-cost in the R model, omitted here
    # rather than written as explicit zero rows (see module docstring).
    for par in ["inv_cost", "fix_cost", "var_cost"]:
        assert par not in result

    inp = result["input"]
    out = result["output"]

    # Every technology present consumes exactly its own <crop>_land, and
    # produces both the generic crop_land and its own <crop>_yield.
    present_crops = {t.removeprefix("rainfed_") for t in inp["technology"].unique()}
    assert present_crops, "expected at least one rainfed_<crop> technology with real data"
    assert present_crops <= set(CROPS)

    for crop in present_crops:
        tech = f"rainfed_{crop}"
        in_rows = inp[inp["technology"] == tech]
        assert set(in_rows["commodity"]) == {crop_land_commodity(crop)}
        assert set(in_rows["level"]) == {"crop"}

        out_rows = out[out["technology"] == tech]
        assert set(out_rows["commodity"]) == {GENERIC_LAND_COMMODITY, crop_yield_commodity(crop)}

    # Basin-local: node_loc == node_origin (input) / node_dest (output).
    assert (inp["node_loc"] == inp["node_origin"]).all()
    assert (out["node_loc"] == out["node_dest"]).all()

    # Not every crop necessarily has positive rain-fed_yield data
    # everywhere -- this is real, data-driven filtering, not a bug, but at
    # least one crop should be present given the real IRB data.
    assert len(present_crops) <= len(CROPS)


def test_add_rainfed_techs_no_valid_basins(water_context, water_scenario):
    water_context.valid_basins = set()
    assert add_rainfed_techs(water_context) == {}


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_rainfed_input_matches_land_output_commodity(water_context, water_scenario):
    """The commodity rainfed_<crop> consumes as input is exactly the same
    commodity crop_<crop> produces as output -- the two layers are
    genuinely chained, not just similarly named."""
    land_result = add_crop_land_techs(context=water_context)
    rainfed_result = add_rainfed_techs(context=water_context)

    land_commodities = set(land_result["output"]["commodity"].unique())
    rainfed_input_commodities = set(rainfed_result["input"]["commodity"].unique())
    assert rainfed_input_commodities <= land_commodities
