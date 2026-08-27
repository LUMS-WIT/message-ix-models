"""Crop / rainfed / irrigation technology port of the legacy R+GAMS
indus_ix model's crop submodel. See :mod:`.land`, :mod:`.rainfed`,
:mod:`.irrigation_tech` for the three technology layers, and :mod:`.common`
for shared configuration/data readers.
"""

from .irrigation_tech import (
    add_irrigation_techs,
    irrigation_commodity_names,
    irrigation_relation_names,
    irrigation_technology_names,
)
from .land import add_crop_land_techs, crop_commodity_names, crop_technology_names
from .rainfed import add_rainfed_techs, rainfed_commodity_names, rainfed_technology_names

__all__ = [
    "add_crop_land_techs",
    "add_irrigation_techs",
    "add_rainfed_techs",
    "crop_commodity_names",
    "crop_technology_names",
    "irrigation_commodity_names",
    "irrigation_relation_names",
    "irrigation_technology_names",
    "rainfed_commodity_names",
    "rainfed_technology_names",
]
