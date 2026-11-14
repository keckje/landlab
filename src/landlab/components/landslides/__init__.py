# from .landslide_probability import LandslideProbability
from .landslide_probability_recharge import LandslideProbabilityRecharge as LandslideProbability
from .landslide_probability_saturated_thickness import LandslideProbabilitySaturatedThickness

__all__ = ["LandslideProbability",
           # "LandslideProbabilityRecharge",
           "LandslideProbabilitySaturatedThickness"]
