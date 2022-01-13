import pytest

import numpy as np
from landlab import RasterModelGrid
from landlab.components import LandslideProbability


@pytest.fixture
def ls_prob():
    grid = RasterModelGrid((20, 20), xy_spacing=10e0)
    grid.add_zeros("topographic__slope", at="node", dtype=float)
    grid.add_zeros("topographic__specific_contributing_area", at="node")
    grid.add_zeros("soil__transmissivity", at="node")
    grid.add_zeros("soil__saturated_hydraulic_conductivity", at="node")
    grid.add_zeros("soil__mode_total_cohesion", at="node")
    grid.add_zeros("soil__minimum_total_cohesion", at="node")
    grid.add_zeros("soil__maximum_total_cohesion", at="node")
    grid.add_zeros("soil__internal_friction_angle", at="node")
    grid.add_zeros("soil__density", at="node")
    grid.add_zeros("soil__thickness", at="node")

    return LandslideProbability(grid)


@pytest.fixture
def example_raster_model_grid():
    grid_3 = RasterModelGrid((5, 4), xy_spacing=(0.2, 0.2))
    gridnum = grid_3.number_of_nodes
    np.random.seed(seed=7)
    grid_3.add_zeros("soil__saturated_hydraulic_conductivity", at="node")
    grid_3.at_node["topographic__slope"] = np.random.rand(gridnum)
    scatter_dat = np.random.randint(1, 10, gridnum).astype(float)
    grid_3.at_node["topographic__specific_contributing_area"] = np.sort(
        np.random.randint(30, 900, gridnum).astype(float)
    )
    grid_3.at_node["soil__transmissivity"] = np.sort(
        np.random.randint(5, 20, gridnum).astype(float), -1
    )
    grid_3.at_node["soil__mode_total_cohesion"] = np.sort(
        np.random.randint(30, 900, gridnum).astype(float)
    )
    grid_3.at_node["soil__minimum_total_cohesion"] = (
        grid_3.at_node["soil__mode_total_cohesion"] - scatter_dat
    )
    grid_3.at_node["soil__maximum_total_cohesion"] = (
        grid_3.at_node["soil__mode_total_cohesion"] + scatter_dat
    )
    grid_3.at_node["soil__internal_friction_angle"] = np.sort(
        np.random.randint(26, 37, gridnum).astype(float)
    )
    grid_3.at_node["soil__thickness"] = np.sort(
        np.random.randint(1, 10, gridnum).astype(float)
    )
    grid_3.at_node["soil__density"] = 2000.0 * np.ones(gridnum)
    
    return (grid_3)