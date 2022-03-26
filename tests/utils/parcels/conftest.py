import numpy as np
import pytest

from landlab.components import FlowDirectorSteepest, NetworkSedimentTransporter
from landlab.data_record import DataRecord
from landlab.grid.network import NetworkModelGrid
from landlab.utils.parcels import BedParcelInitializer

@pytest.fixture()
def example_nmg():
    y_of_node = (0, 100, 200, 200, 300, 400, 400, 125)
    x_of_node = (0, 0, 100, -50, -100, 50, -150, -100)
    nodes_at_link = ((1, 0), (2, 1), (1, 7), (3, 1), (3, 4), (4, 5), (4, 6))

    grid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)

    # add variables to grid
    grid.add_field(
        "topographic__elevation", [0.0, 0.1, 0.3, 0.2, 0.35, 0.45, 0.5, 0.6], at="node"
    )
    grid.add_field(
        "bedrock__elevation", [0.0, 0.1, 0.3, 0.2, 0.35, 0.45, 0.5, 0.6], at="node"
    )
    grid.add_field(
        "reach_length",
        [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
        at="link",
    )  # m

    grid.add_field("channel_width", 15 * np.ones(grid.size("link")), at="link")

    grid.add_field(
        "flow_depth", 0.01121871 * np.ones(grid.size("link")), at="link"
    )  # Why such an odd, low flow depth? Well, it doesn't matter... and oops.-AP
    
    grid.add_field(
        "channel_slope", 
        [0.05,0.1,0.07,0.1,0.08,0.12,0.1],
        at = "link",
        )    
    
    grid.add_field(
        "drainage_area", 
        [1.7,0.2,1,0.3,0.8,0.2,0.4],
        at = "link",
        ) # km2

    return grid

@pytest.fixture()
def example_flow_director(example_nmg):

    fd = FlowDirectorSteepest(example_nmg)
    fd.run_one_step()
    return fd

@pytest.fixture()
def example_parcel_initializer(example_nmg):
    num_starting_parcels = 2
    parcel_initializer = BedParcelInitializer(example_nmg,
                                           median_number_of_starting_parcels = num_starting_parcels)
    return parcel_initializer

@pytest.fixture
def example_parcels(example_nmg):
    num_starting_parcels = 2
    parcel_initializer = BedParcelInitializer(example_nmg,
                                           median_number_of_starting_parcels = num_starting_parcels)
    parcel_volume = 1
    d50_hydraulic_geometry = [0.18,-0.12]
    parcels = parcel_initializer(discharge_at_link=None,user_parcel_volume=parcel_volume,
                                     user_d50=d50_hydraulic_geometry)    
    return parcels
