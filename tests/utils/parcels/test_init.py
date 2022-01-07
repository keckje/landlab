import pytest
import numpy as np

from landlab.utils.parcels import BedParcelInitializer
from landlab import RasterModelGrid

def test_basic_init(example_nmg):
    num_starting_parcels = 2
    _ = BedParcelInitializer(example_nmg,
                             median_number_of_starting_parcels = num_starting_parcels)

def test_grid_is_nmg():
    """test ValueError exception is raised when nmg is a mg"""
    nmg = RasterModelGrid((5, 5))
    num_starting_parcels = 2
    with pytest.raises(ValueError):
        initialize_parcels = BedParcelInitializer(nmg,
                                                  median_number_of_starting_parcels = num_starting_parcels)

# add other exceptions to bed_parcel_initializer, tests go below