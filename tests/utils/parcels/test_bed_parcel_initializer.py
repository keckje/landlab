# -*- coding: utf-8 -*-
"""
Unit tests for bed_parcel_initializer.py
"""

import pytest
import numpy as np

from landlab import RasterModelGrid
from landlab.utils.parcels import BedParcelInitializer
from landlab.utils.parcels import (
                                    parcel_characteristics,
                                    determine_approx_parcel_volume,
                                    calc_total_parcel_volume,
                                    calc_d50_grain_size,
                                    calc_d50_grain_size_hydraulic_geometry,
                                    )



def test_call_hydraulic_geometry(example_parcel_initializer):
    parcel_volume = 1
    d50_hydraulic_geometry = [0.18,-0.12]
    _ = example_parcel_initializer(discharge_at_link=None,user_parcel_volume=parcel_volume,
                                     user_d50=d50_hydraulic_geometry)
    
def test_call_discharge(example_parcel_initializer,example_nmg):
    parcel_volume = 1
    discharge_at_link = np.full(example_nmg.number_of_links, 10.0)  # m^3 / s
    _ = example_parcel_initializer(discharge_at_link=discharge_at_link,user_parcel_volume=parcel_volume,
                                     user_d50=None)


def test_d50_not_specified(example_parcel_initializer):
    """test ValueError exception raised if d50 input is None"""    
    with pytest.raises(ValueError):
        parcel_volume = 1
        parcels = example_parcel_initializer(discharge_at_link=None,user_parcel_volume=parcel_volume,
                                     user_d50=None)

        
def test_calc_d50_grain_size(example_nmg, example_parcel_initializer):
    """test calc d50 grain size give correct values"""
    correct_values = np.array([0.195218575, #  see manual computations.xlsx
                             0.317133511,
                             0.247064737,
                             0.317133511,
                             0.271272098,
                             0.360303937,
                             0.317133511])
    discharge_at_link = np.full(example_nmg.number_of_links, 10.0)
    d50 = calc_d50_grain_size(discharge_at_link, 
                    example_parcel_initializer._grid.at_link["channel_width"],
                    example_parcel_initializer._grid.at_link["channel_slope"],
                    mannings_n=example_parcel_initializer._mannings_n,
                    gravity=example_parcel_initializer._gravity,
                    rho_water=example_parcel_initializer._rho_water,
                    rho_sediment=example_parcel_initializer._rho_sediment,
                    tau_50=example_parcel_initializer._tau_50,    
                )
    print(d50)
    np.testing.assert_almost_equal(d50, correct_values)
    
    
# def test_calc_d50_grain_size_hydraulic_geometry():
#     """test for error in d50 as a function of dominant flow calculation """

# def test_calc_total_parcel_volume():
#     """test for error in parcel volume calculation"""

# def test_det_apporx_parcel_volume():
#     """test for error in apromimate parcel volume calculation"""
    
# def test_parcel_characteristics():
#     """check format and values in each key of the variabes dictionary"""
#     #many assert statements
# def test_all_items_in_data_record
#     """check that vvalues in variables dictionary are correctly converted to the datarecord"""
#     #many assert statements    

    