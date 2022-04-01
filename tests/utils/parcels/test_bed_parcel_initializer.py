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
                                    calc_D50_grain_size,
                                    calc_D50_grain_size_hydraulic_geometry,
                                    )



def test_call_hydraulic_geometry(example_parcel_initializer,example_nmg):
    parcel_volume = 1
    D50_hydraulic_geometry = [0.18,-0.12]
    _ = example_parcel_initializer(discharge_at_link=None,user_parcel_volume=parcel_volume,
                                     user_D50=D50_hydraulic_geometry)
    
def test_call_discharge(example_parcel_initializer,example_nmg):
    parcel_volume = 1
    discharge_at_link = np.full(example_nmg.number_of_links, 10.0)  # m^3 / s
    _ = example_parcel_initializer(discharge_at_link=discharge_at_link,user_parcel_volume=parcel_volume,
                                     user_D50=None)


def test_D50_not_specified(example_parcel_initializer):
    """test ValueError exception raised if D50 input is None"""    
    with pytest.raises(ValueError):
        parcel_volume = 1
        parcels = example_parcel_initializer(discharge_at_link=None,user_parcel_volume=parcel_volume,
                                     user_D50=None)

        
def test_calc_D50_grain_size(example_nmg, example_parcel_initializer):
    """test calc D50 grain size give correct values"""
    correct_values = np.array([0.195218575, #  see manual computations.xlsx
                             0.317133511,
                             0.247064737,
                             0.317133511,
                             0.271272098,
                             0.360303937,
                             0.317133511])
    discharge_at_link = np.full(example_nmg.number_of_links, 10.0)
    D50 = calc_D50_grain_size(discharge_at_link, 
                    example_parcel_initializer._grid.at_link["channel_width"],
                    example_parcel_initializer._grid.at_link["channel_slope"],
                    mannings_n=example_parcel_initializer._mannings_n,
                    gravity=example_parcel_initializer._gravity,
                    rho_water=example_parcel_initializer._rho_water,
                    rho_sediment=example_parcel_initializer._rho_sediment,
                    tau_50=example_parcel_initializer._tau_50,    
                )
    np.testing.assert_almost_equal(D50, correct_values)
    
class Test_CalcD50GrainSizeHydraulicGeometry(object):    
    def test_normal_1(self):
        """test normal values 1"""
        D50e = 0.13182567
        a = 0.1; b = 0.12; CA = 10 # km^2
        D50 = calc_D50_grain_size_hydraulic_geometry(user_D50 = [a,b] ,drainage_area = CA)
        np.testing.assert_allclose(D50, D50e, rtol = 1e-4)
    
    def test_normal_2(self):
        """test normal values 1"""
        D50e = 0.1
        a = 0.1; b = 0; CA = 10 # km^2
        D50 = calc_D50_grain_size_hydraulic_geometry(user_D50 = [a,b] ,drainage_area = CA)
        np.testing.assert_allclose(D50, D50e, rtol = 1e-4)
        
        
    def test_bad_1(self):
        """"""
        a = 0.1; b = 0; CA = 10 # km^2    
        with pytest.raises(ValueError) as exc_info:
            D50 = calc_D50_grain_size_hydraulic_geometry(user_D50 = a, drainage_area = CA)
        msg_e = "coefficient and exponent of hydraulic geometry relation for D50 must be a list of length 1 or 2"
        assert exc_info.match(msg_e)

class Test_CalcD50GrainSize(object):
    def test_normal_1(self):
        D50e = 0.079037
        dominant_discharge = 1.2; width = 3; slope = 0.03; mannings_n = 0.03
        gravity = 9.81; rho_water = 1000; rho_sediment = 2700; tau_50 = 0.045    
        D50 = calc_D50_grain_size(dominant_discharge, width, slope, mannings_n,
                                  gravity, rho_water, rho_sediment, tau_50)
        np.testing.assert_allclose(D50, D50e, rtol = 1e-4)
        
    def test_normal_2(self):
        D50e = 0.314653
        dominant_discharge = 1.2; width = 3; slope = 0.03; mannings_n = 0.3
        gravity = 9.81; rho_water = 1000; rho_sediment = 2700; tau_50 = 0.045    
        D50 = calc_D50_grain_size(dominant_discharge, width, slope, mannings_n,
                                  gravity, rho_water, rho_sediment, tau_50)
        np.testing.assert_allclose(D50, D50e, rtol = 1e-4)

def test_calc_total_parcel_volume():
    vole = 12.5
    vol = calc_total_parcel_volume(5, 5, 0.5)
    np.testing.assert_allclose(vol, vole, rtol = 1e-4)

def test_det_approx_parcel_volume():
    total_parcel_volume_at_link = 10
    median_number_of_starting_parcels = 50
    pvole = 0.2
    pvol = determine_approx_parcel_volume(total_parcel_volume_at_link, median_number_of_starting_parcels)
    np.testing.assert_allclose(pvol, pvole, rtol = 1e-4)
    

class Test_BedParcelInitializer(object):
    def test_normal_1(self, example_nmg):
        """minimum attributes specified, most attributes should use
        defaults specified at instantiation"""
        np.random.seed(seed=5)
        grid = example_nmg

        initialize_parcels = BedParcelInitializer(grid, median_number_of_starting_parcels=1)
        
        parcels = initialize_parcels(discharge_at_link=1)

               
        # create expected values and get values from datarecord
        GEe = np.expand_dims(["link"]*8, axis=1).astype(object)
        GE = parcels.dataset['grid_element']
        EIe = np.expand_dims(np.array([0,1,2,3,4,5,5,6]), axis=1)
        EI = parcels.dataset['element_id']
        SLe = np.array([0,1,2,3,4,5,5,6])
        SL = parcels.dataset['starting_link']
        ARe = np.ones(8)*0
        AR = parcels.dataset['abrasion_rate']
        De = np.ones(8)*2650
        D = parcels.dataset['density']
        TAe = np.array([[ 0.08074127], [ 0.7384403 ], [ 0.44130922], [ 0.15830987],
                        [ 0.87993703], [ 0.27408646], [ 0.41423502], [ 0.29607993]])
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(8), axis=1)
        AL = parcels.dataset['active_layer']
        LLe = np.array([[ 0.62878791], [ 0.57983781], [ 0.5999292 ], [ 0.26581912], 
                        [ 0.28468588], [ 0.25358821], [ 0.32756395], [ 0.1441643 ]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.06802885], [ 0.06232028], [ 0.37674903], [ 0.06607135],
                    [ 0.07391346], [ 0.29280262], [ 0.04609956], [ 0.05135768]])
        D = parcels.dataset['D']
        Ve = np.array([[ 100372.02376292], [ 100372.02376292], [ 100372.02376292],
                       [ 100372.02376292], [ 100372.02376292], [ 100372.02376292],
                       [ 100372.02376292], [ 100372.02376292]])
        V = parcels.dataset['volume']  
        
        assert list(GE.values) == list(GEe)
        np.testing.assert_allclose(EI, EIe, rtol = 1e-4)
        np.testing.assert_allclose(SL, SLe, rtol = 1e-4)
        np.testing.assert_allclose(AR, ARe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)
        np.testing.assert_allclose(TA, TAe, rtol = 1e-4)
        np.testing.assert_allclose(AL, ALe, rtol = 1e-4)
        np.testing.assert_allclose(LL, LLe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)        
        np.testing.assert_allclose(V, Ve, rtol = 1e-4)
        
    def test_normal_2(self, example_nmg):
        """specify parcel volume"""
        np.random.seed(seed=5)
        grid = example_nmg

        initialize_parcels = BedParcelInitializer(grid, median_number_of_starting_parcels=1, 
                                                  abrasion_rate = 0.1, )
        
        parcels = initialize_parcels(discharge_at_link=1)

               
        # create expected values and get values from datarecord
        GEe = np.expand_dims(["link"]*8, axis=1).astype(object)
        GE = parcels.dataset['grid_element']
        EIe = np.expand_dims(np.array([0,1,2,3,4,5,5,6]), axis=1)
        EI = parcels.dataset['element_id']
        SLe = np.array([0,1,2,3,4,5,5,6])
        SL = parcels.dataset['starting_link']
        ARe = np.ones(8)*0
        AR = parcels.dataset['abrasion_rate']
        De = np.ones(8)*2650
        D = parcels.dataset['density']
        TAe = np.array([[ 0.08074127], [ 0.7384403 ], [ 0.44130922], [ 0.15830987],
                        [ 0.87993703], [ 0.27408646], [ 0.41423502], [ 0.29607993]])
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(8), axis=1)
        AL = parcels.dataset['active_layer']
        LLe = np.array([[ 0.62878791], [ 0.57983781], [ 0.5999292 ], [ 0.26581912], 
                        [ 0.28468588], [ 0.25358821], [ 0.32756395], [ 0.1441643 ]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.06802885], [ 0.06232028], [ 0.37674903], [ 0.06607135],
                    [ 0.07391346], [ 0.29280262], [ 0.04609956], [ 0.05135768]])
        D = parcels.dataset['D']
        Ve = np.array([[ 100372.02376292], [ 100372.02376292], [ 100372.02376292],
                       [ 100372.02376292], [ 100372.02376292], [ 100372.02376292],
                       [ 100372.02376292], [ 100372.02376292]])
        V = parcels.dataset['volume']  
        
        assert list(GE.values) == list(GEe)
        np.testing.assert_allclose(EI, EIe, rtol = 1e-4)
        np.testing.assert_allclose(SL, SLe, rtol = 1e-4)
        np.testing.assert_allclose(AR, ARe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)
        np.testing.assert_allclose(TA, TAe, rtol = 1e-4)
        np.testing.assert_allclose(AL, ALe, rtol = 1e-4)
        np.testing.assert_allclose(LL, LLe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)        
        np.testing.assert_allclose(V, Ve, rtol = 1e-4)
            