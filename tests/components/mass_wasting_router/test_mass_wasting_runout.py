import numpy as np
import pandas as pd
import pytest

from landlab import FieldError, RasterModelGrid

# scour_entrain_deposit
@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class TestScourEntrainDeposit(object):
    
    def test_arn_arv_values_1(self):
        """given a typical list of recieiving nodes and volumes,
        determine E,D and qso -normal 1"""
        
    def test_arn_arv_values_2(self):
        """given a typical list of recieiving nodes and volumes,
        determine E,D and qso -normal 2"""
    
    
    def test_arv_less_than_thresh(self):
        """incoming volume is less than the force to stop threshold -special"""
        
    def test_vin_zero(self):
        """what happens if incoming volyme is zero? -boundary"""
        
    def test_vin_negative(self):
        """throws an exception if negative incoming volymes"""


# update_dem
@pytest.mark.xfail(reason = "TDD, class is not yet implemented")
class TestUpdateDEM(object):

    def test_qs1_deta_value_1(self):
        """test dem updates correctly given depth of debris flow and deta"""
        
    def test_qs1_deta_value_2(self):
        """test dem updates correctly given depth of debris flow and deta"""
        
    def test_qs1_deta_value_1_opt2(self):
        """test dem updates correctly given depth of debris flow and deta"""
        
    def test_qs1_deta_value_2_opt2(self):
        """test dem updates correctly given depth of debris flow and deta"""
                
    def test_not_a_float_or_int(self):
        """test throws an error if not a real number"""
        
    
# deposit settles
@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class TestDepositSettles(object):
    def test_dem_1(self):
        """test topographic__elevation and soil__thickness change correctly"""

    def test_dem_2(self):
        """test topographic__elevation and soil__thickness change correctly"""
    
    def test_flat_dem(self):
        # boundary problem
        
    def test_one_high_all_flat_dem(self):
        # special
        
    def test_dem_1_different_order(self):
        # normal 
    
    def test_dem_2_different_order(self):
        # normal 
    
    def test_no_downslope_cells(self):
        # boundary
    
    def test_not_a_real_number(self):
        # bad value
        
# scour
@pytest.mark.xfail(reason = "TDD, test class is not yet implemented") 
class TestScour(object):

    def test_opt1(self):
        
    def test_opt2(self):
        
    def test_opt2_hs_is_zero(self):

    


