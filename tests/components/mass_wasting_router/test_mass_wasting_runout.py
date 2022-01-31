import numpy as np
import pandas as pd
import pytest

from landlab import FieldError, RasterModelGrid


@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_scour_entrain_deposit_updatePD(object):
    
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



@pytest.mark.xfail(reason = "TDD, class is not yet implemented")
class Test_update_dem(object):

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



@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_settle(object):
    def test_dem_1(self):
        """test topographic__elevation and soil__thickness change correctly"""

    def test_dem_2(self):
        """test topographic__elevation and soil__thickness change correctly"""
    
    def test_flat_dem(self):
        """"""
        # boundary problem
        
    def test_one_high_all_flat_dem(self):
        """"""
        # special
        
    def test_dem_1_different_order(self):
        """"""
        # normal 
    
    def test_dem_2_different_order(self):
        """"""
        # normal 
    
    def test_no_downslope_cells(self):
        """"""
        # boundary
    
    def test_not_a_real_number(self):
        """"""
        # bad value
        

# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented") 
class Test_scour(object):

    def test_opt1_normal_1(self, example_square_MWRu):
        """""" 
        n = 24
        qsi = 2
        slope = 0.087489
        opt = 1
        depth = qsi        
        E = example_square_MWRu._scour(n, depth, slope, opt = opt)
        expected_E = 0.1017184
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)
        
        example_square_MWRu._scour()
    def test_opt1_normal_1(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.087489
        example_square_MWRu.slpc = 0.01
        opt = 1
        depth = qsi    
        Dp = 0.25
        E = example_square_MWRu._scour(n, depth, slope, opt = opt)
        expected_E = 0.144256
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)
        
    def test_opt2_normal_1(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.087489
        example_square_MWRu.slpc = 0.01
        opt = 2
        pd_in = 0.25
        depth = qsi        
        E = example_square_MWRu._scour(n, depth, slope, opt = opt,pd_in = pd_in)
        expected_E = 0.031997
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)

    def test_opt2_normal_2(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.087489
        example_square_MWRu.slpc = 0.1
        opt = 2
        pd_in = 0.25
        depth = qsi        
        E = example_square_MWRu._scour(n, depth, slope, opt = opt,pd_in = pd_in)
        expected_E = 0.050712
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)

    def test_opt2_boundary_1(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.05
        example_square_MWRu.slpc = 0.05
        opt = 2
        pd_in = 0.25
        depth = qsi        
        E = example_square_MWRu._scour(n, depth, slope, opt = opt,pd_in = pd_in)
        expected_E = 0.039494
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)
    
# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented") 
class Test_deposit(object):
    
    def test_deposit_L_normal_1(self, example_MWRu_2):
        qsi = 2
        slpn = 0.05
        D = example_MWRu_2._deposit_L_metric( qsi, slpn)
        expected_D = 1.5
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)
        
    def test_deposit_L_normal_2(self, example_MWRu_2):
        qsi = 2
        slpn = 0.2
        D = example_MWRu_2._deposit_L_metric( qsi, slpn)
        expected_D = 0
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)

    def test_deposit_L_boundary_1(self, example_MWRu_2):
        qsi = 2
        slpn = 0.1
        D = example_MWRu_2._deposit_L_metric( qsi, slpn)
        expected_D = 0
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)

    def test_deposit_L_boundary_2(self, example_MWRu_2):
        qsi = 2
        slpn = 0
        D = example_MWRu_2._deposit_L_metric( qsi, slpn)
        expected_D = 2
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)        


    def test_deposit_friction_angle_normal_1(self, example_MWRu_2):
        qsi = 2
        zi = 50
        zo = 50.1
        D = example_MWRu_2._deposit_friction_angle(qsi,zi,zo)
        expected_D = 1.3
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  
        
    def test_deposit_friction_angle_normal_2(self, example_MWRu_2):
        qsi = 2
        zi = 50
        zo = 49.9
        D = example_MWRu_2._deposit_friction_angle(qsi,zi,zo)
        expected_D = 1.2
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  

    def test_deposit_friction_angle_boundary_1(self, example_MWRu_2):
        qsi = 2
        zi = 50
        zo = 50
        D = example_MWRu_2._deposit_friction_angle(qsi,zi,zo)
        expected_D = 1.25
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  

    
    def test_deposit_friction_angle_special_1(self, example_MWRu_2):
        qsi = 2
        zi = 50
        zo = 47
        D = example_MWRu_2._deposit_friction_angle(qsi,zi,zo)
        expected_D = 0
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  

    def test_determine_zo_normal_1(self, example_square_MWRu):
        example_square_MWRu.itL = 1
        example_square_MWRu.run_one_step(dt = 0)
        
        n = 24
        qsi = 0.2
        zi = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        zo = example_square_MWRu._determine_zo(n, zi, qsi)
        expected_zo = 5.5
        np.testing.assert_allclose(zo, expected_zo,rtol = 1e-4) 


    def test_determine_zo_normal_2(self, example_square_MWRu):
        example_square_MWRu.itL = 1
        example_square_MWRu.run_one_step(dt = 0)
        
        n = 24
        qsi = 2
        zi = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        zo = example_square_MWRu._determine_zo(n, zi, qsi)
        expected_zo = 6.3
        np.testing.assert_allclose(zo, expected_zo,rtol = 1e-4) 

    def test_determine_zo_boundary_1(self, example_square_MWRu):
        example_square_MWRu.itL = 1
        example_square_MWRu.run_one_step(dt = 0)
        
        n = 24
        qsi = 0
        zi = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        zo = example_square_MWRu._determine_zo(n, zi, qsi)
        expected_zo = 5
        np.testing.assert_allclose(zo, expected_zo,rtol = 1e-4)


    def test_determine_zo_special_1(self, example_square_MWRu):
        example_square_MWRu.routing_surface = "energy__elevation"
        example_square_MWRu.itL = 1
        example_square_MWRu.settle_deposit = True
        example_square_MWRu.run_one_step(dt = 0)
        
        # make node 24 a pit
        example_square_MWRu._grid.at_node['topographic__elevation'][24] = 3
        n = 24
        qsi = 0
        zi = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        zo = example_square_MWRu._determine_zo(n, zi, qsi)
        expected_zo = None
        assert(zo, expected_zo)            

    def test_deposit_normal_1(self, example_square_MWRu):
        """use default iteration limit (1000), look at deposition near outlet"""

        example_square_MWRu.run_one_step(dt = 0)
                
        n = 9    
        qsi = 2
        slpn = example_square_MWRu._grid.at_node['topographic__steepest_slope'][n].max()
        D = example_square_MWRu._deposit(qsi,slpn,n)
        expected_D = 1.299793
        
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)
    
        
    def test_deposit_boundary(self, example_square_MWRu):
        """use default iteration limit (1000), look at deposition near outlet"""
        example_square_MWRu.run_one_step(dt = 0)
                
        n = 10    
        qsi = 2
        # slope here happens to be zero
        slpn = example_square_MWRu._grid.at_node['topographic__steepest_slope'][n].max()
        D = example_square_MWRu._deposit(qsi,slpn,n)
        expected_D = 1.452528
        
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)


    def test_deposit_special(self, example_square_MWRu):
        """use default iteration limit (1000), look at deposition near outlet,
        routing_surface = topographic__elevation"""
        
        example_square_MWRu.routing_surface = "topographic__elevation"
        example_square_MWRu.run_one_step(dt = 0)
                
        n = 10    
        qsi = 2
        # slope here happens to be zero
        slpn = example_square_MWRu._grid.at_node['topographic__steepest_slope'][n].max()
        D = example_square_MWRu._deposit(qsi,slpn,n)
        expected_D = 2
        
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)
  

# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_particle_diameter_in(object):

    def test_normal_values_1(self, example_MWRu_2):
        """"""
        example_MWRu_2.itL = 0       
        example_MWRu_2.run_one_step(dt = 0)

        n = 571
        vin = np.sum(example_MWRu_2.arv[example_MWRu_2.arn == n])
        pd_in = example_MWRu_2._particle_diameter_in(n,vin)
        expected_pd_in = 0.147599
        
        np.testing.assert_allclose(pd_in, expected_pd_in,rtol = 1e-4)
        
    def test_normal_values_2(self, example_MWRu_2):
        """"""

        example_MWRu_2.itL = 0       
        example_MWRu_2.run_one_step(dt = 0)

        n = 570
        vin = np.sum(example_MWRu_2.arv[example_MWRu_2.arn == n])
        pd_in = example_MWRu_2._particle_diameter_in(n,vin)

        expected_pd_in = 0.23667
        
        np.testing.assert_allclose(pd_in, expected_pd_in,rtol = 1e-4)
        
        
    def test_special_values_1(self, example_MWRu_2):
        """"""
        example_MWRu_2.itL = 8       
        example_MWRu_2.run_one_step(dt = 0)        
        
        n = 517
        vin = np.sum(example_MWRu_2.arv[example_MWRu_2.arn == n])
        pd_in = example_MWRu_2._particle_diameter_in(n,vin)

        expected_pd_in = 0
        
        np.testing.assert_allclose(pd_in, expected_pd_in,rtol = 1e-4)
        

    def test_bad_values_1(self, example_MWRu_2):
        """"""
        example_MWRu_2.itL = 8       
        example_MWRu_2.run_one_step(dt = 0)   

        with pytest.raises(ValueError) as exc_info:
            n = 517
            vin = np.nan
            pd_in = example_MWRu_2._particle_diameter_in(n,vin)

        assert exc_info.match("in-flowing volume is nan or inf")    
        
        
# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_particle_diameter_out(object):
    def test_normal_values_1(self, example_MWRu):
        pd_up = 0.1
        pd_in = 0.5
        qsi = 2
        E = 0.2
        D = 0.3
        
        pd_out = example_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)
        
        expected_pd_out = np.array([0.4579])
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)
        
    def test_normal_values_2(self,example_MWRu):
        pd_up = 0.001
        pd_in = 1
        qsi = 3
        E = 0.5
        D = 0.3
       
        pd_out = example_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)
        
        expected_pd_out = np.array([0.8439])
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)
        
        
    def test_bad_values_1(self,example_MWRu):        
        with pytest.raises(ValueError) as exc_info:
            pd_up = 0.001
            pd_in = np.nan
            qsi = 3
            E = 0.5
            D = 0.3
               
            pd_out = example_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)           
 
        assert exc_info.match("out-flowing particle diameter is zero, negative, nan or inf")        
        

    def test_bad_values_2(self,example_MWRu):        
        with pytest.raises(ValueError) as exc_info:
            pd_up = 0.001
            pd_in = -.05
            qsi = 3
            E = 0.5
            D = 0.3
               
            pd_out = example_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)           
 
        assert exc_info.match("out-flowing particle diameter is zero, negative, nan or inf")  

    

# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_particle_diameter_node:
    def test_normal_values_1(self,example_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        n = 405
        pd_in = 0.2
        D = 0.5
        E = 0.5
        # grid node depth = 1, node particle diameter = 0.075 
        
        pd_out = example_MWRu._particle_diameter_node(n,pd_in,E,D)
        
        expected_pd_out = 0.1375
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)

    def test_normal_values_2(self,example_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        n = 405
        pd_in = 0.2
        D = 0.3
        E = 1
         
        
        pd_out = example_MWRu._particle_diameter_node(n,pd_in,E,D)
        
        expected_pd_out = 0.2
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)

    def test_special_values_1(self,example_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        n = 405
        pd_in = 0.2
        D = 0
        E = 1    

        pd_out = example_MWRu._particle_diameter_node(n,pd_in,E,D)

        expected_pd_out = 0
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)

    def test_bad_values_1(self,example_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        with pytest.raises(ValueError) as exc_info:
            n = 405
            pd_in = np.nan
            D = 0.5
            E = 0.5
            # grid node depth = 1, node particle diameter = 0.075 
            
            pd_out = example_MWRu._particle_diameter_node(n,pd_in,E,D)
            
        assert exc_info.match("node particle diameter is negative, nan or inf")

    def test_bad_values_2(self,example_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        with pytest.raises(ValueError) as exc_info:
            n = 405
            pd_in = np.inf
            D = 0.5
            E = 0.5
            # grid node depth = 1, node particle diameter = 0.075 
            
            pd_out = example_MWRu._particle_diameter_node(n,pd_in,E,D)
            
        assert exc_info.match("node particle diameter is negative, nan or inf")  