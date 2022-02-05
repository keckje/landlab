import numpy as np
import pandas as pd
import pytest

from landlab import FieldError, RasterModelGrid
from landlab.components import FlowDirectorMFD
from landlab.components.mass_wasting_router import MassWastingRunout


class Test__prep_initial_mass_wasting_material(object):
        
        def test_single_node(self, example_square_MWRu):
            """test receiving nodes and volumes from initial mass wasting cells
            are correctly prepared"""
            example_square_MWRu.itL = 0
            example_square_MWRu.run_one_step(dt = 0)
            rn = example_square_MWRu.arn
            rv = example_square_MWRu.arv
            rn_e = np.array([31, 30, 32])
            rv_e = np.array([65.3453,  11.5515,  23.1030])
            np.testing.assert_allclose(rn, rn_e, rtol = 1e-4)
            np.testing.assert_allclose(rv, rv_e, rtol = 1e-4)
       
        def test_two_nodes(self, example_square_mg):
            """test receiving nodes and volumes from initial mass wasting cells
            are correctly prepared, two initial nodes"""
            example_square_mg.at_node['mass__wasting_id'][np.array([31, 38])] = \
            np.array([1,1])
            npu = [1] 
            nid = [1] 
            slpc = [0.03]   
            SD = 0.01
            cs = 0.02
        
            mw_dict = {'critical slope':slpc, 'minimum flux':SD,
                        'scour coefficient':cs}
            
            release_dict = {'number of pulses':npu, 'iteration delay':nid }
                
            example_square_MWRu = MassWastingRunout(example_square_mg,
                                                    release_dict,mw_dict, 
                                                    save = True,
                                                    routing_surface = "energy__elevation", 
                                                    settle_deposit = True)
            
            example_square_MWRu.itL = 0
            example_square_MWRu.run_one_step(dt = 0)
            rn = example_square_MWRu.arn
            rv = example_square_MWRu.arv
            rn_e = np.array([ 24,  23,  25,  31,  30,  32,])
            rv_e = np.array([ 58.578644, 20.710678, 20.710678, 54.681075, 18.771714,
                             26.547212])
            np.testing.assert_allclose(rn, rn_e, rtol = 1e-4)
            np.testing.assert_allclose(rv, rv_e, rtol = 1e-4)            


class Test_scour_entrain_deposit_updatePD(object):
    
    def test_normal_1(self, example_square_MWRu):
        """functions within scour_entrain_deposit_updatePD are
        tested below. This test checks that the output of those
        functions are correctly stored the class variable nudat"""
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(dt = 0)      
        nodes = example_square_MWRu.nudat[:,0].astype(float)
        deta = example_square_MWRu.nudat[:,1].astype(float)
        qso = example_square_MWRu.nudat[:,2] .astype(float)
        rn = np.hstack(example_square_MWRu.nudat[:,3]).astype(float)
        n_pd = example_square_MWRu.nudat[:,4].astype(float)
        nodes_e = np.array([30,31,32])
        deta_e = np.array([-0.036031, -0.042085, -0.029911])
        qso_e = np.array([0.151547, 0.695539, 0.260941])
        rn_e = np.array([31, 23, 38, 24, 24, 23, 25, 31, 25, 24])
        n_pd_e = np.array([0.090969, 0.148153, 0.124476])
        
        np.testing.assert_array_almost_equal(nodes, nodes_e) 
        np.testing.assert_array_almost_equal(deta, deta_e)
        np.testing.assert_array_almost_equal(qso, qso_e)
        np.testing.assert_array_almost_equal(rn, rn_e)
        np.testing.assert_array_almost_equal(n_pd, n_pd_e)
        
    def test_normal_1(self, example_square_MWRu):
        """functions within scour_entrain_deposit_updatePD are
        tested below. This test checks that the output of those
        functions are correctly stored the class variable nudat.
        Special case that qsi is less than the minimum flux
        threshold"""
        nn = example_square_MWRu._grid.number_of_nodes
        example_square_MWRu._grid.at_node['soil__thickness'] = np.ones(nn)*0.01
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(dt = 0)      
        nodes = example_square_MWRu.nudat[:,0].astype(float)
        deta = example_square_MWRu.nudat[:,1].astype(float)
        qso = example_square_MWRu.nudat[:,2] .astype(float)
        rn = np.hstack(example_square_MWRu.nudat[:,3]).astype(float)
        n_pd = example_square_MWRu.nudat[:,4].astype(float)
        nodes_e = np.array([30,31,32])
        deta_e = np.array([0.001155,  0.006535,  0.00231])
        qso_e = np.array([0, 0, 0])
        rn_e = np.array([31, 23, 24, 24, 23, 25, 31, 25, 24])
        n_pd_e = np.array([0.098587,  0.154623,  0.131993])
        
        np.testing.assert_array_almost_equal(nodes, nodes_e) 
        np.testing.assert_array_almost_equal(deta, deta_e)
        np.testing.assert_array_almost_equal(qso, qso_e)
        np.testing.assert_array_almost_equal(rn, rn_e)
        np.testing.assert_array_almost_equal(n_pd, n_pd_e) 
    

class Test_vin_qsi(object):
    def test_normal_1(self, example_square_MWRu):
        example_square_MWRu.itL = 2       
        example_square_MWRu.run_one_step(dt = 0)      
        n = 25
        v_to_nodes = np.hstack(example_square_MWRu.arv_r[1][1])
        q_to_nodes = v_to_nodes/(example_square_MWRu._grid.dx**2) 
        nodes = np.hstack(example_square_MWRu.arn_r[1][1])
        v_e = np.sum(v_to_nodes[nodes==n])
        q_e = np.sum(q_to_nodes[nodes==n])

        v = example_square_MWRu.vqdat[2,1]
        q = example_square_MWRu.vqdat[2,2]
        
        np.testing.assert_allclose(v_e, v, rtol = 1e-4)   
        np.testing.assert_allclose(q_e, q, rtol = 1e-4)

class Test_update_E_dem(object):
    def test_normal_1(self, example_square_MWRu):
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(dt = 0)
        n = 30
        el = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        qsi = example_square_MWRu.vqdat[0][2]
        E_e = el+qsi
        E = example_square_MWRu._grid.at_node['energy__elevation'][n]
        np.testing.assert_allclose(E_e, E, rtol = 1e-4)    


class Test_update_dem(object):
    def test_normal_1(self, example_square_MWRu):
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(dt = 0)
        n = 30
        eli = example_square_MWRu._grid.at_node['topographic__initial_elevation'][n]
        deta = example_square_MWRu.nudat[0][1]
        el_e = eli+deta
        el = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        np.testing.assert_allclose(el_e, el, rtol = 1e-4)


class Test_update_channel_particle_diameter(object):
    def test_normal_1(self, example_square_MWRu):
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(dt = 0)
        n = 30
        pd = example_square_MWRu._grid.at_node['particle__diameter'][n]
        pd_e = 0.09096981
        np.testing.assert_allclose(pd_e, pd, rtol = 1e-4)


# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_settle(object):
    def test_normal_1(self, example_flat_mg):
        """test topographic__elevation and soil__thickness change correctly"""
        mg = example_flat_mg
        n = 12
        mg.at_node['topographic__elevation'][12] = 20
        fd = FlowDirectorMFD(mg, diagonals=True, partition_method = 'slope')
        fd.run_one_step()
        rn = mg.at_node.dataset['flow__receiver_node'].values[n]
        npu = [1] 
        nid = [1] 
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
        
        mw_dict = {'critical slope':slpc, 'minimum flux':SD, 'scour coefficient':cs}        
        release_dict = {'number of pulses':npu, 'iteration delay':nid }        
        example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                          routing_surface = "energy__elevation", settle_deposit = True)

        example_MWRu.D_L = [19] # deposition depth
        example_MWRu._settle([n])
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        expected_r_ne = np.array([2.36927701,2.369277016,2.369277016,2.369277016,
                                  1.968222984,1.968222984,1.968222984,1.968222984])
        expected_ne = 10.65
        np.testing.assert_allclose(rn_e, expected_r_ne, rtol = 1e-4)
        np.testing.assert_allclose(n_e, expected_ne, rtol = 1e-4)
        
    

    def test_normal_2(self, example_flat_mg):
        """test topographic__elevation and soil__thickness change correctly when
        settling is limited by the deposition depth"""
        mg = example_flat_mg
        n = 12
        mg.at_node['topographic__elevation'][12] = 20
        fd = FlowDirectorMFD(mg, diagonals=True, partition_method = 'slope')
        fd.run_one_step()
        rn = mg.at_node.dataset['flow__receiver_node'].values[n]
        npu = [1] 
        nid = [1] 
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
        
        mw_dict = {'critical slope':slpc, 'minimum flux':SD, 'scour coefficient':cs}        
        release_dict = {'number of pulses':npu, 'iteration delay':nid }        
        example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                          routing_surface = "energy__elevation", settle_deposit = True)

        example_MWRu.D_L = [5] # deposition depth
        example_MWRu._settle([n])
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        expected_r_ne = np.array([1.732233698, 1.732233698, 1.732233698, 1.732233698, 1.517766302, 1.517766302, 1.517766302, 1.517766302])
        expected_ne = 15
        np.testing.assert_allclose(rn_e, expected_r_ne, rtol = 1e-4)
        np.testing.assert_allclose(n_e, expected_ne, rtol = 1e-4) 

    def test_special_1(self, example_flat_mg):
        """test topographic__elevation and soil__thickness change correctly when
        settling is limited by the deposition depth"""
        mg = example_flat_mg
        n = 12
        mg.at_node['topographic__elevation'][12] = 1.3
        fd = FlowDirectorMFD(mg, diagonals=True, partition_method = 'slope')
        fd.run_one_step()
        rn = mg.at_node.dataset['flow__receiver_node'].values[n]
        npu = [1] 
        nid = [1] 
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
        
        mw_dict = {'critical slope':slpc, 'minimum flux':SD, 'scour coefficient':cs}        
        release_dict = {'number of pulses':npu, 'iteration delay':nid }        
        example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                          routing_surface = "energy__elevation", settle_deposit = True)

        example_MWRu.D_L = [0.3] # deposition depth
        example_MWRu._settle([n])
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        expected_r_ne = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        expected_ne = 1.3
        np.testing.assert_allclose(rn_e, expected_r_ne, rtol = 1e-4)
        np.testing.assert_allclose(n_e, expected_ne, rtol = 1e-4)     
 
    def test_bumpy_normal_1(self, example_bumpy_mg):
        """test topographic__elevation and soil__thickness change correctly"""
        mg = example_bumpy_mg
        n = 12
        
        mg.at_node['topographic__elevation'][12] = 20
        fd = FlowDirectorMFD(mg, diagonals=True, partition_method = 'slope')
        fd.run_one_step()
        rn = mg.at_node.dataset['flow__receiver_node'].values[n]
        npu = [1] 
        nid = [1] 
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
        
        mw_dict = {'critical slope':slpc, 'minimum flux':SD, 'scour coefficient':cs}        
        release_dict = {'number of pulses':npu, 'iteration delay':nid }        
        example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                          routing_surface = "energy__elevation", settle_deposit = True)

        example_MWRu.D_L = [19] # deposition depth
        example_MWRu._settle([n])
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        expected_r_ne = np.array([7.922500926, 8.851539316, 6.064424145, 3.277308974, 11.45159692, 9.55195179, 3.853016403, 5.75266153])
        expected_ne = 13.275
        np.testing.assert_allclose(rn_e, expected_r_ne, rtol = 1e-4)
        np.testing.assert_allclose(n_e, expected_ne, rtol = 1e-4)


    def test_bumpy_special_1(self, example_bumpy_mg):
        """test topographic__elevation and soil__thickness change correctly"""
        mg = example_bumpy_mg
        n = 12
        
        mg.at_node['topographic__elevation'][12] = 10
        fd = FlowDirectorMFD(mg, diagonals=True, partition_method = 'slope')
        fd.run_one_step()
        rn = mg.at_node.dataset['flow__receiver_node'].values[n]
        npu = [1] 
        nid = [1] 
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
        
        mw_dict = {'critical slope':slpc, 'minimum flux':SD, 'scour coefficient':cs}        
        release_dict = {'number of pulses':npu, 'iteration delay':nid }        
        example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                          routing_surface = "energy__elevation", settle_deposit = True)

        example_MWRu.D_L = [3] # deposition depth
        example_MWRu._settle([n])
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        expected_r_ne = np.array([ 7.227742,  8.151828,  5.379571,  2.607313,  1.,  9.053679,3.375756,  5.268397])
        expected_ne = 7.9357
        np.testing.assert_allclose(rn_e, expected_r_ne, rtol = 1e-4)
        np.testing.assert_allclose(n_e, expected_ne, rtol = 1e-4)


class Test_scour(object):

    def test_opt1_normal_1(self, example_square_MWRu):
        """""" 
        n = 24
        qsi = 2
        slope = 0.087489
        opt = 1
        depth = qsi
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(dt = 0)       
        E = example_square_MWRu._scour(n, depth, slope, opt = opt)
        expected_E = 0.1017184
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)


    def test_opt1_normal_2(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.57735
        example_square_MWRu.slpc = 0.01
        opt = 1
        depth = qsi    
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(dt = 0)       
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
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(dt = 0)               
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
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(dt = 0)               
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
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(dt = 0)               
        E = example_square_MWRu._scour(n, depth, slope, opt = opt,pd_in = pd_in)
        expected_E = 0.039494
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)


class Test_deposit(object):
    
    def test_deposit_L_normal_1(self, example_square_MWRu):
        qsi = 2
        slpn = 0.02
        D = example_square_MWRu._deposit_L_metric( qsi, slpn)
        expected_D = 1.1111
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)
        
    def test_deposit_L_normal_2(self, example_square_MWRu):
        qsi = 2
        slpn = 0.2
        D = example_square_MWRu._deposit_L_metric( qsi, slpn)
        expected_D = 0
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)

    def test_deposit_L_boundary_1(self, example_square_MWRu):
        qsi = 2
        slpn = 0.03
        D = example_square_MWRu._deposit_L_metric( qsi, slpn)
        expected_D = 0
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)

    def test_deposit_L_boundary_2(self, example_square_MWRu):
        qsi = 2
        slpn = 0
        D = example_square_MWRu._deposit_L_metric( qsi, slpn)
        expected_D = 2
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)        


    def test_deposit_friction_angle_normal_1(self, example_square_MWRu):
        qsi = 2
        zi = 50
        zo = 50.1
        D = example_square_MWRu._deposit_friction_angle(qsi,zi,zo)
        expected_D = 1.2
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  
        
    def test_deposit_friction_angle_normal_2(self, example_square_MWRu):
        qsi = 2
        zi = 50
        zo = 49.9
        D = example_square_MWRu._deposit_friction_angle(qsi,zi,zo)
        expected_D = 1.1
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  

    def test_deposit_friction_angle_boundary_1(self, example_square_MWRu):
        qsi = 2
        zi = 50
        zo = 50
        D = example_square_MWRu._deposit_friction_angle(qsi,zi,zo)
        expected_D = 1.15
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  

    
    def test_deposit_friction_angle_special_1(self, example_square_MWRu):
        qsi = 2
        zi = 50
        zo = 47
        D = example_square_MWRu._deposit_friction_angle(qsi,zi,zo)
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
        expected_D = 1.38982
        
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)
    
        
    def test_deposit_boundary(self, example_square_MWRu):
        """use default iteration limit (1000), look at deposition near outlet"""
        example_square_MWRu.run_one_step(dt = 0)
                
        n = 10    
        qsi = 2
        # slope here happens to be zero
        slpn = example_square_MWRu._grid.at_node['topographic__steepest_slope'][n].max()
        D = example_square_MWRu._deposit(qsi,slpn,n)
        expected_D = 1.344888
        
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)


    def test_deposit_special(self, example_square_MWRu):
        """use default iteration limit (1000), look at deposition near outlet,
        routing_surface = topographic__elevation, which should use L_deposit"""
        
        example_square_MWRu.routing_surface = "topographic__elevation"
        example_square_MWRu.run_one_step(dt = 0)
                
        n = 10    
        qsi = 2
        # slope here happens to be zero
        slpn = 0 
        D = example_square_MWRu._deposit(qsi,slpn,n)
        expected_D = 2
        
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)
  

# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_particle_diameter_in(object):

    def test_normal_values_1(self, example_square_MWRu):
        """"""
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(dt = 0)

        n = 24
        vin = np.sum(example_square_MWRu.arv[example_square_MWRu.arn == n])
        pd_in = example_square_MWRu._particle_diameter_in(n,vin)
        expected_pd_in = 0.161067
        
        np.testing.assert_allclose(pd_in, expected_pd_in,rtol = 1e-4)
        
    def test_normal_values_2(self, example_square_MWRu):
        """"""

        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(dt = 0)

        n = 31
        vin = np.sum(example_square_MWRu.arv[example_square_MWRu.arn == n])
        pd_in = example_square_MWRu._particle_diameter_in(n,vin)

        expected_pd_in = 0.155339
        
        np.testing.assert_allclose(pd_in, expected_pd_in,rtol = 1e-4)
        
        
    def test_special_values_1(self, example_square_MWRu):
        """no incoming volume"""
        example_square_MWRu.itL = 0       
        example_square_MWRu.run_one_step(dt = 0)        
        
        n = 24
        vin = np.sum(example_square_MWRu.arv[example_square_MWRu.arn == n])
        pd_in = example_square_MWRu._particle_diameter_in(n,vin)

        expected_pd_in = 0
        
        np.testing.assert_allclose(pd_in, expected_pd_in,rtol = 1e-4)
        

    def test_bad_values_1(self, example_square_MWRu):
        """incoming is np.nan"""
        example_square_MWRu.itL = 8       
        example_square_MWRu.run_one_step(dt = 0)   

        with pytest.raises(ValueError) as exc_info:
            n = 24
            vin = np.nan
            pd_in = example_square_MWRu._particle_diameter_in(n,vin)

        assert exc_info.match("in-flowing volume is nan or inf")    
        
        
# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_particle_diameter_out(object):
    def test_normal_values_1(self, example_square_MWRu):
        pd_up = 0.1
        pd_in = 0.5
        qsi = 2
        E = 0.2
        D = 0.3
        
        pd_out = example_square_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)
        
        expected_pd_out = np.array([0.4579])
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)
        
    def test_normal_values_2(self, example_square_MWRu):
        pd_up = 0.001
        pd_in = 1
        qsi = 3
        E = 0.5
        D = 0.3
       
        pd_out = example_square_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)
        
        expected_pd_out = np.array([0.8439])
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)
        
        
    def test_bad_values_1(self, example_square_MWRu):        
        with pytest.raises(ValueError) as exc_info:
            pd_up = 0.001
            pd_in = np.nan
            qsi = 3
            E = 0.5
            D = 0.3
               
            pd_out = example_square_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)           
 
        assert exc_info.match("out-flowing particle diameter is zero, negative, nan or inf")        
        

    def test_bad_values_2(self,example_square_MWRu):        
        with pytest.raises(ValueError) as exc_info:
            pd_up = 0.001
            pd_in = -.05
            qsi = 3
            E = 0.5
            D = 0.3
               
            pd_out = example_square_MWRu._particle_diameter_out(pd_up,pd_in,qsi,E,D)           
 
        assert exc_info.match("out-flowing particle diameter is zero, negative, nan or inf")  


# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_particle_diameter_node:
    def test_normal_values_1(self, example_square_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        n = 24
        pd_in = 0.2
        D = 0.5
        E = 0.5
        # grid node depth = 1, node particle diameter = 0.075 
        
        pd_out = example_square_MWRu._particle_diameter_node(n,pd_in,E,D)
        
        expected_pd_out = 0.215913
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)

    def test_normal_values_2(self, example_square_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        n = 24
        pd_in = 0.2
        D = 0.3
        E = 1
                 
        pd_out = example_square_MWRu._particle_diameter_node(n,pd_in,E,D)
        
        expected_pd_out = 0.2
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)

    def test_special_values_1(self, example_square_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        n = 24
        pd_in = 0.2
        D = 0
        E = 1    

        pd_out = example_square_MWRu._particle_diameter_node(n,pd_in,E,D)

        expected_pd_out = 0
    
        np.testing.assert_allclose(pd_out, expected_pd_out,rtol = 1e-4)

    def test_bad_values_1(self, example_square_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        with pytest.raises(ValueError) as exc_info:
            n = 24
            pd_in = np.nan
            D = 0.5
            E = 0.5
            # grid node depth = 1, node particle diameter = 0.075 
            
            pd_out = example_square_MWRu._particle_diameter_node(n,pd_in,E,D)
            
        assert exc_info.match("node particle diameter is negative, nan or inf")

    def test_bad_values_2(self, example_square_MWRu):
        # grid node depth = 1, node particle diameter = 0.075
        with pytest.raises(ValueError) as exc_info:
            n = 24
            pd_in = np.inf
            D = 0.5
            E = 0.5
            
            pd_out = example_square_MWRu._particle_diameter_node(n,pd_in,E,D)
            
        assert exc_info.match("node particle diameter is negative, nan or inf")  