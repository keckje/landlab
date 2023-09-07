import numpy as np
import pandas as pd
import pytest

from landlab import FieldError, RasterModelGrid
from landlab.components import FlowDirectorMFD
from landlab.components.mass_wasting_router import MassWastingRunout


class Test__virtual_laboratory_smoke_tests():
    
    def test_pile_runout(self, example_pile_MWRu):
        """smoke test using a pile of debris. Check that profile of final
        topographic surface doesn't change and that mass is always conserved"""
        MWRu = example_pile_MWRu
        MWRu.run_one_step(run_id = 0)
        c = np.array(list(MWRu.runout_evo_maps[0].keys())).max()
        # profile check
        pf = MWRu.topo_evo_maps[0][c][MWRu.pf]
        pf_e = nparray([1.        ,  1.        ,  1.        ,  1.        ,  1.,
                1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                1.00689131,  1.15535961,  1.19989349,  1.21732111,  1.30154153,
                1.3445694 ,  1.34283688,  1.48417702,  1.58725816,  1.73725816,
                1.7912149 ,  1.78716792,  1.90855312,  2.03704731,  2.04748121,
                2.49346051,  2.11886755,  2.08088422,  2.02381708,  1.85815936,
                1.61798616,  1.60479334,  1.6049865 ,  1.42619691,  1.37090398,
                1.3102595 ,  1.28800606,  1.30505667,  1.19631867,  1.11827782,
                1.08073868,  1.        ,  1.        ,  1.        ,  1.        ,
                1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                1.        ])
        # mass conservation check
        DEMi = MWRu._grid.at_node['topographic__initial_elevation']
        DEMf = MWRu._grid.at_node['topographic__elevation']
        DEMdf = DEMf-DEMi
        
        np.testing.assert_allclose(pf, pf_e, atol = 1e-4)
        np.testing.assert_allclose(0, DEMdf.sum(), atol = 1e-4)
        
    #def test_narrow_flume_runout():
        
    #def test_wide_flume_runout():
        
class Test__mass_conserved():
        
    def test_full_runout(self, example_square_MWRu):
        """check that cumulative topographic change at end of model run (len(rn) = 0) 
        equals zero"""
        example_square_MWRu.run_one_step(run_id = 0)
        mg = example_square_MWRu._grid
        diff = mg.at_node['topographic__elevation'] - mg.at_node['topographic__initial_elevation']
        np.testing.assert_allclose(0, diff.sum(), atol = 1e-4)
        
    def test_pause_at_middle_of_runout(self, example_square_MWRu):
        """pause model in the middle of modeled runout and check that cumulative 
        topographic change plus flux equals zero"""
        example_square_MWRu.itL = 3
        example_square_MWRu.run_one_step(run_id = 0)
        mg = example_square_MWRu._grid
        # cumulative topographic change [L]
        diff = mg.at_node['topographic__elevation'] - mg.at_node['topographic__initial_elevation']
        # cumulative flux [L]
        qs = example_square_MWRu.arqso.sum()
        np.testing.assert_allclose(0, diff.sum()+qs, atol = 1e-4)        

class Test__prep_initial_mass_wasting_material():
    
    def test_single_node_positive(self, example_square_MWRu):
        """Test correct number of receiving nodes and volumes from 
        initial mass wasting cells are positive"""
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(run_id = 0)
        rn = example_square_MWRu.arn
        rqso = example_square_MWRu.arqso
        assert len(rn) == 3
        assert rqso.all() > 0


    def test_two_nodes_positive(self, example_square_mg):
        """Test correct number of receiving nodes and volumes from 
        initial mass wasting cells are positive"""
        example_square_mg.at_node['mass__wasting_id'][np.array([31, 38])] = \
        np.array([1,1])
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
    
        mw_dict = {'critical slope':slpc, 'threshold flux':SD,
                    'scour coefficient':cs}
            
        example_square_MWRu = MassWastingRunout(example_square_mg,
                                                mw_dict, 
                                                save = True,
                                                effective_qsi = False,
                                                grain_shear = False)
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(run_id = 0)
        rn = example_square_MWRu.arn
        rqso = example_square_MWRu.arqso
        assert len(rn) == 4
        assert (rqso > 0).all()


    def test_single_node(self, example_square_MWRu):
        """Test receiving nodes and volumes from initial mass wasting cells
        are correct, one initial mass wasting node"""
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(run_id = 0)
        rn = example_square_MWRu.arn
        rqso = example_square_MWRu.arqso
        
        # determine manually from flow directions (mass wasting id = node 38)
        mask = example_square_MWRu._grid.at_node['flow__link_to_receiver_node'][38] != -1
        rn_e = example_square_MWRu._grid.at_node['flow__receiver_node'][38][mask]
        rn_slope = example_square_MWRu._grid.at_node['topographic__steepest_slope'][38][mask]
        rqso_e = rn_slope/rn_slope.sum()
        np.testing.assert_allclose(rn, rn_e, rtol = 1e-4)
        np.testing.assert_allclose(rqso, rqso_e, rtol = 1e-4)
        
        
    def test_two_nodes(self, example_square_mg):
        """Test receiving nodes and volumes from initial mass wasting cells
        are correct, two initial mass wasting nodes"""
        example_square_mg.at_node['mass__wasting_id'][np.array([31, 38])] = \
        np.array([1,1])
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
    
        mw_dict = {'critical slope':slpc, 'threshold flux':SD,
                    'scour coefficient':cs}
            
        example_square_MWRu = MassWastingRunout(example_square_mg,
                                                mw_dict, 
                                                save = True,
                                                effective_qsi = False,
                                                grain_shear = False)
        
        # get flow direction and proportions for lowest landslide node, which is 
        # determined from the top surface of the debriton
        mask_1 = example_square_MWRu._grid.at_node['flow__link_to_receiver_node'][31] != -1
        rn_e_1 = example_square_MWRu._grid.at_node['flow__receiver_node'][31][mask_1]
        rn_slope_1 = example_square_MWRu._grid.at_node['topographic__steepest_slope'][31][mask_1]
        
        # now get flow direction and proportions of second node, which is 
        # which is determined after the debriton has been removed, from
        # bottom surface of the debriton
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(run_id = 0)
        rn = example_square_MWRu.arn
        rqso = example_square_MWRu.arqso
        mask_2 = example_square_MWRu._grid.at_node['flow__link_to_receiver_node'][38] != -1
        rn_e_2 = example_square_MWRu._grid.at_node['flow__receiver_node'][38][mask_2]
        rn_slope_2 = example_square_MWRu._grid.at_node['topographic__steepest_slope'][38][mask_2]
        
        rn_e = np.concatenate((rn_e_1, rn_e_2))
        rqso_e = np.concatenate((rn_slope_1/rn_slope_1.sum(), rn_slope_2/rn_slope_2.sum()))

        np.testing.assert_allclose(rn, rn_e, rtol = 1e-4)
        np.testing.assert_allclose(rqso, rqso_e, rtol = 1e-4)            


class Test_E_A_qso_determine_attributes():

    
    def test_normal_1(self, example_square_MWRu):
        """Test that output of _E_A_qso_determine_attributes shows correct directional
        change (e.g., positive negative or no change)
        """
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)      
        nodes = example_square_MWRu.nudat[:,0].astype(float)
        deta = example_square_MWRu.nudat[:,1].astype(float)
        qso = example_square_MWRu.nudat[:,2].astype(float)
        qsi = example_square_MWRu.nudat[:,3].astype(float)
        E = example_square_MWRu.nudat[:,4].astype(float)
        A = example_square_MWRu.nudat[:,5].astype(float)

        key = 'particle__diameter'
        pd_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        pd_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        pd_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
        key = 'organic__content'
        oc_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        oc_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        oc_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
        
        assert (deta < 0).all() # should erode
        assert (qso > 0).all() # outflow should be positive
        assert (qsi > 0).all() # incoming flow should be positive
        np.testing.assert_array_almost_equal(E, deta*-1) # erosion depth should equal change in eta
        assert (pd_n > 0).all() # grain size at each rn > 0
        np.testing.assert_array_almost_equal(pd_n, pd_up) # grain size that is added to debriton same as grain size at node
        np.testing.assert_array_almost_equal(pd_in.mean(), 0.16452507) # all grain sizes delivered from same node, should be same 
        assert (oc_n > 0).all() # organic content at each rn > 0
        np.testing.assert_array_almost_equal(oc_n, oc_up)# organic content that is added to debriton same as at node
        np.testing.assert_array_almost_equal(oc_in.mean(), 0.08003922) # all organic content delivered from same donor node, should be same 


    def test_normal_2(self, example_square_MWRu):
        """Smoke test that the output of _E_A_qso_determine_attributes 
        have not changed and are stored in the class variable nudat as expected"""
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)      
        nodes = example_square_MWRu.nudat[:,0].astype(float)
        deta = example_square_MWRu.nudat[:,1].astype(float)
        qso = example_square_MWRu.nudat[:,2].astype(float)
        qsi = example_square_MWRu.nudat[:,3].astype(float)
        E = example_square_MWRu.nudat[:,4].astype(float)
        A = example_square_MWRu.nudat[:,5].astype(float)

        key = 'particle__diameter'
        pd_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        pd_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        pd_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
        key = 'organic__content'
        oc_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        oc_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        oc_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
                
        nodes_e = np.array([30,31,32])
        deta_e = np.array([-0.01225359, -0.06877944, -0.04888238])
        qso_e = np.array([ 0.127769, 0.72223323, 0.27991319])
        qsi_e = np.array([ 0.1155154, 0.65345379, 0.2310308])
        E_e = np.array([ 0.01225359, 0.06877944, 0.04888238])
        A_e = np.array([0, 0, 0])
        pd_n_e = np.array([ 0.09096982, 0.14815318, 0.12447694])
        pd_up_e = np.array([ 0.09096982, 0.14815318, 0.12447694])
        pd_in_e = np.array([ 0.16452507, 0.16452507, 0.16452507])
        oc_n_e = np.array([0.0370377, 0.02489513, 0.04734116])
        oc_up_e = np.array([0.0370377, 0.02489513, 0.04734116])
        oc_in_e = np.array([0.08003922, 0.08003922, 0.08003922])
        
        np.testing.assert_array_almost_equal(nodes, nodes_e) 
        np.testing.assert_array_almost_equal(deta, deta_e)
        np.testing.assert_array_almost_equal(qso, qso_e)
        np.testing.assert_array_almost_equal(qsi, qsi_e)
        np.testing.assert_array_almost_equal(pd_n, pd_n_e)
        np.testing.assert_array_almost_equal(pd_up, pd_up_e)
        np.testing.assert_array_almost_equal(pd_in, pd_in_e)
        np.testing.assert_array_almost_equal(oc_n, oc_n_e)
        np.testing.assert_array_almost_equal(oc_up, oc_up_e)
        np.testing.assert_array_almost_equal(oc_in, oc_in_e)


    def test_special_1(self, example_square_MWRu):
        """Test that output of _E_A_qso_determine_attributes shows correct directional
        change (e.g., positive negative or no change)
        Special case that qsi is less than the minimum flux
        threshold"""
        nn = example_square_MWRu._grid.number_of_nodes
        example_square_MWRu._grid.at_node['soil__thickness'] = np.ones(nn)*0.01
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)      
        nodes = example_square_MWRu.nudat[:,0].astype(float)
        deta = example_square_MWRu.nudat[:,1].astype(float)
        qso = example_square_MWRu.nudat[:,2] .astype(float)
        qsi = example_square_MWRu.nudat[:,3].astype(float)
        E = example_square_MWRu.nudat[:,4].astype(float)
        A = example_square_MWRu.nudat[:,5].astype(float)

        key = 'particle__diameter'
        pd_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        pd_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        pd_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
        key = 'organic__content'
        oc_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        oc_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        oc_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
                
        nodes_e = np.array([30, 31, 32])
        deta_e = np.array([0.00115515, 0.00653454, 0.00231031])
        qso_e = np.array([0, 0, 0])
        qsi_e = np.array([0.00115515, 0.00653454, 0.00231031])
        E_e = np.array([0, 0, 0])
        A_e = np.array([0.00115515, 0.00653454, 0.00231031])
        pd_n_e = np.array([0.09858671, 0.15462344, 0.13199288])
        pd_up_e = np.array([ 0, 0, 0])
        pd_in_e = np.array([0.16452507, 0.16452507, 0.16452507])
        oc_n_e = np.array([0.04149065, 0.04668837, 0.05347769])
        oc_up_e = np.array([0, 0, 0])
        oc_in_e = np.array([0.08003922, 0.08003922, 0.08003922])
        
        np.testing.assert_array_almost_equal(deta, qsi) # change eta equals qsi
        np.testing.assert_array_almost_equal(qso, np.array([0, 0, 0])) # no qso
        np.testing.assert_array_almost_equal(E, np.array([0, 0, 0])) # no erosion
        np.testing.assert_array_almost_equal(deta, A) # change eta equal aggradation
        np.testing.assert_array_almost_equal(pd_up, np.array([0, 0, 0])) # pd_up is an array of zeros
        np.testing.assert_array_almost_equal(oc_up, np.array([0, 0, 0])) # oc_up is an array of zeros
        
        
    def test_special_2(self, example_square_MWRu):
        """Smoke test that the output of function _scour_entrain_deposit_updatePD 
        have not changed and stored in the class variable nudat as expected
        Special case that qsi is less than the minimum flux
        threshold"""
        nn = example_square_MWRu._grid.number_of_nodes
        example_square_MWRu._grid.at_node['soil__thickness'] = np.ones(nn)*0.01
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)      
        nodes = example_square_MWRu.nudat[:,0].astype(float)
        deta = example_square_MWRu.nudat[:,1].astype(float)
        qso = example_square_MWRu.nudat[:,2] .astype(float)
        qsi = example_square_MWRu.nudat[:,3].astype(float)
        E = example_square_MWRu.nudat[:,4].astype(float)
        A = example_square_MWRu.nudat[:,5].astype(float)

        key = 'particle__diameter'
        pd_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        pd_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        pd_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
        key = 'organic__content'
        oc_n = np.array([d[key] for d in  example_square_MWRu.nudat[:,6]]).astype(float)
        oc_up = np.array([d[key] for d in  example_square_MWRu.nudat[:,7]]).astype(float)
        oc_in = np.array([d[key] for d in  example_square_MWRu.nudat[:,8]]).astype(float)
                
        nodes_e = np.array([30, 31, 32])
        deta_e = np.array([0.00115515, 0.00653454, 0.00231031])
        qso_e = np.array([0, 0, 0])
        qsi_e = np.array([0.00115515, 0.00653454, 0.00231031])
        E_e = np.array([0, 0, 0])
        A_e = np.array([0.00115515, 0.00653454, 0.00231031])
        pd_n_e = np.array([0.09858671, 0.15462344, 0.13199288])
        pd_up_e = np.array([ 0, 0, 0])
        pd_in_e = np.array([0.16452507, 0.16452507, 0.16452507])
        oc_n_e = np.array([0.04149065, 0.04668837, 0.05347769])
        oc_up_e = np.array([0, 0, 0])
        oc_in_e = np.array([0.08003922, 0.08003922, 0.08003922])
        
        np.testing.assert_array_almost_equal(nodes, nodes_e) 
        np.testing.assert_array_almost_equal(deta, deta_e)
        np.testing.assert_array_almost_equal(qso, qso_e)
        np.testing.assert_array_almost_equal(qsi, qsi_e)
        np.testing.assert_array_almost_equal(E, E_e)
        np.testing.assert_array_almost_equal(A, A_e)
        np.testing.assert_array_almost_equal(pd_n, pd_n_e)
        np.testing.assert_array_almost_equal(pd_up, pd_up_e)
        np.testing.assert_array_almost_equal(pd_in, pd_in_e)
        np.testing.assert_array_almost_equal(oc_n, oc_n_e)
        np.testing.assert_array_almost_equal(oc_up, oc_up_e)
        np.testing.assert_array_almost_equal(oc_in, oc_in_e)        
    

class Test_determine_qsi(object):
    def test_normal_1(self, example_square_MWRu):
        """test ouput of function  _determine_qsi is correct and stored
        in class variable vqdat as expected"""
        example_square_MWRu.itL = 2       
        example_square_MWRu.run_one_step(run_id = 0)      
        n = 25
        qs_to_nodes = np.hstack(example_square_MWRu.arqso_r[1][1])
        nodes = np.hstack(example_square_MWRu.arn_r[1][1])
        qsi_e = np.sum(qs_to_nodes[nodes==n])
        qsi = example_square_MWRu.qsi_dat[2,1]
  
        np.testing.assert_allclose(qsi_e, qsi, rtol = 1e-4)


class Test_update_E_dem(object):
    def test_normal_1(self, example_square_MWRu):
        """run one iteration, check qsi+initial dem matches energy__elevation"""
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)
        n = 30
        el = example_square_MWRu.topo_evo_maps[0][0][n] # elevation of initial topo
        qsi = example_square_MWRu.qsi_dat[0][1]
        E_e = el+qsi
        E = example_square_MWRu._grid.at_node['energy__elevation'][n]
        np.testing.assert_allclose(E_e, E, rtol = 1e-4)    

#@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_update_dem(object):
    def test_normal_1(self, example_square_MWRu):
        """test topographic dem updated correctly"""
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)
        n = 30
        eli = example_square_MWRu._grid.at_node['topographic__initial_elevation'][n]
        deta = example_square_MWRu.nudat[0][1]
        el_e = eli+deta
        el = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        np.testing.assert_allclose(el_e, el, rtol = 1e-4)


#@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_update_channel_particle_diameter(object):
    def test_normal_1(self, example_square_MWRu):
        """test particle diameter updated correctly"""
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)
        n = 30
        pd = example_square_MWRu._grid.at_node['particle__diameter'][n]
        pd_e = 0.09096981
        np.testing.assert_allclose(pd_e, pd, rtol = 1e-4)


#@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_settle(object):
    def test_normal_1(self, example_flat_mg):
        """test topographic__elevation and soil__thickness change correctly"""
        mg = example_flat_mg
        n = 12
        mg.at_node['topographic__elevation'][12] = 20
        fd = FlowDirectorMFD(mg, diagonals=True, partition_method = 'slope') # why are flow proportions not equal? diagonals are slightly further from centralnode and have smaller slope
        fd.run_one_step()
        rn = mg.at_node.dataset['flow__receiver_node'].values[n]
        npu = [1] 
        nid = [1] 
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
        mofd = 4
        
        
        
        mw_dict = {'critical slope':slpc, 'threshold flux':SD,
                   'scour coefficient':cs, 
                   'max observed flow depth':mofd}               
        example_MWRu = MassWastingRunout(mg, mw_dict, save = True,
                                           settle_deposit = True)
        example_MWRu.arn_u = np.array([n])# np.unique(rn) # set the array of unique receiver nodes
        example_MWRu.D_L = [19] # deposition depth at node 
        
        example_MWRu._settle() # run the settle function
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        
        expected_rn_e = np.array([2.36927701,2.369277016,2.369277016,2.369277016,
                                  1.968222984,1.968222984,1.968222984,1.968222984])
        expected_n_e = 10.65
        np.testing.assert_allclose(rn_e, expected_rn_e, rtol = 1e-4)
        np.testing.assert_allclose(n_e, expected_n_e, rtol = 1e-4)
        
    

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
        mofd = 4
        
        mw_dict = {'critical slope':slpc, 'threshold flux':SD,
                   'scour coefficient':cs, 
                   'max observed flow depth':mofd}               
        example_MWRu = MassWastingRunout(mg, mw_dict, save = True,
                                           settle_deposit = True)

        example_MWRu.arn_u = np.array([n])# np.unique(rn) # set the array of unique receiver nodes
        example_MWRu.D_L = [5] # deposition depth at node 
        example_MWRu._settle() # run the settle function
        # example_MWRu.D_L = [5] # deposition depth
        # example_MWRu._settle([n])
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
        mofd = 4
        
        mw_dict = {'critical slope':slpc, 'threshold flux':SD,
                   'scour coefficient':cs, 
                   'max observed flow depth':mofd}               
        example_MWRu = MassWastingRunout(mg, mw_dict, save = True,
                                           settle_deposit = True)

        example_MWRu.arn_u = np.array([n])# np.unique(rn) # set the array of unique receiver nodes
        example_MWRu.D_L = [0.3] # deposition depth at node 
        example_MWRu._settle() # run the settle function
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
        mofd = 4
        
        mw_dict = {'critical slope':slpc, 'threshold flux':SD,
                   'scour coefficient':cs, 
                   'max observed flow depth':mofd}               
        example_MWRu = MassWastingRunout(mg, mw_dict, save = True,
                                           settle_deposit = True)

        example_MWRu.arn_u = np.array([n])# np.unique(rn) # set the array of unique receiver nodes
        example_MWRu.D_L = [19] # deposition depth at node 
        example_MWRu._settle() # run the settle function
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        expected_r_ne = np.array([8.2139975, 9.12061308, 6.40076635, 3.68091962,
                11.59429483, 9.72636035, 4.1225569, 5.99049138])
        expected_ne = 11.15
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
        slpc = [0.03]   
        SD = 0.01
        cs = 0.02
        mofd = 4
        
        mw_dict = {'critical slope':slpc, 'threshold flux':SD,
                   'scour coefficient':cs, 
                   'max observed flow depth':mofd}               
        example_MWRu = MassWastingRunout(mg, mw_dict, save = True,
                                           settle_deposit = True)

        example_MWRu.arn_u = np.array([n])# np.unique(rn) # set the array of unique receiver nodes
        example_MWRu.D_L = [3] # deposition depth at node 
        example_MWRu._settle() # run the settle function
        rn_e = mg.at_node['topographic__elevation'][rn]
        n_e = mg.at_node['topographic__elevation'][n]
        expected_r_ne = np.array([ 7.33097686, 8.220651243, 5.551628106, 2.88260497, 1, 9.078000214, 3.546078728, 5.390059875])
        expected_ne = 7
        np.testing.assert_allclose(rn_e, expected_r_ne, rtol = 1e-4)
        np.testing.assert_allclose(n_e, expected_ne, rtol = 1e-4)

#@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_erosion(object):

    def test_quasi_normal_1(self, example_square_MWRu):
        """""" 
        n = 24
        qsi = 2
        slope = 0.087489
        opt = 1
        depth = qsi
        example_square_MWRu.itL = 0
        example_square_MWRu.grain_shear = False
        example_square_MWRu.run_one_step(run_id = 0)       
        E = example_square_MWRu._erosion(n, depth, slope)
        expected_E = 0.1017184
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)


    def test_quasi_normal_1(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.57735
        example_square_MWRu.slpc = 0.01
        opt = 1
        depth = qsi    
        example_square_MWRu.itL = 0
        example_square_MWRu.grain_shear = False
        example_square_MWRu.run_one_step(run_id = 0)      
        E = example_square_MWRu._erosion(n, depth, slope)
        expected_E = 0.144256
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)
        
    def test_inertial_normal_1(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.087489
        example_square_MWRu.slpc = 0.01
        depth = qsi
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(run_id = 0)
        att_in = {'particle__diameter': 0.25}               
        E = example_square_MWRu._erosion(n, depth, slope, att_in = att_in)
        expected_E = 0.065142
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)

    def test_inertial_normal_2(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.087489
        example_square_MWRu.slpc = 0.1
        depth = qsi
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(run_id = 0)
        att_in = {'particle__diameter': 0.25}                
        E = example_square_MWRu._erosion(n, depth, slope, att_in = att_in)
        expected_E = 0.065142
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)

    def test_inertial_boundary_1(self, example_square_MWRu):
        """"""
        n = 24
        qsi = 2
        slope = 0.05
        example_square_MWRu.slpc = 0.05
        depth = qsi
        example_square_MWRu.itL = 0
        example_square_MWRu.run_one_step(run_id = 0)
        att_in = {'particle__diameter': 0.25}                
        E = example_square_MWRu._erosion(n, depth, slope, att_in = att_in)
        expected_E = 0.058276
        np.testing.assert_allclose(E[0], expected_E, rtol = 1e-4)

# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_aggradation(object):

    def test_deposit_L_normal_1(self, example_square_MWRu):
        qsi = 2
        slpn = 0.02
        D = example_square_MWRu._deposit_L_metric(qsi, slpn)
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
        n = 24
        qsi = 2
        zi = 50
        zo = 50.1
        flat_dem = np.ones(example_square_MWRu._grid.number_of_nodes)*zo
        example_square_MWRu._grid.at_node['topographic__elevation'] = flat_dem
        example_square_MWRu._grid.at_node['topographic__elevation'][n] = zi
        D = example_square_MWRu._deposit_friction_angle(qsi, n)
        expected_D = 1.066667
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  
        
    def test_deposit_friction_angle_normal_2(self, example_square_MWRu):
        n = 24
        qsi = 2
        zi = 50
        zo = 49.9
        flat_dem = np.ones(example_square_MWRu._grid.number_of_nodes)*zo
        example_square_MWRu._grid.at_node['topographic__elevation'] = flat_dem
        example_square_MWRu._grid.at_node['topographic__elevation'][n] = zi
        D = example_square_MWRu._deposit_friction_angle(qsi, n)
        expected_D = 0.8
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  

    def test_deposit_friction_angle_boundary_1(self, example_square_MWRu):
        n = 24
        qsi = 2
        zi = 50
        zo = 50
        flat_dem = np.ones(example_square_MWRu._grid.number_of_nodes)*zo
        example_square_MWRu._grid.at_node['topographic__elevation'] = flat_dem
        example_square_MWRu._grid.at_node['topographic__elevation'][n] = zi
        D = example_square_MWRu._deposit_friction_angle(qsi, n)
        expected_D = 0.95
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  
    
    def test_deposit_friction_angle_special_1(self, example_square_MWRu):
        n = 24
        qsi = 2
        zi = 50
        zo = 47
        flat_dem = np.ones(example_square_MWRu._grid.number_of_nodes)*zo
        example_square_MWRu._grid.at_node['topographic__elevation'] = flat_dem
        example_square_MWRu._grid.at_node['topographic__elevation'][n] = zi
        D = example_square_MWRu._deposit_friction_angle(qsi, n)
        expected_D = 0
        np.testing.assert_allclose(D, expected_D,rtol = 1e-4)  

    def test_aggradation_normal_1(self, example_square_MWRu):
        example_square_MWRu.deposition_rule == "both"
        qsi = 2
        n = 24
        zi = 50
        zo = 50.1
        flat_dem = np.ones(example_square_MWRu._grid.number_of_nodes)*zo
        example_square_MWRu._grid.at_node['topographic__elevation'] = flat_dem
        example_square_MWRu._grid.at_node['topographic__elevation'][n] = zi
        example_square_MWRu._update_topographic_slope()
        slpn = example_square_MWRu._grid.at_node['topographic__steepest_slope'][n].max()
        A = example_square_MWRu._aggradation(qsi,slpn,n)                
        A_L = example_square_MWRu._deposit_L_metric(qsi, slpn)
        A_f = example_square_MWRu._deposit_friction_angle(qsi, n)
        expected_A = min(A_L,A_f)
        np.testing.assert_allclose(A, expected_A, rtol = 1e-4)
        
    def test_aggradation_normal_2(self, example_square_MWRu):
        example_square_MWRu.deposition_rule == "both"
        qsi = 2
        n = 24
        zi = 50
        zo = 49.9
        flat_dem = np.ones(example_square_MWRu._grid.number_of_nodes)*zo
        example_square_MWRu._grid.at_node['topographic__elevation'] = flat_dem
        example_square_MWRu._grid.at_node['topographic__elevation'][n] = zi
        example_square_MWRu._update_topographic_slope()
        slpn = example_square_MWRu._grid.at_node['topographic__steepest_slope'][n].max()
        A = example_square_MWRu._aggradation(qsi,slpn,n)                
        A_L = example_square_MWRu._deposit_L_metric(qsi, slpn)
        A_f = example_square_MWRu._deposit_friction_angle(qsi, n)
        expected_A = min(A_L,A_f)
        np.testing.assert_allclose(A, expected_A, rtol = 1e-4)

    def test_determine_zo_normal_1(self, example_square_MWRu):
        example_square_MWRu.itL = 1
        example_square_MWRu.run_one_step(run_id = 0)        
        n = 24
        qsi = 0.2
        zi = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        zo = example_square_MWRu._determine_zo(n, zi, qsi)
        expected_zo = 5
        np.testing.assert_allclose(zo, expected_zo,rtol = 1e-4) 

    def test_determine_zo_normal_2(self, example_square_MWRu):
        example_square_MWRu.itL = 1
        example_square_MWRu.run_one_step(run_id = 0)
        
        n = 24
        qsi = 2
        zi = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        zo = example_square_MWRu._determine_zo(n, zi, qsi)
        expected_zo = 5
        np.testing.assert_allclose(zo, expected_zo,rtol = 1e-4) 

    def test_determine_zo_boundary_1(self, example_square_MWRu):
        example_square_MWRu.itL = 1
        example_square_MWRu.run_one_step(run_id = 0)
        
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
        example_square_MWRu.run_one_step(run_id = 0)
        
        # make node 24 a pit
        example_square_MWRu._grid.at_node['topographic__elevation'][24] = 3
        n = 24
        qsi = 0
        zi = example_square_MWRu._grid.at_node['topographic__elevation'][n]
        zo = example_square_MWRu._determine_zo(n, zi, qsi)
        expected_zo = None
        assert(zo, expected_zo)            


# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_attributes_in(object):

    def test_normal_values_1(self, example_square_MWRu):
        """"""
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)

        n = 24
        qsi = np.sum(example_square_MWRu.arqso[example_square_MWRu.arn == n])
        att_in = example_square_MWRu._attributes_in(n,qsi)
        
        expected_pd_in = 0.161438
        expected_oc_in = 0.074791
        
        np.testing.assert_allclose(att_in['particle__diameter'], expected_pd_in,rtol = 1e-4)
        np.testing.assert_allclose(att_in['organic__content'], expected_oc_in,rtol = 1e-4)
        
    def test_normal_values_2(self, example_square_MWRu):
        """"""

        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)

        n = 31
        qsi = np.sum(example_square_MWRu.arqso[example_square_MWRu.arn == n])
        att_in = example_square_MWRu._attributes_in(n,qsi)

        expected_pd_in = 0.15751
        expected_oc_in = 0.074886
        
        np.testing.assert_allclose(att_in['particle__diameter'], expected_pd_in,rtol = 1e-4)
        np.testing.assert_allclose(att_in['organic__content'], expected_oc_in,rtol = 1e-4)
        
        
    def test_special_values_1(self, example_square_MWRu):
        """no incoming volume"""
        example_square_MWRu.itL = 0       
        example_square_MWRu.run_one_step(run_id = 0)        
        
        n = 24
        qsi = np.sum(example_square_MWRu.arqso[example_square_MWRu.arn == n])
        att_in = example_square_MWRu._attributes_in(n,qsi)

        expected_pd_in = 0
        expected_oc_in = 0
        
        np.testing.assert_allclose(att_in['particle__diameter'], expected_pd_in,rtol = 1e-4)
        np.testing.assert_allclose(att_in['organic__content'], expected_oc_in,rtol = 1e-4)
        

    def test_bad_values_1(self, example_square_MWRu):
        """incoming is np.nan"""
        example_square_MWRu.itL = 8       
        example_square_MWRu.run_one_step(run_id = 0)   

        with pytest.raises(ValueError) as exc_info:
            n = 24
            qsi = np.nan
            att_in = example_square_MWRu._attributes_in(n,qsi)

        assert exc_info.match("in-flowing flux is nan or inf")    
        
        
#@pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_attribute_out(object):
    def test_normal_values_1(self, example_square_MWRu):
        # use inputs from MWR 
        example_square_MWRu.itL = 1       
        example_square_MWRu.run_one_step(run_id = 0)
        qsi = example_square_MWRu.nudat[:,3][0]
        E = example_square_MWRu.nudat[:,4][0]
        A = example_square_MWRu.nudat[:,5][0]
        att_up = example_square_MWRu.nudat[:,7][0]
        att_in = example_square_MWRu.nudat[:,8][0]
        
        att_out = example_square_MWRu._attribute_out(att_up,att_in,qsi,E,A)
        
        expected_pd_out = np.array([0.157470803])
        expected_oc_out = np.array([0.075915194])
    
        np.testing.assert_allclose(att_out['particle__diameter'], expected_pd_out,rtol = 1e-4)
        np.testing.assert_allclose(att_out['organic__content'], expected_oc_out,rtol = 1e-4)
        
    def test_normal_values_2(self, example_square_MWRu):
        # make up values
        qsi = 3
        E = 2
        A = 1
        att_up = {'particle__diameter': 0.5, 'organic__content': 0.05}
        att_in = {'particle__diameter': 0.25, 'organic__content': 0.2}
       
        att_out = example_square_MWRu._attribute_out(att_up,att_in,qsi,E,A)
        
        expected_pd_out = np.array([0.375])
        expected_oc_out = np.array([0.125])
    
        np.testing.assert_allclose(att_out['particle__diameter'], expected_pd_out,rtol = 1e-4)
        np.testing.assert_allclose(att_out['organic__content'], expected_oc_out,rtol = 1e-4)
        
        
    def test_bad_values_1(self, example_square_MWRu):        
        with pytest.raises(ValueError) as exc_info:
            qsi = 3
            E = 2
            A = 1
            att_up = {'particle__diameter': 0.5, 'organic__content': 0.05}
            att_in = {'particle__diameter': np.nan, 'organic__content': 0.2}
               
            att_out = example_square_MWRu._attribute_out(att_up,att_in,qsi,E,A)           

        assert exc_info.match("out-flowing {} is zero, negative, nan or inf".format('particle__diameter'))        
        

    def test_bad_values_2(self,example_square_MWRu):        
        with pytest.raises(ValueError) as exc_info:
            qsi = 3
            E = 2
            A = 1
            att_up = {'particle__diameter': 0.5, 'organic__content': np.nan}
            att_in = {'particle__diameter': 0.25, 'organic__content': 0.2}
               
            att_out = example_square_MWRu._attribute_out(att_up,att_in,qsi,E,A)           

        assert exc_info.match("out-flowing {} is zero, negative, nan or inf".format('organic__content'))       


# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")         
class Test_attribute_node:
    def test_normal_values_1(self, example_square_MWRu):
        # normal values
        n = 24
        att_in = {'particle__diameter': 0.25, 'organic__content': 0.2}
        A = 0.5
        E = 0.5
        
        att_node = example_square_MWRu._attributes_node(n,att_in,E,A)
        expected_att_node = 0.240913
    
        np.testing.assert_allclose(att_node['particle__diameter'], expected_att_node,rtol = 1e-4)

    def test_normal_values_2(self, example_square_MWRu):
        # normal values
        n = 24
        att_in = {'particle__diameter': 0.25, 'organic__content': 0.2}
        A = 0.3
        E = 1
                 
        att_node = example_square_MWRu._attributes_node(n,att_in,E,A)
        
        expected_att_node = 0.25
    
        np.testing.assert_allclose(att_node['particle__diameter'], expected_att_node,rtol = 1e-4)

    def test_special_values_1(self, example_square_MWRu):
        # deposition depth is 0
        n = 24
        att_in = {'particle__diameter': 0.25, 'organic__content': 0.2}
        A = 0
        E = 1    

        att_node = example_square_MWRu._attributes_node(n,att_in,E,A)

        expected_att_node = 0
    
        np.testing.assert_allclose(att_node['particle__diameter'], expected_att_node, rtol = 1e-4)

    def test_bad_values_1(self, example_square_MWRu):
        # incoming particle diameter is np.nan
        with pytest.raises(ValueError) as exc_info:
            n = 24
            att_in = {'particle__diameter': np.nan, 'organic__content': 0.2}
            A = 0.5
            E = 0.5
            # grid node depth = 1, node particle diameter = 0.075 
            
            att_node = example_square_MWRu._attributes_node(n,att_in,E,A)
            
        assert exc_info.match("node particle diameter is negative, nan or inf")

    def test_bad_values_2(self, example_square_MWRu):
        # incoming particle diameter is np.inf
        with pytest.raises(ValueError) as exc_info:
            n = 24
            att_in = {'particle__diameter': np.inf, 'organic__content': 0.2}
            A = 0.5
            E = 0.5
            
            att_node = example_square_MWRu._attributes_node(n,att_in,E,A)
            
        assert exc_info.match("node particle diameter is negative, nan or inf")  
        
