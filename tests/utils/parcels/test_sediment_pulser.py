import pytest
import numpy as np
import pandas as pd

# from landlab.utils.parcels import SedimentPulserEachParcel, SedimentPulserAtLinks

from landlab.utils.parcels.sediment_pulser_at_links_v2 import SedimentPulserAtLinks
from landlab.utils.parcels.sediment_pulser_each_parcel_v2 import SedimentPulserEachParcel

from landlab import RasterModelGrid

from landlab.utils.parcels import (
                                    calc_lognormal_distribution_parameters,
                                    )

def always_time_to_pulse(time):
    return True

def time_to_pulse_list(time):
    Ptime = [19,20,22,23,24,75,76]
    return time in Ptime



def test_calc_lognormal_distribution_parameters():
    """check lognormal distribution mean and standard deviation with values 
    calculated in excel"""
    mu_x = 0.1; sigma_x = 0.05
    mu_y, sigma_y = calc_lognormal_distribution_parameters(mu_x,sigma_x)
    mu_y_e = -2.41416; sigma_y_e = 0.472381
    np.testing.assert_allclose(np.array([mu_y,sigma_y]), 
                               np.array([mu_y_e, sigma_y_e]), rtol = 1e-4)
     
    mu_x = 0.33; sigma_x = 0.33
    mu_y, sigma_y = calc_lognormal_distribution_parameters(mu_x,sigma_x)
    mu_y_e = -1.45524; sigma_y_e = 0.832555
    np.testing.assert_allclose(np.array([mu_y,sigma_y]), 
                               np.array([mu_y_e, sigma_y_e]), rtol = 1e-4)    
    
# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_SedimentPulserAtLinks(object):
    def test_normal_1(self, example_nmg):
        """only time specified, links and number parcels specified,
        should use defaults in base class"""
        grid = example_nmg; time_to_pulse = always_time_to_pulse
        np.random.seed(seed=5)

        make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
        
        time = 11
        links = [0]
        n_parcels_at_link = [10]
        parcels = make_pulse(time = time, links = links, n_parcels_at_link = n_parcels_at_link)  
                
        # create expected values and get values from datarecord
        GEe = np.expand_dims(["link"]*10, axis=1)
        GE = parcels.dataset['grid_element']
        EIe = np.expand_dims(np.zeros(10).astype(int), axis=1)
        EI = parcels.dataset['element_id']
        SLe = np.zeros(10).astype(int)
        SL = parcels.dataset['starting_link']
        ARe = np.expand_dims(np.ones(10)*0, axis=1)
        AR = parcels.dataset['abrasion_rate']
        De = np.expand_dims(np.ones(10)*2650, axis=1)
        D = parcels.dataset['density']
        TAe = np.expand_dims(np.ones(10)*time, axis=1)
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(10), axis=1)
        AL = parcels.dataset['active_layer']
        LLe = np.array([[ 0.44130922],[ 0.15830987],[ 0.87993703],[ 0.27408646],
                        [ 0.41423502],[ 0.29607993],[ 0.62878791],[ 0.57983781],
                        [ 0.5999292 ],[ 0.26581912]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.05475929],[ 0.0356878 ],[ 0.16503787],[ 0.03728132],
                       [ 0.04556139],[ 0.10310904],[ 0.02589628],[ 0.03088314],
                       [ 0.04757508],[ 0.0357076 ]])
        D = parcels.dataset['D']
        Ve = np.expand_dims(np.ones(10)*0.5, axis=1)
        V = parcels.dataset['volume']        
        
        # test
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
        """only D50 specified, all other parcel attributes should 
        use defaults in base class"""

        grid = example_nmg; time_to_pulse = always_time_to_pulse
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
        time = 11
        links = [2, 6]
        n_parcels_at_link = [2, 3]
        D50 = [0.3, 0.12]
        parcels = make_pulse(time = time, links=links, n_parcels_at_link = n_parcels_at_link,
                           D50 = D50)

        # create expected values and get values from datarecord
        GEe = np.expand_dims(["link"]*5, axis=1)
        GE = parcels.dataset['grid_element']
        EIe = np.expand_dims(np.array([2,2,6,6,6]), axis=1)
        EI = parcels.dataset['element_id']
        SLe = np.array([2,2,6,6,6])
        SL = parcels.dataset['starting_link']
        ARe = np.expand_dims(np.ones(5)*0, axis=1)
        AR = parcels.dataset['abrasion_rate']
        De = np.expand_dims(np.ones(5)*2650, axis=1)
        D = parcels.dataset['density']
        TAe = np.expand_dims(np.ones(5)*time, axis=1)
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(5), axis=1)
        AL = parcels.dataset['active_layer']
        LLe = np.array([[ 0.2968005 ], [ 0.18772123], [ 0.08074127], [ 0.7384403 ],
                        [ 0.44130922]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.31194296], [ 0.28881969], [ 0.21180913], [ 0.10941075], 
                       [ 0.11960177]])
        D = parcels.dataset['D']
        Ve = np.expand_dims(np.ones(5)*0.5, axis=1)
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
        
    def test_normal_3(self, example_nmg):
        """two pulses. First, only time, links and number of parcels specified, 
        uses defaults in base class for all other parcel attributes
        second, two parcels in link two and three parcels in link six 
        are added and all attributes specified"""
        
        grid = example_nmg; time_to_pulse = always_time_to_pulse
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
        
        time = 11
        links = [0]
        n_parcels_at_link = [2]
        parcels = make_pulse(time = time, links = links,
                           n_parcels_at_link = n_parcels_at_link)
        
        time = 12
        links = [2, 6]
        n_parcels_at_link = [2, 3]
        D50 = [0.3, 0.12]
        D_sd =  [0.2, 0.1]   
        parcel_volume = [1, 0.5]
        rho_sediment = [2650, 2500]
        abrasion_rate = [.1, .3]
        parcels = make_pulse(time = time, links = links, n_parcels_at_link = n_parcels_at_link,
                           D50 = D50, D_sd = D_sd, parcel_volume = parcel_volume,
                           rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)
        
        # create expected values and get values from datarecord
        GEe = [['link', np.nan], ['link', np.nan], [np.nan, 'link'],[np.nan, 'link'],
                        [np.nan, 'link'], [np.nan, 'link'],[np.nan, 'link']]
        GE = parcels.dataset['grid_element']
        EIe = np.array([[  0.,  np.nan], [  0.,  np.nan], [ np.nan,   2.], [ np.nan,   2.],
                        [ np.nan,   6.], [ np.nan,   6.], [ np.nan,   6.]])
        EI = parcels.dataset['element_id']
        SLe = np.array([ 0.,  0.,  2.,  2.,  6.,  6.,  6.])
        SL = parcels.dataset['starting_link']
        ARe = np.array([[ 0. ,  np.nan], [ 0. ,  np.nan], [ np.nan,  0.1], [ np.nan,  0.1],
                        [ np.nan,  0.3], [ np.nan,  0.3], [ np.nan,  0.3]])
        AR = parcels.dataset['abrasion_rate']
        De = np.array([[ 2650.,    np.nan], [ 2650.,    np.nan], [   np.nan,  2650.],
                       [   np.nan,  2650.], [   np.nan,  2500.], [   np.nan,  2500.],
                       [   np.nan,  2500.]])
        D = parcels.dataset['density']
        TAe = np.array([[ 11.,  np.nan], [ 11.,  np.nan], [ np.nan,  12.], [ np.nan,  12.],
                        [ np.nan,  12.], [ np.nan,  12.], [ np.nan,  12.]])
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.array([[  1.,  np.nan], [  1.,  np.nan], [ np.nan,   1.], [ np.nan,   1.],
                        [ np.nan,   1.], [ np.nan,   1.], [ np.nan,   1.]])
        AL = parcels.dataset['active_layer']
        LLe = np.array([[ 0.20671916,         np.nan], [ 0.91861091,         np.nan],
                        [        np.nan,  0.08074127], [        np.nan,  0.7384403 ],
                        [        np.nan,  0.44130922], [        np.nan,  0.15830987],
                        [        np.nan,  0.87993703]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.05475929,         np.nan], [ 0.0356878 ,         np.nan],
                       [        np.nan,  1.09001571], [        np.nan,  0.21423009],
                       [        np.nan,  0.09982434], [        np.nan,  0.29090581],
                       [        np.nan,  0.04763353]])
        D = parcels.dataset['D']
        Ve = np.array([[ 0.5,  np.nan], [ 0.5,  np.nan], [ np.nan,  1. ], [ np.nan,  1. ],
                       [ np.nan,  0.5], [ np.nan,  0.5], [ np.nan,  0.5]])
        V = parcels.dataset['volume']        
        
        # test
        assert str(GE.values.tolist()) == str(GEe)
        np.testing.assert_allclose(EI, EIe, rtol = 1e-4)
        np.testing.assert_allclose(SL, SLe, rtol = 1e-4)
        np.testing.assert_allclose(AR, ARe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)
        np.testing.assert_allclose(TA, TAe, rtol = 1e-4)
        np.testing.assert_allclose(AL, ALe, rtol = 1e-4)
        np.testing.assert_allclose(LL, LLe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)        
        np.testing.assert_allclose(V, Ve, rtol = 1e-4)  

    def test_special_1(self, example_nmg):
        """user entered time is not a pulse time, calling instance returns
        the original parcels datarecord, which is None if there is no
        original datarecord"""

        grid = example_nmg; time_to_pulse = time_to_pulse_list
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
        
        time = 11
        links = [0]
        n_parcels_at_link = [2]
        parcels = make_pulse(time = time, links = links,
                           n_parcels_at_link = n_parcels_at_link)
        
        assert parcels == None
        
    
# @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
class Test_SedimentPulserEachParcel(object):
    def test_normal_1(self, example_nmg):
        """minimum attributes specified in Pulse, attributes should use
        defaults specified at instantiation"""
    
        grid = example_nmg
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserEachParcel(grid)
                
        PulseDF = pd.DataFrame({'pulse_volume': [0.2, 1, 1.1, 0.5],
                                          'link_#': [1, 3, 5, 2],
                                          'normalized_downstream_distance': [0.8,0.7,0.5,0.2]})
        time = 7
        parcels = make_pulse(time, PulseDF)
               
        # create expected values and get values from datarecord
        GEe = np.expand_dims(["link"]*7, axis=1)
        GE = parcels.dataset['grid_element']
        EIe = np.expand_dims(np.array([1,3,3,5,5,5,2]), axis=1)
        EI = parcels.dataset['element_id']
        SLe = np.array([1,3,3,5,5,5,2])
        SL = parcels.dataset['starting_link']
        ARe = np.expand_dims(np.ones(7)*0, axis=1)
        AR = parcels.dataset['abrasion_rate']
        De = np.expand_dims(np.ones(7)*2650, axis=1)
        D = parcels.dataset['density']
        TAe = np.expand_dims(np.ones(7)*time, axis=1)
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(7), axis=1)
        AL = parcels.dataset['active_layer']
        LLe = np.array([[0.8], [0.7], [0.7],[0.5],[0.5],[0.5],[0.2]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.05475929],[ 0.0356878 ],[ 0.16503787],[ 0.03728132],
                       [ 0.04556139],[ 0.10310904],[ 0.02589628]])
        D = parcels.dataset['D']
        Ve = np.array([[0.2],[0.5],[0.5],[0.5],[0.5],[0.1],[0.5]])
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
        """all attributes specified in Pulse"""
        
        grid = example_nmg
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserEachParcel(grid)        
        
        PulseDF = pd.DataFrame({'pulse_volume': [0.2, 1, 1.1, 0.5],
                                  'link_#': [1, 3, 5, 2],
                                  'normalized_downstream_distance': [0.8,0.7,0.5,0.2],
                                  'D50': [0.15, 0.2, 0.22, 0.1],
                                  'D_sd': [0, 0, 0, 0],
                                  'abrasion_rate': [0.01, 0.02, 0.005, 0.03],
                                  'density': [2650, 2300, 2750, 2100],
                                  'parcel_volume': [0.1, 1, 1, 0.2]
                                  })
        time = 7
        parcels = make_pulse(time, PulseDF)
        
        
        # create expected values and get values from datarecord
        GEe = np.expand_dims(["link"]*8, axis=1)
        GE = parcels.dataset['grid_element']
        EIe = np.expand_dims(np.array([1,1,3,5,5,2,2,2]), axis=1)
        EI = parcels.dataset['element_id']
        SLe = np.array([1,1,3,5,5,2,2,2])
        SL = parcels.dataset['starting_link']
        ARe = np.expand_dims(np.array([0.01,0.01,0.02,0.005,0.005,0.03,0.03,
                                       0.03]), axis=1)
        AR = parcels.dataset['abrasion_rate']
        De = np.expand_dims(np.array([2650, 2650, 2300, 2750, 2750, 2100, 2100,
                                       2100]), axis=1)
        D = parcels.dataset['density']
        TAe = np.expand_dims(np.ones(8)*time, axis=1)
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(8), axis=1)
        AL = parcels.dataset['active_layer']
        LLe = np.array([[0.8], [0.8], [0.7],[0.5],[0.5],[0.2],[0.2],[0.2]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.15], [ 0.15], [ 0.2 ], [ 0.22], [ 0.22], [ 0.1 ], 
                       [ 0.1 ], [ 0.1 ]])
        D = parcels.dataset['D']
        Ve = np.array([[0.1],[0.1],[1],[1],[0.1],[0.2],[0.2],[0.1]])
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
        
        
    def test_normal_3(self, example_nmg):
        """two pulses. First, only minimum attributes specified, 
        second, two parcels in link two and three parcels in link six 
        are added and all attributes specified"""
        
        grid = example_nmg
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserEachParcel(grid)        
        
        PulseDF = pd.DataFrame({'pulse_volume': [0.2, 1],
                                  'link_#': [1, 3],
                                  'normalized_downstream_distance': [0.8,0.7],
                                  })
        time = 7
        parcels = make_pulse(time, PulseDF)
        
        PulseDF = pd.DataFrame({'pulse_volume': [1.1, 0.5],
                                  'link_#': [5, 2],
                                  'normalized_downstream_distance': [0.5,0.2],
                                  'D50': [0.22, 0.1],
                                  'D_sd': [0, 0],
                                  'abrasion_rate': [0.005, 0.03],
                                  'density': [2750, 2100],
                                  'parcel_volume': [1, 0.2]
                                  })
        time = 8
        parcels = make_pulse(time, PulseDF)      
        
        # create expected values and get values from datarecord
        GEe = [['link', np.nan],['link', np.nan],['link', np.nan],[np.nan, 'link'],
               [np.nan, 'link'],[np.nan, 'link'],[np.nan, 'link'],[np.nan, 'link']]
        GE = parcels.dataset['grid_element']
        EIe = np.array([[  1.,  np.nan],[  3.,  np.nan],[  3.,  np.nan],
                        [ np.nan,   5.],[ np.nan,   5.],
                        [ np.nan,   2.],[ np.nan,   2.],[ np.nan,   2.]])
        EI = parcels.dataset['element_id']
        SLe = np.array([ 1, 3, 3, 5, 5, 2, 2, 2])
        SL = parcels.dataset['starting_link']
        ARe = np.array([[ 0.   ,    np.nan],[ 0.   ,    np.nan],[ 0.   ,    np.nan],
                        [   np.nan,  0.005],[   np.nan,  0.005],[   np.nan,  0.03 ],
                        [   np.nan,  0.03 ],[   np.nan,  0.03 ]])
        AR = parcels.dataset['abrasion_rate']
        De = np.array([[ 2650.,    np.nan],[ 2650.,    np.nan],[ 2650.,    np.nan], 
                       [   np.nan,  2650.],[   np.nan,  2650.],[   np.nan,  2650.],
                       [   np.nan,  2650.],[   np.nan,  2650.]])
        D = parcels.dataset['density']
        TAe = np.array([[  7.,  np.nan],[  7.,  np.nan],[  7.,  np.nan],[ np.nan,   8.],
                        [ np.nan,   8.],[ np.nan,   8.],[ np.nan,   8.],[ np.nan,   8.]])
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.array([[  1.,  np.nan],[  1.,  np.nan],[  1.,  np.nan],[ np.nan,   1.],
                        [ np.nan,   1.],[ np.nan,   1.],[ np.nan,   1.],[ np.nan,   1.]])
        AL = parcels.dataset['active_layer']
        LLe = np.array([[ 0.8,  np.nan],[ 0.7,  np.nan],[ 0.7,  np.nan],[ np.nan,  0.5],
                        [ np.nan,  0.5],[ np.nan,  0.2],[ np.nan,  0.2],[ np.nan,  0.2]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.05475929, np.nan], [ 0.0356878 , np.nan],[ 0.16503787, np.nan],
                       [ np.nan, 0.22], [np.nan, 0.22], [np.nan, 0.1], [np.nan,  0.1], [np.nan, 0.1]])
        D = parcels.dataset['D']
        Ve = np.array([[ 0.2,  np.nan], [ 0.5,  np.nan], [ 0.5,  np.nan], [ np.nan,  1. ],
                       [ np.nan,  0.1], [ np.nan,  0.2], [ np.nan,  0.2], [ np.nan,  0.1]])
        V = parcels.dataset['volume']        
        
        # test
        assert str(GE.values.tolist()) == str(GEe)
        np.testing.assert_allclose(EI, EIe, rtol = 1e-4)
        np.testing.assert_allclose(SL, SLe, rtol = 1e-4)
        np.testing.assert_allclose(AR, ARe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)
        np.testing.assert_allclose(TA, TAe, rtol = 1e-4)
        np.testing.assert_allclose(AL, ALe, rtol = 1e-4)
        np.testing.assert_allclose(LL, LLe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)        
        np.testing.assert_allclose(V, Ve, rtol = 1e-4)        
        
        
    def test_normal_4(self, example_nmg):
        """Series of pulses using both the SedimentPulserAtlinks and 
        SedimentPulserEachParcel."""
        
        grid = example_nmg
        np.random.seed(seed=5)
        
        # define the initial parcels datarecord
        make_pulse_links = SedimentPulserAtLinks(grid, time_to_pulse=always_time_to_pulse)
        
        # pulse 1
        time = 7
        links = [0]
        n_parcels_at_link = [2]
        parcels = make_pulse_links(time = time, links = links,
                           n_parcels_at_link = n_parcels_at_link)

        # datarecord is input to SedimentPulserEachParcel, now both pulser
        # instances will point to the same datarecord when called
        make_pulse_pulseDF = SedimentPulserEachParcel(parcels = parcels, grid = grid)
        
        
        # pulse 2
        PulseDF = pd.DataFrame({'pulse_volume': [1.1],
                                  'link_#': [5],
                                  'normalized_downstream_distance': [0.5],
                                  'D50': [0.22],
                                  'D_sd': [0],
                                  'abrasion_rate': [0.005],
                                  'density': [2750],
                                  'parcel_volume': [1]
                                  })
        time = 8
        parcels = make_pulse_pulseDF(time, PulseDF) 

        # pulse 3
        time = 9
        links = [5]
        n_parcels_at_link = [1]
        parcels = make_pulse_links(time = time, links = links,
                           n_parcels_at_link = n_parcels_at_link)
        
        # pulse 4
        PulseDF = pd.DataFrame({'pulse_volume': [0.5],
                          'link_#': [6],
                          'normalized_downstream_distance': [0.2],
                          })
        time = 10
        parcels = make_pulse_pulseDF(time, PulseDF) 

        # create expected values and get values from datarecord
        GEe = [['link', np.nan, np.nan, np.nan],
               ['link', np.nan, np.nan, np.nan],
               [np.nan, 'link', np.nan, np.nan],
               [np.nan, 'link', np.nan, np.nan],
               [np.nan, np.nan, 'link', np.nan],
               [np.nan, np.nan, np.nan, 'link']]
        GE = parcels.dataset['grid_element']
        EIe = np.array([[  0.,  np.nan,  np.nan,  np.nan],
                        [  0.,  np.nan,  np.nan,  np.nan],
                        [ np.nan,   5.,  np.nan,  np.nan],
                        [ np.nan,   5.,  np.nan,  np.nan],
                        [ np.nan,  np.nan,   5.,  np.nan],
                        [ np.nan,  np.nan,  np.nan,   6.]])
        EI = parcels.dataset['element_id']
        SLe = np.array([ 0, 0, 5, 5, 5, 6])
        SL = parcels.dataset['starting_link']
        ARe = np.array([[ 0.   ,    np.nan,    np.nan,    np.nan],
                       [ 0.   ,    np.nan,    np.nan,    np.nan],
                       [   np.nan,  0.005,    np.nan,    np.nan],
                       [   np.nan,  0.005,    np.nan,    np.nan],
                       [   np.nan,    np.nan,  0.   ,    np.nan],
                       [   np.nan,    np.nan,    np.nan,  0.   ]])
        AR = parcels.dataset['abrasion_rate']
        De = np.array([[ 2650.,    np.nan,    np.nan,    np.nan],
                       [ 2650.,    np.nan,    np.nan,    np.nan],
                       [   np.nan,  2650.,    np.nan,    np.nan],
                       [   np.nan,  2650.,    np.nan,    np.nan],
                       [   np.nan,    np.nan,  2650.,    np.nan],
                       [   np.nan,    np.nan,    np.nan,  2650.]])
        D = parcels.dataset['density']
        TAe = np.array([[  7.,  np.nan,  np.nan,  np.nan],
                       [  7.,  np.nan,  np.nan,  np.nan],
                       [ np.nan,   8.,  np.nan,  np.nan],
                       [ np.nan,   8.,  np.nan,  np.nan],
                       [ np.nan,  np.nan,   9.,  np.nan],
                       [ np.nan,  np.nan,  np.nan,  10.]])
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.array([[  1.,  np.nan,  np.nan,  np.nan],
                       [  1.,  np.nan,  np.nan,  np.nan],
                       [ np.nan,   1.,  np.nan,  np.nan],
                       [ np.nan,   1.,  np.nan,  np.nan],
                       [ np.nan,  np.nan,   1.,  np.nan],
                       [ np.nan,  np.nan,  np.nan,   1.]])
        AL = parcels.dataset['active_layer']
        LLe = np.array([[ 0.20671916,         np.nan,         np.nan,         np.nan],
                       [ 0.91861091,         np.nan,         np.nan,         np.nan],
                       [        np.nan,  0.5       ,         np.nan,         np.nan],
                       [        np.nan,  0.5       ,         np.nan,         np.nan],
                       [        np.nan,         np.nan,  0.2968005 ,         np.nan],
                       [        np.nan,         np.nan,         np.nan,  0.2       ]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.05475929,         np.nan,         np.nan,         np.nan],
                       [ 0.0356878 ,         np.nan,         np.nan,         np.nan],
                       [        np.nan,  0.22      ,         np.nan,         np.nan],
                       [        np.nan,  0.22      ,         np.nan,         np.nan],
                       [        np.nan,         np.nan,  0.04556139,         np.nan],
                       [        np.nan,         np.nan,         np.nan,  0.10310904]])
        D = parcels.dataset['D']
        Ve = np.array([[ 0.5,  np.nan,  np.nan,  np.nan],
                       [ 0.5,  np.nan,  np.nan,  np.nan],
                       [ np.nan,  1. ,  np.nan,  np.nan],
                       [ np.nan,  0.1,  np.nan,  np.nan],
                       [ np.nan,  np.nan,  0.5,  np.nan],
                       [ np.nan,  np.nan,  np.nan,  0.5]])
        V = parcels.dataset['volume']  

        # test
        assert str(GE.values.tolist()) == str(GEe)
        np.testing.assert_allclose(EI, EIe, rtol = 1e-4)
        np.testing.assert_allclose(SL, SLe, rtol = 1e-4)
        np.testing.assert_allclose(AR, ARe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)
        np.testing.assert_allclose(TA, TAe, rtol = 1e-4)
        np.testing.assert_allclose(AL, ALe, rtol = 1e-4)
        np.testing.assert_allclose(LL, LLe, rtol = 1e-4)
        np.testing.assert_allclose(D, De, rtol = 1e-4)        
        np.testing.assert_allclose(V, Ve, rtol = 1e-4) 

    def test_special_1(self, example_nmg):
        """test exception raised if instance is called without specifying the
        pulserDF"""
        
        grid = example_nmg
        make_pulse = SedimentPulserEachParcel(grid)               
        PulseDF = None
        time = 7
        
        with pytest.raises(ValueError) as exc_info:
            parcels = make_pulse(time, PulseDF)  
        msg_e = "PulseDF was not specified"
        assert exc_info.match(msg_e)
        
        
    def test_special_2(self, example_nmg):
        """test that calling with an empty PulseDF returns the existing
        datarecord"""
        
        grid = example_nmg
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserEachParcel(grid)
                
        PulseDF = pd.DataFrame({'pulse_volume': [0.2, 1, 1.1, 0.5],
                                          'link_#': [1, 3, 5, 2],
                                          'normalized_downstream_distance': [0.8,0.7,0.5,0.2]})
        # call SedimentPulserEachParcel using setup in normal_test_1
        time1 = 7
        parcels = make_pulse(time1, PulseDF)
        
        # call again, using an empty PulserDF
        time2 = 8
        parcels = make_pulse(time2, pd.DataFrame([]))
               
        # create expected values and get values from datarecord
        GEe = np.expand_dims(["link"]*7, axis=1)
        GE = parcels.dataset['grid_element']
        EIe = np.expand_dims(np.array([1,3,3,5,5,5,2]), axis=1)
        EI = parcels.dataset['element_id']
        SLe = np.array([1,3,3,5,5,5,2])
        SL = parcels.dataset['starting_link']
        ARe = np.expand_dims(np.ones(7)*0, axis=1)
        AR = parcels.dataset['abrasion_rate']
        De = np.expand_dims(np.ones(7)*2650, axis=1)
        D = parcels.dataset['density']
        TAe = np.expand_dims(np.ones(7)*time1, axis=1)
        TA = parcels.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(7), axis=1)
        AL = parcels.dataset['active_layer']
        LLe = np.array([[0.8], [0.7], [0.7],[0.5],[0.5],[0.5],[0.2]])
        LL = parcels.dataset['location_in_link']
        De = np.array([[ 0.05475929],[ 0.0356878 ],[ 0.16503787],[ 0.03728132],
                       [ 0.04556139],[ 0.10310904],[ 0.02589628]])
        D = parcels.dataset['D']
        Ve = np.array([[0.2],[0.5],[0.5],[0.5],[0.5],[0.1],[0.5]])
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
              