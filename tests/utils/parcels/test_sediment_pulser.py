import pytest
import numpy as np

from landlab.utils.parcels import SedimentPulserEachParcel, SedimentPulserAtLinks

from landlab import RasterModelGrid

from landlab.utils.parcels import (
                                    calc_lognormal_distribution_parameters,
                                    )

def always_time_to_pulse(time):
    return True

def time_to_pulse_list(time):
    Ptime = [19,20,22,23,24,75,76]
    return time in Ptime


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
        pulse = make_pulse(time = time, links = links, n_parcels_at_link = n_parcels_at_link)  
                
        # create expected values and get values from instance
        GEe = np.expand_dims(["link"]*10, axis=1)
        GE = pulse.dataset['grid_element']
        EIe = np.expand_dims(np.zeros(10).astype(int), axis=1)
        EI = pulse.dataset['element_id']
        SLe = np.zeros(10).astype(int)
        SL = pulse.dataset['starting_link']
        ARe = np.expand_dims(np.ones(10)*0, axis=1)
        AR = pulse.dataset['abrasion_rate']
        De = np.expand_dims(np.ones(10)*2650, axis=1)
        D = pulse.dataset['density']
        TAe = np.expand_dims(np.ones(10)*time, axis=1)
        TA = pulse.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(10), axis=1)
        AL = pulse.dataset['active_layer']
        LLe = np.array([[ 0.44130922],[ 0.15830987],[ 0.87993703],[ 0.27408646],
                        [ 0.41423502],[ 0.29607993],[ 0.62878791],[ 0.57983781],
                        [ 0.5999292 ],[ 0.26581912]])
        LL = pulse.dataset['location_in_link']
        De = np.array([[ 0.05475929],[ 0.0356878 ],[ 0.16503787],[ 0.03728132],
                       [ 0.04556139],[ 0.10310904],[ 0.02589628],[ 0.03088314],
                       [ 0.04757508],[ 0.0357076 ]])
        D = pulse.dataset['D']
        Ve = np.expand_dims(np.ones(10)*0.5, axis=1)
        V = pulse.dataset['volume']        
        
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
        d50 = [0.3, 0.12]
        pulse = make_pulse(time = time, links=links, n_parcels_at_link = n_parcels_at_link,
                           d50 = d50)

        # create expected values and get values from instance
        GEe = np.expand_dims(["link"]*5, axis=1)
        GE = pulse.dataset['grid_element']
        EIe = np.expand_dims(np.array([2,2,6,6,6]), axis=1)
        EI = pulse.dataset['element_id']
        SLe = np.array([2,2,6,6,6])
        SL = pulse.dataset['starting_link']
        ARe = np.expand_dims(np.ones(5)*0, axis=1)
        AR = pulse.dataset['abrasion_rate']
        De = np.expand_dims(np.ones(5)*2650, axis=1)
        D = pulse.dataset['density']
        TAe = np.expand_dims(np.ones(5)*time, axis=1)
        TA = pulse.dataset['time_arrival_in_link']
        ALe = np.expand_dims(np.ones(5), axis=1)
        AL = pulse.dataset['active_layer']
        LLe = np.array([[ 0.2968005 ], [ 0.18772123], [ 0.08074127], [ 0.7384403 ],
                        [ 0.44130922]])
        LL = pulse.dataset['location_in_link']
        De = np.array([[ 0.31194296], [ 0.28881969], [ 0.21180913], [ 0.10941075], 
                       [ 0.11960177]])
        D = pulse.dataset['D']
        Ve = np.expand_dims(np.ones(5)*0.5, axis=1)
        V = pulse.dataset['volume']  
        
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
        pulse = make_pulse(time = time, links = links,
                           n_parcels_at_link = n_parcels_at_link)
        
        time = 12
        links = [2, 6]
        n_parcels_at_link = [2, 3]
        d50 = [0.3, 0.12]
        std_dev =  [0.2, 0.1]   
        parcel_volume = [1, 0.5]
        rho_sediment = [2650, 2500]
        abrasion_rate = [.1, .3]
        pulse = make_pulse(time = time, links = links, n_parcels_at_link = n_parcels_at_link,
                           d50 = d50, std_dev = std_dev, parcel_volume = parcel_volume,
                           rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)
        
        # create expected values and get values from instance
        GEe = [['link', np.nan], ['link', np.nan], [np.nan, 'link'],[np.nan, 'link'],
                        [np.nan, 'link'], [np.nan, 'link'],[np.nan, 'link']]
        GE = pulse.dataset['grid_element']
        EIe = np.array([[  0.,  np.nan], [  0.,  np.nan], [ np.nan,   2.], [ np.nan,   2.],
                        [ np.nan,   6.], [ np.nan,   6.], [ np.nan,   6.]])
        EI = pulse.dataset['element_id']
        SLe = np.array([ 0.,  0.,  2.,  2.,  6.,  6.,  6.])
        SL = pulse.dataset['starting_link']
        ARe = np.array([[ 0. ,  np.nan], [ 0. ,  np.nan], [ np.nan,  0.1], [ np.nan,  0.1],
                        [ np.nan,  0.3], [ np.nan,  0.3], [ np.nan,  0.3]])
        AR = pulse.dataset['abrasion_rate']
        De = np.array([[ 2650.,    np.nan], [ 2650.,    np.nan], [   np.nan,  2650.],
                       [   np.nan,  2650.], [   np.nan,  2500.], [   np.nan,  2500.],
                       [   np.nan,  2500.]])
        D = pulse.dataset['density']
        TAe = np.array([[ 11.,  np.nan], [ 11.,  np.nan], [ np.nan,  12.], [ np.nan,  12.],
                        [ np.nan,  12.], [ np.nan,  12.], [ np.nan,  12.]])
        TA = pulse.dataset['time_arrival_in_link']
        ALe = np.array([[  1.,  np.nan], [  1.,  np.nan], [ np.nan,   1.], [ np.nan,   1.],
                        [ np.nan,   1.], [ np.nan,   1.], [ np.nan,   1.]])
        AL = pulse.dataset['active_layer']
        LLe = np.array([[ 0.20671916,         np.nan], [ 0.91861091,         np.nan],
                        [        np.nan,  0.08074127], [        np.nan,  0.7384403 ],
                        [        np.nan,  0.44130922], [        np.nan,  0.15830987],
                        [        np.nan,  0.87993703]])
        LL = pulse.dataset['location_in_link']
        De = np.array([[ 0.05475929,         np.nan], [ 0.0356878 ,         np.nan],
                       [        np.nan,  1.09001571], [        np.nan,  0.21423009],
                       [        np.nan,  0.09982434], [        np.nan,  0.29090581],
                       [        np.nan,  0.04763353]])
        D = pulse.dataset['D']
        Ve = np.array([[ 0.5,  np.nan], [ 0.5,  np.nan], [ np.nan,  1. ], [ np.nan,  1. ],
                       [ np.nan,  0.5], [ np.nan,  0.5], [ np.nan,  0.5]])
        V = pulse.dataset['volume']        
        
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
        """user entered time not a pulse time, calling instance returns
        the original parcels datarecord, which is None if there is no
        original datarecord"""

        grid = example_nmg; time_to_pulse = time_to_pulse_list
        np.random.seed(seed=5)
        
        make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
        
        time = 11
        links = [0]
        n_parcels_at_link = [2]
        pulse = make_pulse(time = time, links = links,
                           n_parcels_at_link = n_parcels_at_link)
        
        assert pulse == None
        



def test_calc_lognormal_distribution_parameters():
    """check lognormal distribution mean and sigma with values calculated
    in excel"""
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
    

    
# # @pytest.mark.xfail(reason = "TDD, test class is not yet implemented")
# class Test_SedimentPulserAtLinks(object):
#     def test_normal_1(self, example_nmg):
    
