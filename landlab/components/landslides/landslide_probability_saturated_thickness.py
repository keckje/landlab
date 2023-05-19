import copy

import numpy as np
import scipy.constants
from scipy import interpolate
from statsmodels.distributions.empirical_distribution import ECDF

from landlab.components.landslides.landslide_probability_base import LandslideProbabilityBase

class LandslideProbabilitySaturatedThickness(LandslideProbabilityBase):
    """Landlab component designed to calculate probability of failure at
    each grid node based on the infinite slope stability model
    stability index (Factor of Safety).
    
    Relative wetness and factor-of-safety are based on the infinite slope
    stability model driven by topographic, soil and **saturated thickness** data provided
    by user as inputs to the component. For each node, component simulates mean
    relative wetness as well as the probability of saturation based on Monte Carlo
    simulation of relative wetness where the probability is the number of
    iterations with relative wetness >= 1.0 divided by the number of iterations.
    Probability of failure for each node is also simulated in the Monte Carlo
    simulation as the number of iterations with factor-of-safety <= 1.0
    divided by the number of iterations.
    
    The main method of the LandslideProbabilitySaturatedThickness class is
    `calculate_landslide_probability()``, which calculates the mean soil
    relative wetness, probability of soil saturation, and probability of
    failure at each node based on a Monte Carlo simulation.
    
    **Usage:**
    
    Option 1 - Event soil saturated zone thickness 

    .. code-block:: python

        LandslideProbability(grid,
                             number_of_iterations=250,
                             saturated__thickness_distribution = 'event'


    Option 2 - Lognormal_spatial soil saturated zone thickness 

    .. code-block:: python

        LandslideProbability(grid,
                             number_of_iterations=250,
                             saturated__thickness_distribution = 'lognormal_spatial'
                             saturated__thickness_mean = mean__saturated_thickness_np_array,
                             saturated__thickness_standard_deviation = stdev__saturated_thickness_np_array

    """
    
    # component name
    _name = "Landslide Probability from Soil Saturated Thickness"

    _unit_agnostic = False

    __version__ = "1.1"
    
    _info = LandslideProbabilityBase._info
    
    _info["saturated__thickness"] = {
        "dtype": float,
        "intent": "in",
        "optional": True,
        "units": "m",
        "mapping": "node",
        "doc": "thickness of the soil saturated zone at each node, can be the output from a distributed hydrology model in response to a specific precipitation event",
    }
    

    def __init__(
        self,
        grid,
        number_of_iterations=250,
        g=scipy.constants.g,
        saturated__thickness_distribution = "event",
        saturated__thickness_mean = None,
        saturated__thickness_standard_deviation = None,
        seed=0,
        save = False
        ):
        """
        Parameters
        ----------
        grid: RasterModelGrid
            A raster grid.
        number_of_iterations: int, optional
            Number of iterations to run Monte Carlo simulation (default=250).
        saturated__thickness_distribution: str
             single word indicating distribution of soil saturation zone thickness, 
             either 'event' or 'lognormal_spatial'. 
        saturated__thickness_mean: float, numpy array (m)
            Mean of saturated thickness at each node. Array length equal to number
            of raster model grid nodes. (default = None)
        saturated__thickness_standard_deviation: float, numpy array (m)
            Standard deviation of saturated thickness at each node. Array length 
            equal to number of raster model grid nodes.  (default = None)
        g: float, optional (m/sec^2)
            acceleration due to gravity.
        seed: int, optional
            seed for random number generation. if seed is assigned any value
            other than the default value of zero, it will create different
            sequence. To create a certain sequence repititively, use the same
            value as input for seed.
        """

        super().__init__(grid, 
                         number_of_iterations, 
                         g,
                         seed,
                         save)
        
        self._saturated__thickness_distribution = saturated__thickness_distribution
        self._prep_saturated_thickness(saturated__thickness_mean, saturated__thickness_standard_deviation)
        
        
    def _prep_saturated_thickness(self, saturated__thickness_mean, saturated__thickness_standard_deviation):
        
        if self._saturated__thickness_distribution == "event":     
            if (self._grid.at_node["saturated__thickness"]<0).any():
                msg = "saturated__thickness cannot be negative"
                raise ValueError(msg)
            if len(self._grid.at_node["saturated__thickness"].shape) > 1:
                msg = "saturated__thickness should be a 1-d array"
                raise ValueError(msg)
        elif self._saturated__thickness_distribution == "lognormal_spatial":
            assert saturated__thickness_mean.shape[0] == (
                self._grid.number_of_nodes
            ), "Input array should be of the length of grid.number_of_nodes!"
            assert saturated__thickness_standard_deviation.shape[0] == (
                self._grid.number_of_nodes
            ), "Input array should be of the length of grid.number_of_nodes!"
                        
            if (saturated__thickness_mean<0).any() or \
            (saturated__thickness_standard_deviation<0).any():
                msg = "negative mean__saturated_thickness and/or stdev__saturated_thicknes"
                raise ValueError(msg)
                
            if (len(saturated__thickness_mean.shape) > 1) or \
                (len(saturated__thickness_standard_deviation.shape) > 1):
                msg = "mean__saturated_thickness and/or stdev__saturated_thickness not a 1-d array"
                raise ValueError(msg)                
            
            self._sat_thickness_mean = saturated__thickness_mean
            self._sat_thickness_stdev = saturated__thickness_standard_deviation
        else:
            msg = "not a saturated thickness distribution option"
            raise ValueError(msg)        
           
            
    def _get_soil_water(self,i):
        
        if self._saturated__thickness_distribution == 'lognormal_spatial':
            # if mean is dry, assume always dry (log mu and sigma can not be determined from mean = 0)
            if (self._sat_thickness_mean[i] == 0):  
                self._satthick = np.zeros(self._n) 
            else:
                mu_lognormal = np.log(
                    (self._sat_thickness_mean[i] ** 2)
                    / np.sqrt(self._sat_thickness_stdev[i] ** 2 + self._sat_thickness_mean[i] ** 2)
                )
                sigma_lognormal = np.sqrt(
                    np.log(
                        (self._sat_thickness_stdev[i] ** 2) / (self._sat_thickness_mean[i] ** 2) + 1
                    )
                )
                self._satthick = np.random.lognormal(mu_lognormal, sigma_lognormal, self._n)     
                
                
    def _compute_rel_wetness(self,i):
        """compute relative wetness: relative wetness is stochastically determined 
        from the user selected recharge pdf for each iteration"""
        if self._saturated__thickness_distribution == 'event':
            # relative wetness is a single value and determined from the
            # saturated thickness at the grid node
            self._rel_wetness = ((self._grid.at_node["saturated__thickness"][i]) / 
                                 self._hs)        
        elif self._saturated__thickness_distribution == 'lognormal_spatial': 
            # relative wetness is an array of values, equal in length to the 
            # number of iterations (self._n). The values are stochastically 
            # determined from a lognormal pdf of saturated zone thickness 
            # at the grid node in the function _get_soil_water
            self._rel_wetness = (self._satthick) / self._hs   