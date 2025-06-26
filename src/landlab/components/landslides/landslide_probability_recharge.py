import copy

import numpy as np
import scipy.constants
from scipy import interpolate
from statsmodels.distributions.empirical_distribution import ECDF

from landlab.components.landslides.landslide_probability_base import LandslideProbabilityBase


class LandslideProbabilityRecharge(LandslideProbabilityBase):
    """Landlab component designed to calculate probability of failure at
    each grid node based on the infinite slope stability model
    stability index (Factor of Safety).

    Relative wetness and factor-of-safety are based on the infinite slope
    stability model driven by topographic, soil and **recharge** data provided
    by user as inputs to the component. For each node, component simulates mean
    relative wetness as well as the probability of saturation based on Monte Carlo
    simulation of relative wetness where the probability is the number of
    iterations with relative wetness >= 1.0 divided by the number of iterations.
    Probability of failure for each node is also simulated in the Monte Carlo
    simulation as the number of iterations with factor-of-safety <= 1.0
    divided by the number of iterations.

    The main method of the LandslideProbability class is
    `calculate_landslide_probability()``, which calculates the mean soil
    relative wetness, probability of soil saturation, and probability of
    failure at each node based on a Monte Carlo simulation.
    
    
    The main method of the LandslideProbabilityRecharge class is
    `calculate_landslide_probability()``, which calculates the mean soil
    relative wetness, probability of soil saturation, and probability of
    failure at each node based on a Monte Carlo simulation.

    **Usage:**

    Option 1 - Uniform recharge

    .. code-block:: python

        LandslideProbabilityRecharge(grid,
                             number_of_iterations=250,
                             groundwater__recharge_distribution='uniform',
                             groundwater__recharge_min_value=5.,
                             groundwater__recharge_max_value=121.)

    Option 2 - Lognormal recharge

    .. code-block:: python

        LandslideProbabilityRecharge(grid,
                             number_of_iterations=250,
                             groundwater__recharge_distribution='lognormal',
                             groundwater__recharge_mean=30.,
                             groundwater__recharge_standard_deviation=0.25)

    Option 3 - Lognormal_spatial recharge

    .. code-block:: python

        LandslideProbabilityRecharge(grid,
                             number_of_iterations=250,
                             groundwater__recharge_distribution='lognormal_spatial',
                             groundwater__recharge_mean=np.random.randint(20, 120, grid_size),
                             groundwater__recharge_standard_deviation=np.random.rand(grid_size))

    Option 4 - Data_driven_spatial recharge

    .. code-block:: python

        LandslideProbabilityRecharge(grid,
                             number_of_iterations=250,
                             groundwater__recharge_distribution='data_driven_spatial',
                             groundwater__recharge_HSD_inputs=[HSD_dict,
                                                               HSD_id_dict,
                                                               fract_dict])
    
    """
    
    # component name
    _name = "Landslide Probability from Soil Recharge"

    _unit_agnostic = False

    __version__ = "1.1"
    
    _info = LandslideProbabilityBase._info
    
    
    def __init__(
        self,
        grid,
        number_of_iterations=250,
        g=scipy.constants.g,
        groundwater__recharge_distribution="uniform",
        groundwater__recharge_min_value=20.0,
        groundwater__recharge_max_value=120.0,
        groundwater__recharge_mean=None,
        groundwater__recharge_standard_deviation=None,
        groundwater__recharge_HSD_inputs=[],
        seed=0,
        ):
        
        """
        Parameters
        ----------
        grid: RasterModelGrid
            A raster grid.
        number_of_iterations: int, optional
            Number of iterations to run Monte Carlo simulation (default=250).
        groundwater__recharge_distribution: str, optional
            single word indicating recharge distribution, either 'uniform',
            'lognormal', 'lognormal_spatial' or 'data_driven_spatial'.
            Set as None to use saturated__thickness_distribution.
            (default='uniform')
        groundwater__recharge_min_value: float, optional (mm/d)
            minium groundwater recharge for 'uniform' (default=20.)
        groundwater__recharge_max_value: float, optional (mm/d)
            maximum groundwater recharge for 'uniform' (default=120.)
        groundwater__recharge_mean: float, optional (mm/d)
            mean groundwater recharge for 'lognormal'
            and 'lognormal_spatial' (default=None)
        groundwater__recharge_standard_deviation: float, optional (mm/d)
            standard deviation of groundwater recharge for 'lognormal'
            and 'lognormal_spatial' (default=None)
        groundwater__recharge_HSD_inputs: list, optional
            list of 3 dictionaries in order (default=[]) - HSD_dict
            {Hydrologic Source Domain (HSD) keys: recharge numpy array values},
            {node IDs keys: list of HSD_Id values}, HSD_fractions {node IDS
            keys: list of HSD fractions values} (none)
            Note: this input method is a very specific one, and to use this method,
            one has to refer Ref 1 & Ref 2 mentioned above, as this set of
            inputs require rigorous pre-processing of data.
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
                         seed)
        
        self._groundwater__recharge_distribution = groundwater__recharge_distribution
        self._prep_recharge(groundwater__recharge_mean, 
                            groundwater__recharge_standard_deviation)
        
        
    def _prep_recharge(self, groundwater__recharge_mean, 
                       groundwater__recharge_standard_deviation):
        # Uniform distribution
        if self._groundwater__recharge_distribution == "uniform":
            self._recharge_min = groundwater__recharge_min_value
            self._recharge_max = groundwater__recharge_max_value
            self._Re = np.random.uniform(
                self._recharge_min, self._recharge_max, size=self._n
            )
            self._Re /= 1000.0  # Convert mm to m
        # Lognormal Distribution - Uniform in space
        elif self._groundwater__recharge_distribution == "lognormal":
            assert (
                groundwater__recharge_mean is not None
            ), "Input mean of the distribution!"
            assert (
                groundwater__recharge_standard_deviation is not None
            ), "Input standard deviation of the distribution!"
            self._recharge_mean = groundwater__recharge_mean
            self._recharge_stdev = groundwater__recharge_standard_deviation
            self._mu_lognormal = np.log(
                (self._recharge_mean**2)
                / np.sqrt(self._recharge_stdev**2 + self._recharge_mean**2)
            )
            self._sigma_lognormal = np.sqrt(
                np.log((self._recharge_stdev**2) / (self._recharge_mean**2) + 1)
            )
            self._Re = np.random.lognormal(
                self._mu_lognormal, self._sigma_lognormal, self._n
            )
            self._Re /= 1000.0  # Convert mm to m
        # Lognormal Distribution - Variable in space
        elif self._groundwater__recharge_distribution == "lognormal_spatial":
            assert groundwater__recharge_mean.shape[0] == (
                self._grid.number_of_nodes
            ), "Input array should be of the length of grid.number_of_nodes!"
            assert groundwater__recharge_standard_deviation.shape[0] == (
                self._grid.number_of_nodes
            ), "Input array should be of the length of grid.number_of_nodes!"
            self._recharge_mean = groundwater__recharge_mean
            self._recharge_stdev = groundwater__recharge_standard_deviation
        # Custom HSD inputs - Hydrologic Source Domain -> Model Domain
        elif self._groundwater__recharge_distribution == "data_driven_spatial":
            self._HSD_dict = groundwater__recharge_HSD_inputs[0]
            self._HSD_id_dict = groundwater__recharge_HSD_inputs[1]
            self._fract_dict = groundwater__recharge_HSD_inputs[2]
            self._interpolate_HSD_dict()
        else:
            msg = "not a recharge distribution option"
            raise ValueError(msg)
            

    def _get_soil_water(self,i):
        """get recharge values based on distribution type"""
        if self._groundwater__recharge_distribution == "data_driven_spatial":
            self._calculate_HSD_recharge(i)
            self._Re /= 1000.0  # mm->m
        elif self._groundwater__recharge_distribution == "lognormal_spatial":
            mu_lognormal = np.log(
                (self._recharge_mean[i] ** 2)
                / np.sqrt(self._recharge_stdev[i] ** 2 + self._recharge_mean[i] ** 2)
            )
            sigma_lognormal = np.sqrt(
                np.log(
                    (self._recharge_stdev[i] ** 2) / (self._recharge_mean[i] ** 2) + 1
                )
            )
            self._Re = np.random.lognormal(mu_lognormal, sigma_lognormal, self._n)
            self._Re /= 1000.0  # Convert mm to m
                   
    
    def _compute_rel_wetness(self, i):
        """compute relative wetness: relative wetness is stochastically determined 
        from the user selected recharge pdf for each iteration
        input i is not used in landslide_probability_recharge"""
        self._rel_wetness = ((self._Re) / self._T) * ( 
            self._a / np.sin(np.arctan(self._theta))
        ) 
   
        
    def _interpolate_HSD_dict(self):
        """Method to extrapolate input data.

        This method uses a non-parametric approach to expand the input
        recharge array to the length of number of iterations. Output is
        a new dictionary of interpolated recharge for each HSD id.
        """
        HSD_dict = copy.deepcopy(self._HSD_dict)
        # First generate interpolated Re for each HSD grid
        Yrand = np.sort(np.random.rand(self._n))
        # n random numbers (0 to 1) in a column
        for vkey in HSD_dict.keys():
            if isinstance(HSD_dict[vkey], int):
                continue  # loop back up if value is integer (e.g. -9999)
            Re_temp = HSD_dict[vkey]  # an array of annual Re for 1 HSD grid
            Fx = ECDF(Re_temp)  # instantiate to get probabilities with Re
            Fx_ = Fx(Re_temp)  # probability array associated with Re data
            # interpolate function based on recharge data & probability
            f = interpolate.interp1d(
                Fx_, Re_temp, bounds_error=False, fill_value=min(Re_temp)
            )
            # array of Re interpolated from Yrand probabilities (n count)
            Re_interpolated = f(Yrand)
            # replace values in HSD_dict with interpolated Re
            HSD_dict[vkey] = Re_interpolated

        self._interpolated_HSD_dict = HSD_dict


    def _calculate_HSD_recharge(self, i):
        """Method to calculate recharge based on upstream fractions.

        This method calculates the resultant recharge at node i of the
        model domain, using recharge of contributing HSD ids and the
        areal fractions of upstream contributing HSD ids. Output is a
        numpy array of recharge at node i.
        """
        store_Re = np.zeros(self._n)
        HSD_id_list = self._HSD_id_dict[i]
        fract_list = self._fract_dict[i]
        for j in range(0, len(HSD_id_list)):
            Re_temp = self._interpolated_HSD_dict[HSD_id_list[j]]
            fract_temp = fract_list[j]
            Re_adj = Re_temp * fract_temp
            store_Re = np.vstack((store_Re, np.array(Re_adj)))
        self._Re = np.sum(store_Re, 0)