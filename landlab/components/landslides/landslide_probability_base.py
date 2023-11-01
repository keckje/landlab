import copy

import numpy as np
import scipy.constants
from scipy import interpolate
from statsmodels.distributions.empirical_distribution import ECDF

from landlab import Component


class LandslideProbabilityBase(Component):
    """base class for Landslide probability component using the infinite slope stability
       model.

       
    """

    # component name
    _name = "Landslide Probability"

    _unit_agnostic = False

    __version__ = "1.1"

    _cite_as = """
    @article{strauch2018hydroclimatological,
      author = {Strauch, Ronda and Istanbulluoglu, Erkan and Nudurupati,
      Sai Siddhartha and Bandaragoda, Christina and Gasparini, Nicole M and
      Tucker, Gregory E},
      title = {{A hydroclimatological approach to predicting regional landslide
      probability using Landlab}},
      issn = {2196-6311},
      doi = {10.5194/esurf-6-49-2018},
      pages = {49--75},
      number = {1},
      volume = {6},
      journal = {Earth Surface Dynamics},
      year = {2018}
    }
    """
    _info = {
        "landslide__probability_of_failure": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "None",
            "mapping": "node",
            "doc": "number of times FS is <=1 out of number of iterations user selected",
   
        },
        "soil__density": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "kg/m3",
            "mapping": "node",
            "doc": "wet bulk density of soil",
        },
        "soil__internal_friction_angle": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "degrees",
            "mapping": "node",
            "doc": "critical angle just before failure due to friction between particles",
        },
        "soil__maximum_total_cohesion": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "Pa or kg/m-s2",
            "mapping": "node",
            "doc": "maximum of combined root and soil cohesion at node",
        },
        "soil__mean_relative_wetness": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "None",
            "mapping": "node",
            "doc": "Indicator of soil wetness; relative depth perched water table within the soil layer",
        },
        "soil__minimum_total_cohesion": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "Pa or kg/m-s2",
            "mapping": "node",
            "doc": "minimum of combined root and soil cohesion at node",
        },
        "soil__mode_total_cohesion": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "Pa or kg/m-s2",
            "mapping": "node",
            "doc": "mode of combined root and soil cohesion at node",
        },
        "soil__probability_of_saturation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "None",
            "mapping": "node",
            "doc": "number of times relative wetness is >=1 out of number of iterations user selected",
        },
        "soil__saturated_hydraulic_conductivity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m/day",
            "mapping": "node",
            "doc": "mode rate of water transmitted through soil - provided if transmissivity is NOT provided to calculate tranmissivity  with soil depth",
        },
        "soil__thickness": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "soil depth to restrictive layer",
        },
        "soil__transmissivity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m2/day",
            "mapping": "node",
            "doc": "mode rate of water transmitted through a unit width of saturated soil - either provided or calculated with Ksat and soil depth",
        },
        "topographic__slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "tan theta",
            "mapping": "node",
            "doc": "gradient of the ground surface",
        },
        "topographic__specific_contributing_area": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "specific contributing (upslope area/cell face ) that drains to node",
        }
    }
        
    
    def __init__(
        self,
        grid,
        number_of_iterations=250,
        g=scipy.constants.g,
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
        g: float, optional (m/sec^2)
            acceleration due to gravity.
        seed: int, optional
            seed for random number generation. if seed is assigned any value
            other than the default value of zero, it will create different
            sequence. To create a certain sequence repititively, use the same
            value as input for seed.
        """
    
        # Initialize seeded random number generation
        self._seed_generator(seed)
        
        super().__init__(grid)
        
        # Store parameters and do unit conversions
        self._n = int(number_of_iterations)
        self._g = g
        self._save = save
        # Check if all output fields are initialized
        self.initialize_output_fields()
        
        # Create a switch to imply whether Ksat is provided and
        # check that either soil__saturated_hydraulic_conductivity or
        # soil__transmissivity are provided
        if self._grid.has_field('node', 'soil__saturated_hydraulic_conductivity'):
            self._Ksat_provided = 1  # True
        elif self._grid.has_field('node', 'soil__transmissivity'):
            self._Ksat_provided = 0  # False
        else:
            msg = "no 'soil__transmissivity' or 'soil__saturated_hydraulic_conductivity' field"             
            raise ValueError(msg) 
            
        self._nodal_values = self._grid.at_node
        # create list for storing soil depth and Factor of Safety values from 
        # each iteration
        if self._save:
            self._hs_list = [] # figure out a way to get rid of these
            self._FS_list = []


    def _seed_generator(self, seed=0):
        """Method to initiate random seed.

        Seed the random-number generator. This method will create the
        same sequence again by re-seeding with the same value (default
        value is zero). To create a sequence other than the default,
        assign non-zero value for seed.
        """
        np.random.seed(seed)


    def _define_soil_parameters(self,i):
        """generate distributions to sample from to provide input parameters
        currently triangle distribution using mode, min, & max"""
        self._a = np.float32(
            self._grid.at_node["topographic__specific_contributing_area"][i]
        )
        self._theta = np.float32(self._grid.at_node["topographic__slope"][i])
        self._Tmode = np.float32(self._grid.at_node["soil__transmissivity"][i])
        if self._Ksat_provided:
            self._Ksatmode = np.float32(
                self._grid.at_node["soil__saturated_hydraulic_conductivity"][i]
            )
        self._Cmode = np.float32(self._grid.at_node["soil__mode_total_cohesion"][i])
        self._Cmin = np.float32(self._grid.at_node["soil__minimum_total_cohesion"][i])
        self._Cmax = np.float32(self._grid.at_node["soil__maximum_total_cohesion"][i])
        self._phi_mode = np.float32(
            self._grid.at_node["soil__internal_friction_angle"][i]
        )
        self._rho = np.float32(self._grid.at_node["soil__density"][i])
        self._hs_mode = np.float32(self._grid.at_node["soil__thickness"][i])
      
   
    def _get_random_soil_parameter_values(self):
       """use soil parameters to generate random soil parameter values"""
       self._C = np.random.triangular(
           self._Cmin, self._Cmode, self._Cmax, size=self._n
       )
       
       # phi - internal angle of friction provided in degrees
       phi_min = self._phi_mode - 0.18 * self._phi_mode
       phi_max = self._phi_mode + 0.32 * self._phi_mode
       self._phi = np.random.triangular(phi_min, self._phi_mode, phi_max, size=self._n)
       # soil thickness
       # hs_min = min(0.005, self._hs_mode-0.3*self._hs_mode) # Alternative
       hs_min = self._hs_mode - 0.3 * self._hs_mode
       hs_max = self._hs_mode + 0.1 * self._hs_mode
       
       # for evolving terrains, nodes with 0 soil depth (post-landslide nodes), 
       # a depth of 0 causes error. Set a small soil depth in landslide areas.
       try: 
           self._hs = np.random.triangular(hs_min, self._hs_mode, hs_max, size=self._n)
       except:
           self._hs = np.random.triangular(.001, 0.005, .01, size=self._n)               
       self._hs[self._hs <= 0.0] = 0.005
       if self._Ksat_provided:
           # Hydraulic conductivity (Ksat)
           Ksatmin = self._Ksatmode - (0.3 * self._Ksatmode)
           Ksatmax = self._Ksatmode + (0.1 * self._Ksatmode)
           self._Ksat = np.random.triangular(
               Ksatmin, self._Ksatmode, Ksatmax, size=self._n
           )
           self._T = self._Ksat * self._hs
       else:
           # Transmissivity (T)
           self._Ksat = np.ones(self._n)*np.nan # Ksat not provided
           Tmin = self._Tmode - (0.3 * self._Tmode)
           Tmax = self._Tmode + (0.1 * self._Tmode)
           self._T = np.random.triangular(Tmin, self._Tmode, Tmax, size=self._n)

       # calculate Factor of Safety for n number of times
       # calculate components of FS equation
       self._C_dim = self._C / (
           self._hs * self._rho * self._g
       )  # dimensionless cohesion
   

    def _compute_probability_of_saturation(self):
        """calculate probability of saturation"""
        countr = 0
        for val in self._rel_wetness:  # find how many RW values >= 1
            if val >= 1.0:
                countr = countr + 1  # number with RW values (>=1)
        # probability: No. high RW values/total No. of values (n)
        self._soil__probability_of_saturation = np.float32(countr) / self._n
        # Maximum Rel_wetness = 1.0
        np.place(self._rel_wetness, self._rel_wetness > 1, 1.0)
        self._soil__mean_relative_wetness = np.mean(self._rel_wetness)

    
    def _compute_FS(self):
        """compute the factor of safety"""
        Y = np.tan(np.radians(self._phi)) * (1 - (self._rel_wetness * 0.5))
        # convert from degrees; 0.5 = water to soil density ratio
        # calculate Factor-of-safety
        self._FS = (self._C_dim / np.sin(np.arctan(self._theta))) + (
            np.cos(np.arctan(self._theta)) * (Y / np.sin(np.arctan(self._theta)))
        )

    def _get_soil_water(self,i):
        raise NotImplementedError('not implemented in the base class')
        
    def _compute_rel_wetness(self,i):
        raise NotImplementedError('not implemented in the base class')
                              
    def calculate_factor_of_safety(self, i):
        """Method to calculate factor of safety.

        Method calculates factor-of-safety stability index by using
        node specific parameters, creating distributions of these parameters,
        and calculating the index by sampling these distributions 'n' times.

        The index is calculated from the 'infinite slope stabilty
        factor-of-safety equation' in the format of Pack RT, Tarboton DG,
        and Goodwin CN (1998),The SINMAP approach to terrain stability mapping.

        Parameters
        ----------
        i: int
            index of core node ID.
        """
        self._define_soil_parameters(i)
        
        self._get_soil_water(i)
        
        self._get_random_soil_parameter_values()
        
        self._compute_rel_wetness(i)
        
        self._compute_probability_of_saturation()
        
        self._compute_FS()
        
        
        # store "self._n" soil depth and factor of safety values used to determine
        # probability at node i
        if self._save:
            self._hs_list.append(self._hs)
            self._FS_list.append(self._FS)
        count = 0
        for val in self._FS:  # find how many FS values <= 1
            if val <= 1.0:
                count = count + 1  # number with unstable FS values (<=1)
        # probability: No. unstable values/total No. of values (n)
        self._landslide__probability_of_failure = np.float32(count) / self._n
        

    def calculate_landslide_probability(self):
        """Main method of Landslide Probability class.

        Method creates arrays for output variables then loops through
        all the core nodes to run the method
        'calculate_factor_of_safety.' Output parameters probability of
        failure, mean relative wetness, and probability of saturation
        are assigned as fields to nodes.
        """
        # Create arrays for data with -9999 as default to store output
        self._mean_Relative_Wetness = np.full(self._grid.number_of_nodes, -9999.0)
        self._prob_fail = np.full(self._grid.number_of_nodes, -9999.0)
        self._prob_sat = np.full(self._grid.number_of_nodes, -9999.0)

        # Run factor of safety Monte Carlo for all core nodes in domain
        # i refers to each core node id
        for i in self._grid.core_nodes:
            self.calculate_factor_of_safety(i)
            # Populate storage arrays with calculated values
            self._mean_Relative_Wetness[i] = self._soil__mean_relative_wetness
            self._prob_fail[i] = self._landslide__probability_of_failure
            self._prob_sat[i] = self._soil__probability_of_saturation
                        
        # Values can't be negative
        self._mean_Relative_Wetness[self._mean_Relative_Wetness < 0.0] = 0.0
        self._prob_fail[self._prob_fail < 0.0] = 0.0
        # assign output fields to nodes
        self._grid.at_node["soil__mean_relative_wetness"] = self._mean_Relative_Wetness
        self._grid.at_node["landslide__probability_of_failure"] = self._prob_fail
        self._grid.at_node["soil__probability_of_saturation"] = self._prob_sat
        # arrays of n iterations of factor of safety values and soil depth for 
        # each core node
        if self._save:
            self._FSarray = np.array(self._FS_list)
            self._hsarray = np.array(self._hs_list)