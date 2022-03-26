import numpy as np

from landlab.data_record import DataRecord
from landlab.utils.parcels.sediment_pulser_base import SedimentPulserBase

_OUT_OF_NETWORK = -2

class SedimentPulserAtLinks(SedimentPulserBase):
    '''
    Send a pulse of parcels to specific links of a channel network. 
    User specifies the pulse with a list of link IDs and list of the number of 
    parcels sent to each link. Other attributes can also be specified. Pulse 
    location within the links is random.
    
    This utility runs the landlab DataRecord "add_item" method on a DataRecord 
    that has been prepared for the NetworkSedimentTransporter. In the 
    NetworkSedimentTransporter, sediment "parcels" are the items added to the 
    DataRecord
    
    SedimentPulserAtLinks is instantiated by specifying the network model grid
    it will pulse the parcels into and times when a pulse accurs. 
    If parcel attributes are constant with time and uniform 
    accross the basin, these constant-uniform-attirbutes can be defined 
    when SedimentPulserAtLinks is instantiated.
    
    SedimentPulserAtLinks is run (adds parcels to DataRecord) by calling the 
    instance with a list of links and a list of the number of parcels added to each link.

    If parcel attributes do vary with location and time, the user specifies 
    the varying parcel attributes each time the instance is called with a list for each
    attribute, with length equal to the length of the number of links included in
    the pulse.

    
    Parameters
    ----------
    
    Instantiation parameters:
        
    grid : ModelGrid
        landlab *ModelGrid* to place sediment parcels on.
    parcels: landlab DataRecord 
        Tracks parcel location and variables
    time_to_pulse: function, optional
        defines the condition when a pulse occurs using the _pulse_characteristics
        method. If not specified, a pulse occurs when instance is run
    d50: float, optional
        median grain size [m]
    std_dev: float, optional
        standard deviation of grain sizes [m]
    rho_sediment : float, optional
        Sediment grain density [kg / m^3].
    parcel_volume : float, optional
        parcel volume used for all parcels that do not have a specified volume [m^3]
    abrasion_rate: list of floats, optional
        rate that grain size decreases with distance along channel [mm/km?]
     
    
    Run instance parameters:
    time : integer or datetime64 value equal to nst.time
        time that the pulse is triggered in the network sediment transporter
    links : list of integers
        link ID # that parcels are added too
    n_parcels_at_link: list of integers
        number of parcels added to each link listed in links
    d50 : list of floats, optional 
        median grain size of parcels added to each link listed in links
    std_dev : list of floats, optional
        grain size standard deviation of parcels added to each link listed in links
    parcel_volume : list of floats, optional
        volume of each parcel added to link listed in links

        
    
    

    Examples
    --------
    >>> from landlab import NetworkModelGrid
    >>> from landlab.utils.parcels.sediment_pulser_at_links import SedimentPulserAtLinks

    >>> y_of_node = (0, 100, 200, 200, 300, 400, 400, 125)
    >>> x_of_node = (0, 0, 100, -50, -100, 50, -150, -100)
    >>> nodes_at_link = ((1, 0), (2, 1), (1, 7), (3, 1), (3, 4), (4, 5), (4, 6))
    >>> grid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)
    >>> grid.at_link["channel_width"] = np.full(grid.number_of_links, 1.0)  # m
    >>> grid.at_link["channel_slope"] = np.full(grid.number_of_links, .01)  # m / m
    >>> grid.at_link["reach_length"] = np.full(grid.number_of_links, 100.0)  # m
    >>> def time_to_pulse(time):
    ...     return True
    
        Instantiate 'SedimentPulserAtLinks' utility for the network model
        grid and pulse criteria

    >>> make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
    
        Run the instance with inputs for the time, link location, number of
        parcels using the instance grain characterstics for all links

    >>> time = 10
    >>> links = [2, 6]
    >>> n_parcels_at_link = [20, 25]
    >>> pulse1 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
                   d50=d50, std_dev=std_dev,parcel_volume = parcel_volume)        
        
        check contents of DataRecord
    
    >>> print(pulse1.variable_names)
    >>> print(pulse1.dataset['D'])
    >>> print(pulse1.dataset['time_arrival_in_link'])
    
        Run the instance with inputs for the time, link location, number of
        parcels and grain charactersitics specific to each link
    
    >>> time = 11
    >>> links = [2, 6]
    >>> n_parcels_at_link = [20, 25]
    >>> d50 = [0.3, 0.12]
    >>> std_dev =  [0.2, 0.1]   
    >>> parcel_volume = [1, 0.5]
    >>> rho_sediment = [2650, 2500]
    >>> abrasion_rate = [.1, .3]
    >>> pulse2 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
                   d50=d50, std_dev=std_dev,parcel_volume = parcel_volume, 
                   rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)
        
        check contents of DataRecord
    
    >>> print(pulse2.variable_names)
    >>> print(pulse2.dataset['D'])
    >>> print(pulse2.dataset['time_arrival_in_link'])
    '''

    def __init__(
        self,
        grid,
        time_to_pulse = None,
        **kwgs
        ):
               
        SedimentPulserBase.__init__(self, grid, **kwgs)
        
        # set time_to_pulse to True if not specified
        if time_to_pulse is None:
            self._time_to_pulse = lambda time: True
        else:
            self._time_to_pulse = time_to_pulse
        

    def __call__(self, time, links = None, n_parcels_at_link = None, d50 = None, 
                 std_dev = None, rho_sediment = None, parcel_volume = None, 
                 abrasion_rate = None):
        

        # check user provided links and number of parcels
        assert (links is not None and n_parcels_at_link is not None
                ), "must provide links and number of parcels entered into each link"                
         
        links = np.array(links)
        n_parcels_at_link  =np.array(n_parcels_at_link)

        # any other parameters are not specified with __Call__ method, use default values
        # default values are specified in the base class and assumed uniform accross 
        # the channel network (all links use the same parameter values)
        if d50 is None: 
            d50 = np.full_like(links, self._d50, dtype=float)
        else:
            d50 = np.array(d50)        
        
        if std_dev is None: 
            std_dev = np.full_like(links, self._std_dev, dtype=float)
        else:
            std_dev = np.array(std_dev)        
        
        if rho_sediment is None: 
            rho_sediment = np.full_like(links, self._rho_sediment, dtype=float)
        else:
            rho_sediment = np.array(rho_sediment)                
        
        if parcel_volume is None: 
            parcel_volume = np.full_like(links, self._parcel_volume, dtype=float)
        else:
            parcel_volume = np.array(parcel_volume)             
        
        if abrasion_rate is None: 
            abrasion_rate = np.full_like(links, self._abrasion_rate, dtype=float)
        else:
            abrasion_rate = np.array(abrasion_rate)

        # before running, check that no inputs < 0 
        if (np.array([d50, std_dev, rho_sediment, parcel_volume, abrasion_rate]) < 0).any():# check for negative inputs         
           raise AssertionError("parcel attributes cannot be less than zero")
                                
        # before running, check if time to pulse            
        if not self._time_to_pulse(time):
            # if not time to pulse, return the existing parcels
            print('user provided time not a time-to-pulse, parcels have not changed')
            
            return self._parcels
        
        # convert d50 and std_dev to log normal distribution parameters
        d50_log, std_dev_log = self.calc_lognormal_distribution_parameters(mu_x = d50, 
                                                                      sigma_x = std_dev)  

        # create times for DataRecord
        variables, items = self._sediment_pulse_stochastic(
            time,
            links,
            n_parcels_at_link,
            parcel_volume,
            d50_log,
            std_dev_log,
            abrasion_rate,
            rho_sediment
        )

        # if DataRecord does not exist, create one
        if self._parcels is None:
            self._parcels = DataRecord(
                self._grid,
                items=items,
                time=[time],
                data_vars=variables,
                dummy_elements={"link": [_OUT_OF_NETWORK]},
            )
        # else, add parcels to existing DataRecrod
        else:
            self._parcels.add_item(time=[time], new_item=items, new_item_spec=variables)
            
        return self._parcels

    def _sediment_pulse_stochastic(self,
        time,
        links,
        n_parcels_at_link,
        parcel_volume,
        d50_log,
        std_dev_log,
        abrasion_rate,
        rho_sediment
        ):
        """
        specify attributes of pulses added to a Network Model Grid DataRecord 
        at stochastically determined link locations 

        Parameters
        ----------
        time : int, string, or datetime
            time that pulse enters the channel network
        links : list or int
            ID(s) of links that will recieve a pulse of sediment
        n_parcels_at_link : list or int
            DESCRIPTION.
        parcel_volume : TYPE
            DESCRIPTION.
        d50_log : TYPE
            DESCRIPTION.
        std_dev_log : TYPE
            DESCRIPTION.
        abrasion_rate : TYPE
            DESCRIPTION.
        rho_sediment : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.
        dict
            DESCRIPTION.

        """
                
        # Create np array for each paracel attribute. Length of array is equal 
        # to the number of parcels (i.e., attribute values are specified for each
        # parcel)
        
        # link id, logD50 and volume
        element_id = np.empty(np.sum(n_parcels_at_link),dtype=int)
        grain_size = np.empty(np.sum(n_parcels_at_link))
        volume = np.empty(np.sum(n_parcels_at_link))       
        offset = 0    
        for link, n_parcels in enumerate(n_parcels_at_link):
            element_id[offset:offset + n_parcels] = links[link]
            grain_size[offset:offset + n_parcels] = np.random.lognormal(
                d50_log[link], std_dev_log[link], n_parcels
            )
            volume[offset:offset + n_parcels] = parcel_volume[link] % n_parcels
            offset += n_parcels
        starting_link = element_id.copy()
        
        # abrasion rate and density
        abrasion_rate_L = []
        density_L = []
        for c, ei in enumerate(np.unique(element_id)):
            element_id_subset = element_id[element_id == ei]    
            abrasion_rate_L = abrasion_rate_L+list(np.full_like(element_id_subset, abrasion_rate[c], dtype=float))
            density_L = density_L+list(np.full_like(element_id_subset, rho_sediment[c], dtype=float))
        abrasion_rate = np.array(abrasion_rate_L) # np.full_like(element_id_subset, abrasion_rate, dtype=float)
        density = np.array(density_L) # np.full_like(element_id, rho_sediment, dtype=float)
    
        element_id = np.expand_dims(element_id, axis=1)
        grain_size = np.expand_dims(grain_size, axis=1)
        volume = np.expand_dims(volume, axis=1)
        abrasion_rate = np.expand_dims(abrasion_rate, axis=1)
        density = np.expand_dims(density, axis=1)
        
    
        # time of arrivial (time instance called)
        time_arrival_in_link = np.full(np.shape(element_id), time, dtype=float)
        
        # link location (distance from link inlet / link length) is stochastically determined
        location_in_link = np.expand_dims(np.random.rand(np.sum(n_parcels_at_link)), axis=1)
    
        # All parcels in pulse are in the active layer (1) rather than subsurface (0)
        active_layer = np.ones(np.shape(element_id))
        
        # specify that parcels are in the links of the network model grid
        grid_element = ["link"]*np.size(element_id)
        grid_element = np.expand_dims(grid_element, axis=1)
           
        return {
            "starting_link": (["item_id"], starting_link),
            "abrasion_rate": (["item_id", "time"], abrasion_rate),
            "density": (["item_id", "time"], density),
            "time_arrival_in_link": (["item_id", "time"], time_arrival_in_link),
            "active_layer": (["item_id", "time"], active_layer),
            "location_in_link": (["item_id", "time"], location_in_link),
            "D": (["item_id", "time"], grain_size),
            "volume": (["item_id", "time"], volume),
        }, {"grid_element": grid_element, "element_id": element_id}




# def calc_lognormal_distribution_parameters(mu_x, sigma_x):
#     '''
    
#     lognormal distribution parameters determined from mean and standard
#     deviation following Maidment, 1990, Chapter 18, eq. 18.2.6 

#     Parameters
#     ----------
#     mu_x : float
#         mean grain size.
#     sigma_x : float
#         standard deviation of grain sizes.

#     Returns
#     -------
#     mu_y : float
#         mean of natural log of grain size
#     sigma_y : float
#         standard deviation of natural log of grain sizes.

#     '''
#     sigma_y = (np.log(((sigma_x**2)/(mu_x**2))+1))**(1/2)
#     mu_y = np.log(mu_x)-(sigma_y**2)/2        

    
#     return mu_y, sigma_y
