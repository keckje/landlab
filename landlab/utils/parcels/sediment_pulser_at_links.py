import numpy as np

from landlab.data_record import DataRecord
from landlab.utils.parcels.sediment_pulser_base import SedimentPulserBase

_OUT_OF_NETWORK = -2

class SedimentPulserAtLinks(SedimentPulserBase):
    '''
    Send a pulse of parcels to specific links (reaches) of a channel network. 
    Pulse location within each link is random.
    
    This utility runs the landlab DataRecord "add_item" method on a DataRecord 
    that has been prepared for the NetworkSedimentTransporter. 
    
    In the NetworkSedimentTransporter, sediment "parcels" are the items added
    to the DataRecord
    
    SedimentPulserAtLinks is instantiated by specifying grain characteristics
    and the criteria for when a pulse accurs.
    
    SedimentPulserAtLinks is run (adds parcels to DataRecrod) by calling the 
    SedimentPulserAtLinks instance with  a list of links and a list of the number 
    of parcels added to each link. Parcels are then randomly placed into each link. 

    Optionallly, the user can specify the grain characteristics of the pulse for
    each link by providing alist of grain characterstics such as D50 and abrasion rates
    in a list equal in length to the length of the list of links.
    
    
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
    D50: float, optional
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
    D50 : list of floats, optional 
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
                   D50=D50, std_dev=std_dev,parcel_volume = parcel_volume)        
        
        check contents of DataRecord
    
    >>> print(pulse1.variable_names)
    >>> print(pulse1.dataset['D'])
    >>> print(pulse1.dataset['time_arrival_in_link'])
    
        Run the instance with inputs for the time, link location, number of
        parcels and grain charactersitics specific to each link
    
    >>> time = 11
    >>> links = [2, 6]
    >>> n_parcels_at_link = [20, 25]
    >>> D50 = [0.3, 0.12]
    >>> std_dev =  [0.2, 0.1]   
    >>> parcel_volume = [1, 0.5]
    >>> rho_sediment = [2650, 2500]
    >>> abrasion_rate = [.1, .3]
    >>> pulse2 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
                   D50=D50, std_dev=std_dev,parcel_volume = parcel_volume, 
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
        

    def __call__(self, time, links=[0], n_parcels_at_link=[100], D50 = None, 
                 std_dev = None, rho_sediment = None, parcel_volume = None, 
                 abrasion_rate = None):
        
        # prepare inputs for pulse_characteritics function

        # convert input lists to np.array if not already
        links = np.array(links)
        n_parcels_at_link  =np.array(n_parcels_at_link)

        # if parameters are not specified with __Call__ method, use default values
        # default values are specified in the base class and assumed uniform accross 
        # the channel network (all links use the same parameter values)
        if D50 is None: 
            D50 = np.full_like(links, self._D50, dtype=float)
        else:
            D50 = np.array(D50)        
        
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
            parcel_volume = np.array(rho_sediment)             
        
        if abrasion_rate is None: 
            abrasion_rate = np.full_like(links, self._abrasion_rate, dtype=float)
        else:
            abrasion_rate = np.array(abrasion_rate)

        # before running, check inputs < 0 
        if (np.array([D50,std_dev,rho_sediment, parcel_volume, abrasion_rate]) < 0).any():# check for negative inputs         
           raise AssertionError("grain size median grain size standard deviation and parcel volume can not be less than zero")
                                
        # before running, check if time to pulse            
        if not self._time_to_pulse(time):
            # if not time to pulse, return the existing parcels
            print('user provided time not a time-to-pulse')
            return self._parcels
        
        # convert D50 and std_dev to log normal distribution parameters
        D50_log, std_dev_log = calc_lognormal_distribution_parameters(mu_x = D50, 
                                                                      sigma_x = std_dev)  
        # approximate d84
        d84 = D50 * std_dev
        
        # approximate active layer depth
        active_layer_depth = d84 * 2.0

        # approxmate total parcel volume in each reach
        total_parcel_volume_at_link = calc_total_parcel_volume( 
            self._grid.at_link["channel_width"][links],
            self._grid.at_link["reach_length"][links],
            active_layer_depth,
        )
        
        # create times for DataRecord
        variables, items = self._sediment_pulse_stochastic(
            time,
            links,
            n_parcels_at_link,
            total_parcel_volume_at_link,
            parcel_volume,
            D50_log,
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
        total_parcel_volume_at_link,
        parcel_volume,
        D50_log,
        std_dev_log,
        abrasion_rate,
        rho_sediment
        ):
        '''
        specify attributes of pulses added to a Network Model Grid DataRecord 
        at stochastically determined link locations 
        
    
        Parameters
        ----------
        time : integer or datetime64 value equal to nst.time
            time that the pulse is triggered in the network sediment transporter
        SedimentPulseDF : pandas dataframe
            each row contains information on the deposition location and volume of
            a single pulse of sediment. The pulse is divided into n number of 
            parcels, where n equals the np.ceil(sediment pulse volume / parcel volume)
            
            The SedimentPulseDF must include the following columns:
                'link_#','vol [m^3]','link_downstream_distance_ratio'
            
            Optionally, the SedimentPulseDf can include the following columns:
               'D50 [m]', 'std_dev [m]', rho_sediment '[m^3/kg]', 'parcel_volume [m^3]
               
            if the optional columns are not included in SedimentPulseDF, then
            the instance variables are used to define sediment characateristics 
            of each pulse.

    
        Returns
        -------
        tuple: (item_id,variables)
            item_id: dictionary, model grid element and index of element of each parcel
            variables: dictionary, variable values for al new pulses
        '''
                
        # create empty np.arrays to be poplated in for loop 
        element_id = np.empty(np.sum(n_parcels_at_link),dtype=int)
        volume = np.empty(np.sum(n_parcels_at_link))
        grain_size = np.empty(np.sum(n_parcels_at_link))
        
        # if len(list(abrasion_rate)) == 1:
        #     abrasion_rate = (np.full_like(element_id, abrasion_rate, dtype=float))
        # if len(list(abrasion_rate)) == 1:
        
        offset = 0    
        for link, n_parcels in enumerate(n_parcels_at_link):
            element_id[offset:offset + n_parcels] = links[link]
            grain_size[offset:offset + n_parcels] = np.random.lognormal(
                D50_log[link], std_dev_log[link], n_parcels
            )
            volume[offset:offset + n_parcels] = parcel_volume[link] % n_parcels
            offset += n_parcels
        starting_link = element_id.copy()
        
        # abrasion rate and density are presently constant accross network
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
    
        time_arrival_in_link = np.full(np.shape(element_id), time, dtype=float)
        location_in_link = np.expand_dims(np.random.rand(np.sum(n_parcels_at_link)), axis=1)
    
    
        # 1 = active/surface layer; 0 = subsurface layer
        active_layer = np.ones(np.shape(element_id))
    
        grid_element = ["link"]*np.size(element_id)
        grid_element = np.expand_dims(grid_element, axis=1)
        
    
        return {
            "starting_link": (["item_id"], starting_link),
            "abrasion_rate": (["item_id"], abrasion_rate),
            "density": (["item_id"], density),
            # "lithology": (["item_id"], lithology),
            "time_arrival_in_link": (["item_id", "time"], time_arrival_in_link),
            "active_layer": (["item_id", "time"], active_layer),
            "location_in_link": (["item_id", "time"], location_in_link),
            "D": (["item_id", "time"], grain_size),
            "volume": (["item_id", "time"], volume),
        }, {"grid_element": grid_element, "element_id": element_id}




def calc_total_parcel_volume(width, length, sediment_thickness):
    '''
    

    Parameters
    ----------
    width : float
        reach (link) width [m]
    length : float [m]
        link length
    sediment_thickness : float
        thickness of particles

    Returns
    -------
    float
        volume of sediment

    '''
    return width * length * sediment_thickness


def calc_lognormal_distribution_parameters(mu_x, sigma_x):
    '''
    
    lognormal distribution parameters determined from mean and standard
    deviation following Maidment, 1990, Chapter 18, eq. 18.2.6 

    Parameters
    ----------
    mu_x : float
        mean grain size.
    sigma_x : float
        standard deviation of grain sizes.

    Returns
    -------
    mu_y : float
        mean of natural log of grain size
    sigma_y : float
        standard deviation of natural log of grain sizes.

    '''
    sigma_y = (np.log(((sigma_x**2)/(mu_x**2))+1))**(1/2)
    mu_y = np.log(mu_x)-(sigma_y**2)/2        
    
    return mu_y, sigma_y
