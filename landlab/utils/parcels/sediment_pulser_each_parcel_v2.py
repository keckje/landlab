import numpy as np

from landlab.data_record import DataRecord
from landlab.utils.parcels.sediment_pulser_base import SedimentPulserBase

_OUT_OF_NETWORK = -2

class SedimentPulserEachParcel(SedimentPulserBase):
    
    '''
    Send pulses of sediment to specific point locations within the channel 
    network and divide the pulses into parcels   
    
    SedimentPulserEachParcel is instantiated by specifying the network model grid
    it will pulse the parcels into
    
    
    The sediment pulse table is a pandas dataframe. It has columns for the link,
    distance on link, volume of material in pulse, parcel volume that the pulse
    is divided into, and optionally grain characteristics specific to the material 
    in the pulse.
    
    
    SedimentPulserCatalog is run (adds parcels to DataRecrod) by calling the 
    SedimentPulserCatalog instance with  a the time that pulses are added to 
    the channel network and a sediment pulse catalog (pandas dataframe that lists 
    the volume of the pulse,the link number and the distance on the link that 
    the pulse enters the channel network. Grain characteristics of the pulse use
    the instance sediment characteristics.

    Optionallly, the user can include grain characteristics of each pulse in the
    sediment pulse catalog
     
    Parameters
    ----------
    
    Instantiation parameters:
        
    grid : ModelGrid
        landlab *ModelGrid* to place sediment parcels on.
    parcels: landlab DataRecord 
        Tracks parcel location and variables
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
    SedimentPulseDF : pandas dataframe
        each row contains information on the deposition location and volume of
        a single pulse of sediment. The pulse is divided into n number of 
        parcels, where n equals the np.ceil(sediment pulse volume / parcel volume)
        For details on the format of the DataFrame, see docstring below in the
        function _sediment_pulse_dataframe below
        

        

    # Complete pulser at links
    # complete pulser each parcel
    # complete tests
    # draft parcel collector
    
    

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
    
    '''
     
    def __init__(
        self,
        grid,
        **kwgs
        ):
               
        SedimentPulserBase.__init__(self, grid,**kwgs)
        
    
    def __call__(self, time, SedimentPulseDF = None):
        


        if SedimentPulseDF is None: # if no PulseDF provided, raise error. Should at least provide an empty PulseDF
            
            raise('SedimentPulseDF was not specified')
            
        else: # PulseDf was provided
            
            if SedimentPulseDF.empty == True: # if empty, pulser stops, returns the existing parcels, call stops
                return self._parcels
                print('PulserDF is EMPTY')
            
            variables, items = self._sediment_pulse_dataframe(time,  # create variabels and and items needed to create the data record
                SedimentPulseDF,
                point_pulse = True)
            print('created parcel dictionary')
        
        if self._parcels is None: # if no parcels, create parcels
            self._parcels = DataRecord(
                self._grid,
                items=items,
                time=[time],
                data_vars=variables,
                dummy_elements={"link": [_OUT_OF_NETWORK]},
            )
            print('Parcels is NONE')
        else: # else use the add item method to add parcels
            self._parcels.add_item(time=[time], new_item=items, new_item_spec=variables)
            print('added parcels')
        return self._parcels



    def _sediment_pulse_dataframe(self, time, SedimentPulseDF, point_pulse = True):
        
        '''
        specify attributes of pulses added to a Network Model Grid DataRecord 
        at specific channel netowrk locations (link, distance on link) from the
        SedimentPulseDF. At a minimum, SedimentPulseDF must have a column "vol [m^3]"
        
    
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
               'd50 [m]', 'std_dev [m]', rho_sediment '[m^3/kg]', 'parcel_volume [m^3]'
               
            if the optional columns are not included in SedimentPulseDF, then
            the instance variables are used to define sediment characateristics 
            of each pulse.

    
        Returns
        -------
        tuple: (item_id,variables)
            item_id: dictionary, model grid element and index of element of each parcel
            variables: dictionary, variable values for al new pulses
     
    
       ###CHECK, may be inconsitecy in use of parcel and pulse, a pulse should be divided into parcels, more parcels
       than pulses
        '''
        #(1) split pulse table into parcels.
        p_np = [] # list of parcels in each pulse
        volume = np.array([]) # list of parcel volumes from all pulses
        for index, row in SedimentPulseDF.iterrows():
            
            # set the maximum allowable parcel volume using either
            # the default value or value in the pulse table
            if 'parcel_volume [m^3]' in SedimentPulseDF:
                mpv = row['parcel_volume [m^3]']
            else:
                mpv = self._parcel_volume
            
            # split the pulse into parcels
            if row['vol [m^3]'] < mpv:
                # only one partial parcel volume
                v_p = np.array([row['vol [m^3]']])
            else:
                # number of whole parcels
                n_wp = int(np.floor(row['vol [m^3]']/mpv))
                # array of volumes, whole parcels
                v_wp = np.ones(n_wp)*mpv
                # volume of last parcel, a partial parcel
                v_pp = np.array([row['vol [m^3]']%mpv])
                # array of all parcel volumes
                # partial parcel included if volume > 0
                if v_pp>0:
                    v_p = np.concatenate((v_wp, v_pp))
                else:
                    v_p = v_wp
            volume = np.concatenate((volume, v_p))                   
            p_np.append(len(v_p)) #number of parcels in pulse = volume pulse/volume 1 parcel   
        volume = np.expand_dims(volume, axis=1)


        # link location (distance from link inlet / link length) is read from Pulse table      
        LinkDistanceRatio = np.array([]) #create 1 x num_pulse_parcels array that lists distance ratio of each link. 
        for i,val in enumerate(SedimentPulseDF['link_downstream_distance'].values):
            LinkDistanceRatio = np.concatenate((LinkDistanceRatio,np.ones(p_np[i])*val)) #all parcels enter channel at single point           
        location_in_link = np.expand_dims(LinkDistanceRatio, axis=1)
        

        # element id and starting link
        element_id = np.array([]) 
        for i, row in SedimentPulseDF.iterrows():       
            element_id = np.concatenate((element_id,np.ones(p_np[i])*row['link_#']))            
        starting_link = element_id.copy()
        element_id = np.expand_dims(element_id.astype(int), axis=1) #change format to 1Xn array

        # specify that parcels are in the links of the network model grid
        grid_element = ["link"]*np.size(element_id)
        grid_element = np.expand_dims(grid_element, axis=1)
        
        # time of arrivial (time instance called)
        time_arrival_in_link = np.full(np.shape(element_id), time, dtype=float)
            
        # All parcels in pulse are in the active layer (1) rather than subsurface (0)
        active_layer = np.ones(np.shape(element_id))
        
        
        if 'rho sediment' in SedimentPulseDF.columns:
            density = np.array([]) 
            for i, row in SedimentPulseDF.iterrows():       
                density = np.concatenate((density,np.ones(p_np[i])*row['rho sediment']))   
            density = np.expand_dims(density, axis=1)
        else:
            density = self._rho_sedimen * np.ones(np.shape(element_id))          
        

        if 'abrasion rate' in SedimentPulseDF.columns:
            abrasion_rate = np.array([]) 
            for i, row in SedimentPulseDF.iterrows():       
                abrasion_rate = np.concatenate((abrasion_rate,np.ones(p_np[i])*row['abrasion rate']))   
            abrasion_rate = np.expand_dims(abrasion_rate, axis=1)
        else:
            abrasion_rate = self._abrasion_rate* np.ones(np.shape(element_id))           
        
        # grain_size = 0.25 * np.ones(np.shape(element_id))
            
        if 'D50 [m]' in SedimentPulseDF.columns and 'D stdev [m]' in SedimentPulseDF.columns:
            grain_size = np.array([]) 
            for i, row in SedimentPulseDF.iterrows():       
                # det d50 and std
                n_parcels = p_np[i]
                d50 = row['D50 [m]']
                stdv = row['D stdev [m]']
                d50_log, std_dev_log = self.calc_lognormal_distribution_parameters(mu_x = d50, sigma_x = stdv)
                grain_size_pulse = np.random.lognormal(d50_log, std_dev_log, n_parcels)
                grain_size = np.concatenate((grain_size,grain_size_pulse))
        else:
            n_parcels = sum(p_np)
            d50 = self._d50
            stdv = self._std_dev
            d50_log, std_dev_log = self.calc_lognormal_distribution_parameters(mu_x = d50, sigma_x = stdv)
            grain_size = np.random.lognormal(d50_log, std_dev_log, n_parcels)       
        
        grain_size = np.expand_dims(grain_size, axis=1)
        
    

        
        item_id = {"grid_element": grid_element,
                 "element_id": element_id}
        
        ############
        # apply np.expand_dims(element_id, axis=1)...may get rid of the need to define zeros for distance in link before parcel was in DataRecord
    
        #(9) construct dictionary of all parcel variables to be entered into data recorder
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
