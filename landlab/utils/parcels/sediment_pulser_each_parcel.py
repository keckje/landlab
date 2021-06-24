import numpy as np

from landlab.data_record import DataRecord
from landlab.utils.parcels.sediment_pulser_base import SedimentPulserBase

_OUT_OF_NETWORK = -2

class SedimentPulserTable(SedimentPulserBase):
    
    '''
    Send a pulse of sediment to specific locations of a channel network and divide 
    the pulse into parcels
    
    This utility prepares input for and runs the landlab DataRecord "add_item"
    method on a DataRecord that has been prepared for the NetworkSedimentTransporter. 
    In the NetworkSedimentTransporter, the items are sediment "parcels"
    
    SedimentPulserTable is instantiated by specifying general sediment 
    characteritics of pulses of sediment
    
    
    The sediment pulse dataframe is a pandas dataframe. It has columns for the link,
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
        at specific channel netowrk locations (link, distance on link)
        
    
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
        #(1) create parcels for each landslide pulse

        p_np = []
        if 'parcel_volume [m^3]' in SedimentPulseDF:       
            for index, row in SedimentPulseDF.iterrows():
                p_np.append(int(row['vol [m^3]']/row['parcel_volume [m^3]'])) #number of parcels in pulse = volume pulse/volume 1 parcel        
        else:               
            for index, row in SedimentPulseDF.iterrows():
                p_np.append(int(row['vol [m^3]']/self._parcel_volume)) #number of parcels in pulse = volume pulse/volume 1 parcel
     
        num_pulse_parcels = sum(p_np) # total number of parcels that enter network for timestep t
    
       
        LinkDistanceRatio = np.array([]) #create 1 x num_pulse_parcels array that lists distance ratio of each link. 
        for i,val in enumerate(SedimentPulseDF['link_downstream_distance'].values):
            # print(val)
            if point_pulse:
                LinkDistanceRatio = np.concatenate((LinkDistanceRatio,np.ones(p_np[i])*val)) #enter channel at single point
            else:
                # determine length of deposit, and depth and scale spacing for entry points using logspace?
                LinkDistanceRatio = np.concatenate((LinkDistanceRatio,np.linspace(val,0.99,p_np[i]))) #enter channel distributed from deposition point to end of link
        
        new_location_in_link = np.expand_dims(LinkDistanceRatio, axis=1)
        
        #(3)create 1xnum_pulse_parcels array that lists the link each parcel is entered into.
        newpar_element_id = np.array([]) 
        for i, row in SedimentPulseDF.iterrows():       
            newpar_element_id = np.concatenate((newpar_element_id,np.ones(p_np[i])*row['link_#']))        
    
        newpar_element_id = np.expand_dims(newpar_element_id.astype(int), axis=1) #change format to 1Xn array
        new_starting_link = np.squeeze(newpar_element_id)
    
        #(4)create 1xn array of zeros to append to array of distance parcels traveled before tiemestep t (zero because parcels did not exist before timestep)
        newpar_dist = np.zeros(num_pulse_parcels,dtype=int)          
                 
        #(5) create time stamp of zero for each parcel before parcel existed        
        new_time_arrival_in_link = time* np.ones(
            np.shape(newpar_element_id)) #arrives at current time in nst model
        
        #(6) compute total volume of all parcels entered into network during timestep
        new_volume = self._parcel_volume*np.ones(np.shape(newpar_element_id)) #SedimentPulseDF['vol [m^3]'].values /100  # volume of each parcel (m3) divide by 100 because large parcels break model
        #new_volume = np.expand_dims(new_volume, axis=1)
        
        #(7) assign grain properties -lithology ,activity, density, abrasion rate, diameter,- this can come from dataframe parcelDF
        new_lithology = ["pulse_material"] * np.size(
            newpar_element_id)  
        
        new_active_layer = np.ones(
            np.shape(newpar_element_id))  # 1 = active/surface layer; 0 = subsurface layer
        
        new_density = 2650 * np.ones(np.size(newpar_element_id))  # (kg/m3)
            
        new_abrasion_rate = 0 * np.ones(np.size(newpar_element_id))
        
        try:
            p_parcel_D  = SedimentPulseDF['d50 [m]'] # grain size in parcel : Change to read parcelDF
        except:
            p_parcel_D = self._d50
            
        new_D = p_parcel_D * np.ones(np.shape(newpar_element_id))
    
    
        #(8) assign part of grid that parcel is deposited (node vs link)    
        newpar_grid_elements = np.array(
            np.empty(
                (np.shape(newpar_element_id)), dtype=object)) 
        
        newpar_grid_elements.fill("link")
        
        item_id = {"grid_element": newpar_grid_elements,
                 "element_id": newpar_element_id}
    
        #(9) construct dictionary of all parcel variables to be entered into data recorder
        variables = {
            "starting_link": (["item_id"], new_starting_link),
            "abrasion_rate": (["item_id"], new_abrasion_rate),
            "density": (["item_id"], new_density),
            #"lithology": (["item_id"], new_lithology),
            "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link),
            "active_layer": (["item_id", "time"], new_active_layer),
            "location_in_link": (["item_id", "time"], new_location_in_link),
            "D": (["item_id", "time"], new_D),
            "volume": (["item_id", "time"], new_volume),
        }
        
        print('TOTAL PARCEL VOLUME ADDED THIS STEP')
        print(np.nansum(new_volume))
        return variables,item_id
