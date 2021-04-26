import numpy as np

from landlab.data_record import DataRecord


_OUT_OF_NETWORK = -2


class SedimentPulserBase:

    
    """
    This utility is the base class for several classes that 
    prepare input for and run the landlab DataRecord "add_item"
    method on a DataRecord configured for the NetworkSedimentTransporter 
    component. 
    
    In the NetworkSedimentTransporter, items are sediment "parcels". Presently, 
    this utility has two subclasses for adding sediment parcels to the DataRecord:
    
    (1) SedimentPulserAtLinks and (2) SedimentPulserEachParcel

    
    This base class defines a shared __init__ method for the subclasses


    Parameters
    ----------
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
        parcel volume used for all parcels that do not have a specified volume
    abrasion_rate: list of floats, optional
        rate that grain size decreases with distance along channel [mm/km?]
    
    
    

    Examples
    --------
    >>> from landlab import NetworkModelGrid
    >>> from landlab.utils.sediment_pulser_base import SedimentPulserBase

    >>> y_of_node = (0, 100, 200, 200, 300, 400, 400, 125)
    >>> x_of_node = (0, 0, 100, -50, -100, 50, -150, -100)
    >>> nodes_at_link = ((1, 0), (2, 1), (1, 7), (3, 1), (3, 4), (4, 5), (4, 6))
    >>> grid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)
    >>> grid.at_link["channel_width"] = np.full(grid.number_of_links, 1.0)  # m
    >>> grid.at_link["channel_slope"] = np.full(grid.number_of_links, .01)  # m / m
    >>> grid.at_link["reach_length"] = np.full(grid.number_of_links, 100.0)  # m
    >>> make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
    
    >>> Option 1:
    >>> def time_to_pulse(time):
    ...     return True
    >>> make_pulse = SedimentPulser(grid, time_to_pulse=time_to_pulse)
    >>> time = 10
    >>> d50 = [0.3, 0.12]
    >>> num_pulse_parcels = [20, 25]
    >>> P_links = [2, 6]
    >>> P_parcel_volume = [1, 0.5]
    >>> make_pulse(time, d50, num_pulse_parcels,P_links,P_parcel_volume)
    
    >>> Option 2
    >>> parcel_df = pd.DataFrame({'vol [m^3]': [0.5, 0.2, 0.5, 1],
                                  'link_#': [1, 3, 5, 2],
                                  'link_downstream_distance': []})
    >>>
    
    """
    def __init__(
        self,
        grid,
        parcels=None,
        d50 = 0.05,
        std_dev = 0.03,
        rho_sediment=2650.0,
        parcel_volume = 0.5,
        abrasion_rate = 0.0
        ):
        
        self._grid = grid
        self._parcels = parcels
        self._d50 = d50
        self._std_dev = std_dev
        self._rho_sediment = rho_sediment
        self._parcel_volume = parcel_volume
        self._abrasion_rate = abrasion_rate
 
            
    # add checks, messages, prepare tests
    # see NST and Flow Director for test ideas

