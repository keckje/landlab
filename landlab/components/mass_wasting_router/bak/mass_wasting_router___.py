import numpy as np
import pandas as pd
from collections import OrderedDict

from landlab import Component, FieldError
from landlab.components import (FlowDirectorMFD, FlowAccumulator, DepressionFinderAndRouter,FlowDirectorSteepest)
from landlab import imshow_grid, imshow_grid_at_node


from landlab.components.mass_wasting_router.mass_wasting_SED import MassWastingSED
from landlab.utils.grid_t_tools import GridTTools


class MassWastingRouter___(GridTTools):
    

    '''a component that redistributes mass wasting derived sediment through a 
    watershed and determines what portion of and where the sediment enters 
    the channel network

    This component is designed to couple a mass-wasting model with a sediment
    transport model. It has been written using the LandslideProbability component 
    and NetworkSedimentTransporter component. Any other mass-wasting model
    or network scale sediment transport model can be coupled with the mass wasting
    router so lang as the input are formatted correctly.
    
    componenmt overview:
        
    Hillslope scale landslides are interpreted from the a raster model grid 
    "Landslide__Probability" field.

    Attributes of each landslide are summarized from raster model grid fields values
    of all cells in each landslide and used to approximate initial debris flow
    volume and grain size distribution.

    The landslide is routed through the watershed as a debris flow using a cellular
    automata debris flow model that scours, deposits and updates the raster model grid
    field "topographic__elevation".

    An emperical model for sediment delivery to the channel converts zones of
    deposition and scour to a dataframe of sediment location and volumes
    input into the fluvial network.

    This component is unit sensitive. Units specific to each required field are
    listed below.



    TODO: erode channel node elevations using flow rate at grid cell
          record datetime of each timestep
          add parameter for using different router and option to have no terrace cells (for models with larger grid cells)
          clean up doc strings and comments
          tests (review Datacamp class), submittal to LANDLAB
          debris flow erosion model that accounts for available regolith thickness

    WISH LIST
    ADD entrainment model - See Frank et al., 2015, user parameterizes based on literature OR results of model runs in RAMMS
        first draft done

    '''



    _name = 'MassWastingRouter'

    _unit_agnostic = False

    _version = 1.0

    
    _info = {
        
        'mass__wasting_events': {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "1 indicates mass wasting event, 0 is no event",
            },
        
        'mass__wasting_volumes': {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "initial mass wasting volumes",
            },
        
        
        'topographic__elevation': {            
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
            },
        
        'soil__thickness': {            
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "regolith (soil) thickness, measured perpendicular to the \
            land surface and includes all materials above the unweathered bedrock \
            surface, e.g., saprolite, colluvium, alluvium, glacial drift"
            },
        
        "flow__receiver_node": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
            },
        
        'flow__receiver_proportions': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of proportion of flow sent to each receiver.",
            },
        
        'topographic__steepest_slope': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
            
            },

        }

#%%
    def __init__(
            self,
            grid,
            nmgrid,
            Ct = 5000,
            BCt = 100000,
            MW_to_channel_threshold = 50,
            PartOfChannel_buffer = 10, # may not need this parameter
            TerraceWidth = 1,
            probability_threshold = 0.75,  # landslide mapping parameters
            min_mw_cells = 1,
            release_dict = {'number of pulses':8, # MassWastingSED parameters 
                            'iteration delay':5 },
            df_dict = {'critical slope':0.1, 
                       'minimum flux':3,
                       'scour coefficient':0.025},
            FluvialErosionRate = [[0.03,-0.43], [0.01,-0.43]], # Fluvial erosion rate parameters

            parcel_volume = 0.2, # minimum parcel depth, parcels smaller than this are aggregated into larger parcels
            **kwds):

        """

        Parameters
        ----------
        grid:ModelGrid
            Landlab ModelGrid object
        nmgrid:network model grid
            Landlab Network Model Grid object
        movement_type: string
            defines model used for clumping
        material_type: string
            defines model used for clumping

        ###PARAMETERS TO ADD###
            deposits: datarecord with fields for deposit remaining volume, upstream extent (link, position on link),
            downstream extent, coefficient for exponential decay curve that determines rates material is released, flow rate
            that causes release (effective flow approximation), time since deposition

            entrainment parameters

            runout parameters

            velocity parameters

        """

        # call __init__ from parent classes
        super().__init__(grid, nmgrid, Ct, BCt, MW_to_channel_threshold,
                         PartOfChannel_buffer, TerraceWidth)


        #   landslide probability
        if 'MW__probability' in grid.at_node:
            self.mwprob = grid.at_node['MW__probability']
        else:
            raise FieldError(
                'A MW__probability field is required as a component input!')

        #   high probability mass wasting cells
        if 'high__MW_probability' in grid.at_node:
            self.hmwprob = grid.at_node['high__MW_probability']
        else:
            self.hmwprob = grid.add_zeros('node',
                                        'high__MW_probability')


        #   mass wasting clumps
        if 'mass__wasting_clumps' in grid.at_node:
            self.mwclump = grid.at_node['mass__wasting_clumps']
        else:
            self.mwclump = grid.add_zeros('node',
                                        'mass__wasting_clumps')

        #   soil thickness
        if 'soil__thickness' in grid.at_node:
            self.Soil_h = grid.at_node['soil__thickness']
        else:
            raise FieldError(
                'A soil__thickness field is required as a component input!')


        # years since disturbance
        if 'years__since_disturbance' in grid.at_node:
            self.years_since_disturbance = self._grid.at_node['years__since_disturbance']
        else:
            self.years_since_disturbance = 25*np.ones(self.dem.shape) #


        self._grid.add_field('topographic__initial_elevation',
                        self._grid.at_node['topographic__elevation'],
                        at='node',
                        copy = True,clobber=True)


        ### prep LandslideMapper
        self.MW_to_C_threshold = MW_to_channel_threshold # maximum distance [m] from channel for downslope clumping
        self.probability_threshold = probability_threshold # probability of landslide threshold
        self.min_mw_cells = min_mw_cells # minimum number of cells to be a mass wasting clump

        ### prep MassWastingRunout
        self._time_idx = 0 # index
        self._time = 0.0 # duration of model run (hours, excludes time between time steps)
        # TODO need to keep track of actual time (add difference between date of each iteration)

        ### prep MassWastingEroder
        self.C_a = FluvialErosionRate[0][0]
        self.C_b = FluvialErosionRate[0][1]
        self.T_a = FluvialErosionRate[1][0]
        self.T_b = FluvialErosionRate[1][1]
        self.ErRate = 0 # debris flow scour depth for simple runout model

        # minimum parcel size (parcels smaller than this are aggregated into a single parcel this size)
        self.parcel_volume = parcel_volume

        # dictionaries of variables for each iteration, for plotting
        self.LS_df_dict = {} #  a single dataframe for each storm, each clumps is a row, columns are average attributes of all cells in the clump
        self.LSclump_dict = {} # a single dictionary for each storm, each key is a dataframe with attributes of all cells in the clump
        self.DF_DEM_dif_dict = {} # dem differening during DF for plots
        self.DFcells_dict = {} # all precipitions during the debris flow routing algorithm
        self.StormDEM_dict = {} # DEM following debris flow and landslides
        self.FED_all = {} # all fluvial erosion depths
        self.FENodes_all = {} # all fluvial erosion nodes
        self.parcelDF_dict = {} #  parcels dataframe, created from aggregated fluvial erosion nodes

        
        ### create initial values
        self._extractLSCells() # initial high landslide probability grid cells
        self.parcelDF = pd.DataFrame([]) # initial parcel DF
        self.dem_initial = self._grid.at_node['topographic__initial_elevation'].copy() # set initial elevation
        self.dem_previous_time_step = self._grid.at_node['topographic__initial_elevation'].copy() # set previous time step elevation
        self.dem_dz_cumulative = self.dem - self.dem_initial
        self.dem_mw_dzdt = self.dem_dz_cumulative # initial cells of deposition and scour - none, TODO, make this an optional input
        self.DistNodes = np.array([])
        self._grid.at_node['mass__wasting_events'] = np.zeros(self.nodes.shape[0]).astype(int)
        self._grid.at_node['mass__wasting_volumes'] = np.zeros(self.nodes.shape[0])



        ###instantiate LandslideMapper
        
        
        ###instantiate MassWastingRunout
        self.DebrisFlows = MassWastingSED(self._grid,release_dict,df_dict)
        
        
        ###instantiate MassWastingEroder
        
        # move to MWRu and include in MWR
        ## define bedload and debris flow channel nodes
        ## channel
        self._ChannelNodes()        
        
        # move to MWE
        ###instantiate the 
        #### Define the raster model grid representation of the network model grid
        out = self._LinktoNodes(linknodes = self.linknodes,
                                active_links = self._nmgrid.active_links,
                                nmgx = self.nmgridx, nmgy = self.nmgridy)

        self.Lnodelist = out[0]
        self.Ldistlist = out[1]
        self.xyDf = pd.DataFrame(out[2])



        ## terrace
        self._TerraceNodes()

        ## define fluvial erosion rates of channel and terrace nodes (no fluvial erosion on hillslopes)
        self._DefineErosionRates()

        ## create the nmg to rmg node mapper
        self._NMG_node_to_RMG_node_mapper()

        ## define nmg node elevation based on rmg channel nodes
        self._update_NMG_nodes()



        ### instantiate MassWastingSED

        


        self.parcelDF_dict[self._time_idx] = self.parcelDF.copy() # save a copy of each pulse



    def _update_NMG_nodes(self):
        '''
        updates the elevation of the nmg nodes based on the closest channel rmg node
        updates the link slopes based on the updated nmg node elevations

        Returns
        -------
        None.

        '''

        # update elevation
        for i, node in enumerate(self.nmg_nodes):
            RMG_node = self.NMGtoRMGnodeMapper[i]
            self._nmgrid.at_node['topographic__elevation'][i] = self._grid.at_node['topographic__elevation'][RMG_node]

        # update slope
        nmg_fd = FlowDirectorSteepest(self._nmgrid, "topographic__elevation")
        nmg_fd.run_one_step()

    def _multidirectionflowdirector(self):
        '''
        removes rmg fields created by d8 flow director, runs multidirectionflowdirector

        '''
        self._grid.delete_field(loc = 'node', name = 'flow__sink_flag')
        self._grid.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
        self._grid.delete_field(loc = 'node', name = 'flow__receiver_node')
        self._grid.delete_field(loc = 'node', name = 'topographic__steepest_slope')
        # run flow director, add slope and receiving node fields
        fd = FlowDirectorMFD(self._grid, diagonals=True,
                              partition_method = 'slope')
        fd.run_one_step()

    def _d8flowdirector(self):
        '''
        removes fields created by multidirectionflowdirector, runs d8flow director

        '''
        self._grid.delete_field(loc = 'node', name = 'flow__sink_flag')
        self._grid.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
        self._grid.delete_field(loc = 'node', name = 'flow__receiver_node')
        self._grid.delete_field(loc = 'node', name = 'topographic__steepest_slope')
        try:
            self._grid.delete_field(loc = 'node', name = 'flow__receiver_proportions') # not needed?
        except:
            None

        fr = FlowAccumulator(self._grid,'topographic__elevation',flow_director='D8')
        fr.run_one_step()

        df_4 = DepressionFinderAndRouter(self._grid)
        df_4.map_depressions()

    def run_one_step(self, dt):
        """Run MassWastingRouter forward in time.

        When the MassWastingRouter runs forward in time the following
        steps occur:

            1. If there are landslides to be routed:
                a. Determines extent of each landslide
                b. Routes the landslide through the raster model grid dem and
                    determines the raster model grid deposition location
                c. Converts the raster model grid deposition deposition
                    location to a network model grid deposition location

            2. If there are landslide deposits to erode
                a. looks up time since deposition and remaining volume, determines number of parcels to release
                b. checks if volume is greater than number of parcels
                c. if parcels to be released, releases parcels at a random location within the length of the deposit
                d. updates the remaining volume and time since deposit


        Parameters
        ----------
        dt : float
            Duration of time to run the NetworkSedimentTransporter forward.


        """
        
        self.mwprob = self.grid.at_node['MW__probability'] #  update mw probability variable
        # self.hmwprob = self.grid.at_node['high__MW_probability']  #update boolean landslide field
        self._extractLSCells()

        # determine extent of landslides
        self.MappLandslidess.run_one_step(dt)
        print('landslides mapped')

        
    if self._this_timesteps_landslides.any():

            # run multi-flow director needed for debris flow routing
            self._multidirectionflowdirector()
            # print('multiflowdirection')
            # route debris flows, update dem
            
            self.DebrisFlows.run_one_step(dt)
            
            self.DFcells_dict[self._time_idx] = self.DebrisFlows.DFcells
            self.StormDEM_dict[self._time_idx] = self.DebrisFlows._grid.at_node['topographic__elevation'].copy()                
            # print('scour and deposition')

            # subtract previous storm dem from this dem
            self._dem_dz()
            # print('dem differencing to determine mass-wasting deposition and scour zones')

            # update NMG node elevation
            self._update_NMG_nodes()
            # print('updated NMG node elevation')

            # reset landslide probability field
            self._grid.at_node['MW__probability'] = np.zeros(self._grid.at_node['topographic__elevation'].shape[0])

        # determine time since each cell was disturbed by mass wasting process
        
        self.FluvialErosion.run_one_step(dt)

        # re-run d8flowdirector
        # (clumping and down-slope distance computations require d-8 flow direction)
        self._d8flowdirector()
        # print('reset flow directions to d8')


        self._time += dt  # cumulative modeling time (not time or time stamp)
        self._time_idx += 1  # update iteration index

