import numpy as np
import pandas as pd
from collections import OrderedDict

from landlab import Component, FieldError
from landlab.components import (FlowDirectorMFD, FlowAccumulator, DepressionFinderAndRouter, FlowDirectorSteepest)
from landlab import imshow_grid, imshow_grid_at_node

from landlab.components.mass_wasting_router.landslide_mapper import LandslideMapper as LM
from landlab.components.mass_wasting_runout import MassWastingRunout as MWRu
from landlab.components.mass_wasting_router.mass_wasting_eroder import MassWastingEroder as MWE
from landlab.utils.channel_network_grid_tools import ChannelNetworkToolsMapper


class MassWastingRouter(Component):
    

    '''a component that redistributes mass wasting derived sediment through a 
    watershed by determining what portion of and where the mass wasting sediment 
    enters and travels through the channel network

    This component is designed to couple a mass-wasting model with a fluvial sediment
    transport model. It was developed using the LandslideProbability component 
    and NetworkSedimentTransporter component. Any other mass-wasting model
    or network scale sediment transport model may be coupled with the mass wasting
    router so lang as the inputs and outputs are formatted correctly.
    
    component overview:
        
    Hillslope scale landslides are interpreted from the a raster model grid 
    "Landslide__Probability" or "Factor__of_safety" field.

    Attributes of each landslide are summarized from all raster model grid cells 
    within the landslide and used to approximate an initial landslide 
    volume (and grain size).

    The landslide is routed through the watershed as a debris flow using a cellular
    automata debris flow model that scours, deposits and updates the raster model grid
    field "topographic__elevation".

    An emperical model for sediment delivery to the channel converts zones of
    deposition and scour to a pandas dataframe of sediment location and volumes.

    This component is unit sensitive. Units specific to each required field are
    listed below.

    author: Jeff Keck
    '''
    _name = 'MassWastingRouter'

    _unit_agnostic = False

    _version = 1.0

    _info = {
        
        'mass__wasting_events': {
            "dtype": int,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "1 indicates mass wasting event, 0 is no event",
            },
        
        'mass__wasting_volumes': {
            "dtype": float,
            "intent": "in",
            "optional": True,
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
            MW_to_channel_threshold = 50, # landslide mapping
            terrace_width = 1,
            mass_wasting_threshold = 0.75, 
            min_mw_cells = 1,
            mw_dict = {'critical_slope':[0.1], 
                       'threshold_flux':0.03,
                       'erosion_coefficient':0.02},
            fluvial_erosion_rate = [[0.03,-0.43], [0.01,-0.43]], # mass wasting eroder

            parcel_volume = 0.2, # minimum parcel depth, parcels smaller than this are aggregated into larger parcels
            **kwds):

        """

        Parameters
        ----------
        grid:ModelGrid
            Landlab ModelGrid object
        nmgrid:network model grid
            Landlab Network Model Grid object
        Ct: float
            Contributing area threshold at which channel begin (colluvial channels)
        BCt: float
            Contributing area threshold at which cascade channels begin, which is 
            assumed to be the upper limit of frequent bedload transport
            
        LANDSLIDE MAPPER
        
        MW_to_channel_threshold: float
            Threshold distance between downslope end of landslide and channel.
            Landslides within this distance of the channel area assumed to extend
            to the channel (rather than fail and runout over the downslope regolith)
            [m]
        terrace_width: int
            Width of terrace that boarders the fullvial channel network [cells]
        mass_wasting_threshold: float
            Threshold value of mass_wasting metric above which, the cell is assumed
            to fail.
        min_mw_cells: int
            minimum number of adjacent cells needed for a group of cell to be 
            considered a landslide
        
        
        MASS WASTING RUNOUT
        
        mw_dict : dictionary of key word arguements and values for MassWastingRunout
                    
        
        MASS WASTING ERODER
        
        fluvial_erosion_rate: list of lists
            Each list is the coeficient and exponent of a negative power function
            that predicts fluvial erosion rate [m/storm] as a fuction of time
            since the cell was distrubed by a debris flow or extreme flow. 
            The first list is for the channel cells. The second list is for the 
            terrace cells.
        parcel_volume: float
            minimum parcel depth, parcels smaller than this are aggregated into 
            larger parcels [m3]
        """

        super().__init__(grid)

        if 'topographic__elevation' in grid.at_node:  # redundant
            self.dem = grid.at_node['topographic__elevation']
        else:
            raise FieldError(
                'A topography is required as a component input!')


        # years since disturbance
        if 'years__since_disturbance' in grid.at_node:
            self.years_since_disturbance = self._grid.at_node['years__since_disturbance']
        else:
            self.years_since_disturbance = 25*np.ones(self.dem.shape) #


        self._grid.add_field('topographic__initial_elevation',
                        self._grid.at_node['topographic__elevation'],
                        at='node',
                        copy = True,clobber=True)

        # nodes, reshaped in into m*n,1 array like other mg fields
        self.nodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1],1)
        self.rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1]) #nodes in single column array

        ### network model grid characteristics
        self._nmgrid = nmgrid
        self.nmg_nodes = nmgrid.nodes

        ### Channel extraction parameters
        self.Ct = Ct # Channel initiation threshold [m2]   
        self.BCt = BCt # CA threshold for channels that typically transport bedload [m2] 

        ### prep time
        self._time_idx = 0 # index
        self._time = 0.0 # duration of model run (hours, excludes time between time steps)
        # TODO need to keep track of actual time (add difference between date of each iteration)

        # dictionaries of variables for each iteration, for plotting
        self.LS_df_dict = {} #  a single dataframe for each storm, each clumps is a row, columns are average attributes of all cells in the clump
        self.LSclump_dict = {} # a single dictionary for each storm, each key is a dataframe with attributes of all cells in the clump
        # self.DF_DEM_dif_dict = {} # dem differening during DF for plots
        self.DFcells_dict = {} # all precipitions during the debris flow routing algorithm
        self.StormDEM_dict = {} # DEM following debris flow and landslides
        self.dem_dz_dict = {} # change in DEM elevation for each model iteration
        self.FED_all = {} # all fluvial erosion depths
        self.FENodes_all = {} # all fluvial erosion nodes
        self.parcelDF_dict = {} #  parcels dataframe, created from aggregated fluvial erosion nodes

        ### initial values
        self._grid.at_node['mass__wasting_events'] = np.zeros(self.nodes.shape[0]).astype(int)
        self._grid.at_node['mass__wasting_volumes'] = np.zeros(self.nodes.shape[0])
      
        # instantiate channel network grid tools
        self.gt = ChannelNetworkToolsMapper(grid = grid, nmgrid = nmgrid, Ct = Ct, BCt = BCt)

        ## define colluvial and fluvial channel networks in the raster model grid
        self.gt.extract_channel_nodes(Ct,BCt)        

        ## create the nmg to rmg node mapper
        self.gt.map_rmg_nodes_to_nmg_nodes()

        # define nmg node elevation based on rmg channel nodes...this is only needed 
        # during run one step, move to run_one_step, slope is then recomputed in 
        # NST's run_one_step
        self.gt.transfer_rmg_channel_node_field_to_nmg_node_field()

        ### class instance of LandslideMapper      
        self.Landslides = LM(self._grid,
             Ct = Ct, 
             BCt  = BCt,
             MW_to_channel_threshold = MW_to_channel_threshold, 
             mass_wasting_threshold = mass_wasting_threshold, 
             min_mw_cells = min_mw_cells,
             )
        print('instantiated LM')
        
        ### class instance of MassWastingRunout
        # add grid to mw_dict
        mw_dict['grid'] = self._grid
        # save needs to be on when run with MWRo
        mw_dict['save'] = True
        # set initial mass wasting id
        self._grid.at_node['mass__wasting_id'] = np.zeros(self._grid.number_of_nodes).astype(int)
        # set initial recieiver nodes
        self._grid.at_node['flow__receiver_proportions'] = np.zeros((self._grid.number_of_nodes,8))
        self.DebrisFlows = MWRu(**mw_dict)       
        print('instantiated MWRunout')

        ### class instance of MassWastingEroder
        self.DepositEroder = MWE(
                    self._grid,
                    self._nmgrid,
                    Ct = Ct,
                    BCt = BCt,
                    terrace_width = terrace_width,#self._nmgrid,
                    fluvial_erosion_rate = fluvial_erosion_rate, # Fluvial erosion rate parameters
                    parcel_volume = parcel_volume, # minimum parcel depth, parcels smaller than this are aggregated into larger parcels
                    )

        
        self.xyDf_t = self.DepositEroder.gti.xyDf_t
        print('instantiated eroder')

    def _transfer_rmg_node_field_to_nmg_node_field(self):
        '''
        updates the elevation of the nmg nodes based on the closest channel rmg node
        updates the link slopes based on the updated nmg node elevations
        move to channel_network_grid_tools?

        Returns
        -------
        None.
        '''

        # update elevation
        for i, node in enumerate(self.nmg_nodes):
            RMG_node = self.gt.NMGtoRMGnodeMapper[i]
            self._nmgrid.at_node['topographic__elevation'][i] = self._grid.at_node['topographic__elevation'][RMG_node]

        # update slope # may need to turn off because this can lead to negative slope, which throws off NST and flow_resistance_RR in DHG
        # doesn't seem to work
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
                b. if parcels to be released, releases parcels at a random location within the length of the deposit
                c. updates the remaining volume and time since deposit


        Parameters
        ----------
        dt : float
            Time since the last bedload transporting storm


        """
        
        ## map landslides, if any
        self.Landslides.run_one_step()
        self.LS_df_dict[self._time_idx] = self.Landslides.LS_df.copy()
        self.LSclump_dict[self._time_idx] = self.Landslides.LSclump.copy()
        # print('ran landslide mapper')
        # print(self.Landslides.LS_df)
        
        if self.Landslides.LS_df.empty is False:
            
            ## run multi-flow director needed for debris flow routing
            self._multidirectionflowdirector()
            # print('multiflowdirection')
            
            ## route debris flows, update dem   
            self.dem_before_MWR = self.DebrisFlows._grid.at_node['topographic__elevation'].copy()
            self.DebrisFlows.run_one_step()
            self.DFcells_dict[self._time_idx] = self.DebrisFlows.saver.arn_r
            self._grid.at_node['MW__disturbed'] = np.abs(self.dem_before_MWR - self.DebrisFlows._grid.at_node['topographic__elevation'])>0 # may not need to call dem from MWR here
            self.StormDEM_dict[self._time_idx] = self.DebrisFlows._grid.at_node['topographic__elevation'].copy()  # copy of DEM before fluvial erosion              
            # print('scour and deposition')
       
            ## update NMG node elevation
            self._transfer_rmg_node_field_to_nmg_node_field()
            # print('updated NMG node elevation')
               
            ## reset landslide probability field
            self._grid.at_node['MW__probability'] = np.zeros(self._grid.at_node['topographic__elevation'].shape[0])
        else:
            self.StormDEM_dict[self._time_idx] = self.DebrisFlows._grid.at_node['topographic__elevation'].copy()
            self._grid.at_node['MW__disturbed'] = np.zeros(self._grid.at_node['topographic__elevation'].shape[0], dtype=bool)

        ## determine time since each cell was disturbed by mass wasting process, erode deposits  
        print('iteration {}, ready to run eroder'.format(self._time_idx))
        self.DepositEroder.run_one_step(dt)
        self.FED_all[self._time_idx] = self.DepositEroder.FED
        self.FENodes_all[self._time_idx] = self.DepositEroder.FENodes
        self.parcelDF_dict[self._time_idx] = self.DepositEroder.parcelDF.copy()
        self.dem_dz_dict[self._time_idx] = self.DepositEroder.dem_mw_dzdt
        
        ## re-run d8flowdirector
        ## (clumping and down-slope distance computations require d-8 flow direction)
        self._d8flowdirector()
        # print('reset flow directions to d8')

        self._time += dt  # cumulative modeling time (not time or time stamp)
        self._time_idx += 1  # update iteration index

