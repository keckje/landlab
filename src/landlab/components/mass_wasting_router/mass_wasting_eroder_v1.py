import numpy as np
import pandas as pd
from collections import OrderedDict

from landlab import Component, FieldError
from landlab.components import (FlowDirectorMFD, FlowAccumulator, DepressionFinderAndRouter,FlowDirectorSteepest)
from landlab import imshow_grid, imshow_grid_at_node
# from landlab.utils.channel_network_grid_tools import ChannelNetworkGridTools
from landlab.utils.channel_network_grid_tools import ChannelNetworkToolsInterpretor, ChannelNetworkToolsMapper

class MassWastingEroder(Component):
    

    '''a component that converts zones of deposition and scour recorded on a 
    raster model grid representation of a watershed into sediment pulses that 
    can then be inserted into a network model grid representation of the channel 
    network.

    This component is unit sensitive. Units specific to each required field are
    listed below.


    TODO: erode channel node elevations using flow rate at grid cell
          record datetime of each timestep
          add parameter for using different router and option to have no terrace cells (for models with larger grid cells)

    author: Jeff Keck
    '''

    _name = 'MassWastingEroder'

    _unit_agnostic = False

    _version = 1.0

    
    _info = {
        
        
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
            }

#%%
    def __init__(
            self,
            grid,
            nmgrid,
            Ct = 5000,
            BCt = 100000,
            TerraceWidth = 1,
            fluvial_erosion_rate = [[1,1], [1,1]], # [[0.03,-0.43], [0.01,-0.43]], # Fluvial erosion rate parameters
            parcel_volume = 0.2, # minimum parcel depth, parcels smaller than this are aggregated into larger parcels
            gti = None,
            **kwds):

        """

        Parameters
        ----------
        grid:ModelGrid
            Landlab ModelGrid object
        nmgrid:network model grid
            Landlab Network Model Grid object


        """

        # call __init__ from parent classes
        # super().__init__(grid, nmgrid, Ct, BCt)
        super().__init__(grid)

        
        if 'topographic__elevation' in grid.at_node:
            self.dem = grid.at_node['topographic__elevation']
        else:
            raise FieldError(
                'A topography is required as a component input!')


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

    
        if 'topographic__initial_elevation' in grid.at_node:
            self._grid.add_field('topographic__initial_elevation', 
                           grid.at_node['topographic__initial_elevation'],
                           at='node',
                           copy = True,clobber=True)
        else:
            self._grid.add_field('topographic__initial_elevation',
                            self._grid.at_node['topographic__elevation'],
                            at='node',
                            copy = True,clobber=True)


        #   flow receiver node
        if 'flow__receiver_node' in grid.at_node:
            self.frnode = grid.at_node['flow__receiver_node']
        else:
            raise FieldError(
                'A flow__receiver_node field is required as a component input!')  

        # instantiate channel network grid tools or use  provided instance
        if gti != None:
            self.gti = gti
            self.gtm = ChannelNetworkToolsMapper(grid = grid, nmgrid = nmgrid)
        else:
            self.gti = ChannelNetworkToolsInterpretor(grid = grid, nmgrid = nmgrid, Ct = Ct,BCt = BCt)
            self.gtm = ChannelNetworkToolsMapper(grid = grid, nmgrid = nmgrid)
            
        # self.gt = ChannelNetworkGridTools(grid = grid, nmgrid = nmgrid, Ct = Ct,BCt = BCt)

        self.rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1]) #nodes in single column array
               
        ### NM Grid characteristics
        self._nmgrid = nmgrid
        # network model grid characteristics       
        self.linknodes = nmgrid.nodes_at_link #links as ordered by read_shapefile       
        # network model grid node coordinates, translated to origin of 0,0, used to map grid to nmg
        self.nmgridx = nmgrid.x_of_node
        self.nmgridy = nmgrid.y_of_node
        self.linklength = nmgrid.length_of_link 
        # self.nmg_nodes = nmgrid.nodes

        ### Channel extraction parameters
        self.TerraceWidth = TerraceWidth # distance from channel grid cells that are considered terrace grid cells [# cells]     
         
        ### fluvial erosion
        self.C_a = fluvial_erosion_rate[0][0]
        self.C_b = fluvial_erosion_rate[0][1]
        self.T_a = fluvial_erosion_rate[1][0]
        self.T_b = fluvial_erosion_rate[1][1]
        self._no_longer_disturbed_year = 5
        self._disturbance_rule = 'minimum_erosion'#'maximum_time'#'minimum_erosion'
        # minimum parcel size (parcels smaller than this are aggregated into a single parcel this size)
        self.parcel_volume = parcel_volume



        #### these functions may have already been run
        if not hasattr(gti,"ChannelNodes"):
            self.gti.extract_channel_nodes(Ct,BCt)
        if not hasattr(gti,"TerraceNodes"):
            self.gti.extract_terrace_nodes()
        if not hasattr(gti,"xyDf"):
            out = self.gtm.map_nmg_links_to_rmg_nodes(linknodes = self.linknodes,
                                    active_links = self._nmgrid.active_links,
                                    nmgx = self.nmgridx, nmgy = self.nmgridy)
    
            self.Lnodelist = out[0] # all nodes that coincide with link
            self.Ldistlist = out[1] # the downstream distance of each node that coincide with link
            self.xyDf = pd.DataFrame(out[2])

        ## define fluvial erosion rates of channel and terrace nodes (no fluvial erosion on hillslopes)
        self._DefineErosionRates()


        ### create initial values
        # self.parcelDF = pd.DataFrame([]) # initial parcel DF
        self.dem_initial = self._grid.at_node['topographic__initial_elevation'].copy() # set initial elevation
        self.dem_previous_time_step = self._grid.at_node['topographic__initial_elevation'].copy() # set previous time step elevation
        self.dem_dz_cumulative = self.dem - self.dem_initial
        self.dem_mw_dzdt = self.dem_dz_cumulative.copy() # initial cells of deposition and scour - none, TODO, make this an optional input
        self.DistNodes = np.array([])
        self.FED = np.array([])
        
        self.TN = self.gti.TerraceNodes
        self.CN = self.gti.ChannelNodes
        self.channel_nodes = np.unique(np.concatenate((self.TN,self.CN)))
        
    def _DefineErosionRates(self):
        '''defines the coefficient and exponent of a negative power function that
        predicts fluvial erosion rate per storm [L/storm] as a funciton of time
        since the last disturbance.

        function is defined for all channel nodes, including the debris flow channel
        nodes.

        Erosion rate is specified as the parameters of a negative power function
        that predicts fluvial erosion rate [m/storm] relative to time since
        disturbance, indpendent of the magnitude of the flow event.


        Returns
        -------
        None.

        '''
        coefL = []

        # coeficients of fluvial erosion/storm as a function of time


        for i in self.rnodes:
            if i in self.gti.TerraceNodes:
                coefL.append(np.array([self.T_a, self.T_b]))
            elif i in self.gti.ChannelNodes:
                coefL.append(np.array([self.C_a, self.C_b]))
            else:
                coefL.append(np.array([0, 0]))


        self.fluvial_erosion_rate = np.array(coefL,dtype = 'float')

        
    def _dem_dz(self):
        """determine change in dem since the last time step

        Returns
        -------
        None.

        """

        self.dem_mw_dzdt = self._grid.at_node['topographic__elevation'] - self.dem_previous_time_step
        # print('max change dem due to debris flows')
        # print(self.dem_mw_dzdt.max())


    def _find_recently_disturbed(self):
        """create a boolean mask, True values represent grid cells that were 
        disturbed between this and the last model iteration"""

        # add a check for duplicate nodes, is len(CN)+len(TN) same as np.unique(np.concatenate((TN,CN)))
        
        self.recently_disturbed = np.abs(self.dem_mw_dzdt) > 0.01#[channel_nodes]

    def _TimeSinceDisturbance(self, dt):
        """Years since a cell was disturbed advance in time increases period of
        time since the storm event and the last storm event (ts_n - ts_n-1)
        if landslides, time since disturbance in any cell that has a change
        in elevation caused by the debris flows is set to 0 (fluvial erosion does not count)
        
        dt: float
            number of years since the last storm event

        """
        # all grid cells advance forward in time by amount dt
        self.years_since_disturbance+=dt
        # find cells whose elevation changed bdetween this and the last model iteration, 
        # set time to disturbance to a small number in units of years, 13 days selected 
        # to match max observed single day sediment transport following disturbance
        self.years_since_disturbance[self.recently_disturbed] = 13/365

    def _NoLongerDisturbed(self):
        """remove nodes from DistNodes array (list of node ids who represent 
        recently disturbed cells whose time since disturbance exceeds a threshold duration"""
        if self._disturbance_rule == 'maximum_time':
            disturbed_node_mask = self.years_since_disturbance[self.DistNodes] < self._no_longer_disturbed_year
        
        elif self._disturbance_rule == 'minimum_erosion':
            disturbed_node_mask = np.abs(self.dem_mw_dzdt[self.DistNodes]) > 0.01
        
        
        self.DistNodes = self.DistNodes[disturbed_node_mask]


    def _FluvialErosion(self, erosion_model = 'time'):
        '''determine pulse location based on the time since mw caused disturbance
        small changes in cell elevation are ignored

        pulse volume is determined as a function of the time since the mass wasting
        disturbance

        this is run each time step

        NOTE: percentage of debris flow or terrace that becomes a pulse
        of sediment is determined independently of flow magnitude
        '''
        # change in dpeth less than this is ignored
        # dmin = 0.1

        # rnodes = self._grid.nodes.reshape(mg.shape[0]*mg.shape[1]) # reshame mg.nodes into 1d array

        if erosion_model == 'shear_stress':
            if np.any(self.dem_mw_dzdt > 0):
                DistMask = self.dem_mw_dzdt > 0
                self.FENodes = self.rnodes[DistMask]
                self.FED = self.dem_mw_dzdt[DistMask] #change to this as maximum for shear stress based estimate
                self.FEV = self.FED*self._grid.dx*self._grid.dx

        if erosion_model == 'time':

            # if np.any(np.abs(self.dem_mw_dzdt) > 0): # if there are no disturbed cells
                # disturbance (dz>0) mask
            # DistMask = np.abs(self.dem_mw_dzdt) > 0 # # find cells whose elevation changed bdetween this and the last model iteratio,these are the cell that will fluvially erode
            print('number of disturbed cells: {}'.format(self.recently_disturbed.sum()))

            # two options for determining disturbed nodes:
            # 1. new deposit and scour cell ids are appeneded to list of disturbed cells # just use cells that have change
            # only consider disturbed nodes from last landslide event
            self.DistNodes = self.rnodes[self.recently_disturbed]# # nodes that have not been disturbed for a while are never removed from DistNodes

            # 2. include previously disturbed nodes and new landslide disturbed nodes
            # NewDistNodes = self.rnodes[self.recently_disturbed] # all node ids that have disturbance larger than minimum value
            # self.DistNodes = np.unique(np.concatenate((self.DistNodes, NewDistNodes))).astype(int)
            
            # create a mask to only keep the channel nodes
            cnodes_mask = np.isin(self.DistNodes,self.channel_nodes)
            self.DistNodes = self.DistNodes[cnodes_mask]
            # get the fluvial erosion rate parameters of each disturbed cells
            self.FERateC  = self.fluvial_erosion_rate[self.DistNodes]
            
            # remove non-channel nodes from list of erosion rates # this should be done before determing the recently_distrubed cells
            # cnd_msk = FERateC[:,0] > 0 # channel nodes coefficient are greater than 0
            # self.FERateC = FERateC[cnd_msk] 
            
            # array of all fluvial erosion nodes
            self.FENodes = self.DistNodes#[cnd_msk]
            # print('no disturbed cells')

            # if there are fluvial erosion nodes, erode them
            if len(self.FENodes)>0:

                FERateCs = np.stack(self.FERateC) # change format..this may not be needed

                # get time since distubrance [years] at each fluvial erosion node
                self.YSD = self.years_since_disturbance[self.FENodes]
                # print('years since disturbance: {}'.format(self.YSD))
                
                # compute the erosion depth at each FE node  #TODO: apply this to terrace cells only
                self.FED = FERateCs[:,0]*self.YSD**FERateCs[:,1]#(times the ratio of the flow rate to the 2 year flow rate)

                # # force cells that have not been disturbed longer than ___ years to have no erosion
                # self.FED[self.YSD>3] = 0

                # Where FED larger than regolith thickness, FED equals regolith thickness (fresh bedrock is not eroded)
                MxErMask = self._grid.at_node['soil__thickness'][self.FENodes]< self.FED
                self.FED[MxErMask] = self._grid.at_node['soil__thickness'][self.FENodes][MxErMask]

                # update regolith depth :
                self._grid.at_node['soil__thickness'][self.FENodes] = self._grid.at_node['soil__thickness'][self.FENodes].copy()-self.FED

                # update dem
                self._grid.at_node['topographic__elevation'][self.FENodes] = self._grid.at_node['topographic__elevation'][self.FENodes].copy()-self.FED


                # self.FED_all[self._time_idx] = self.FED
                # self.FENodes_all[self._time_idx] = self.FENodes


                # convert depth to volume
                self.FEV = self.FED*self._grid.dx*self._grid.dx

            else:
                print('no disturbed cells to fluvially erode')
                self.FENodes = np.array([])


    def _parcelAggregator(self):
        '''
        reduces the number of parcels entered into a channel network by aggregating
        parcels into larger parcels

        TODO: (1) make self.parcel_volume automatically determined based on a maximum
        number of parcels per iteration
        (2) add constrain that parcels separated  farther than a specified distance can not be aggregated

        Returns
        -------
        None.

        '''
        if len(self.FENodes)>0:
            # aggregates parcels into
            FEV_ag = []
            FENodes_ag = []
            c = 0 # advance through FED using counter c
            while c < len(self.FEV):
                v = self.FEV[c]
                if v < self.parcel_volume:
                    vsum = v
                    ag_node_L = [] # list of nodes aggregated into one parcel
                    while vsum < self.parcel_volume:
                        vsum = vsum+v # add depth of nodes
                        ag_node_L.append(self.FENodes[c])  #add node to list of nodes in parcel
                        c+=1
                        if c >= len(self.FEV):
                            break
                        v = clumps.DepositEroder.FEV[c]
                else:
                    ag_node_L = [self.FENodes[c]]
                    vsum = v
                    c+=1

                FEV_ag.append(vsum) # cumulative depth in each aggregated parcel (will be greater than d_min)
                FENodes_ag.append(ag_node_L[-1]) # the last node in the list is designated as the deposition node

            self.FEV = np.array(FEV_ag)
            self.FENodes = np.array(FENodes_ag)
        else:
            print('no disturbed cells to aggregate')


    def _parcelDFmaker(self):
        '''
        from the list of cell locations of the pulse and the volume of the pulse,
        convert to a dataframe of pulses (ParcelDF) that is the input for pulser
        '''

        def LDistanceRatio(row):
            '''
            # determine deposition location on reach - ratio of distance from
              inlet to deposition location in link to length of link

            '''
            # return row['link_downstream_distance [m]']/self.linklength[int(row['link_#'])]
            return row['link_downstream_distance']/self.linklength[int(row['link_#'])]
        if len(self.FENodes) == 0:
            FEDn = []
            parcelDF = pd.DataFrame([])
            self.parcelDF = parcelDF


        else:
            Lmwlink = []
            for h, FEDn in enumerate(self.FENodes): #for each cell deposit

                depXY = [self._grid.node_x[FEDn],self._grid.node_y[FEDn]] #deposition location x and y coordinate

                #search cells of links to find closest link grid


                #compute distance between deposit and all network cells
                def Distance(row):
                    return ((row['x']-depXY[0])**2+(row['y']-depXY[1])**2)**.5

                nmg_dist = self.xyDf.apply(Distance,axis=1)

                offset = nmg_dist.min()
                mdn = self.xyDf[nmg_dist == offset] #minimum distance node


                #find link that contains raster grid cell

                search = mdn['node'].iloc[0] #node number, first value if more than one grid cell is min dist from debris flow
                for i, sublist in enumerate(self.Lnodelist): #for each list of nodes (corresponding to each link i)
                    if search in sublist: #if node is in list, then
                        link_n = sublist#Lnodelist[i]
                        en = link_n.index(search)
                        link_d = self.Ldistlist[i]
                        ld = link_d[en]
                        linkID = i
                        mwlink = OrderedDict({'mw_unit':h,'pulse_volume':self.FEV[h],'raster_grid_cell_#':FEDn,'link_#':linkID,'link_cell_#':search,'raster_grid_to_link_offset [m]':offset,'link_downstream_distance':ld})
                        # mwlink = OrderedDict({'mw_unit':h,'vol [m^3]':self.FEV[h],'raster_grid_cell_#':FEDn,'link_#':linkID,'link_cell_#':search,'raster_grid_to_link_offset [m]':offset,'link_downstream_distance [m]':ld})
                        Lmwlink.append(mwlink)
                        break #for now use the first link found - later, CHANGE this to use largest order channel
                    else:
                        if i ==  len(self.Lnodelist):
                            print(' DID NOT FIND A LINK NODE THAT MATCHES THE DEPOSIT NODE ')


            parcelDF = pd.DataFrame(Lmwlink)
            pLinkDistanceRatio = parcelDF.apply(LDistanceRatio,axis=1)
            pLinkDistanceRatio.name = 'normalized_downstream_distance'
            self.parcelDF = pd.concat([parcelDF,pLinkDistanceRatio],axis=1)



        # self.parcelDF_dict[self._time_idx] = self.parcelDF.copy() # save a copy of each pulse


    def run_one_step(self, dt):
        """erode the channel and terrace nodes as a function of time since the 
        last disturance or/and the flow rate at the node.

        Parameters
        ----------
        dt : float
            Duration of time to run the NetworkSedimentTransporter forward.


        """

        # subtract previous storm dem from this dem
        self._dem_dz()

        # find which cell changed in elevation due to landslide processes
        self._find_recently_disturbed()
        
        # determine time since each cell was disturbed by mass wasting process
        self._TimeSinceDisturbance(dt)
        # print('determed time since last mass wasting disturbance')
        # fluvially erode any recently disturbed cells, create lists of
        # cells and volume at each cell that enters the channel network
        
        # remove cells that have are no longer disturbed (TimeSinceDisturbance>threshold)
        # only run this if there are DistNodes
        # if len(self.DistNodes) != 0:
        #     self._NoLongerDisturbed()
        
        # before fluvial erosion of DEM, make a copy, disturbed cells are caused
        # by MWR and fluvial erosion
        self.dem_previous_time_step = self._grid.at_node['topographic__elevation'].copy()
        
        self._FluvialErosion()
        # print('fluvial erosion')

        # self._parcelAggregator()
        # # print('aggregating parcels')

        # convert list of cells and volumes to a dataframe compatiable with
        # the sediment pulser utility
        self._parcelDFmaker()
        
        # after fluvial erosion of DEM, make a copy, disturbed cells identified next 
        # iteration are any cells that changed elevation following MWR
        # self.dem_previous_time_step = self._grid.at_node['topographic__elevation'].copy()
        
        self.dem_dz_cumulative = self._grid.at_node['topographic__elevation'] - self.dem_initial
        

