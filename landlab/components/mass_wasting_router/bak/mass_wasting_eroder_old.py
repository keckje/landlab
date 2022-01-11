import numpy as np
import pandas as pd
from collections import OrderedDict

from landlab import Component, FieldError
from landlab.components import (FlowDirectorMFD, FlowAccumulator, DepressionFinderAndRouter,FlowDirectorSteepest)
from landlab import imshow_grid, imshow_grid_at_node


from landlab.utils.grid_t_tools import GridTTools


class MassWastingEroder(GridTTools):
    

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
            FluvialErosionRate = [[1,1], [1,1]], # [[0.03,-0.43], [0.01,-0.43]], # Fluvial erosion rate parameters
            parcel_volume = 0.2, # minimum parcel depth, parcels smaller than this are aggregated into larger parcels
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
        super().__init__(grid, nmgrid, Ct, BCt)


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





        ### fluvial erosion
        self.C_a = FluvialErosionRate[0][0]
        self.C_b = FluvialErosionRate[0][1]
        self.T_a = FluvialErosionRate[1][0]
        self.T_b = FluvialErosionRate[1][1]

        # minimum parcel size (parcels smaller than this are aggregated into a single parcel this size)
        self.parcel_volume = parcel_volume


        #### Define the raster model grid representation of the network model grid
        out = self._LinktoNodes(linknodes = self.linknodes,
                                active_links = self._nmgrid.active_links,
                                nmgx = self.nmgridx, nmgy = self.nmgridy)

        self.Lnodelist = out[0]
        self.Ldistlist = out[1]
        self.xyDf = pd.DataFrame(out[2])

        ## define bedload and debris flow channel nodes
        ## channel
        self._ChannelNodes()

        ## terrace
        self._TerraceNodes()

        ## define fluvial erosion rates of channel and terrace nodes (no fluvial erosion on hillslopes)
        self._DefineErosionRates()



        ### create initial values
        # self.parcelDF = pd.DataFrame([]) # initial parcel DF
        self.dem_initial = self._grid.at_node['topographic__initial_elevation'].copy() # set initial elevation
        self.dem_previous_time_step = self._grid.at_node['topographic__initial_elevation'].copy() # set previous time step elevation
        self.dem_dz_cumulative = self.dem - self.dem_initial
        self.dem_mw_dzdt = self.dem_dz_cumulative # initial cells of deposition and scour - none, TODO, make this an optional input
        self.DistNodes = np.array([])
        self.FED = np.array([])

        
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
            if i in self.TerraceNodes:
                coefL.append(np.array([self.T_a, self.T_b]))
            elif i in self.ChannelNodes:
                coefL.append(np.array([self.C_a, self.C_b]))
            else:
                coefL.append(np.array([0, 0]))


        self.FluvialErosionRate = np.array(coefL,dtype = 'float')


    def _TerraceNodes(self):
        """MWR

        """
     
        for i in range(self.TerraceWidth):
            if i == 0:
                # diagonal adjacent nodes to channel nodes
                AdjDN =np.ravel(self._grid.diagonal_adjacent_nodes_at_node[self.BedloadChannelNodes])  
                # adjacent nodes to channel nodes
                AdjN = np.ravel(self._grid.adjacent_nodes_at_node[self.BedloadChannelNodes])
            elif i>0:
                # diagonal adjacent nodes to channel nodes
                AdjDN = self._grid.diagonal_adjacent_nodes_at_node[TerraceNodes] 
                # adjacent nodes to channel nodes
                AdjN = self._grid.adjacent_nodes_at_node[TerraceNodes]            
            
            # all adjacent nodes to channel nodes
            AllNodes = np.concatenate((AdjN,AdjDN))
            # unique adjacent nodes
            AllNodes = np.unique(AllNodes)
            # unique adjacent nodes, excluding all channel nodes.
            TerraceNodes = AllNodes[np.in1d(AllNodes,self.ChannelNodes,invert = True)]
        
        t_x = self._grid.node_x[TerraceNodes]
        t_y = self._grid.node_y[TerraceNodes]
        self.xyDf_t = pd.DataFrame({'x':t_x, 'y':t_y})
        self.TerraceNodes = TerraceNodes
        
        
    def _dem_dz(self):
        """determine change in dem since the last time step

        Returns
        -------
        None.

        """

        self.dem_mw_dzdt = self._grid.at_node['topographic__elevation'] - self.dem_previous_time_step
        # print('max change dem due to debris flows')
        # print(self.dem_mw_dzdt.max())


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
        self.years_since_disturbance[self.dem_mw_dzdt != 0] = 13/365 # 13 days selected to match max observed single day sediment transport following disturbance


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
            DistMask = np.abs(self.dem_mw_dzdt) > 0 # deposit/scour during last debris flow was greater than threhsold value.
            self.NewDistNodes = self.rnodes[DistMask] # all node ids that have disturbance larger than minimum value

            #new deposit and scour cell ids are appeneded to list of disturbed cells
            self.DistNodes = np.unique(np.concatenate((self.DistNodes,self.rnodes[DistMask]))).astype(int)
            FERateC  = self.FluvialErosionRate[self.DistNodes] # fluvial erosion rate coefficients
            cnd_msk = FERateC[:,0] > 0 # channel nodes only mask, no ersion applied to hillslope nodes
            self.FERateC = FERateC[cnd_msk] # coefficients of erosion function for channel nodes

            self.FENodes = self.DistNodes[cnd_msk]
            # print('no disturbed cells')

            if len(self.FENodes)>0:

                FERateCs = np.stack(self.FERateC) # change format
                # determine erosion depth


                # time since distubrance [years]
                self.YSD = self.years_since_disturbance[self.FENodes]

                # fluvial erosion rate (depth for storm) = a*x**b, x is time since distubrance #TODO: apply this to terrace cells only
                self.FED = FERateCs[:,0]*self.YSD**FERateCs[:,1]

                # force cells that have not been disturbed longer than ___ years to have no erosion
                # FED[YSD>3] = 0

                # Where FED larger than regolith thickness, FED equals regolith thickness (fresh bedrock is not eroded)
                MxErMask = self._grid.at_node['soil__thickness'][self.FENodes]< self.FED
                self.FED[MxErMask] = self._grid.at_node['soil__thickness'][self.FENodes][MxErMask]

                # update regolith depth :
                self._grid.at_node['soil__thickness'][self.FENodes] = self._grid.at_node['soil__thickness'][self.FENodes]-self.FED

                # update dem
                self._grid.at_node['topographic__elevation'][self.FENodes] = self._grid.at_node['topographic__elevation'][self.FENodes]-self.FED


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

        TODO: make self.parcel_volume automatically determined based on a maxmum
        number of parcels per iteration

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
            return row['link_downstream_distance [m]']/self.linklength[int(row['link_#'])]

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
                        mwlink = OrderedDict({'mw_unit':h,'vol [m^3]':self.FEV[h],'raster_grid_cell_#':FEDn,'link_#':linkID,'link_cell_#':search,'raster_grid_to_link_offset [m]':offset,'link_downstream_distance [m]':ld})
                        Lmwlink.append(mwlink)
                        break #for now use the first link found - later, CHANGE this to use largest order channel
                    else:
                        if i ==  len(self.Lnodelist):
                            print(' DID NOT FIND A LINK NODE THAT MATCHES THE DEPOSIT NODE ')


            parcelDF = pd.DataFrame(Lmwlink)
            pLinkDistanceRatio = parcelDF.apply(LDistanceRatio,axis=1)
            pLinkDistanceRatio.name = 'link_downstream_distance'
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

        # determine time since each cell was disturbed by mass wasting process
        self._TimeSinceDisturbance(dt)
        # print('determed time since last mass wasting disturbance')
        # fluvially erode any recently disturbed cells, create lists of
        # cells and volume at each cell that enters the channel network
        self._FluvialErosion()
        # print('fluvial erosion')

        # self._parcelAggregator()
        # print('aggregating parcels')
        
        self.dem_previous_time_step = self._grid.at_node['topographic__elevation'].copy()
        
        self.dem_dz_cumulative = self._grid.at_node['topographic__elevation'] - self.dem_initial

        # convert list of cells and volumes to a dataframe compatiable with
        # the sediment pulser utility
        self._parcelDFmaker()
        


