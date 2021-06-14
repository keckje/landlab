# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:36:00 2019

@author: keckj

TO DO:
    1. ADD bedload and debris flow networks
    2. Use debris flow network to determine clumping extent
    3. add new routing fuction to MWR
    4. create function of converting dem differences into a dataframe of pulses
    
    
    ADD entrainment model - See Frank et al., 2015, user parameterizes based on literature OR results of model runs in RAMMS
        first draft done
    Add function for breaking debris flow volume into parcels and distributing them along channel based on debris flow geometry that user can aparameterize based on deposit geometry - See ParcelLocation.py
        done
        
    Add function for continued erosion of remaining flood plain deposit that stochasticly release sediment from length of deposit at rate defined by an exponential decay curve that use can parameterize based on field observations    
    
    Make Raster to Network Model grid function work for squiggly lines => see network model grid plotting tool for method to get curved line representation
        not necessary
        
    tests, submittal to LANDLAB
"""

from landlab import Component, FieldError
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

from landlab.components import FlowDirectorMFD, FlowAccumulator
from landlab import imshow_grid
from landlab import imshow_grid_at_node


print('nnnnnnnnnnn')

class MassWastingRouter(Component):
    
    '''
        (000) assumption - dem changes are controlled by debris flow deposition. DEm changes caused by scour minor - not included
        (0) Given a landslide probability map, initial_dem and maps of soil and geologic properties
        (1) Determines mass wasting unit area,volume, material
        (2) Routes mass wasting unit downslope
        (3) erodes and deposits, updating the dem
        (5) erodes DEM, # updating DEM, note dem changes limited to 1 cell width
        (6) Converts deposition location from raster model grid to network model grid
        (7) 

        (9) transfer of deposit to nmg
                make this the "rasterdeposit_to link"
                
                reqiored raster model grid fields:
                    
                initial_dem
                new_dem 
                difference
                max difference (initially 0)
                time since difference >= max difference (initially 0)
                probability pulse
                    
                for each time step
                    subtracts initial_dem from dem
                    
                    immediate transfer to network:
                        if raster grid cell is link cell, a percent of volume is entered into the channel network
                    
                    delayed transfer to network
                        if >= max_dfference (once max difference, vegetation killed, so underlying deposit is unstable again)
                            time since difference >= max difference = 0
                            volume = (new_dem-initial_dem)*cell area
                            determine probability of pulse from grid cell during flow event Q P = f(time since max difference,Q/Qbf), Adjust probability funciton so that material is released to match field observed rate.
                            if probability pulse > th and cell volume greater than pulse
                                pulse
                                    convert pulse location to link location
                                update DEM (subtract pulse volume)


        (10) how to deal with parcels upstream of deposiiton?: All parcels removed or ignored? Possibly ignored since likely few parcels in supply limited channels/and new debris is left in channel
    
    Parameters
    ----------
    grid: ModelGrid 
        Landlab Model Grid object with node fields: topographic__elevation, flow_reciever node, masswasting_nodes
    nmgrid: NetworkModelGrid
        Landlab Network Model Grid object

    
    -future Parameters-
    movement_type: string
        defines model used for clumping
    material_type: string
        defines model used for clumping

    
    Construction:
    
        MWR = MassWastingRouter(grid, nmgrid)
              
    '''
    

    
    _name = 'MassWastingRouter'
    
    _unit_agnostic = False
    
    _version = 1.0
    
    _info = {}
    
    # TO DO: move below to _info 
    _input_var_names = (
        'topographic__elevation', 'flow__receiver_node','MW__probability'
    )
    
    _output_var_names = (
    'high__MW_probability','mass__wasting_clumps'
    )
    
    _var_units = {
    'topographic__elevation': 'm',
    'flow__receiver_node': 'node number',
    'MW__probability': 'annual probability of occurance',
    'high__MW_probability': 'binary',    
    'mass__wasting_clumps': 'number assigned to mass wasting clump',
    }
    
    _var_mapping = {
    'topographic__elevation': 'node',
    'flow__receiver_node': 'node',
    'MW__probability': 'node',
    'high__MW_probability':'node',
    'mass__wasting_clumps': 'node'
    }
    
    _var_doc = {
    'topographic__elevation':
        'elevation of the ground surface relative to some datum',
    'flow__receiver_node': 'node that receives flow from each node',
    'MW__probability':
        'annual probability of shallow landslide occurance',
    'high__MW_probability':'nodes input into mass wasting clumping component',
    'mass__wasting_clumps': 
        'unstable grid cells that are assumed to fail as one, single mass'
    }
        
    
    
    def __init__(
            self, 
            grid,
            nmgrid,
            Ct = 5000,
            BCt = 100000,
            MW_to_channel_threshold = 50,
            PartOfChannel_buffer = 10, # may not need this parameter
            TerraceWidth = 1,
            FluvialErosionRate = [[0.03,-0.43], [0.01,-0.43]],
            probability_threshold = 0.75,
            min_mw_cells = 1,
            mass_wasting_type='debrisflow', # use these to use a pre-set set of parameters for router
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
        
        # Store grid, parameters and fields in grid
        
        super().__init__(grid)
        
        # Create fields needed for MWR
        #   Elevation
        if 'topographic__elevation' in grid.at_node:
            self.dem = grid.at_node['topographic__elevation']
        else:
            raise FieldError(
                'A topography is required as a component input!')
        
        #   flow receiver node
        if 'flow__receiver_node' in grid.at_node:
            self.frnode = grid.at_node['flow__receiver_node']
        else:
            raise FieldError(
                'A flow__receiver_node field is required as a component input!')  
        
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
                        copy = True)

        # prep MWR 

        # time
        self._time_idx = 0
        self._time = 0.0

        ### Grid characteristics
        self.gr = grid.shape[0] #number of rows
        self.gc = grid.shape[1] #number of columns
        self.dx = grid.dx #width of cell
        self.dy = grid.dy #height of cell
            
        receivers = self.frnode #receiver nodes (node that receives runoff from node)

        # nodes, reshaped in into m*n,1 array like other mg fields
        self.nodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1],1)
        self.rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1]) #nodes in single column array
                
        self.xdif = grid.node_x[receivers]-grid.node_x[self.rnodes] # change in x from node to receiver node
        self.ydif = (grid.node_y[receivers]-grid.node_y[self.rnodes])*-1 #, change in y from node to receiver node...NOTE: flip direction of y axis so that up is positve
       
        # grid node coordinates, translated to origin of 0,0
        self.gridx = grid.node_x#-grid.node_x[0] 
        self.gridy = grid.node_y#-grid.node_y[0]
        
        # extent of each cell in grid        
        self.ndxe = self.gridx+self.dx/2
        self.ndxw = self.gridx-self.dx/2
        self.ndyn = self.gridy+self.dy/2
        self.ndys = self.gridy-self.dy/2
        
        ### clumping
        self.Ct = Ct # Channel initiation threshold [m2]   
                  
        self.MW_to_C_threshold = MW_to_channel_threshold # maximum distance [m] from channel for downslope clumping        
        self.POCbuffer = PartOfChannel_buffer # distance [m] from a channel cell that is considered part of the channel (used for determining distance between landslide and channel)
                     
        self.probability_threshold = probability_threshold # probability of landslide threshold      
        self.min_mw_cells = min_mw_cells # minimum number of cells to be a mass wasting clump
        
        ### runout
        self._method = 'ScourAndDeposition'

        ### fluvial erosion            
        self.BCt = BCt # CA threshold for channels that typically transport bedload [m2] 
        self.TerraceWidth = TerraceWidth # distance from channel grid cells that are considered terrace grid cells [# cells] 
        self.C_a = FluvialErosionRate[0][0]
        self.C_b = FluvialErosionRate[0][1]
        self.T_a = FluvialErosionRate[1][0]
        self.T_b = FluvialErosionRate[1][1]
        
        self.ErRate = 0 # debris flow scour depth for simple runout model
        
        # dictionaries of variables for each iteration, for plotting
        self.DFcells_dict = {} # dictionary of all computational cells during the debris flow routing algorithm
        self.parcelDF_dict = {} # dictionary of the parcel dataframe
        self.LS_df_dict = {}
        self.LSclump_dict = {}        
        #### Define the raster model grid representation of the channel network
        
        ## nmg geometry and conversion to raster model grid equivalent
        self._nmgrid = nmgrid
        # network model grid characteristics       
        self.linknodes = self._nmgrid.nodes_at_link #links as ordered by read_shapefile       
        # network model grid node coordinates, translated to origin of 0,0, used to map grid to nmg
        self.nmgridx = self._nmgrid.x_of_node
        self.nmgridy = self._nmgrid.y_of_node
        self.linklength = nmgrid.length_of_link
        
        self._LinktoNodes()
                        
        ## define bedload and debris flow channel nodes       
        ## channel
        self._ChannelNodes()        
        ## terrace
        self._TerraceNodes()
        ## define fluvial erosion rates of channel and terrace nodes (no fluvial erosion on hillslopes)
        self._DefineErosionRates()
        
        
        ### Prep MWR
        # extract cells from landslide probability map that will be used as unstable cells
        self._extractLSCells() 
        # create initial parcel DF        
        self.parcelDF = pd.DataFrame([]) 
        
        self.dem_initial = self._grid.at_node['topographic__initial_elevation'].copy() # set initial elevation
        self.dem_previous_time_step = self._grid.at_node['topographic__initial_elevation'].copy() # set previous time step elevation
        self.dem_dz_cumulative = self.dem - self.dem_initial
        self.dem_mw_dzdt = self.dem_dz_cumulative # initial cells of deposition and scour - none, TODO, make this an optional input        
        self.DistNodes = np.array([])


    def _LinktoNodes(self):
        '''
        #convert links to coincident nodes
            #loop through all links in network grid to determine raster grid cells that coincide with each link
            #and equivalent distance from upstream node on link
        '''
        
        def LinktoNodes_code(linknodes, active_links, nmgx, nmgy):
            Lnodelist = [] #list of lists of all nodes that coincide with each link
            Ldistlist = [] #list of lists of all nodes that coincide with each link
            xdDFlist = []
            Lxy= [] #list of all nodes the coincide with the network links
                  
            for k,lk in enumerate(linknodes) : #for each link in network grid
                
                linkID = active_links[k] #link id (indicie of link in nmg fields)
                
                lknd = lk #link node numbers
                
                x0 = nmgx[lknd[0]] #x and y of downstream link node
                y0 = nmgy[lknd[0]]
                x1 = nmgx[lknd[1]] #x and y of upstream link node
                y1 = nmgy[lknd[1]]
                
                #create 1000 points along domain of link
                X = np.linspace(x0,x1,1000)
                Xs = X-x0 #change begin value to zero
                
                #determine distance from upstream node to each point
                #y value of points
                if Xs.max() ==0: #if a vertical link (x is constant)
                    vals = np.linspace(y0,y1,1000)
                    dist = vals-y0 #distance along link, from downstream end upstream
                    dist = dist.max()-dist #distance from updtream to downstream
                else: #all their lines
                    vals = y0+(y1-y0)/(x1-x0)*(Xs)
                    dist = ((vals-y0)**2+Xs**2)**.5
                    dist = dist.max()-dist #distance from updtream to downstream
                
                    
              
               #match points along link (vals) with grid cells that coincide with link
                
                nodelist = [] #list of nodes along link
                distlist = [] #list of distance along link corresponding to node
                # print(vals)
                for i,v in enumerate(vals):
                    
                    x = X[i]
                    
                    mask = (self.ndyn>=v) & (self.ndys<=v) & (self.ndxe>=x) & (self.ndxw<=x)  #mask - use multiple boolian tests to find cell that contains point on link
                    
                    node = self.nodes[mask] #use mask to extract node value
                    # print(node)
                    if node.shape[0] > 1:
                        node = np.array([node[0]])
                    # print(node)
                    # create list of all nodes that coincide with linke
                    if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                        nodelist.append(node[0][0])  
                        distlist.append(dist[i])
                        xy = {'linkID':linkID,
                            'node':node[0][0],
                              'x':self.gridx[node[0][0]],
                              'y':self.gridy[node[0][0]]}
                        Lxy.append(xy)
                
                Lnodelist.append(nodelist)
                Ldistlist.append(distlist)
                
            return (Lnodelist, Ldistlist, Lxy)

        # if NetType == 'bedload':
        linknodes = self.linknodes
        active_links = self._nmgrid.active_links
        nmgx = self.nmgridx
        nmgy = self.nmgridy

        out = LinktoNodes_code(linknodes, active_links, nmgx, nmgy)

        self.Lnodelist = out[0]
        self.Ldistlist = out[1]
        self.xyDf = pd.DataFrame(out[2])
                   
    def _ChannelNodes(self):
        
        # to top of debris flow channel (top colluvial channel)
        ChannelNodeMask = self._grid.at_node['drainage_area'] > self.Ct
        df_x = self._grid.node_x[ChannelNodeMask]
        df_y = self._grid.node_y[ChannelNodeMask]
        self.xyDf_d = pd.DataFrame({'x':df_x, 'y':df_y})
        self.ChannelNodes = self.rnodes[ChannelNodeMask] 
        
        # to top of bedload channels (~top cascade channels)
        BedloadChannelNodeMask = self._grid.at_node['drainage_area'] > self.BCt
        bc_x = self._grid.node_x[BedloadChannelNodeMask]
        bc_y = self._grid.node_y[BedloadChannelNodeMask]
        self.xyDf_bc = pd.DataFrame({'x':bc_x, 'y':bc_y})
        self.BedloadChannelNodes = self.rnodes[BedloadChannelNodeMask] 

    def _TerraceNodes(self):
     
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

    
    def _DefineErosionRates(self):
        '''
        sets the erosion rate of the terrace and channel cells.
        
        channel cells include the debris flow channels.
        
        terrace nodes are all nodes adjacent to the bedload channel cells.
        
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

    
    def _min_distance_to_network(self, cellid, ChType = 'debrisflow'):
        def distance_to_network(row):
            '''
            compute distance between a cell and the nearest debris flow network 
            cell used to determine clump distance to colluvial channel network
            for clumping algorithm
            
            ChType = debrisflow: uses debris flow network
            ChType = nmg: uses network model grid network
            
            '''
            return ((row['x']-self.gridx[cellid])**2+(row['y']-self.gridy[cellid])**2)**.5
        
        if ChType == 'debrisflow':
            nmg_dist = self.xyDf_d.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDf_d[nmg_dist == offset] # minimum distance node and node x y        
        elif ChType == 'nmg':
            nmg_dist = self.xyDf.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDf[nmg_dist == offset] # minimum distance node and node x y    

        return offset, mdn


    def _downslopecells(self,StartCell):
        '''
        compute distance between a given cell and the nearest downslope channel 
        network cell. distance is computed to POCbuffer distance from channel network
        cell. Track which cells are downslope
        
        channel network is the colluvial channel network
        
        this function is used to determine the lower extent of a landslide.
        
        use this to determine 
        
        (1) check if clump meets distance threshold (is close enough to channel network)
         to causethe hillslope below to fail         
        
        (2) if it does, get list of all unique cells and add to clump
        
        '''

        loc = []
        dist = []
        c = 0
        
        loc.append(int(StartCell))
        dist.append((self.xdif[int(StartCell)]**2+self.ydif[int(StartCell)]**2)**.5)
                       
        flow = True
        
        while flow == True:
            
            slope  = self.grid.at_node['topographic__slope'][loc[c]]
            #compute distance between deposit and all debris flow network cells
            cdist, nc = self._min_distance_to_network(loc[c],  ChType = 'debrisflow')
            #TO DO need to change so that if distance to network is less than minimum, then stop
            
            
            #if loc[c] not in self.xyDf['node'].values: # channel network doesnt always match DEM
            # print(cdist)
            if cdist > self.POCbuffer: # downslope distance measured to POCbuffer from channel    
                loc.append(self.grid.at_node['flow__receiver_node'][loc[c]])
                dist.append((self.xdif[self.grid.at_node['flow__receiver_node'][loc[c]]]**2+self.ydif[self.grid.at_node['flow__receiver_node'][loc[c]]]**2)**.5)
                c=c+1
                if loc[-1] == loc[-2]: # check that runout is not stuck at same node # NEED TO DEBUG
                    break
            else:
                flow = False
        
        Dista = np.array(dist)
        Dist = Dista.sum()
        return (Dist,loc)
        
    
    def _extractLSCells(self):
        '''
        extract areas that exceed annual probability threshold
        '''
        a = self.mwprob 
        
        type(a) # np array
        
        # keep as (nxr)x1 array, convert to pd dataframe
        a_df = pd.DataFrame(a)
                    
        mask = a_df > self.probability_threshold
        
        # boolean list of which grid cells are landslides
        self._this_timesteps_landslides = mask.values

        
        a_th = a_df[mask] #Make all values less than threshold NaN
        
        a_thg = a_th.values
        
        #add new field to grid that contains all cells with annual probability greater than threshold
        self.hmwprob = a_thg #consider removing this instance variable
        
        self.grid.at_node['high__MW_probability'] = self.hmwprob
                   
        #create mass wasting unit list           
        self.LS_cells = self.nodes[mask]
        # print(self.LS_cells)            


    
    def AdjCells(self,n):
        '''                
        returns cell numbers of cells adjacent to cell (n) AND the flow direction indexes
        
        gr - number of rows in grid
        gc - number of columns in grid
        n - cell number other cells are adjacent to
        ac - node numbers
        acn - a
        
        #change to use mg.adjacent_nodes_at_node[] and mg.diagonal_adjacent_nodes_at_node[]

        '''
        gc = self.gc
        gr = self.gr
        
        u = n-gc
        ur = n-gc+1
        r = n+1
        br = n+gc+1
        b = n+gc
        bl = n+gc-1
        l = n-1
        ul = n-gc-1   
        
        
        if n == 0:                      #top left corner
            ac = [r,br,b]
            acn = [2,3,4]               #flow directio index
        elif n == gc-1:                 #top right corner
            ac = [b,bl,l]
            acn = [4,5,6]
        elif n == gr*gc-1:              #bottom right corner
            ac = [u,l,ul]  
            acn = [0,6,7]
        elif n == gr*gc-gc:             #bottom left corner
            ac = [u,ur,r]
            acn = [0,1,2]
        elif n<gc-1 and n>0:            #top side
            ac = [r,br,b,bl,l]
            acn = [2,3,4,5,6]
        elif n%gc == gc-1:              #right side
            ac = [u,b,bl,l,ul]
            acn = [0,4,5,6,7]
        elif n<gr*gc-1 and n>gr*gc-gc:  #bottom side
            ac = [u,ur,r,l,ul]
            acn = [0,1,2,6,7]
        elif n%gc == 0:                 #left side
            ac = [u,ur,r,br,b]
            acn = [0,1,2,3,4]
        else:                           #inside grid
            ac = [u,ur,r,br,b,bl,l,ul]
            acn = [0,1,2,3,4,5,6,7]
            
        return ac,acn #adjacent cell numbers and cell numbers for direction grid

    
    def NotDivergent(self,n,ac,acn):
        '''        
        takes mass wasting cell and adjacent cell numbers as input
        returns boolian list of cells that are not divergent (convergent or planar)
        
        xdif - change in x in slope direction to receiving cell
        ydif -  change in y in slope direciton to receiving cell
        n - mass wasting unit cell
        ac - grid number of adjacent cells
        acn - adjacent cell number
        '''
        
        xd = self.xdif[n]
        yd = self.ydif[n]
        
        '''       
        functions that check if cell is not divergent based on center cell direction
        
        Input to functions:
        x - adjacent cell number (ordered 0 to 7, with 0 north of center, clockwise)
        yda - change in y direction at adjacent cell to it's receiving cell
        xda - change in x direction at adjacent cell to it's receiving cell
            '''
    
        def pyzx(x,yda,xda): #positve y zero x  (slopes north)
    
            return {            
                0: True,
                1: (yda > 0 and xda == 0) or (yda > 0 and xda < 0) or (yda == 0 and xda < 0),    
                2: (yda > 0 and xda == 0) or (yda > 0 and xda < 0) or (yda == 0 and xda < 0),
                3: (yda > 0 and xda == 0) or (yda > 0 and xda < 0),
                4: (yda > 0 and xda == 0),
                5: (yda > 0 and xda == 0) or (yda > 0 and xda > 0),
                6: (yda > 0 and xda == 0) or (yda > 0 and xda > 0) or (yda == 0 and xda > 0),
                7: (yda > 0 and xda == 0) or (yda > 0 and xda > 0) or (yda == 0 and xda > 0),
                    
            }[x]
        
        def nyzx(x,yda,xda): #negative y, zero x (slopes south)
            
            return {
                0: (yda < 0 and xda == 0),
                1: (yda < 0 and xda == 0) or (yda < 0 and xda < 0),
                2: (yda < 0 and xda == 0) or (yda < 0 and xda < 0) or (yda == 0 and xda < 0),
                3: (yda < 0 and xda == 0) or (yda < 0 and xda < 0) or (yda == 0 and xda < 0),
                4: True,
                5: (yda < 0 and xda == 0) or (yda < 0 and xda > 0) or (yda == 0 and xda > 0),
                6: (yda < 0 and xda == 0) or (yda < 0 and xda > 0) or (yda == 0 and xda > 0),
                7: (yda < 0 and xda == 0) or (yda < 0 and xda > 0),
                
            }[x]
            
        def zypx(x,yda,xda): #zero y, positive x (slopes east)
            
            return {
                0: (yda == 0 and xda > 0) or (yda < 0 and xda > 0) or (yda < 0 and xda == 0),
                1: (yda == 0 and xda > 0) or (yda < 0 and xda > 0) or (yda < 0 and xda == 0),
                2: True,
                3: (yda == 0 and xda > 0) or (yda > 0 and xda > 0) or (yda > 0 and xda == 0),
                4: (yda == 0 and xda > 0) or (yda > 0 and xda > 0) or (yda > 0 and xda == 0),
                5: (yda == 0 and xda > 0) or (yda > 0 and xda > 0),
                6: (yda == 0 and xda > 0),
                7: (yda == 0 and xda > 0) or (yda < 0 and xda > 0),
                
            }[x]
        
        def zynx(x,yda,xda): #zero y, negative x (slopes west)
            
            return {
                0: (yda == 0 and xda < 0) or (yda < 0 and xda < 0) or (yda < 0 and xda == 0),
                1: (yda == 0 and xda < 0) or (yda < 0 and xda < 0),
                2: (yda == 0 and xda < 0),
                3: (yda == 0 and xda < 0) or (yda > 0 and xda < 0),
                4: (yda == 0 and xda < 0) or (yda > 0 and xda < 0) or (yda > 0 and xda == 0), 
                5: (yda == 0 and xda < 0) or (yda > 0 and xda < 0) or (yda > 0 and xda == 0),
                6: True,
                7: (yda == 0 and xda < 0) or (yda < 0 and xda < 0) or (yda < 0 and xda == 0),
                
            }[x]
        
        def pypx(x,yda,xda): #positive y, positive x (slopes northeast)
            
            return {
                0: (yda > 0 and xda > 0) or (yda == 0 and xda > 0) or (yda < 0 and xda > 0), 
                1: True,
                2: (yda > 0 and xda > 0) or (yda > 0 and xda == 0) or (yda > 0 and xda < 0),
                3: (yda > 0 and xda > 0) or (yda > 0 and xda == 0) or (yda > 0 and xda < 0), 
                4: (yda > 0 and xda > 0) or (yda > 0 and xda == 0), 
                5: (yda > 0 and xda > 0), 
                6: (yda > 0 and xda > 0) or (yda == 0 and xda > 0), 
                7: (yda > 0 and xda > 0) or (yda == 0 and xda > 0) or (yda < 0 and xda > 0), 
                
            }[x]
        
        def pynx(x,yda,xda): # positive y, negative x (slopes northwest)
            
            return {
                0: (yda > 0 and xda < 0) or (yda == 0 and xda < 0) or (yda < 0 and xda < 0), 
                1: (yda > 0 and xda < 0) or (yda == 0 and xda < 0) or (yda < 0 and xda < 0), 
                2: (yda > 0 and xda < 0) or (yda == 0 and xda < 0), 
                3: (yda > 0 and xda < 0),  
                4: (yda > 0 and xda < 0) or (yda > 0 and xda == 0), 
                5: (yda > 0 and xda < 0) or (yda > 0 and xda == 0) or (yda > 0 and xda > 0), 
                6: (yda > 0 and xda < 0) or (yda > 0 and xda == 0) or (yda > 0 and xda > 0), 
                7: True, 
                
            }[x]
        
        def nypx(x,yda,xda): #negative y, postive x (slopes southeast)
            
            return {
                0: (yda < 0 and xda > 0) or (yda < 0 and xda == 0),  
                1: (yda < 0 and xda > 0) or (yda < 0 and xda == 0) or (yda < 0 and xda < 0), 
                2: (yda < 0 and xda > 0) or (yda < 0 and xda == 0) or (yda < 0 and xda < 0), 
                3: True, 
                4: (yda < 0 and xda > 0) or (yda == 0 and xda > 0) or (yda > 0 and xda > 0),
                5: (yda < 0 and xda > 0) or (yda == 0 and xda > 0) or (yda > 0 and xda > 0), 
                6: (yda < 0 and xda > 0) or (yda == 0 and xda > 0), 
                7: (yda < 0 and xda > 0), 
                
            }[x]
        
        def nynx(x,yda,xda): #negative y, negative x (slopes southwest)
            
            return {
                0: (yda < 0 and xda < 0) or (yda < 0 and xda == 0), 
                1: (yda < 0 and xda < 0), 
                2: (yda < 0 and xda < 0) or (yda == 0 and xda < 0), 
                3: (yda < 0 and xda < 0) or (yda == 0 and xda < 0) or (yda > 0 and xda < 0), 
                4: (yda < 0 and xda < 0) or (yda == 0 and xda < 0) or (yda > 0 and xda < 0), 
                5: True, 
                6: (yda < 0 and xda < 0) or (yda < 0 and xda == 0) or (yda < 0 and xda > 0),
                7: (yda < 0 and xda < 0) or (yda < 0 and xda == 0) or (yda < 0 and xda > 0), 
                
            }[x]
        
            
        ydal = self.ydif[ac]
        xdal = self.xdif[ac]
        
        #create mask for adjacent cells using  check if not divergent (True) or divergent (false)
        if yd > 0 and xd == 0:
            ac_m = np.array(list(map(pyzx,acn,ydal,xdal)))
        elif yd < 0 and xd == 0:
            ac_m = np.array(list(map(nyzx,acn,ydal,xdal)))        
        elif yd == 0 and xd > 0:
            ac_m = np.array(list(map(zypx,acn,ydal,xdal)))
        elif yd == 0 and xd < 0:
            ac_m = np.array(list(map(zynx,acn,ydal,xdal)))
        elif yd > 0 and xd > 0:
            ac_m = np.array(list(map(pypx,acn,ydal,xdal)))
        elif yd > 0 and xd < 0:
            ac_m = np.array(list(map(pynx,acn,ydal,xdal)))
        elif yd < 0 and xd > 0:
            ac_m = np.array(list(map(nypx,acn,ydal,xdal)))
        elif yd < 0 and xd < 0:
            ac_m = np.array(list(map(nynx,acn,ydal,xdal)))
        else: # if yd == 0 and xd == 0, flat ground? # NEED TO DETERMINE WHEN THIS ARISES
            print('WARNING, CELL '+str(n)+'HAS INDETERMINANT FLOW DIRECTION')
            ac_m = np.array(list(map(nynx,acn,ydal,xdal)))
        
        return ac_m

    
    def aLScell(self,cl):
        """
        returns boolian list of adjacent cells that are also mass wasting cells 
        """
        cl_m = []
        for c,j in enumerate(cl): #for each adjacent and not divergent cell, check if a mass wasting cell  
            
            tv1 = self.LS_cells[self.LS_cells==j] #check if cell number in mass wasting cell list
            
            if len(tv1) != 0: #if mass wasting cell (is included in LS_cells), add to into mass wasting unit LSd[i]
                
                cl_m.append(True)
            else:
                cl_m.append(False)
    
        return cl_m
    
    
    
    def GroupAdjacentLScells(self,lsc):
        '''
        returns list of cells to clump (adjacent, not divergent) with input cell (lsc)
        '''
        
        
        #(1) determine adjacent cells
        Adj = self.AdjCells(lsc)
        # print('adjcells')
        ac = Adj[0] #grid cell numbers of adjacent cells
        acn = Adj[1] #adjacent cell numbers based on local number schem
        
        #(2) mask for list of adject cells: keep only ones that are not divergent
        ac_m = self.NotDivergent(lsc,ac,acn) 
        # print('notdivergent')
        andc = list(np.array(ac)[ac_m]) #adjacent not divergent cells
        
        #(3) check if mass wasting cells
        andc_m = self.aLScell(andc) 
        # print('checkmasswastingcell')
        #final list of cells 
        group = [lsc] 
        
        group.extend(list(np.array(andc )[andc_m]))#adjacent, notdivergent and landslide cells
        
    
        
        return group
        
    
    # def DownslopeCells(self,group):
    #     '''
    #     determines cells downslope of mass wasting unit and upslope of nearest channel
    #     '''
        
    
    
    ##Use above functions to clump groups of cells into mass wasting units
    
    def _MassWastingExtent(self):
        '''
       
        TODO: look into replaceing pandas dataframe and pd.merge(inner) with np.in1d(array1,array2) 
            and pd.erge(outer) call with np.unique(), get rid of dataframe operations
        

        '''
        
        #single cell clump
        groupALS_l = [] 
        for i, v in enumerate(self.LS_cells):
            
            group = self.GroupAdjacentLScells(v)
            if len(group) >= self.min_mw_cells: # at least this number of cells to be a clump
                groupALS_l.append(group) # list of all single cell clumps
        
        # print('groupadjacentcells')
        
        groupALS_lc = groupALS_l*1  #copy of list, to keep original
        self.groupALS_lc = groupALS_lc
        
        #create the mass wasting clump
        LSclump = {} # final mass wasting clump dicitonary
        
        ll = len(groupALS_lc)
        c=0
        while ll > 0:
            
            # first: clump single cell clumps into one clump if they share cells
            
            # use first cell clump in list to searchps
            df1 = pd.DataFrame({'cell':list(groupALS_lc[0])})            
            i =1
            while i < ll: # for each cell clump in groupALS_lc
                #print(i)
                df2 = pd.DataFrame({'cell':list(groupALS_lc[i])}) # set cell list i as df2
                # check if df1 and df2 share cells (if no shared cells, dfm.shape = 0)  
                dfm = pd.merge(df1,df2, on = 'cell', how= "inner") 
            
                if dfm.shape[0] > 0: # if shared cells
                    groupALS_lc.remove(groupALS_lc[i]) # cell list i is removed from groupALS_lc
                    ll = len(groupALS_lc)
                    # cell list df2 is combined with df1, dupilicate cells are list onely once,
                    # and the combined list becomes df1 using pd.merge method
                    df1 = pd.merge(df1,df2, on = 'cell', how= "outer") 
                
                i+=1
            
            # second, add downslope cells to mass wasting clump
            # for each cell in df1
                # determine downslope flow paths (cellsD)
            # determine minimum downslope distance to channel network 
            distL = []
            cellsD = {}
            for cell in df1['cell'].values:
                dist, cells = self._downslopecells(cell)
                distL.append(dist)
                cellsD[cell] = cells
            
            md = np.array(distL).min() # minimum distance to downstream channel

            # if md < MW_to_C_threshold, clump downslope cells         
            if md <= self.MW_to_C_threshold: 
            # downhill gridcells fail with landslide cells (added to clump)
                for cell in list(cellsD.keys()): # for each list of downstream nodes
                    
                    # TODO may need to remove any ls cells that are in downslope path
                    # lscls  = self.aLScell(cellsD[cell])                    
                    # np.array(andc )[andc_m]
                    
                    # convert list to df
                    df2 = pd.DataFrame({'cell':list(cellsD[cell])})
                                                         
                    # use pd merge to add unique cell values
                    df1 = pd.merge(df1,df2, on = 'cell', how= "outer") 
            

            # from ids of all cells in clump df1, get attributes of each cell, organize into dataframe
            # save in dictionary

            dff = pd.DataFrame({'cell':list(df1.cell),
                                'x location':list(self._grid.node_x[df1.cell]),
                                'y location':list(self._grid.node_y[df1.cell]),
                                'elevation [m]':list(self._grid.at_node['topographic__elevation'][df1.cell]),
                                'soil thickness [m]':list(self._grid.at_node['soil__thickness'][df1.cell]),
                                'slope [m/m]':list(self._grid.at_node['topographic__slope'][df1.cell])    
                                })
            LSclump[c] = dff

            # subtract 90% of soil depth from dem, 
            self._grid.at_node['topographic__elevation'][df1.cell] = self._grid.at_node['topographic__elevation'][df1.cell] - 0.9*self._grid.at_node['soil__thickness'][df1.cell]
       
            
            # set soil thickness to 0 at landslide cells
            self._grid.at_node['soil__thickness'][df1.cell] =0      
            
            # once all lists that share cells are appended to clump df1                   
            # remove initial cell list from list
            groupALS_lc.remove(groupALS_lc[0])    
            
            #update length of remaining 
            ll = len(groupALS_lc)

            c=c+1
        
    
        # summarize characteristics of each mass wasting unit in a dataframe
        LS = OrderedDict({})
        cellarea = self._grid.area_of_cell[0]#100 #m2
        
        for key, df in LSclump.items():
            slp = df['slope [m/m]'].mean() #mean slope of cells
            st = df['soil thickness [m]'].mean() #mean soil thickness of cells
            rc = df[df['elevation [m]']==df['elevation [m]'].min()] # choose cell with lowest elevation as initiation point of slide
            rc = rc.iloc[0]
            vol = st*cellarea*df.shape[0]
            #rc = rc.set_index(pd.Index([0]))
            LS[key] = [rc['cell'], rc['x location'], rc['y location'], rc['elevation [m]'], st, slp, vol]
        
        try:
            LS_df = pd.DataFrame.from_dict(LS, orient = 'index', columns = LSclump[0].columns.to_list()+['vol [m^3]'])
        except:
            LS_df = pd.DataFrame([])
        
        #create mass__wasting_clumps field
        for c, nv in enumerate(LSclump.values()):
            self.mwclump[list(nv['cell'].values)] = c+1 #plus one so that first value is 1
        
        self.grid.at_node['mass__wasting_clumps'] = self.mwclump
        self.LS_df = LS_df
        self.LSclump = LSclump
        self.LS_df_dict[self._time_idx] = LS_df.copy()
        self.LSclump_dict[self._time_idx] = LSclump.copy()        
        # return groupALS_l,LSclump,LS_df #clumps around a single cell, clumps for each landslide, summary of landslide


   
    def _MassWastingRunout(self, DistT =2000 , SlopeT = .15, ):
        '''
        compute mass wasting unit runout from volume and location.
        Distance traveled computed using algorith described in Benda and Cundy, 1990
        Maximum distance computed using variation of Crominas, 1996
        
        TO DO: 
            remove material at landslide
            entrains, add volume to debris flow, removes elevation from DEM
        
        '''
        Locd = {}
        Disd = {}
        if self.LS_df.empty is True:
            Loc_df =pd.DataFrame([])
            Dis_df =pd.DataFrame([])
            Dist =pd.DataFrame([])
        else:
            
            for index, row in self.LS_df.iterrows():
                
                loc = []
                dist = []
                c = 0
                
                loc.append(int(row['cell']))
                dist.append((self.xdif[int(row['cell'])]**2+self.ydif[int(row['cell'])]**2)**.5)
                
                # TO DO add user input that specifies distance based on Crominas, 1996.
                # Need elevation of highest landslide cell and channel at base of shortest
                # hillslope path
                # V = row['vol [m^3]'] 
                # el = row['elevation [m]']
                # H = el - self.minZ
                # A = .866#5#.866
                # B = -.1#-.2
                # MaxD = 1000#H/(A*V**B) #max distance based on Crominas, 1996
                # volslide = V
                
                
                #list of conditionals
                cond = [lambda x: DistT>x[0] and SlopeT<x[1], 
                        lambda x: SlopeT<x,
                        lambda x: DistT>x]
                              
               
                flow = True                
                while flow == True:
                    slope  = self.grid.at_node['topographic__slope'][loc[c]]
                    # volslide = volslide+volslide*.1 # slide accretes
                    
                    # select conditional based on user input
                    if DistT and SlopeT:
                        conditional = cond[0]
                        cx = [sum(dist),slope]
                    elif SlopeT:
                        conditional = cond[1]
                        cx = slope
                    elif DistT:
                        conditional = cond[2]
                        cx = sum(dist)
                        
                    if conditional(cx):

                        loc.append(self.grid.at_node['flow__receiver_node'][loc[c]])

                        dist.append((self.xdif[self.grid.at_node['flow__receiver_node'][loc[c]]]**2+self.ydif[self.grid.at_node['flow__receiver_node'][loc[c]]]**2)**.5)
                        
                        # erode DEM
                        self.grid.at_node['topographic__elevation'][loc[c]] = self.grid.at_node['topographic__elevation'][loc[c]]-self.ErRate
                        c=c+1
                        if loc[-1] == loc[-2]: #check that runout is not stuck at same node # NEED TO DEBUG
                            break
                    else:
                        flow = False
                
                Locd[index] = loc
                Disd[index] = dist
            
            
            Loc_df = pd.DataFrame.from_dict(Locd, orient = 'index')
            Dis_df = pd.DataFrame.from_dict(Disd, orient = 'index')
            Dist = Dis_df.sum(axis=1)
        
        self.Locd = Locd
        self.Disd = Disd
        self.Dist = Dist
        
        # return Locd,Loc_df,Disd,Dis_df,Dist               
        # return Locd,Disd,Dist


    def _RasterOutputToLink(self):
        '''
        convert output of MassWastingRunout from raster grid coordinates to a
        location on the network model grid
        Locd: Landslide run out path - dictionary that contains a list of the raster grid cell numbers included in the run out path of each landslide
        LS_df: Landslide catalogue for watershed that includes volume of landslide
        
        '''
           
        #for each deposit, find closest link cell, record link, link cell and offset from original raster model cell
        if len(self.Locd) == 0:
            dep = []
            parcelDF = pd.DataFrame([])
        Lmwlink = []
        for h, val in enumerate(OrderedDict(self.Locd)): #for each runout path
            dep = self.Locd[h][-1] # deposition location node: last value in runout path
            
            depXY = [self.gridx[dep],self.gridy[dep]] #deposition location x and y coordinate
            
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
                    break #for now use the first link found - later, CHANGE this to use largest order channel
            
            mwlink = OrderedDict({'mw_unit':h,'vol [m^3]':self.LS_df['vol [m^3]'][h],'raster_grid_cell_#':dep,'link_#':linkID,'link_cell_#':search,'raster_grid_to_link_offset [m]':offset,'link_downstream_distance [m]':ld})
            
            Lmwlink.append(mwlink)
        
        parcelDF = pd.DataFrame(Lmwlink)
        
            #(2) determine deposition location on reach - ratio of distance from inlet to deposition location in link to length of link
        def LDistanceRatio(row):
            return row['link_downstream_distance [m]']/self.linklength[int(row['link_#'])] 
        
        # print(parcelDF)
        pLinkDistanceRatio = parcelDF.apply(LDistanceRatio,axis=1)
        pLinkDistanceRatio.name = 'link_downstream_distance'
        
        parcelDF = pd.concat([parcelDF,pLinkDistanceRatio],axis=1)
        
        self.parcelDF = parcelDF
        
        # return dep, nodelist, parcelDF, Lnodelist



    def _MassWastingScourAndDeposition(self):
        # print('self._time_idx is: '+str(self._time_idx))
        # save data for plots
        # if self._time_idx == 1:
        SavePlot = True
        # else:
        #     SavePlot = False
            
        self.DFcells = {}
        self.RunoutPlotInterval = 2
        
        # depostion
        slpc = 0.1# critical slope at which debris flow stops
        SV = 2 # material stops at cell when volume is below this
        
        # release parameters for landslide
        nps = 8 # number of pulses, list for each landslide
        nid = 5 # delay between pulses (iterations), list for each landslide
                
        if self.LS_df.empty is True:
            print('no debris flows to route')
        else:
            
            ivL = self.LS_df['vol [m^3]'][0::2].values.astype('int')
            innL = self.LS_df['cell'][0::2].values.astype('int')       
        
            cL = {}
            
            # ivL and innL can be a list of values. For each value in list:
            for i,inn in enumerate(innL):
                print(i)
                cL[i] = []
                
                # # release parameters for landslide
                # nps = 8 # number of pulses, list for each landslide
                # nid = 5 # delay between pulses (iterations), list for each landslide
                
                # set up initial landslide cell
                iv =ivL[i]/nps# initial volume (total volume/number of pulses) 
            
                
                # initial receiving nodes (cells) from landslide
                rn = self._grid.at_node.dataset['flow__receiver_node'].values[inn]
                self.rn = rn
                rni = rn[np.where(rn != -1)]
                
                # initial receiving proportions from landslide
                rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[inn]
                rp = rp[np.where(rp > 0)]
                rvi = rp*iv # volume sent to each receving cell
            
                # now loop through each receiving node, 
                # determine next set of recieving nodes
                # repeat until no more receiving nodes (material deposits)
                
                self.DFcells[inn] = []
                slpr = []
                c = 0
                c2=0
                arn = rni
                arv = rvi
                while len(arn)>0:# or c <300:
            
                    
                    # time to pulse conditional statement
                    # add pulse (fraction) of total landslide volume
                    # at interval "nid" until total number of pulses is less than total
                    # for landslide volume (nps*nid)
            
                    if ((c+1)%nid ==0) & (c2<=nps*nid):
                        arn = np.concatenate((arn,rni))
                        arv = np.concatenate((arv,rvi))
                        
                    arn_ns = np.array([])
                    arv_ns = np.array([])
                    # print('arn:'+str(arn))
                    
                    # for each unique cell in receiving node list arn
                    # print('arn')
                    # print(arn)
                    for n in np.unique(arn):
                        n = int(n)
                        # 
                        dmx = self.dem_dz_cumulative[n]+self.Soil_h[n]
                        # slope at cell (use highes slope)
                        slpn = self._grid.at_node['topographic__steepest_slope'][n].max()
                        
                        # incoming volume: sum of all upslope volume inputs
                        vin = np.sum(arv[arn == n])
                        
                        # determine deposition volume following Campforts, et al., 2020            
                        # since using volume (rather than volume/dx), L is (1-(slpn/slpc)**2) 
                        # rather than mg.dx/(1-(slpn/slpc)**2)           
                        Lnum = np.max([(1-(slpn/slpc)**2),0])
                        dpd = vin*Lnum # deposition
                        
                        #determine erosion volume (function of slope)
                        er = dmx-dmx*Lnum
                        
                        
                        # volumetric balance at cell
                        
                        # determine volume sent to downslope cells
                        
                        # additional constraint to control debris flow behavoir
                        # if flux to a cell is below threshold, debris is forced to stop
                        # print('incoming volume')
                        # print(vin)
                        if vin <=SV:
                            dpd = vin # all volume that enter cell is deposited 
                            vo = 0 # debris stops, so volume out is 0
            
                            # determine change in cell height
                            deta = (dpd)/(self.dx*self.dy) # (deposition)/cell area
            
                        else:
                            vo = vin-dpd+er # vol out = vol in - vol deposited + vol eroded
                            
                            # determine change in cell height
                            deta = (dpd-er)/(self.dx*self.dy) # (deposition-erosion)/cell area
            
                        # print(deta)
                        # update raster model grid
                        self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta
                        # print(mg.at_node['topographic__elevation'][n])
                
                
                        # build list of receiving nodes and receiving volumes for next iteration        
                
                        # material stops at node if transport volume is 0 OR node is a 
                        # boundary node
                        if vo>0 and n not in self._grid.boundary_nodes: 
                            
                
                            th = 0
                            # receiving proportion of volume from cell n to each downslope cell
                            rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[n]
                            rp = rp[np.where(rp > th)]
                            # print(rp)        
                            
                            # receiving nodes (cells)
                            rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
                            rn = rn[np.where(rn != -1)]            
                            # rn = rn[np.where(rp > th)]
                            
                            # receiving volume
                            rv = rp*vo
                       
                            # store receiving nodes and volumes in temporary arrays
                            arn_ns =np.concatenate((arn_ns,rn), axis = 0) # next step receiving node list
                            arv_ns = np.concatenate((arv_ns,rv), axis = 0) # next steip receiving node incoming volume list
                            
            
                        
                    # once all cells in iteration have been evaluated, temporary receiving
                    # node and node volume arrays become arrays for next iteration
                    arn = arn_ns
                    arv = arv_ns
                    
                    # update DEM slope 
                    # fd = FlowDirectorDINF(mg) # update slope
                    fd = FlowDirectorMFD(self._grid, diagonals=True,
                                    partition_method = 'slope')
                    fd.run_one_step()
                    
                    
    
                    
                    if c%self.RunoutPlotInterval == 0:
                        cL[i].append(c)
                        
                        if SavePlot:
                            try:
                                # save DEM difference (DEM for iteration C - initial DEM)
                                _ = self._grid.add_field( 'topographic__elevation_d_'+str(i)+'_'+str(c)+'_'+str(self._time_idx),
                                                self._grid.at_node['topographic__elevation']-self.dem_initial,
                                                at='node')
                    
                                self.DFcells[inn].append(arn.astype(int))
                            except:
                                print('maps already written')
                    c+=1
                    
                    # update pulse counter
                    c2+=nid
                    # print('COUnTER')
                    # print(c)
                    # print(c2)
               
        # determine change in dem caused by mass wasting since last time step (fluvial erosion is not counted)
        self.dem_mw_dzdt = self._grid.at_node['topographic__elevation'] - self.dem_previous_time_step
        print('max change dem')
        print(self.dem_mw_dzdt.max())
        self.DFcells_dict[self._time_idx] = self.DFcells
                   
    
    def _TimeSinceDisturbance(self,dt):
        '''
        All cells move foward amount dt years
        
        if landslides, time since disturbance in any cell that has a change 
        in elevation caused by the debris flows is set to 0 (fluvial erosion does not count)

        '''
        # all grid cells advance forward in time by amount dt
        self.years_since_disturbance+=dt
        
        if self._this_timesteps_landslides.any():   
            # all grid cells that have scour or erosion from landslide or debris flow set to zero   
            self.years_since_disturbance[self.dem_mw_dzdt != 0] = 13/365 # 13 days selected to match max observed single day sediment transport following disturbance
              
    
    def _FluvialErosion(self):
        '''
        determine pulse location based on the time since mw caused disturbance
        small changes in cell elevation are ignored
        
        pulse volume is determined as a function of the time since the mw
        disturbance
        
        this is run each time step
                        
        NOTE: percentage of debris flow or terrace that becomes a pulse 
        of sediment is determined independently of flow magnitude
        '''
        # change in dpeth less than this is ignored
        dmin = 0.01
            
        # rnodes = self._grid.nodes.reshape(mg.shape[0]*mg.shape[1]) # reshame mg.nodes into 1d array
        
        # disturbance (dz>0) mask
        DistMask = np.abs(self.dem_mw_dzdt) >dmin # deposit/scour during last debris flow was greater than threhsold value. 
        
        if np.any(np.abs(self.dem_mw_dzdt) > dmin): # if there are no disturbed cells 
            self.NewDistNodes = self.rnodes[DistMask]
            
            #new deposit and scour cell ids are appeneded to list of disturbed cells
            self.DistNodes = np.unique(np.concatenate((self.DistNodes,self.rnodes[DistMask]))).astype(int)
            
            FERateC  = self.FluvialErosionRate[self.DistNodes] 
        
            # remove cells that do not fluvially erode (hillslope nodes)
            self.FENodes = self.DistNodes[FERateC[:,0] > 0] 
            FERateC = np.stack(FERateC[FERateC[:,0] > 0])
            
            if len(self.FENodes) !=0:# if there are no disturbed channel cells 
                # determine erosion depth
                
                # max fluvial erosion depth (same as max fluvial erosion depthj)
                FEDmax = self.dem_dz_cumulative[self.FENodes]+self._grid.at_node['soil__thickness'][self.FENodes] # deposition since initial dem + soil depth 
                
                # erosion rate in router is not tied to soil depth and may be less than zero, set to zero if that is the case
                FEDmax[FEDmax<0] = 0
                
                # time since distubrance [years]
                YSD = self.years_since_disturbance[self.FENodes]
                
                # fluvial erosion rate (depth for storm) = a*x**b, x is time since distubrance #TODO: apply this to terrace cells only
                FED = FERateC[:,0]*YSD**FERateC[:,1]
                
                # TODO make new erosion function for channel cells that erodes as a function of flow shear stress
                # FED
                
                self.FED = FED
                self.FEDmax = FEDmax
                
                # apply criteria erosion can't exceed maxim
                FED[FED>FEDmax] = FEDmax[FED>FEDmax]
                
                # convert depth to volume        
                self.FEV = FED*self._grid.dx*self._grid.dx
        else:
            print('no disturbed cells to fluvially erode')
            self.FENodes = np.array([])
        
        # print(self.FEV)
        # print(self.FENodes)
        # return (FENodes, FEV)
        
    
    
    def _parcelDFmaker(self):
        '''
        from the list of cell locations of the pulse and the volume of the pulse, 
        convert to a dataframe of pulses (ParcelDF) that is the input for pulser
        '''
    
        if len(self.FENodes) == 0:
            FEDn = []
            parcelDF = pd.DataFrame([])
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
                    break #for now use the first link found - later, CHANGE this to use largest order channel
            
            mwlink = OrderedDict({'mw_unit':h,'vol [m^3]':self.FEV[h],'raster_grid_cell_#':FEDn,'link_#':linkID,'link_cell_#':search,'raster_grid_to_link_offset [m]':offset,'link_downstream_distance [m]':ld})
            
            Lmwlink.append(mwlink)
        
        parcelDF = pd.DataFrame(Lmwlink)
        
        
        #(2) determine deposition location on reach - ratio of distance from inlet to deposition location in link to length of link
        
        
        def LDistanceRatio(row):
            return row['link_downstream_distance [m]']/self.linklength[int(row['link_#'])] 
        
        # print(parcelDF)
        pLinkDistanceRatio = parcelDF.apply(LDistanceRatio,axis=1)
        pLinkDistanceRatio.name = 'link_downstream_distance'
        
        self.parcelDF = pd.concat([parcelDF,pLinkDistanceRatio],axis=1)
        self.parcelDF_dict[self._time_idx] = self.parcelDF.copy() # save a copy of each pulse


        
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
        # run flow director, add slope and receiving node fields
        fr = FlowAccumulator(self._grid,'topographic__elevation',flow_director='D8')
        fr.run_one_step()

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


        self._time += dt  # update mass wasting router time
        self._time_idx += 1  # update time index
        self.mwprob = self.grid.at_node['MW__probability'] #  update mw probability variable
        self.hmwprob = self.grid.at_node['high__MW_probability']  #update boolean landslide field
        self._extractLSCells()

        
        if self._method == 'simple':
            if self._this_timesteps_landslides.any():
            
                # determine extent of landslides
                self._MassWastingExtent()
                print('masswastingextent')            
                # determine runout pathout
                self._MassWastingRunout()
                print('masswastingrounout')
                # convert mass wasting deposit location and attributes to parcel attributes  
                self._RasterOutputToLink()
                print('rasteroutputtolink')
        
            else:
                self.parcelDF = pd.DataFrame([])
                print('No landslides to route this time step, checking for terrace deposits')
        if self._method == 'ScourAndDeposition':

            if self._this_timesteps_landslides.any():

                # determine extent of landslides
                self._MassWastingExtent()
                print('masswastingextent')     
                # run multi-flow director needed for debris flow routing
                self._multidirectionflowdirector()   
                # route debris flows, update dem to account for scour and 
                # deposition caused by debris flow and landslide processes
                self._MassWastingScourAndDeposition()
                print('scour and deposition') 
            # determine time since each cell was disturbed
            self._TimeSinceDisturbance(dt)
            # fluvially erode any recently disturbed cells, create lists of 
            # cells and volume at each cell that enters the channel network
            self._FluvialErosion()
            print('fluvial erosion')
            # set dem as dem as dem representing the previous time timestep
            self.dem_previous_time_step = self._grid.at_node['topographic__elevation'].copy() # set previous time step elevation
            # compute the cumulative vertical change of each grid cell
            self.dem_dz_cumulative = self._grid.at_node['topographic__elevation'] - self.dem_initial
            # convert list of cells and volumes to a dataframe compatiable with
            # the sediment pulser utility
            self._parcelDFmaker()
            # re-run d8flowdirector for clumping and distance computations
            self._d8flowdirector()
        
 
            
            # terrace eroder