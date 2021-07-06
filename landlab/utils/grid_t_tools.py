# -*- coding: utf-8 -*-

from landlab import Component, FieldError
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy as sc


class GridTTools(Component):    
    '''
    base class for MassWastingRouter and DHSVMtoLandlab. This class contains
    methods for converting field values on the network model grid to equivalent
    raster model grid node values. Both the MassWastingRouter and DHSVMtoLandlab
    components rely on these methods to translate modeled values in the the 
    raster model grid to the network model grid or vice versa.
    
    The run_one_step method is not implemented. Calling GridTTools establishes
    class variables related to the dimensions and geometry of the raster model
    grid and network model grid. Both the MassWastingRouter and DHSVMtoLandlab
    use the GridTTools call method when instantiated
    
    
    Parameters
    ----------
    grid : raster model grid
        A grid.
    nmgrid : network model grid
        A network model grid
    Ct: float
        Contributing area threshold at which channel begin (colluvial channels)
    BCt: float
        Contributing area threshold at which cascade channels begin, which is 
        assumed to be the upper limit of frequent bedload transport
    MW_to_channel_threshold: float
        A maximum distance, above which, cells downslope of a landslide do not
        fail with the landslide. This threshold is set based on review of failure
        behavoir in the basin
    PartOfChannel_buffer: float
        buffer distance from the centerline of the channel that is used to identify
        nodes that are part of the channel network
    TerraceWidth: int
        width from channel cells in number of cells considered terrace cells.
        Defaul value is 1 (i.e. all cells directly adjacent to the channels
                           cells are considered terrace cells)
    
    author: Jeff Keck

    '''

   
    def __init__(
            self, 
            grid,
            nmgrid,
            Ct = 5000,
            BCt = 100000,
            MW_to_channel_threshold = 50,
            PartOfChannel_buffer = 10,
            TerraceWidth = 1,
            **kwds):
        
        super().__init__(grid)

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


        
        ### RM Grid characteristics
        self.ncn = len(grid.core_nodes) # number core nodes
        self.gr = grid.shape[0] #number of rows
        self.gc = grid.shape[1] #number of columns
        self.dx = grid.dx #width of cell
        self.dy = grid.dy #height of cell

        # receivers = self.frnode #receiver nodes (node that receives runoff from node) = self.frnode #receiver nodes (node that receives runoff from node)

        # nodes, reshaped in into m*n,1 array like other mg fields
        self.nodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1],1)
        self.rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1]) #nodes in single column array
                
        self.xdif = grid.node_x[self.frnode]-grid.node_x[self.rnodes] # change in x from node to receiver node
        self.ydif = (grid.node_y[self.frnode]-grid.node_y[self.rnodes])*-1 #, change in y from node to receiver node...NOTE: flip direction of y axis so that up is positve
       
        # grid node coordinates, translated to origin of 0,0
        self.gridx = grid.node_x#-grid.node_x[0] 
        self.gridy = grid.node_y#-grid.node_y[0]
        
        # extent of each cell in grid        
        self.ndxe = self.gridx+self.dx/2
        self.ndxw = self.gridx-self.dx/2
        self.ndyn = self.gridy+self.dy/2
        self.ndys = self.gridy-self.dy/2



        ### NM Grid characteristics
        self._nmgrid = nmgrid
        # network model grid characteristics       
        self.linknodes = nmgrid.nodes_at_link #links as ordered by read_shapefile       
        # network model grid node coordinates, translated to origin of 0,0, used to map grid to nmg
        self.nmgridx = nmgrid.x_of_node
        self.nmgridy = nmgrid.y_of_node
        self.linklength = nmgrid.length_of_link 
        self.nmg_nodes = nmgrid.nodes
        
        ### Channel extraction parameters
        self.Ct = Ct # Channel initiation threshold [m2]   
        self.POCbuffer = PartOfChannel_buffer # distance [m] from a channel cell that is considered part of the channel (used for determining distance between landslide and channel)
        self.BCt = BCt # CA threshold for channels that typically transport bedload [m2] 
        self.TerraceWidth = TerraceWidth # distance from channel grid cells that are considered terrace grid cells [# cells] 
         
      

    def _ChannelNodes(self):
        """MWR, DtoL
        change to 'fluvial channel' and 'channel'
        """
        
        # to top of debris flow channel (top colluvial channel)
        ChannelNodeMask = self._grid.at_node['drainage_area'] > self.Ct
        df_x = self._grid.node_x[ChannelNodeMask]
        df_y = self._grid.node_y[ChannelNodeMask]
        self.xyDf_df = pd.DataFrame({'x':df_x, 'y':df_y})
        self.ChannelNodes = self.rnodes[ChannelNodeMask] 
        
        # to top of bedload channels (~top cascade channels)
        BedloadChannelNodeMask = self._grid.at_node['drainage_area'] > self.BCt
        bc_x = self._grid.node_x[BedloadChannelNodeMask]
        bc_y = self._grid.node_y[BedloadChannelNodeMask]
        self.xyDf_bc = pd.DataFrame({'x':bc_x, 'y':bc_y})
        self.BedloadChannelNodes = self.rnodes[BedloadChannelNodeMask] 


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
    
        
    def _LinktoNodes(self, linknodes, active_links, nmgx, nmgy):
        '''MWR, DtoL
        #convert links to coincident nodes
            #loop through all links in network grid to determine raster grid cells that coincide with each link
            #and equivalent distance from upstream node on link
        '''
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

            
    def _DHSVM_network_to_NMG_Mapper(self):    
        
        '''DtoL
        determine the closest dhsvm nmg (nmg_d) link to each link in the landlab
        nmg. Note, the landlab nmg id of the dhsmv network is used, not the DHSVM
        network id (arcID) To translate the nmg_d link id to the DHSVM network id: 
            self.nmg_d.at_link['arcid'][i], where i is the nmg_d link id.
        '''
        
        #compute distance between deposit and all network cells
        def Distance(row):
            return ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5
        
        # for each node equivalent of each nmg link, find the closest nmg_d node
        # and record the equivalent nmg_d link id 
        
        LinkMapper ={}
        LinkMapL = {}
        for i, sublist in enumerate(self.Lnodelist):# for each list of link nodes
            LinkL = []
            for j in sublist: # for each node associated with the nmg link
                XY = [self.gridx[j], self.gridy[j]] # get x and y coordinate of node
                nmg_d_dist = self.xyDf_d.apply(Distance,axis=1) # compute the distance to all dhsvm nodes
                offset = nmg_d_dist.min() # find the minimum distance
                mdn = self.xyDf_d['linkID'].values[(nmg_d_dist == offset).values][0]# link id of minimum distance node
                LinkL.append(mdn)
            LinkMapL[i] = LinkL
            LinkMapper[i] = np.argmax(np.bincount(np.array(LinkL))) # dhsvm link that has most link ids in link list
            print('nmg link '+ str(i)+' mapped to equivalent DHSVM link')
            
        # return (Mapper, MapL)
        self.LinkMapper = LinkMapper
        self.LinkMapL = LinkMapL

    def _DHSVM_network_to_RMG_Mapper(self):
        """DtoL

        """
        
        #compute distance between deposit and all network cells
        def Distance(row):
            return ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5        

        # for each node in the channel node list record the equivalent nmg_d link id 

        NodeMapper ={}
        ncn = self.ChannelNodes.shape[0] # number of channel nodes
        for i, node in enumerate(self.ChannelNodes):# for each node in the channel node list
            XY = [self.gridx[i], self.gridy[i]] # get x and y coordinate of node
            nmg_d_dist = self.xyDf_d.apply(Distance,axis=1) # compute the distance to all dhsvm nodes
            offset = nmg_d_dist.min() # find the minimum distance
            mdn = self.xyDf_d['linkID'].values[(nmg_d_dist == offset).values][0]# link id of minimum distance node
            NodeMapper[i] = mdn # dhsmve link for node i
            if i%20 == 0:
                print(str(np.round((i/ncn)*100))+' % RMG nodes mapped to equivalent DHSVM link')
            
        self.NodeMapper = NodeMapper
        

    def _NMG_node_to_RMG_node_mapper(self):
        """MWR, DtoL

        """
            
        #compute distance between deposit and all network cells
        def Distance(row):
            return ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5        

        # for each node in the channel node list record the equivalent nmg_d link id 

        NodeMapper ={}
        for i, node in enumerate(self.nmg_nodes):# for each node network modelg grid
            XY = [self._nmgrid.node_x[i], self._nmgrid.node_y[i]] # get x and y coordinate of node
            nmg_node_dist = self.xyDf_df.apply(Distance,axis=1) # compute the distance to all channel nodes
            offset = nmg_node_dist.min() # find the minimum distance between node and channel nodes
            mdn = self.xyDf_df.index[(nmg_node_dist == offset).values][0]# index of minimum distance node
            NodeMapper[i] = self.ChannelNodes[mdn] # dhsmve link for node i
            
        self.NMGtoRMGnodeMapper = NodeMapper

    def _min_distance_to_network(self, cellid, ChType = 'debrisflow'):
        def distance_to_network(row):
            '''GTT only
            compute distance between a cell and the nearest debris flow network 
            cell used to determine clump distance to colluvial channel network
            for clumping algorithm
            
            ChType = debrisflow: uses debris flow network
            ChType = nmg: uses network model grid network
            
            TODO: change to "fluvial channel network" and "channel network" options
            
            '''
            return ((row['x']-self.gridx[cellid])**2+(row['y']-self.gridy[cellid])**2)**.5
        
        if ChType == 'debrisflow':
            nmg_dist = self.xyDf_df.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDf_df[nmg_dist == offset] # minimum distance node and node x y        
        elif ChType == 'nmg':
            nmg_dist = self.xyDf.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDf[nmg_dist == offset] # minimum distance node and node x y    

        return offset, mdn


    def _downslopecells(self,StartCell):
        '''MWR
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
            
            slope  = self._grid.at_node['topographic__slope'][loc[c]]
            #compute distance between deposit and all debris flow network cells
            cdist, nc = self._min_distance_to_network(loc[c],  ChType = 'debrisflow')
            #TO DO need to change so that if distance to network is less than minimum, then stop
            
            
            #if loc[c] not in self.xyDf['node'].values: # channel network doesnt always match DEM
            # print(cdist)
            if cdist > self.POCbuffer: # downslope distance measured to POCbuffer from channel    
                loc.append(self._grid.at_node['flow__receiver_node'][loc[c]])
                dist.append((self.xdif[self._grid.at_node['flow__receiver_node'][loc[c]]]**2+self.ydif[self._grid.at_node['flow__receiver_node'][loc[c]]]**2)**.5)
                c=c+1
                if loc[-1] == loc[-2]: # check that runout is not stuck at same node # NEED TO DEBUG
                    break
            else:
                flow = False
        
        Dista = np.array(dist)
        Dist = Dista.sum()
        return (Dist,loc)

        
    def AdjCells(self,n):
        '''MWR                
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
        '''MWR        
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

    #interplate function used by other functions
    def intp(self, x,y,x1,message = None): 
        '''ALL
        Parameters
        ----------
        x : list, np.array, pd.series of float or int
            x values used for interpolation
        y : list, np.array, pd.series of float or int
            x values used for interpolation
        x1 : float or int within domain of x
            interpolate at x1
    
        Returns
        -------
        float
            y1, interpolated value at x1
    
        '''  
        if x1 <= min(x):
            y1 = min(y)
            # print('TRIED TO INTERPOLATE AT '+str(x1)+' BUT MINIMUM INTERPOLATION RANGE IS: '+str(min(x)))
            if message:
                print(message)
        elif x1 >= max(x):
            y1 = max(y)
            # print('TRIED TO INTERPOLATE AT '+str(x1)+' BUT MAXIMUM INTERPOLATION RANGE IS: '+str(max(x)))
            if message:
                print(message)
        else:            
            f = sc.interpolate.interp1d(x,y)   
            y1 = f(x1)
        return y1
    
    def run_one_step(self):
        """run_one_step is not implemented for this component."""
        raise NotImplementedError("run_one_step()")       