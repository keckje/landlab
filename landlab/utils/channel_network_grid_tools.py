# -*- coding: utf-8 -*-

from landlab import Component, FieldError
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy as sc


class ChannelNetworkGridTools(Component):    
    '''
    This class contains methods for converting field values on the network model 
    grid to equivalent raster model grid node values. 
    
   
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
            grid = None,
            nmgrid = None,
            Ct = 5000,
            BCt = 100000,
            MW_to_channel_threshold = 50,
            PartOfChannel_buffer = 10,
            TerraceWidth = 1,
            **kwds):
        
        
        
        if grid != None:
            print('grid')
            print(grid)
            print('grid')
            super().__init__(grid)
    
            if 'topographic__elevation' in grid.at_node:
                self.dem = grid.at_node['topographic__elevation']
            else:
                raise FieldError(
                    'A topography is required as a component input!')
            
            # this error goes to LandslideMapper
            #   flow receiver node
            # if 'flow__receiver_node' in grid.at_node:
            #     self.frnode = grid.at_node['flow__receiver_node']
            # else:
            #     raise FieldError(
            #         'A flow__receiver_node field is required as a component input!')  
    
    
            
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

            if 'flow__receiver_node' in grid.at_node: # Landslide Mapper
                self.frnode = grid.at_node['flow__receiver_node']
                    
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


        if nmgrid != None:
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
     
        

    def extract_channel_nodes(self,Ct,BCt):
        """MWR, DtoL
        change to 'fluvial channel' and 'channel'
        """
        
        # to top of debris flow channel (top colluvial channel)
        ChannelNodeMask = self._grid.at_node['drainage_area'] > Ct
        df_x = self._grid.node_x[ChannelNodeMask]
        df_y = self._grid.node_y[ChannelNodeMask]
        self.xyDf_df = pd.DataFrame({'x':df_x, 'y':df_y})
        self.ChannelNodes = self.rnodes[ChannelNodeMask] 
        
        # to top of bedload channels (~top cascade channels)
        BedloadChannelNodeMask = self._grid.at_node['drainage_area'] > BCt
        bc_x = self._grid.node_x[BedloadChannelNodeMask]
        bc_y = self._grid.node_y[BedloadChannelNodeMask]
        self.xyDf_bc = pd.DataFrame({'x':bc_x, 'y':bc_y})
        self.BedloadChannelNodes = self.rnodes[BedloadChannelNodeMask] 


    def extract_terrace_nodes(self):
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
    
        
    
    def map_nmg_links_to_rmg_nodes(self, linknodes, active_links, nmgx, nmgy):
        '''was LinktoNodes() MWR, DtoL
        #convert links to coincident nodes
        for each nmg link, maps all coicident rmg nodes. Output is a list of lists. order of lists is order links
        
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

            
    def map_nmg1_links_to_nmg2_links(self,Lnodelist,xyDf_d):    
        
        '''was DHSVM_network_to_NMG_Mapper
        ### change to onetwork to network mapper
        
        # map_nmg1_links_to_nmg2_links
        
        ### used for matching finer detail network model grid to equivalent 
        coarser scale network model grid
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
        for i, sublist in enumerate(Lnodelist):# for each list of link nodes
            LinkL = []
            for j in sublist: # for each node associated with the nmg link
                XY = [self.gridx[j], self.gridy[j]] # get x and y coordinate of node
                nmg_d_dist = xyDf_d.apply(Distance,axis=1) # compute the distance to all dhsvm nodes
                offset = nmg_d_dist.min() # find the minimum distance
                mdn = xyDf_d['linkID'].values[(nmg_d_dist == offset).values][0]# link id of minimum distance node
                LinkL.append(mdn) # nmg1 link id mapped to each equivalent node of nmg2 link
            LinkMapL[i] = LinkL # list of all nm1 links matched to rmg nodes of nmg2 link
            LinkMapper[i] = np.argmax(np.bincount(np.array(LinkL))) # nm1 link with highest count is nm1 link matched to nm2 link
            print('nmg1 link '+ str(i)+' mapped to equivalent nmg2 link')
            
        return (LinkMapper, LinkMapL)
        # self.LinkMapper = LinkMapper
        # self.LinkMapL = LinkMapL


    def DHSVM_network_to_RMG_Mapper(self,xyDf_d):
        """DtoL
        # this function appears to give same output as map_nmg_links_to_rmg_nodes 
        # try to get rid of this function
        # map_rmg_nodes_to_nmg_links
        
        Determine the closest link to each RMG channel node
        output can be used to map nmg field values to the rmg

        Channel nodes are identified based on a simple contributing area threshold
        using ChannelMapper().  
        
        Channelnodes: np array of all RMG nodes that are channel nodes.
        xyDf_d: pd dataframe with columns: link id, raster model grid node, x, y of node
        
        """
        
        #compute distance between deposit and all network cells
        def Distance(row):
            return ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5        

        # for each node in the channel node list record the equivalent nmg_d link id 

        NodeMapper ={}
        ncn = self.ChannelNodes.shape[0] # number of channel nodes
        for i, node in enumerate(self.ChannelNodes):# for each node in the channel node list
            XY = [self.gridx[i], self.gridy[i]] # get x and y coordinate of node
            nmg_d_dist = xyDf_d.apply(Distance,axis=1) # compute the distance to all dhsvm nodes
            offset = nmg_d_dist.min() # find the minimum distance
            mdn = xyDf_d['linkID'].values[(nmg_d_dist == offset).values][0]# link id of minimum distance node
            NodeMapper[i] = mdn # dhsmve link for node i
            if i%20 == 0:
                print(str(np.round((i/ncn)*100))+' % RMG nodes mapped to equivalent DHSVM link')
            
        self.NodeMapper = NodeMapper       


    def map_rmg_nodes_to_nmg_nodes(self):
        """MWR, DtoL
        #was NMG_node_to_RMG_node_mapper
        find RMG node that is closest to each NMG node
        
        used to map changes in the rmg elevation to the nmg profile
        
        Channel nodes are identified based on a simple contributing area threshold
        using ChannelMapper.
        
        nmg_nodes: list of nmg nodes
        xyDF: pd dataframe with columns: x, y of each RMG channel node
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
            NodeMapper[i] = self.ChannelNodes[mdn] # rmg node mapped to nmg node i
            
        self.NMGtoRMGnodeMapper = NodeMapper




    def min_distance_to_network(self, cellid, ChType = 'debrisflow'):
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



    def transfer_rmg_node_field_to_nmg_node_field(self):
        '''
        # change to transfer_rmg_node_field_to_nmg_node_field
        updates the elevation of the nmg nodes based on the closest channel rmg node
        updates the link slopes based on the updated nmg node elevations
        move to channel_network_grid_tools?

        Returns
        -------
        None.
        '''

        # update elevation
        for i, node in enumerate(self.nmg_nodes):
            RMG_node = self.NMGtoRMGnodeMapper[i]
            self._nmgrid.at_node['topographic__elevation'][i] = self._grid.at_node['topographic__elevation'][RMG_node]



    # def NMG_link_field_to_RMG_node_field(self,field)
    # 

    
    
    def run_one_step(self):
        """run_one_step is not implemented for this component."""
        raise NotImplementedError("run_one_step()")       