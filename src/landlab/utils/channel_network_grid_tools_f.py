"""
    TODO:
    add second network model grid input?
    make all functions use class variables, user just runs the function after class instantiation?
    look at Shelby's "create_network" utility for coding style options
    
    break up into a functions
        1.figure out inputs
        2.rewrite
"""

from landlab import Component, FieldError
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy as sc


# class ChannelNetworkToolsBase():    
#     '''
#     Base class for ChannelNetworkToolsInterpreter and ChannelNetworkToolsMapper.
#     This class estblishes class variables and includes functions shared by both
#     child classes.

#     Parameters
#     ----------
#     grid : raster model grid
#         A grid.
#     nmgrid : network model grid
#         A network model grid
#     CC: float
#         Contributing area threshold at which channel begin (colluvial channels)
#     FC: float
#         Contributing area threshold at which fluvial (e.g., cascade) channels 
#         begin, which is assumed to be the upper limit of frequent bedload transport
#     TerraceWidth: int
#         width from channel cells in number of cells considered terrace cells.
#         Defaul value is 1 (i.e. all cells directly adjacent to the channels
#                            cells are considered terrace cells)
    
#     author: Jeff Keck

#     '''

   
#     def __init__(
#             self, 
#             grid = None,
#             nmgrid = None,
#             CC = 5000,
#             FC = 100000,
#             TerraceWidth = 1,
#             **kwds):
        
#         if grid != None:

#             self._grid = grid
    
#             if 'topographic__elevation' in grid.at_node:
#                 self.dem = grid.at_node['topographic__elevation']
#             else:
#                 raise FieldError(
#                     'A topography is required as a component input!')
            
#             ### raster model grid attributes
#             self.ncn = len(grid.core_nodes) # number core nodes
#             self.gr = grid.shape[0] #number of rows
#             self.gc = grid.shape[1] #number of columns
#             self.dx = grid.dx #width of cell
#             self.dy = grid.dy #height of cell

#             # nodes, reshaped in into m*n,1 array like other mg fields
#             self.nodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1],1)
#             self.rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1]) #nodes in single column array

#             if 'flow__receiver_node' in grid.at_node: # consider moving this to Landslide Mapper
#                 self.frnode = grid.at_node['flow__receiver_node']
#                 self.xdif = grid.node_x[self.frnode]-grid.node_x[self.rnodes] # change in x from node to receiver node
#                 self.ydif = (grid.node_y[self.frnode]-grid.node_y[self.rnodes])*-1 #, change in y from node to receiver node...NOTE: flip direction of y axis so that up is positve
               
#             # grid node coordinates, translated to origin of 0,0
#             self.gridx = grid.node_x#-grid.node_x[0] 
#             self.gridy = grid.node_y#-grid.node_y[0]
            
#             # extent of each cell in grid        
#             self.ndxe = self.gridx+self.dx/2
#             self.ndxw = self.gridx-self.dx/2
#             self.ndyn = self.gridy+self.dy/2
#             self.ndys = self.gridy-self.dy/2

#         if nmgrid != None:
#             ### network model grid attributes
#             self._nmgrid = nmgrid
#             self.linknodes = nmgrid.nodes_at_link #links as ordered by read_shapefile       
#             self.nmgridx = nmgrid.x_of_node
#             self.nmgridy = nmgrid.y_of_node
#             self.linklength = nmgrid.length_of_link 
#             self.nmg_nodes = nmgrid.nodes

#         ### Channel extraction parameters
#         self.CC = CC # Channel initiation threshold [m2]   
#         self.FC = FC # CA threshold for channels that typically transport bedload [m2] 
#         self.TerraceWidth = TerraceWidth # distance from channel grid cells that are considered terrace grid cells [# cells]          

    
#     def extract_channel_nodes(self,CC,FC):
#         """interpret which nodes of the DEM correspond to the fluvial channel 
#         network and entire channel network (including colluvial channels)
#         """
        
#         # entire channel network (all channels below channel head or upper extent 
#         # of the colluvial channel network)
#         ChannelNodeMask = self._grid.at_node['drainage_area'] > CC
#         df_x = self.gridx[ChannelNodeMask]
#         df_y = self.gridy[ChannelNodeMask]
#         self.xyDf_df = pd.DataFrame({'x':df_x, 'y':df_y})
#         self.ChannelNodes = self.rnodes[ChannelNodeMask] 
        
#         # fluvial channel network only (all channels below upper extent of cascade channels)
#         FluvialChannelNodeMask = self._grid.at_node['drainage_area'] > FC
#         bc_x = self.gridx[FluvialChannelNodeMask]
#         bc_y = self.gridy[FluvialChannelNodeMask]
#         self.xyDf_bc = pd.DataFrame({'x':bc_x, 'y':bc_y})
#         self.FluvialChannelNodes = self.rnodes[FluvialChannelNodeMask]   
        
##########################        
def extract_channel_nodes(grid,Ct):
    """interpret which nodes of the DEM represent the channel network 
    (including colluvial channels)
    """
    # define rnodes, nodes in single column array
    rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1])     
    
    # entire channel network (all channels below channel head or upper extent 
    # of the colluvial channel network)
    ChannelNodeMask = grid.at_node['drainage_area'] > Ct
    C_x = grid.node_x[ChannelNodeMask]
    C_y = grid.node_y[ChannelNodeMask]
    xyDf_C = pd.DataFrame({'x':C_x, 'y':C_y})
    ChannelNodes = rnodes[ChannelNodeMask] 
    return ChannelNodes, xyDf_C


def extract_fluvial_channel_nodes(grid,FCt):
    """interpret which nodes of the DEM represent the fluvial channel 
    network
    """
    # define rnodes, nodes in single column array
    rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1])     

    # fluvial channel network only (all channels below upper extent of cascade channels)
    FluvialChannelNodeMask = grid.at_node['drainage_area'] > FCt
    FC_x = grid.node_x[FluvialChannelNodeMask]
    FC_y = grid.node_y[FluvialChannelNodeMask]
    xyDf_FC = pd.DataFrame({'x':FC_x, 'y':FC_y})
    FluvialChannelNodes = rnodes[FluvialChannelNodeMask]
    return FluvialChannelNodes, xyDf_FC


# class ChannelNetworkToolsInterpretor(ChannelNetworkToolsBase):
#     """
#     A set of tools for interpreting the fluvial and enitre channel
#     network (fluvial and colluvial channels, which end at the channel head) and
#     determining distance from a particular node in the watershed to the network


#     Parameters:
#         grid
#         nmgrid
#         **kwgs include:

#             CC: float
#                 Contributing area threshold at which colluvial channel begins.
#             FC: float
#                 Contributing area threshold at which fluvial (e.g., cascade) channels 
#                 begin, which is assumed to be the upper limit of frequent bedload transport
#             TerraceWidth: int
#                 width from channel cells in number of cells considered terrace cells.
#                 Defaul value is 1 (i.e. all cells directly adjacent to the channels
#                                    cells are considered terrace cells)
#     """

    
#     def __init__(self, grid, **kwgs):
#         """instatiate ChannelNetworkToolsInterpretor using base class init"""
#         ChannelNetworkToolsBase.__init__(self, grid, **kwgs)

#     def extract_terrace_nodes(self):
#         """determine which raster model grid nodes coincide with channel terraces,
#         which presently are asssumed to be a fixed width (number of nodes) from
#         the channel nodes
#         """
#         for i in range(self.TerraceWidth):
#             if i == 0:
#                 # diagonal adjacent nodes to channel nodes
#                 AdjDN =np.ravel(self._grid.diagonal_adjacent_nodes_at_node[self.BedloadChannelNodes])  
#                 # adjacent nodes to channel nodes
#                 AdjN = np.ravel(self._grid.adjacent_nodes_at_node[self.BedloadChannelNodes])
#             elif i>0:
#                 # diagonal adjacent nodes to channel nodes
#                 AdjDN = self._grid.diagonal_adjacent_nodes_at_node[TerraceNodes] 
#                 # adjacent nodes to channel nodes
#                 AdjN = self._grid.adjacent_nodes_at_node[TerraceNodes]            
#             # all adjacent nodes to channel nodes
#             AllNodes = np.concatenate((AdjN,AdjDN))
#             # unique adjacent nodes
#             AllNodes = np.unique(AllNodes)
#             # unique adjacent nodes, excluding all channel nodes.
#             TerraceNodes = AllNodes[np.in1d(AllNodes,self.ChannelNodes,invert = True)]
        
#         t_x = self._grid.node_x[TerraceNodes]
#         t_y = self._grid.node_y[TerraceNodes]
#         self.xyDf_t = pd.DataFrame({'x':t_x, 'y':t_y})
#         self.TerraceNodes = TerraceNodes
    


#     def min_distance_to_network(self, cellid, ChType = 'debrisflow'):
#         """determine the distance from a node to the nearest channel node
#             ChType = debrisflow: uses debris flow network
#             ChType = nmg: uses network model grid network
#         """
#         def distance_to_network(row):
#             """compute distance between nodes """
#             return ((row['x']-self.gridx[cellid])**2+(row['y']-self.gridy[cellid])**2)**.5
#         if ChType == 'debrisflow':
#             nmg_dist = self.xyDf_df.apply(distance_to_network,axis=1)
#             offset = nmg_dist.min() # minimum distancce
#             mdn = self.xyDf_df[nmg_dist == offset] # minimum distance node and node x y        
#         elif ChType == 'nmg':
#             nmg_dist = self.xyDf.apply(distance_to_network,axis=1)
#             offset = nmg_dist.min() # minimum distancce
#             mdn = self.xyDf[nmg_dist == offset] # minimum distance node and node x y    
#         return offset, mdn        

#######################
def extract_terrace_nodes(grid,ChannelNodes,FluvialChannelNodes,TerraceWidth):
    """determine which raster model grid nodes coincide with channel terraces,
    which presently are asssumed to be a fixed width (number of nodes) from
    the channel nodes
    """
    for i in range(TerraceWidth):
        if i == 0:
            # diagonal adjacent nodes to fluvial channel nodes
            AdjDN =np.ravel(grid.diagonal_adjacent_nodes_at_node[FluvialChannelNodes])  
            # adjacent nodes to channel fluvial nodes
            AdjN = np.ravel(grid.adjacent_nodes_at_node[FluvialChannelNodes])
        elif i>0:
            # diagonal adjacent nodes to channel nodes
            AdjDN = grid.diagonal_adjacent_nodes_at_node[TerraceNodes] 
            # adjacent nodes to channel nodes
            AdjN = grid.adjacent_nodes_at_node[TerraceNodes]            
        # all adjacent nodes to channel nodes
        AllNodes = np.concatenate((AdjN,AdjDN))
        # unique adjacent nodes
        AllNodes = np.unique(AllNodes)
        # unique adjacent nodes, excluding all channel nodes.
        TerraceNodes = AllNodes[np.in1d(AllNodes,ChannelNodes,invert = True)]
    
    return TerraceNodes

def dist_between_points(x1,y1,x2,y2):
    return (((x2-x1)**2)+((y2-y1)**2))**0.5

def min_distance_to_network(grid, cellid, xyDf):
    """determine the distance from a node to the nearest channel node
    """
    dist_to_channel_nodes = xyDf.apply(lambda row: dist_between_points(grid.node_x[cellid],
                                                                       grid.node_y[cellid], 
                                                                       row['x'],
                                                                       row['y']), axis=1)
    return dist_to_channel_nodes.min() 



class ChannelNetworkToolsMapper(ChannelNetworkToolsBase): 
    """map field values from network modelg grid nodes and links to raster model
    grid nodes and vice versa
    
    Parameters:
        grid
        nmgrid
    """
    
    def __init__(self,grid,**kwgs):
        """instatiate ChannelNetworkToolsMapper using the base class init"""
        ChannelNetworkToolsBase.__init__(self, grid, **kwgs)
        

    def map_nmg_links_to_rmg_nodes(self, linknodes, active_links, nmgx, nmgy):
        '''convert network model grid links to coincident raster model grid nodes.
        Output is a list of lists. order of lists is order of links. Coincident rmg
        nodes are used to determine the link # and location on link at which sediment
        on the rmg is inserted into the nmg.
        '''
        Lnodelist = [] #list of lists of all nodes that coincide with each link
        Ldistlist = [] #list of lists of the distance on the link (measured from upstream link node) for all nodes that coincide with each link
        xdDFlist = []
        Lxy= [] #list of all nodes the coincide with the network links
        #loop through all links in network grid to determine raster grid cells that coincide with each link
        #and equivalent distance from upstream node on link
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
            for i,v in enumerate(vals):
                x = X[i]
                mask = (self.ndyn>=v) & (self.ndys<=v) & (self.ndxe>=x) & (self.ndxw<=x)  #mask - use multiple boolean tests to find cell that contains point on link
                node = self.nodes[mask] #use mask to extract node value
                if node.shape[0] > 1:
                    node = np.array([node[0]])
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
    
    def map_nmg_links_to_rmg_channel_nodes(self, xyDf):
        """find the closest link to each channel node"""
        # for each channel node, get the distance to all link nodes
        # channel node x
        cnode_x = self._grid.node_x[self.ChannelNodes]
        # channel node y
        cnode_y = self._grid.node_y[self.ChannelNodes]
        
        def func(row, xc, yc):
            """distance between channel node and link node"""
            return ((xc-row['x'])**2+(yc-row['y'])**2)**0.5
        
        cn_link_id = []
        for i, cn in enumerate(self.ChannelNodes):
            xc = cnode_x[i]; yc = cnode_y[i]
            dist = xyDf.apply(lambda row: func(row, xc, yc),axis=1)
            # link of the link node with the shortest distance to the channel node is the link
            cn_link_id.append(xyDf['linkID'][dist == dist.min()].values[0])# if more than one pick the first value
        
        # for each link, group all channel nodes assigned to link into list, append to Lnodelist
        b = np.array(cn_link_id)
        Lnodelist = []
        for link in np.unique(b):
            Lnodelist.append(list(self.ChannelNodes[b == link]))
        return Lnodelist

    def map_nmg_links_to_rmg_nodes_DHSVM(self, linknodes, active_links, nmgx, nmgy):
        '''convert network model grid links to coincident raster model grid nodes.
        Output is a list of lists. order of lists is order of links
        '''
        Lnodelist = [] #list of lists of all nodes that coincide with each link
        Ldistlist = [] #list of lists of the distance on the link (measured from upstream link node) for all nodes that coincide with each link
        xdDFlist = []
        Lxy= [] #list of all nodes the coincide with the network links
        #loop through all links in network grid to determine raster grid cells that coincide with each link
        #and equivalent distance from upstream node on link
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
            for i,v in enumerate(vals):
                x = X[i]
                mask = (self.ndyn>=v) & (self.ndys<=v) & (self.ndxe>=x) & (self.ndxw<=x)  #mask - use multiple boolean tests to find cell that contains point on link
                node = self.nodes[mask] #use mask to extract node value
                if node.shape[0] > 1:
                    node = np.array([node[0]])
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
            
            df = pd.DataFrame.from_dict(Lxy)
            df['row'] = self._grid.at_node['DHSVM_row'][df['node']]
            df['column'] = self._grid.at_node['DHSVM_column'][df['node']]
            
            
            
             
        return (Lnodelist, Ldistlist, Lxy, df)

            
    def map_nmg1_links_to_nmg2_links(self,Lnodelist,xyDf_d):    
        
        """map link ids from a finer detail network model grid to equivalent link ids on
        a coarser scale network model grid 
        link to each link in the landlab nmg. Note, the landlab nmg id of the dhsmv
        network is used, not the DHSVM network id (arcID) To translate the nmg_d link
        id to the DHSVM network id: self.nmg_d.at_link['arcid'][i], where i is the nmg_d link id.
        """
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


    def map_nmg_links_to_rmg_channel_nodes_old(self, xyDf):
        """ Determine the closest nmg link (assign a network model grid link) to each 
        rmg channel node. Output can be used to map between nmg link field values and
        rmg node field values.
        
        this is the old version, had a bug, is now fixed
        """
        #compute distance between deposit and all network cells
     
        # for each node in the channel node list record the equivalent nmg_d link id 
        NodeMapper ={}
        ncn = self.ChannelNodes.shape[0] # number of channel nodes
        for i, node in enumerate(self.ChannelNodes):# for each node in the rmg channel node list
            def Distance(row,xc,yx):
                return ((row['x']-xc)**2+(row['y']-yc)**2)**0.5 
            xc = self.gridx[node]; yc = self.gridy[node]            
            nmg_d_dist = xyDf.apply(lambda row: Distance(row, xc, yc),axis=1) # compute the distance to all nmg nodes
            offset = nmg_d_dist.min() # find the minimum distance
            mdn = xyDf['linkID'][nmg_d_dist == offset].values[0]# link id of minimum distance nmg node
            NodeMapper[i] = mdn # dhsmve link for node i
            if i%20 == 0:
                print(str(np.round((i/ncn)*100))+' % RMG nodes mapped to equivalent DHSVM link')
        self.NodeMapper = NodeMapper
        
        b = np.array(list(self.NodeMapper.values()))
        Lnodelist = []
        for link in np.unique(b):
            Lnodelist.append(list(b[b == link]))
            
        self.EquivalentLinkForNode = Lnodelist


    def map_rmg_nodes_to_nmg_nodes(self):
        """ find rmg node that is closest to each nmg node (assign a raster model grid
        node to each network model grid node)
        
        This function can be used to map changes in the rmg topographic__elevation 
        field to the nmg topographic__elevation field
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


    def transfer_rmg_node_field_to_nmg_node_field(self):
        """update the topographic elevation field of nmg using th elevation of the
        equivalent raster model grid node
        """
        # update elevation
        for i, node in enumerate(self.nmg_nodes):
            RMG_node = self.NMGtoRMGnodeMapper[i]
            self._nmgrid.at_node['topographic__elevation'][i] = self._grid.at_node['topographic__elevation'][RMG_node]
            
    
    def transfer_nmg_link_field_to_rmg_node_field(self, nmg_field, rmg_field, link_node_list):
        """
        transfers the field value on each link to each of its coincident rmg
        nodes.

        Returns
        -------
        None.
        """
        
        for i, link in enumerate(self._nmgrid.active_links):
            value = self._nmgrid.at_link[nmg_field][link]
            link_nodes =  link_node_list[i]
            self._grid.at_node[rmg_field][link_nodes] = value
            
#############################
def map_nmg_links_to_rmg_nodes(grid, linknodes, active_links, nmgx, nmgy):
    '''convert network model grid links to coincident raster model grid nodes.
    Output is a list of lists. order of lists is order of links. Coincident rmg
    nodes are used to determine the link # and location on link at which sediment
    on the rmg is inserted into the nmg.
    '''
    
    # extent of each cell in grid        
    ndxe = grid.node_x+grid.dx/2
    ndxw = grid.node_x-grid.dx/2
    ndyn = grid.node_y+grid.dy/2
    ndys = grid.node_y-grid.dy/2
    
    # nodes, reshaped in into m*n,1 array like other mg fields
    nodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1],1)
    
    Lnodelist = [] #list of lists of all nodes that coincide with each link
    Ldistlist = [] #list of lists of the distance on the link (measured from upstream link node) for all nodes that coincide with each link
    xdDFlist = []
    Lxy= [] #list of all nodes the coincide with the network links
    #loop through all links in network grid to determine raster grid cells that coincide with each link
    #and equivalent distance from upstream node on link
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
        for i,v in enumerate(vals):
            x = X[i]
            mask = (ndyn>=v) & (ndys<=v) & (ndxe>=x) & (ndxw<=x)  #mask - use multiple boolean tests to find cell that contains point on link
            node = nodes[mask] #use mask to extract node value
            if node.shape[0] > 1:
                node = np.array([node[0]])
            # create list of all nodes that coincide with linke
            if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                nodelist.append(node[0][0])  
                distlist.append(dist[i])
                xy = {'linkID':linkID,
                    'node':node[0][0],
                      'x':grid.node_x[node[0][0]],
                      'y':grid.node_y[node[0][0]]}
                Lxy.append(xy)
        
        Lnodelist.append(nodelist)
        Ldistlist.append(distlist)
        
    return (Lnodelist, Ldistlist, Lxy)


    
