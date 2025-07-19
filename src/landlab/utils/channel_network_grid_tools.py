from landlab import Component, FieldError 
from landlab.components.flow_director.flow_director_steepest import FlowDirectorSteepest
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy as sc


# """
#     TODO:
#     add second network model grid input?
#     make all functions use class variables, user just runs the function after class instantiation?
#     look at Shelby's "create_network" utility for coding style options
    
#     break up into a functions
#         1.figure out inputs
#         2.rewrite
# """




def _flatten_lol(lol):
    """
    for list (l) in list of lists (lol) for item (i) in l append i"

    Parameters
    ----------
    lol : list of lists

    Returns
    -------
    the list of lists concatenated into a single list
    """
    return [i for l in lol for i in l] 

def get_linknodes(nmgrid):
    """get the downstream and upstream nodes at a link from flow director.
    Linknodes in network model grid (nodes_at_link) may not be correct. Output
    from this function should be used as input for functions that require
    linknodes in ChannelNetworkGridTools
    """
    fd = FlowDirectorSteepest(nmgrid, "topographic__elevation")
    fd.run_one_step()
    upstream_node_id = []; downstream_node_id = []
    for i in range(nmgrid.number_of_links):
        upstream_node_id.append(fd.upstream_node_at_link()[i])
        downstream_node_id.append(fd.downstream_node_at_link()[i])
    # create np array and transpose, each row is {downstream node id, upstream node id]
    linknodes = np.array([downstream_node_id,upstream_node_id]).T 
    return linknodes

def _node_at_coords(grid, x, y):
    """
    given a raster model grid and coordinates x and y, this finds the
    rmg node closest to the x and y

    Parameters
    ----------
    grid : raster model grid
    x : float
    y : float

    Returns
    -------
    node : int
        node id number

    """
    ndxe = grid.node_x+grid.dx/2
    ndxw = grid.node_x-grid.dx/2
    ndyn = grid.node_y+grid.dy/2
    ndys = grid.node_y-grid.dy/2
    # find cell that contains point
    mask = (ndyn>=y) & (ndys<=y) & (ndxe>=x) & (ndxw<=x)  
    # rmg nodes, reshaped in into m*n,1 array like other mg fields
    nodes = grid.nodes.flatten()
    node = nodes[mask] #use mask to extract node value
    if node.shape[0] >= 1: # if at cell boundary, use first node
        node = np.array([node[0]])
    else:
        raise ValueError("coordinates outside of model grid")
    return node

def _link_to_points_and_dist(x0,y0,x1,y1,number_of_points = 1000):
        """
    create a series of point coordinates along a channel reach
    and the distance to each coordinate from the upstream end of 
    the reach

    Parameters
    ----------
    x0 : float 
        upstream end coordinate x
    y0 : float
        upstream end coordinate y
    x1 : float
        downstream end coordinate x
    y1 : float
        downstream end coordinate y
    number_of_points : int
        number of points to create along the reach. The default is 1000.

    Returns
    -------
    X : np array
        x coordinate of points
    Y : np array
        y coordinate of points
    dist : np array
        distance to point from the upstream end of the reach (downstream distance)

    """
        #create 1000 points along domain of link
        X = np.linspace(x0,x1,number_of_points)
        Xs = np.abs(X-x0) #change begin value to zero
        #determine distance from upstream node to each point
        #y value of points
        if Xs.max() ==0: #if a vertical link (x is constant)
            Y = np.linspace(y0,y1,number_of_points) # y 
            dist = np.abs(Y-y0) #distance along link, from downstream end upstream
        else: #all their lines
            Y = y0+(y1-y0)/np.abs(x1-x0)*(Xs) # y
            dist = ((Y-y0)**2+Xs**2)**.5
        return X, Y, dist

class ChannelNetworkToolsBase():    
    '''
    Base class for ChannelNetworkToolsInterpreter and ChannelNetworkToolsMapper.
    This class estblishes class variables and includes functions shared by both
    child classes.

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
            TerraceWidth = 1,
            **kwds):
        
        if grid != None:

            self._grid = grid
    
            if 'topographic__elevation' in grid.at_node:
                self.dem = grid.at_node['topographic__elevation']
            else:
                raise FieldError(
                    'A topography is required as a component input!')
            
            ### raster model grid attributes
            self.ncn = len(grid.core_nodes) # number core nodes
            self.gr = grid.shape[0] #number of rows
            self.gc = grid.shape[1] #number of columns
            self.dx = grid.dx #width of cell
            self.dy = grid.dy #height of cell

            # nodes, reshaped in into m*n,1 array like other mg fields
            self.nodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1],1)
            self.rnodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1]) #nodes in single column array

            if 'flow__receiver_node' in grid.at_node: # consider moving this to Landslide Mapper
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
            ### network model grid attributes
            self._nmgrid = nmgrid
            self.linknodes = nmgrid.nodes_at_link #links as ordered by read_shapefile       
            self.nmgridx = nmgrid.x_of_node
            self.nmgridy = nmgrid.y_of_node
            self.linklength = nmgrid.length_of_link 
            self.nmg_nodes = nmgrid.nodes

        ### Channel extraction parameters
        self.Ct = Ct # Channel initiation threshold [m2]   
        self.BCt = BCt # CA threshold for channels that typically transport bedload [m2] 
        self.TerraceWidth = TerraceWidth # distance from channel grid cells that are considered terrace grid cells [# cells]          

    
    def extract_channel_nodes(self,Ct,BCt):
        """interpret which nodes of the DEM correspond to the fluvial channel 
        network and entire channel network (including colluvial channels)
        """
        
        # entire channel network (all channels below channel head or upper extent 
        # of the colluvial channel network)
        ChannelNodeMask = self._grid.at_node['drainage_area'] > Ct
        df_x = self.gridx[ChannelNodeMask]
        df_y = self.gridy[ChannelNodeMask]
        self.xyDf_df = pd.DataFrame({'x':df_x, 'y':df_y})
        self.ChannelNodes = self.rnodes[ChannelNodeMask] 
        
        # fluvial channel network only (all channels below upper extent of cascade channels)
        BedloadChannelNodeMask = self._grid.at_node['drainage_area'] > BCt
        bc_x = self.gridx[BedloadChannelNodeMask]
        bc_y = self.gridy[BedloadChannelNodeMask]
        self.xyDf_bc = pd.DataFrame({'x':bc_x, 'y':bc_y})
        self.BedloadChannelNodes = self.rnodes[BedloadChannelNodeMask]         


class ChannelNetworkToolsInterpretor(ChannelNetworkToolsBase):
    """
    A set of tools for interpreting the fluvial and enitre channel
    network (fluvial and colluvial channels, which end at the channel head) and
    determining distance from a particular node in the watershed to the network


    Parameters:
        grid
        nmgrid
        **kwgs include:

            Ct: float
                Contributing area threshold at which channel begin (colluvial channels)
            BCt: float
                Contributing area threshold at which cascade channels begin, which is 
                assumed to be the upper limit of frequent bedload transport
            TerraceWidth: int
                width from channel cells in number of cells considered terrace cells.
                Defaul value is 1 (i.e. all cells directly adjacent to the channels
                                   cells are considered terrace cells)
    """

    
    def __init__(self, grid, **kwgs):
        """instatiate ChannelNetworkToolsInterpretor using base class init"""
        ChannelNetworkToolsBase.__init__(self, grid, **kwgs)

    def extract_terrace_nodes(self):
        """determine which raster model grid nodes coincide with channel terraces,
        which presently are asssumed to be a fixed width (number of nodes) from
        the channel nodes
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
            
            # finally, remove any -1 nodes, which represent adjacent nodes outside
            # of the model grid
            TerraceNodes = TerraceNodes[~(TerraceNodes == -1)]
        
        t_x = self._grid.node_x[TerraceNodes]
        t_y = self._grid.node_y[TerraceNodes]
        self.xyDf_t = pd.DataFrame({'x':t_x, 'y':t_y})
        self.TerraceNodes = TerraceNodes
    


    def min_distance_to_network(self, cellid, ChType = 'debrisflow'):
        """determine the distance from a node to the nearest channel node
            ChType = debrisflow: uses debris flow network
            ChType = nmg: uses network model grid network
        """
        def distance_to_network(row):
            """compute distance between nodes """
            return ((row['x']-self.gridx[cellid])**2+(row['y']-self.gridy[cellid])**2)**.5
        if ChType == 'debrisflow':
            nmg_dist = self.xyDf_df.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDf_df[nmg_dist == offset] # minimum distance node and node x y        
        elif ChType == 'nmg':
            nmg_dist = self.xyDf_bc.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDf_bc[nmg_dist == offset] # minimum distance node and node x y    
        return offset, mdn        

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
        

    def map_nmg_links_to_rmg_nodes(self, linknodes, nmgx, nmgy, remove_duplicates = False):
        '''map the network model grid to the coincident raster model grid nodes (i.e.,
        create the nmg rmg nodes). The nmg rmg nodes are defined based on the 
        the rmg node id and the coincident nmg link and distance downstream on that
        link. In Eroder, erosion at
        a channel or terrace is becomes a pulse of sediment and is inserted into
        the network model grid at the link # and downstream location of the closest
        rmg nmg node.(see _parcelDFmaker function in eroder). 
        The pulser utlity uses the dataframe to transfer the depostion 
        to the nmg at the corrisponding link # and downstream distance.
        
        # this code assumes linknodes correctly ordered as [downstream node, upstream node]
        
        # update to remove duplicates
        '''
        Lnodelist = [] #list of lists of all nodes that coincide with each link
        Ldistlist = [] #list of lists of the distance on the link (measured from upstream link node) for all nodes that coincide with each link
        LlinkIDlist = []
        Lxlist = []
        Lylist = []
        Lxy= [] #list of all nodes the coincide with the network links
        #loop through all links in network grid to determine raster grid cells that coincide with each link
        #and equivalent distance from upstream node on link
        for linkID,lknd in enumerate(linknodes) : #for each link in network grid
           
            x1 = nmgx[lknd[0]] #x and y of downstream link node
            y1 = nmgy[lknd[0]]
            x0 = nmgx[lknd[1]] #x and y of upstream link node
            y0 = nmgy[lknd[1]]
            
            # 1000 points generated from upstream node to downstream node
            X, Y, dist = _link_to_points_and_dist(x0,y0,x1,y1,number_of_points = 1000)
          
            nodelist = [] #list of nodes along link
            distlist = [] #list of distance along link corresponding to node
            linkIDlist =[]
            xlist = []
            ylist =[]
            for i,y in enumerate(Y):
                x = X[i]
                node = _node_at_coords(self._grid,x,y)
                if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                    nodelist.append(node[0])  
                    distlist.append(dist[i])
                    linkIDlist.append(linkID)
                    xlist.append(self.gridx[node[0]])
                    ylist.append(self.gridy[node[0]])
                    xy = {'linkID':linkID,
                          'node':node[0],
                          'x':self.gridx[node[0]],
                          'y':self.gridy[node[0]], # add dist
                          'dist':dist[i]}
                    Lxy.append(xy)
            
            Lnodelist.append(nodelist)
            Ldistlist.append(distlist)
            LlinkIDlist.append(linkIDlist)
            Lxlist.append(xlist)
            Lylist.append(ylist)
        # remove duplicates by assiging duplicate node to link with largest mean contributing area.
        if remove_duplicates:
            for link in range(len(linknodes)):
                for other_link in range(len(linknodes)):
                    if link != other_link:
                        link_nodes = Lnodelist[link]
                        other_link_nodes = Lnodelist[other_link]
                        link_a = self._nmgrid.at_link['drainage_area'][link]
                        other_link_a = self._nmgrid.at_link['drainage_area'][other_link]
                        dup = np.intersect1d(link_nodes,other_link_nodes)
                        # if contributing area of link a largest, remove dupilcate nodes from other link
                        if len(dup)>0:
                            print('link {} and link {} have duplicates: {}'.format(link, other_link, dup))
                            if link_a >= other_link_a:
                                mask = ~np.isin(other_link_nodes, dup)
                                Lnodelist[other_link] = list(np.array(other_link_nodes)[mask])
                                Ldistlist[other_link] = list(np.array(Ldistlist[other_link])[mask])
                                LlinkIDlist[other_link] = list(np.array(LlinkIDlist[other_link])[mask])
                                Lxlist[other_link] = list(np.array(Lxlist[other_link])[mask])
                                Lylist[other_link] = list(np.array(Lylist[other_link])[mask])
                            else:
                                mask = ~np.isin(link_nodes, dup)
                                Lnodelist[link] = list(np.array(link_nodes)[mask])                
                                Ldistlist[link] = list(np.array(Ldistlist[link])[mask])
                                LlinkIDlist[link] = list(np.array(LlinkIDlist[link])[mask])
                                Lxlist[link] = list(np.array(Lxlist[link])[mask])
                                Lylist[link] = list(np.array(Lylist[link])[mask])

            Lxy = pd.DataFrame(np.array([_flatten_lol(LlinkIDlist),
                                _flatten_lol(Lnodelist),
                                _flatten_lol(Lxlist),
                                _flatten_lol(Lylist),
                                _flatten_lol(Ldistlist)]).T,
                               columns = ['linkID','node','x','y','dist'])
            Lxy['linkID'] = Lxy['linkID'].astype(int)
            Lxy['node'] = Lxy['node'].astype(int)
            
        return (Lnodelist, Ldistlist, Lxy, LlinkIDlist, Lxlist, Lylist) # coincident rmg node, downstream distance on link of node and dictionary that contains the link id, x and y value of each conicident node
    
    def map_rmg_channel_nodes_to_nmg_rmg_nodes(self, xyDf):
        """for each link i in the nmg create a list of the rmg channel nodes that
        represent the link"""
        # for each channel node, get the distance to all link rmg nodes
        # channel node x
        cnode_x = self._grid.node_x[self.ChannelNodes] # cn, not fcn
        # channel node y
        cnode_y = self._grid.node_y[self.ChannelNodes]
        
        def func(row, xc, yc):
            """distance between channel node and link node"""
            return ((xc-row['x'])**2+(yc-row['y'])**2)**0.5
        
        cn_link_id = []
        for i, cn in enumerate(self.ChannelNodes): # for each channel node
            xc = cnode_x[i]; yc = cnode_y[i]
            # compute the distance to all link rmg nodes
            dist = xyDf.apply(lambda row: func(row, xc, yc),axis=1) 
            # link nmg_rmg node with the shortest distance to the channel node is 
            # assigned to the channel node.
            # if more than one (which can happen because the end and begin of the reaches at a junction overlay the same cell) 
            # pick link with largest contributing area
            dist_min_links = xyDf['linkID'][dist == dist.min()].values
            dmn_cont_area = self._nmgrid.at_link['drainage_area'][dist_min_links]
            dmn_mask = dmn_cont_area == dmn_cont_area.max()
            
            cn_link_id.append(dist_min_links[dmn_mask][0]) # more than one cell from the same link may be the same distance from the node, just pick one
        
        # for each link, group all channel nodes assigned to link into list, append to Lcnodelist
        b = np.array(cn_link_id)
        Lcnodelist = []
        for link in range(self._nmgrid.number_of_links): #np.unique(b):
            Lcnodelist.append(list(self.ChannelNodes[b == link]))
        return Lcnodelist # for each link i, the list of channel nodes assigned to the link

    def map_nmg1_links_to_nmg2_links(self,Lnodelist_nmg1,xyDf_d_nmg2):    
        
        """map link ids from one network model grid (nmg1) to equivalent link ids on
        another network model grid (nmg2) 
        link to each link in the landlab nmg. Note, the landlab nmg id of the dhsvm
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
        for i, sublist in enumerate(Lnodelist_nmg1):# for each nmg1 link i
            LinkL = []
            for j in sublist: # for each rmg node j in the list
                XY = [self.gridx[j], self.gridy[j]] # get x and y coordinate of node
                nmg_d_dist = xyDf_d_nmg2.apply(Distance,axis=1) # compute the distance to all nodes of nmg2 links
                offset = nmg_d_dist.min() # find the minimum distance
                mdn = xyDf_d_nmg2['linkID'].values[(nmg_d_dist == offset).values][0]# nmg2 link id with minimum distance node
                LinkL.append(mdn) # nmg2 link id is appended to a list
            LinkMapL[i] = LinkL # final list of all nmg2 link ids matched to a nmg1 link i rmg node
            LinkMapper[i] = np.argmax(np.bincount(np.array(LinkL))) # nmg2 link with highest count is mapped to nmg1
            print('nmg1 link '+ str(i)+' mapped to equivalent nmg2 link')
        return (LinkMapper, LinkMapL)

    def _dist_func(x1,x2,y1,y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5 


    def map_rmg_channel_nodes_to_nmg_nodes(self):
        """ find rmg channel node that is closest to each nmg node
        
        The mapping dictionary produced by this function can be used to transfer 
        field values from a rmg node to an spatially equivalent nmg node. However,
        if the rmg channel network does not extend to the upper reaches of 
        the nmg channel network, the most upstream nmg nodes may be matched with 
        rmg nodes in the wrong reach or branch of the rmg channel network. This 
        function works best if the rmg channel network roughly underlies the nmg 
        channel network.
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
            mdn = self.xyDf_df.index[(nmg_node_dist == offset).values][0]# index of minimum distance node, use first value
            NodeMapper[i] = self.ChannelNodes[mdn] # rmg node mapped to nmg node i         
        self.NMGtoRMGnodeMapper = NodeMapper



    def transfer_rmg_channel_node_field_to_nmg_node_field(self, field = 'topographic__elevation'):
        """update the topographic elevation field of nmg using th elevation of the
        equivalent raster model grid node
        """
        # update elevation
        for i, node in enumerate(self.nmg_nodes):
            RMG_node = self.NMGtoRMGnodeMapper[i]
            self._nmgrid.at_node[field][i] = self._grid.at_node[field][RMG_node]
            
    
    def transfer_nmg_link_field_to_rmg_channel_node_field(self, nmg_field, rmg_field, Lcnodelist):
        """
        transfers the field value on each link to each rmg channel node assigned
        to the link in map_rmg_channel_nodes_to_nmg_rmg_nodes()

        Returns
        -------
        None.
        """
        for i, link in enumerate(self._nmgrid.active_links):
            value = self._nmgrid.at_link[nmg_field][link]
            link_nodes =  Lcnodelist[i]
            self._grid.at_node[rmg_field][link_nodes] = value
            
            
    def transfer_rmg_channel_node_field_to_nmg_link_field(self, rmg_field, nmg_field, Lcnodelist, metric = 'mean'):
        """
        transfer the max, min, mean or median field value of the rmg 
        channel nodes that represent each nmg link to the nmg link field.
        """
        for i, link in enumerate(self._nmgrid.active_links):
            link_nodes =  Lcnodelist[i]
            if len(link_nodes)>0: # if link_nodes is not empty
                if metric == 'mean':
                    value = self._grid.at_node[rmg_field][link_nodes].mean()
                elif metric == 'max':
                    value = self._grid.at_node[rmg_field][link_nodes].max()
                elif metric == 'min':
                    value = self._grid.at_node[rmg_field][link_nodes].min()
                elif metric == 'median':
                    value = np.median(self._grid.at_node[rmg_field][link_nodes])
                else:
                    raise ValueError('metric "{}" not an option'.format(metric))
                self._nmgrid.at_link[nmg_field][link] = value
            