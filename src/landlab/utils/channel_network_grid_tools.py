from landlab import Component, FieldError 
from landlab.components.flow_director.flow_director_steepest import FlowDirectorSteepest
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy as sc

"""
A collection of tools for interpreting the channel location from a DEM and mapping 
values from different grid representations of a channel network.
"""
#     TODO:

# check DHG for other functions that should be a part of CNGT

# create a function that updates the mappers for a new DEM and rmg channel location - done

# notebooks: notebook that loads the b694 grid and maps links between the two grids

# check test notebook still runs
# consistent naming and format for  - done

# add docstring that explains key terms

# two pull requests

# WHEN pull request for CNGT is done, update MWRo, eroder, mapper, DHG


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


def get_link_nodes(nmgrid):
    """get the downstream (head) and upstream (tail) nodes at a link from 
    flow director. The network model grid nodes_at_link attribute may not be 
    ordered according to flow direction. Output from this function should be 
    used for all ChannelNetworkGridTool functions that require a link_nodes input
    
    Parameters
    ----------
    nmgrid : network model grid
    
    Returns
    -------
    link_nodes : np.array
        for a nmgrid of n nodes, returns a nx2 np array, the ith row of the
        array is the [downstream node id, upstream node id] of the ith link
    """
    
    fd = FlowDirectorSteepest(nmgrid, "topographic__elevation")
    fd.run_one_step()
    upstream_node_id = []; downstream_node_id = []
    for i in range(nmgrid.number_of_links):
        upstream_node_id.append(fd.upstream_node_at_link()[i])
        downstream_node_id.append(fd.downstream_node_at_link()[i])
    # create np array and transpose, each row is [downstream node id, upstream node id]
    link_nodes = np.array([downstream_node_id,upstream_node_id]).T 
    return link_nodes


def _link_to_points_and_dist(x0,y0,x1,y1,number_of_points = 1000):
    """Given two points defined by coordinates x0,y0 and x1,y1, create a series 
    of points between them.
    If x0,y0 and x1,y1 are the head and tail nodes of a link, the distance is the 
    downstream distance along the link.

    Parameters
    ----------
    x0 : float 
        point 1 coordinate x
    y0 : float
        point 1 coordinate y
    x1 : float
        point 2 coordinate x
    y1 : float
        point 2 coordinate y
    number_of_points : int
        number of points to create along the reach. The default is 1000.

    Returns
    -------
    X : np array
        x coordinate of points
    Y : np array
        y coordinate of points
    dist : np array
        linear distance to point from the point 2

    """
    # create number_of_points points along domain of link
    X = np.linspace(x0,x1,number_of_points)
    Xs = np.abs(X-x0) #change begin value to zero
    # determine distance from upstream node to each point
    if Xs.max() == 0: #if a vertical link (x is constant)
        Y = np.linspace(y0,y1,number_of_points) # y 
        dist = np.abs(Y-y0)
    else:
        Y = y0+(y1-y0)/np.abs(x1-x0)*(Xs) # y
        dist = ((Y-y0)**2+Xs**2)**.5
    return X, Y, dist

    
def _dist_func(x0,x1,y0,y1):
    """returns linear distance between two points"""
    return ((x0-x1)**2+(y0-y1)**2)**0.5 


def extract_channel_nodes(grid, Ct):
    """interpret which nodes of the DEM represent the channel network as all nodes 
    that have a drainage area >= to the average drainage area at which 
    channels initiate in the DEM (Ct, based on field or remote sensing evidence). 
    
    Ct = average drainage area at which colluvial channels to get the entire 
    channel network. 
    Ct = the drainage area at which cascade channels typically begin to get 
    a channel network where sediment transport is primarily via fluvial processes
    

    Parameters
    ----------
    grid : raster model grid
        raster model grid with node field "drainage_area"
    Ct : float
        Channel threshold drainage area

    Returns
    -------
    cn : np array of int
         array of all node ids included in the channel network

    """
    cn_mask = grid.at_node['drainage_area'] >= Ct
    cn = grid.nodes.flatten()[cn_mask] 
    return cn


def extract_terrace_nodes(grid, terrace_width, acn, fcn):
    """determine which raster model grid nodes coincide with channel terraces,
    which presently are asssumed to be a fixed width (number of nodes) from
    the channel nodes
    

    Parameters
    ----------
    grid : raster model grid
    terrace_width : int
        Width of terrace in number of nodes. If provided as float, will be rounded
        to nearest int.
    acn : np array 
        array of all node IDs included in the channel network 
    fcn : np array
        array of all node IDs included in the fluvial channel network

    Raises
    ------
    ValueError
        Occurs if terrace width less than 1.

    Returns
    -------
    TerraceNodes : np array
        array of all node IDs included in the terrace 

    """
    terrace_width = np.round(terrace_width).astype(int) # round to int in case provided as float
    
    if terrace_width <1: # check that at least 1
        msg = "terrace width must be 1 or greater"
        raise ValueError(msg)
        
        
    for i in range(terrace_width):
        if i == 0:
            # diagonal adjacent nodes to channel nodes
            AdjDN =np.ravel(grid.diagonal_adjacent_nodes_at_node[fcn])  
            # adjacent nodes to channel nodes
            AdjN = np.ravel(grid.adjacent_nodes_at_node[fcn])
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
        TerraceNodes = AllNodes[np.isin(AllNodes,acn,invert = True)]
        # finally, remove any -1 nodes, which represent adjacent nodes outside
        # of the model grid
        TerraceNodes = TerraceNodes[~(TerraceNodes == -1)]
    t_x = grid.node_x[TerraceNodes]
    t_y = grid.node_y[TerraceNodes]
    xyDF_t = pd.DataFrame({'x':t_x, 'y':t_y})
    TerraceNodes = TerraceNodes
    return TerraceNodes


def min_distance_to_network(grid, acn, node_id):
    """determine the shortest distance (as the crow flies) from a node to the channel network and
    the closest channel node

    Parameters
    ----------
    grid : raster model grid
    acn : list of int
        array of all node ids included in the channel network
    node_id : int
        ID of node from which the distance will be determined

    Returns
    -------
    offset : float
        distance between node and channel network
    mdn : int
        ID of channel node that is closest node

    """
    def distance_to_network(grid, row):
        """compute distance between nodes """
        return _dist_func(row['x'],grid.node_x[node_id],row['y'],grid.node_y[node_id])      
    xyDF = pd.DataFrame(np.array([grid.node_x[acn],grid.node_y[acn]]).T, columns = ['x','y'])
    xyDF.index = acn
    nmg_dist = xyDF.apply(lambda row: distance_to_network(grid, row), axis=1) 
    offset = nmg_dist.min() # minimum distancce
    mdn = xyDF[nmg_dist == offset].index.values # find closest node
    if len(mdn) > 1:
        mdn = mdn[0] # pick first in list if more than one 
    return offset, mdn       


def map_nmg_links_to_rmg_coincident_nodes(grid, nmgrid, link_nodes, remove_duplicates = False):
    """maps the links of the network model grid to all coincident raster model grid 
    nodes. Each coincident raster model grid node is defined in terms of its 
    x and y coordinates, the the link it is mapped to and distance downstream from 
    the upstream end (tail end) of the link. 
    

    Parameters
    ----------
    grid : raster model grid
    nmgrid : network model grid
    link_nodes : np array
        head and tail node of each link
    remove_duplicates : bool
        if True, when two or more links are coincident with the same node,
        the node is assigned to the link with the larges drainage area. If False,
        the node is assigned to each coincident link. The default is False.

    Returns
    -------
    
    nmg_link_to_rmg_coincident_nodes_mapper: pandas dataframe
        each row of the dataframe lists the link ID, the coincident node ID, the 
        x and y coordinates and the downstream distance of the coincident node 
        and the drainage area of the link

    """
    Lnodelist = [] #list of lists of all nodes that coincide with each link
    Ldistlist = [] #list of lists of the distance on the link (measured from upstream link node) for all nodes that coincide with each link
    LlinkIDlist = []
    Lxlist = []
    Lylist = []
    Lxy= [] #list of all nodes the coincide with the network links
    #loop through all links in network grid to determine raster grid cells that coincide with each link
    #and equivalent distance from upstream node on link
    for linkID,lknd in enumerate(link_nodes) : #for each link in network grid
       
        x0 = nmgrid.x_of_node[lknd[0]] #x and y of downstream link node
        y0 = nmgrid.y_of_node[lknd[0]]
        x1 = nmgrid.x_of_node[lknd[1]] #x and y of upstream link node
        y1 = nmgrid.y_of_node[lknd[1]]
        
        # x and y coordinates and downstream distance from the upstream node
        # for 1000 points generated from downstream node to upstream node
        X, Y, dist = _link_to_points_and_dist(x0,y0,x1,y1,number_of_points = 1000)
        dist = dist.max()-dist # convert to downstream distance
        nodelist = [] #list of nodes along link
        distlist = [] #list of distance along link corresponding to node
        linkIDlist =[]
        xlist = []
        ylist =[]
        for i,y in enumerate(Y):
            x = X[i]
            node = grid.find_nearest_node((x,y)) # change to grid.find_nearest_node
            if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                nodelist.append(node)  
                distlist.append(dist[i])
                linkIDlist.append(linkID)
                xlist.append(grid.node_x[node])
                ylist.append(grid.node_y[node])
                xy = {'linkID':linkID,
                      'coincident_node':node,
                      'x':grid.node_x[node],
                      'y':grid.node_y[node], 
                      'dist':dist[i],
                      'drainage_area':nmgrid.at_link['drainage_area'][linkID]}
                Lxy.append(xy)
              
        Lnodelist.append(nodelist)
        Ldistlist.append(distlist)
        LlinkIDlist.append(linkIDlist)
        Lxlist.append(xlist)
        Lylist.append(ylist)
    
    nmg_link_to_rmg_coincident_nodes_mapper = pd.DataFrame(Lxy)
    
    # if remove_duplicates, select link with largest mean contributing area.
    if remove_duplicates:
        for link in range(len(link_nodes)):
            for other_link in range(len(link_nodes)):
                if link != other_link:
                    link_coin_nodes = Lnodelist[link]
                    other_link_coin_nodes = Lnodelist[other_link]
                    link_a = nmgrid.at_link['drainage_area'][link]
                    other_link_a = nmgrid.at_link['drainage_area'][other_link]
                    dup = np.intersect1d(link_coin_nodes,other_link_coin_nodes)
                    # if contributing area of link largest, remove dupilcate nodes from other link
                    if len(dup)>0:
                        print('link {} and link {} have duplicates: {}'.format(link, other_link, dup))
                        if link_a >= other_link_a:
                            mask = ~np.isin(other_link_coin_nodes, dup)
                            Lnodelist[other_link] = list(np.array(other_link_coin_nodes)[mask])
                            Ldistlist[other_link] = list(np.array(Ldistlist[other_link])[mask])
                            LlinkIDlist[other_link] = list(np.array(LlinkIDlist[other_link])[mask])
                            Lxlist[other_link] = list(np.array(Lxlist[other_link])[mask])
                            Lylist[other_link] = list(np.array(Lylist[other_link])[mask])
                        else:
                            mask = ~np.isin(link_coin_nodes, dup)
                            Lnodelist[link] = list(np.array(link_coin_nodes)[mask])                
                            Ldistlist[link] = list(np.array(Ldistlist[link])[mask])
                            LlinkIDlist[link] = list(np.array(LlinkIDlist[link])[mask])
                            Lxlist[link] = list(np.array(Lxlist[link])[mask])
                            Lylist[link] = list(np.array(Lylist[link])[mask])
        LinkIDs = _flatten_lol(LlinkIDlist)
        nmg_link_to_rmg_coincident_nodes_mapper = pd.DataFrame(np.array([LinkIDs,
                            _flatten_lol(Lnodelist),
                            _flatten_lol(Lxlist),
                            _flatten_lol(Lylist),
                            _flatten_lol(Ldistlist),
                           nmgrid.at_link['drainage_area'][np.array(LinkIDs)]]).T,
                           columns = ['linkID','coincident_node','x','y','dist','drainage_area'])
        nmg_link_to_rmg_coincident_nodes_mapper['linkID'] = nmg_link_to_rmg_coincident_nodes_mapper['linkID'].astype(int)
        nmg_link_to_rmg_coincident_nodes_mapper['coincident_node'] = nmg_link_to_rmg_coincident_nodes_mapper['coincident_node'].astype(int)
        
    return nmg_link_to_rmg_coincident_nodes_mapper 





####PULL REQUEST 2

def map_rmg_nodes_to_nmg_links(grid, nmg_link_to_rmg_coincident_nodes_mapper, rmg_nodes, remove_small_tribs = True):
    """
    Parameters
    ----------
    grid : raster model grid
        needs to have node field "drainage_area"
    nmg_link_to_rmg_coincident_nodes_mapper : pandas dataframe
        each row of the dataframe lists the link ID, the coincident node ID, the 
        x and y coordinates and the downstream distance of the coincident node 
        and the drainage area of the link
    rmg_nodes : np.array
        an array of node ids to be mapped to the nmg links
    remove_small_tribs : bool
        If True, first order channels that split from a much higher order channel
        are not matched to a Link. If False, first order channels will be mapped 
        to the same link as the much larger channel they drain into.

    Returns
    -------
    rmg_nodes_to_nmg_links_mapper : pandas dataframe
        each row of the dataframe lists the node ID, the link ID the node has been 
        mapped too, the closest nmg-link-coincident node ID, the drainage area
        of the link and the drainage area of the node

    """

    def dist_between_nmg_and_rmg_nodes(row, xc, yc):
        """distance between channel node and link node"""
        return _dist_func(xc,row['x'],yc,row['y'])
    
    rmg_node_link_id = []
    node_ = []
    dist_ = []
    link_ = []
    for n in rmg_nodes: # for each rmg node
        xc = grid.node_x[n]; yc = grid.node_y[n]
        # compute the distance to all link coincident rmg nodes
        dist = nmg_link_to_rmg_coincident_nodes_mapper.apply(lambda row: dist_between_nmg_and_rmg_nodes(row, xc, yc),axis=1)
        # pick closest coincident node and corresponding link
        # if more than one (which can happen because the confluence between two
        # links overlay the same node), pick link with largest contributing area
        mask = dist == dist.min()
        dist_min_links = nmg_link_to_rmg_coincident_nodes_mapper[['linkID','dist','coincident_node','drainage_area']][mask]
        link = dist_min_links[dist_min_links['drainage_area']==dist_min_links['drainage_area'].max()].head(1)
        link['node_drainage_area'] = grid.at_node['drainage_area'][n] # add node drainage area to attributes included in mapper
        link_.append(link)
        
    rmg_nodes_to_nmg_links_mapper  = pd.concat(link_)
    rmg_nodes_to_nmg_links_mapper['node'] = rmg_nodes
    rmg_nodes_to_nmg_links_mapper =rmg_nodes_to_nmg_links_mapper[['node','linkID','dist','coincident_node','drainage_area','node_drainage_area']].reset_index(drop=True) 

    if remove_small_tribs:# check for small tributary nodes assigned to link and remove them 
        for link in np.unique(nmg_link_to_rmg_coincident_nodes_mapper['linkID'].values):
            # first get the coincident rmg node that represents the inlet to the link
            mask1 = nmg_link_to_rmg_coincident_nodes_mapper['linkID'] == link 
            min_area_node = nmg_link_to_rmg_coincident_nodes_mapper['coincident_node'][mask1].iloc[-1] # channel node assigned to inlet
            # now get the smallest contributing area of the inlet rmg channel node 
            # (rmg channel node mapped to the link inlet coincident node)
            mask2 = rmg_nodes_to_nmg_links_mapper['coincident_node'] == min_area_node 
            min_area = rmg_nodes_to_nmg_links_mapper['node_drainage_area'][mask2].min() # drainage area
            # finally, find any nodes assigned to link that have drainage area 
            # less than the inlet and remove
            mask3 = (rmg_nodes_to_nmg_links_mapper['linkID'] == link) & (rmg_nodes_to_nmg_links_mapper['node_drainage_area']<min_area)
            rmg_nodes_to_nmg_links_mapper = rmg_nodes_to_nmg_links_mapper.drop(rmg_nodes_to_nmg_links_mapper.index[mask3].values)  
    
    return rmg_nodes_to_nmg_links_mapper 


def map_nmg1_links_to_nmg2_links(grid, LxyDF_nmg1, LxyDF_nmg2):    
    
    """given two slighly different network model grids of the same channel network,
    map links from one network model grid (nmg1) to the closest links of the other
    network model grid (nmg2). 
    """

    def distance_between_links(row, XY):
        return _dist_func(row['x'],XY[0],row['y'],XY[1])# ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5

    link_mapper ={}
    link_map_L = {}
    link_offset_L = {}
    for i in np.unique(LxyDF_nmg1['linkID']):# for each nmg1 link i
        sublist = LxyDF_nmg1['coincident_node'].values[LxyDF_nmg1['linkID'] == i] # get all rmg nodes coincident with the nmg1 link i
        LinkL = [] # id of nmg2 link that is closest to nmg1 rmg node
        offsetL = [] # distance of nmg2 link that is closest to nmg1 rmg node 
        DAL = [] # drainage area of nmg2 link that is closest to nmg1 rmg node 
        
        for j in sublist: # for each nmg1 rmg node find the closest nmg2 rmg node
            XY = [grid.node_x[j], grid.node_y[j]] # get x and y coordinate of the rmg node
            nmg_d_dist = LxyDF_nmg2.apply(lambda row: distance_between_links(row, XY),axis=1)# compute the distance from the rmg node to all nmg2 rmg nodes
            offset = nmg_d_dist.min() # find the minimum distance between the nmg1 rmg node and nmg2 rmg nodes
            mdn = LxyDF_nmg2['linkID'].values[(nmg_d_dist == offset).values][0]# get the nmg2 link id with minimum distance node, if more than one, pick the first one
            DA = LxyDF_nmg2['drainage_area'].values[(nmg_d_dist == offset).values][0] # drainage area
            offsetL.append(offset)
            LinkL.append(mdn) 
            DAL.append(DA)
            
        offsets = np.array(offsetL)
        Links = np.array(LinkL)
        DAs = np.array(DAL)
        
        count =  np.bincount(Links) # number of times nmg2 rmg nodes were closest to nmg1 link   

        # nmg2 link with highest count is matched to nmg1 link 
        if (count == count.max()).sum() == 1: 
            Link = np.argmax(count) 
        else: # if two or more nmg2 links have the same count, select the closer one that drains the largest area
             # get the link(s) that have the closest rmg node. 
            Links_ = Links[offsets == offsets.min()]
            if len(Links_)>1: 
                DAs_  = DAs[offsets == offsets.min()] # get subset of drainage areas
                DA_max = DAs_.max()
                Link = Links_[DAs_ == DA_max][0] # if drainage area is same, use the first one
            else:
                Link = Links_[0]
        link_mapper[i] = Link
        link_offset_L[i] = offsets
        link_map_L[i] = Links # final list of all nmg2 link ids matched to a nmg1 link i rmg nodes
        print('nmg1 link '+ str(i)+' mapped to equivalent nmg2 link')
    return (link_mapper, link_map_L, link_offset_L)


def map_rmg_channel_nodes_to_nmg_nodes(grid, nmgrid, acn):
    """ find rmg channel node that is closest to each node at the head and 
    tail of each link in the nmg.
    """
    #compute distance between deposit and all network cells
    def distance_between_links(row, XY):
        return _dist_func(row['x'],XY[0],row['y'],XY[1])       
    # for each node in the channel node list record the equivalent nmg_d link id 
    xyDF = pd.DataFrame(np.array([grid.node_x[acn],grid.node_y[acn]]).T, columns = ['x','y'])
    nmg_node_to_cn_mapper ={}
    for i, node in enumerate(nmgrid.nodes):# for each node network modelg grid
        XY = [nmgrid.node_x[i], nmgrid.node_y[i]] # get x and y coordinate of node
        nmg_node_dist = xyDF.apply(lambda row: distance_between_links(row, XY),axis=1)#.apply(Distance,axis=1) # compute the distance to all channel nodes
        offset = nmg_node_dist.min() # find the minimum distance between node and channel nodes
        mdn = xyDF.index[(nmg_node_dist == offset).values][0]# index of minimum distance node, use first value
        nmg_node_to_cn_mapper[i] = acn[mdn] # rmg node mapped to nmg node i         
    return nmg_node_to_cn_mapper


def transfer_nmg2_link_field_to_nmg1_link_field(nmgrid_2, nmgrid_1, link_mapper, link_field, default_value = np.nan):
    # add field to the nmgrid_2 links if not already present
    if link_field not in nmgrid_2.at_link.keys(): # field not
        nmgrid_2.at_link[link_field] = np.ones(nmgrid_2.number_of_links)*default_value
    
    for i, link in enumerate(nmgrid_2.active_links):
        nmgrid_1_link = link_mapper[link]
        value = nmgrid_1.at_link[link_field][nmgrid_1_link]
        nmgrid_2.at_link[link_field][link] = value 


def transfer_rmg_channel_node_field_to_nmg_node_field(grid, nmgrid, NMGtoRMGnodeMapper, field = 'topographic__elevation'):
    """update the field value of the nmg nodes using the field value at the
    equivalent raster model grid nodes
    """
    # update field value
    for i, node in enumerate(nmgrid.nodes):
        RMG_node = NMGtoRMGnodeMapper[i]
        nmgrid.at_node[field][i] = grid.at_node[field][RMG_node]
        

def transfer_nmg_link_field_to_rmg_channel_node_field(grid, nmgrid, nmg_field, rmg_field, cn_to_nmg_link_mapper, default_value = np.nan):
    """updates the field value of the rmg nodes using the values of each link mapped
    to the nodes

    Returns
    -------
    None.
    """
    # add field to the rmg if not already present
    if rmg_field not in grid.at_node.keys(): # field not
        grid.at_node[rmg_field] = np.ones(grid.number_of_nodes)*default_value
    
    for i, link in enumerate(nmgrid.active_links):
        value = nmgrid.at_link[nmg_field][link]
        link_nodes =  cn_to_nmg_link_mapper['node'][cn_to_nmg_link_mapper['linkID'] == link].values
        grid.at_node[rmg_field][link_nodes] = value
        
        
def transfer_rmg_channel_node_field_to_nmg_link_field(grid, nmgrid, rmg_field, nmg_field, cn_to_nmg_link_mapper, metric = 'mean', default_value = np.nan):
    """updates the field value of the nmg links using the mean, max, minimum or 
    median value of the rmg nodes mapped to each link"""
    # add field to the nmg if not already present
    if nmg_field not in nmgrid.at_link.keys(): # field not
        nmgrid.at_link[nmg_field] = np.ones(nmgrid.number_of_links)*default_value
    
    for i, link in enumerate(nmgrid.active_links):
        link_nodes =  cn_to_nmg_link_mapper['node'][cn_to_nmg_link_mapper['linkID'] == link].values
        if len(link_nodes)>0: # if link_nodes is not empty
            if metric == 'mean':
                value = grid.at_node[rmg_field][link_nodes].mean()
            elif metric == 'max':
                value = grid.at_node[rmg_field][link_nodes].max()
            elif metric == 'min':
                value = grid.at_node[rmg_field][link_nodes].min()
            elif metric == 'median':
                value = np.median(grid.at_node[rmg_field][link_nodes])
            else:
                raise ValueError('metric "{}" not an option'.format(metric))
            nmgrid.at_link[nmg_field][link] = value

def update_rmg_channel_location_and_mapping(grid, nmgrid, Ct, BFt, terrace_width, link_nodes, nmg_link_to_rmg_coincident_nodes_mapper):
    """if the DEM changes, this function remaps channel and terrace nodes and updates 
    the mappers"""
    
    acn = extract_channel_nodes(grid,Ct)
    fcn = extract_channel_nodes(grid,BFt)
    tn = extract_terrace_nodes(grid, terrace_width, acn, fcn)
    cn_to_nmg_link_mapper = map_rmg_nodes_to_nmg_links(grid, nmgrid, nmg_link_to_rmg_coincident_nodes_mapper, acn)
    tn_to_nmg_link_mapper = map_rmg_nodes_to_nmg_links(grid, nmgrid, nmg_link_to_rmg_coincident_nodes_mapper, tn)
    nmg_node_to_cn_mapper = map_rmg_channel_nodes_to_nmg_nodes(grid, nmgrid, acn)
    
    return {'acn':acn,
            'fcn':fcn,
            'tn':tn,
            'nmg_link_to_rmg_coincident_nodes_mapper':nmg_link_to_rmg_coincident_nodes_mapper,
            'cn_to_nmg_link_mapper':cn_to_nmg_link_mapper,
            'tn_to_nmg_link_mapper':tn_to_nmg_link_mapper,            
            'nmg_node_to_cn':nmg_node_to_cn_mapper}
    



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
    terrace_width: int
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
            terrace_width = 1,
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
        self.terrace_width = terrace_width # distance from channel grid cells that are considered terrace grid cells [# cells]          

    
    def extract_channel_nodes(self,Ct,BCt):
        """interpret which nodes of the DEM correspond to the fluvial channel 
        network and entire channel network (including colluvial channels)
        TO FUNCTION => DONE
        """
        
        # entire channel network (all channels below channel head or upper extent 
        # of the colluvial channel network)
        ChannelNodeMask = self._grid.at_node['drainage_area'] > Ct
        df_x = self.gridx[ChannelNodeMask]
        df_y = self.gridy[ChannelNodeMask]
        self.xyDF_df = pd.DataFrame({'x':df_x, 'y':df_y})
        self.ChannelNodes = self.rnodes[ChannelNodeMask] 
        
        # fluvial channel network only (all channels below upper extent of cascade channels)
        BedloadChannelNodeMask = self._grid.at_node['drainage_area'] > BCt
        bc_x = self.gridx[BedloadChannelNodeMask]
        bc_y = self.gridy[BedloadChannelNodeMask]
        self.xyDF_bc = pd.DataFrame({'x':bc_x, 'y':bc_y})
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
            terrace_width: int
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
        TO FUNCTION => DONE
        """
        for i in range(self.terrace_width):
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
        self.xyDF_t = pd.DataFrame({'x':t_x, 'y':t_y})
        self.TerraceNodes = TerraceNodes
    


    def min_distance_to_network(self, cellid, ChType = 'debrisflow'):
        """determine the distance from a node to the nearest channel node
            ChType = debrisflow: uses debris flow network
            ChType = nmg: uses network model grid network
        """

        def distance_to_network(grid, row):
            """compute distance between nodes """
            return _dist_func(row['x'],grid.node_x[cellid],row['y'],grid.node_y[cellid])        

        # dist = xyDF.apply(lambda row: distance_to_network(row, grid),axis=1)         
        if ChType == 'debrisflow':
            nmg_dist = self.xyDF_df.apply(lambda row: distance_to_network(self._grid, row),axis=1) #.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDF_df[nmg_dist == offset] # minimum distance node and node x y        
        elif ChType == 'nmg':
            nmg_dist = self.xyDF_bc.apply(lambda row: distance_to_network(self._grid, row),axis=1) #.apply(distance_to_network,axis=1)
            offset = nmg_dist.min() # minimum distancce
            mdn = self.xyDF_bc[nmg_dist == offset] # minimum distance node and node x y    
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
                node = self._grid.find_nearest_node((x,y))
                if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                    nodelist.append(node)  
                    distlist.append(dist[i])
                    linkIDlist.append(linkID)
                    xlist.append(self.gridx[node])
                    ylist.append(self.gridy[node])
                    xy = {'linkID':linkID,
                          'node':node,
                          'x':self.gridx[node],
                          'y':self.gridy[node], # add dist
                          'dist':dist[i],
                          'drainage_area':self._nmgrid.at_link['drainage_area'][linkID]}
                    Lxy.append(xy)
                
            
            
            Lnodelist.append(nodelist)
            Ldistlist.append(distlist)
            LlinkIDlist.append(linkIDlist)
            Lxlist.append(xlist)
            Lylist.append(ylist)
        LxyDF = pd.DataFrame(Lxy)
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
                        # if contributing area of link largest, remove dupilcate nodes from other link
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
            LinkIDs = _flatten_lol(LlinkIDlist)
            LxyDF = pd.DataFrame(np.array([LinkIDs,
                                _flatten_lol(Lnodelist),
                                _flatten_lol(Lxlist),
                                _flatten_lol(Lylist),
                                _flatten_lol(Ldistlist),
                               self._nmgrid.at_link['drainage_area'][np.array(LinkIDs)]]).T,
                               columns = ['linkID','node','x','y','dist','drainage_area'])
            LxyDF['linkID'] = LxyDF['linkID'].astype(int)
            LxyDF['node'] = LxyDF['node'].astype(int)
        return LxyDF
    
    def map_rmg_channel_nodes_to_nmg_rmg_nodes(self, xyDF):
        """for each link i in the nmg create a list of the rmg channel nodes that
        represent the link"""
        # for each channel node, get the distance to all link rmg nodes
        # channel node x
        cnode_x = self._grid.node_x[self.ChannelNodes] # cn, not fcn
        # channel node y
        cnode_y = self._grid.node_y[self.ChannelNodes]

        def dist_between_nmg_and_rmg_nodes(row, xc, yc):
            """distance between channel node and link node"""
            return _dist_func(xc,row['x'],yc,row['y'])
        
        cn_link_id = []
        for i, cn in enumerate(self.ChannelNodes): # for each channel node
            xc = cnode_x[i]; yc = cnode_y[i]
            # compute the distance to all link rmg nodes
            dist = xyDF.apply(lambda row: dist_between_nmg_and_rmg_nodes(row, xc, yc),axis=1) 
            # link nmg_rmg node with the shortest distance to the channel node is 
            # assigned to the channel node.
            # if more than one (which can happen because the end and begin of the reaches at a junction overlay the same cell) 
            # pick link with largest contributing area
            dist_min_links = xyDF['linkID'][dist == dist.min()].values
            dmn_cont_area = self._nmgrid.at_link['drainage_area'][dist_min_links]
            dmn_mask = dmn_cont_area == dmn_cont_area.max()
            
            cn_link_id.append(dist_min_links[dmn_mask][0]) # more than one cell from the same link may be the same distance from the node, just pick one
        
        # for each link, group all channel nodes assigned to link into list, append to Lcnodelist
        b = np.array(cn_link_id)
        Lcnodelist = []
        for link in range(self._nmgrid.number_of_links): #np.unique(b):
            Lcnodelist.append(list(self.ChannelNodes[b == link]))
        return Lcnodelist # for each link i, the list of channel nodes assigned to the link

    def map_nmg1_links_to_nmg2_links(self,xyDF_nmg1,xyDF_nmg2):    
        
        """map link ids from one network model grid (nmg1) to equivalent link ids on
        another network model grid (nmg2).
        link to each link in the landlab nmg. Note, the landlab nmg id of the dhsvm
        network is used, not the DHSVM network id (arcID) To translate the nmg_d link
        id to the DHSVM network id: self.nmg_d.at_link['arcid'][i], where i is the nmg_d link id.
        """

        def distance_between_links(row, XY):
            return _dist_func(row['x'],XY[0],row['y'],XY[1])# ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5

        LinkMapper ={}
        LinkMapL = {}
        LinkOffsetL = {}
        for i in np.unique(xyDF_nmg1['linkID']):# for each nmg1 link i
            sublist = xyDF_nmg1['node'].values[xyDF_nmg1['linkID'] == i] # get all rmg nodes coincident with the nmg1 link i
            LinkL = [] # id of nmg2 link that is closest to nmg1 rmg node
            offsetL = [] # distance of nmg2 link that is closest to nmg1 rmg node 
            DAL = [] # drainage area of nmg2 link that is closest to nmg1 rmg node 
            
            for j in sublist: # for each nmg1 rmg node find the closest nmg2 rmg node
                XY = [self.gridx[j], self.gridy[j]] # get x and y coordinate of the rmg node
                nmg_d_dist = xyDF_nmg2.apply(lambda row: distance_between_links(row, XY),axis=1)# compute the distance from the rmg node to all nmg2 rmg nodes
                offset = nmg_d_dist.min() # find the minimum distance between the nmg1 rmg node and nmg2 rmg nodes
                mdn = xyDF_nmg2['linkID'].values[(nmg_d_dist == offset).values][0]# get the nmg2 link id with minimum distance node, if more than one, pick the first one
                DA = xyDF_nmg2['drainage_area'].values[(nmg_d_dist == offset).values][0] # drainage area
                offsetL.append(offset)
                LinkL.append(mdn) 
                DAL.append(DA)
                
            offsets = np.array(offsetL)
            Links = np.array(LinkL)
            DAs = np.array(DAL)
            
            count =  np.bincount(Links) # number of times nmg2 rmg nodes were closest to nmg1 link   

            # nmg2 link with highest count is matched to nmg1 link 
            if (count == count.max()).sum() == 1: 
                Link = np.argmax(count) 
            else: # if two or more nmg2 links have the same count, select the closer one that drains the largest area
                 # get the link(s) that have the closest rmg node. 
                Links_ = Links[offsets == offsets.min()]
                if len(Links_)>1: 
                    DAs_  = DAs[offsets == offsets.min()] # get subset of drainage areas
                    DA_max = DAs_.max()
                    Link = Links_[DAs_ == DA_max][0] # if drainage area is same, use the first one
                else:
                    Link = Links_[0]
            LinkMapper[i] = Link
            LinkOffsetL[i] = offsets
            LinkMapL[i] = Links # final list of all nmg2 link ids matched to a nmg1 link i rmg nodes
            print('nmg1 link '+ str(i)+' mapped to equivalent nmg2 link')
        return (LinkMapper, LinkMapL, LinkOffsetL)



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
        def distance_between_links(row, XY):
            return _dist_func(row['x'],XY[0],row['y'],XY[1])       
        # for each node in the channel node list record the equivalent nmg_d link id 
        NodeMapper ={}
        for i, node in enumerate(self.nmg_nodes):# for each node network modelg grid
            XY = [self._nmgrid.node_x[i], self._nmgrid.node_y[i]] # get x and y coordinate of node
            nmg_node_dist = self.xyDF_df.apply(lambda row: distance_between_links(row, XY),axis=1)#.apply(Distance,axis=1) # compute the distance to all channel nodes
            offset = nmg_node_dist.min() # find the minimum distance between node and channel nodes
            mdn = self.xyDF_df.index[(nmg_node_dist == offset).values][0]# index of minimum distance node, use first value
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
            
    
    def transfer_nmg_link_field_to_rmg_channel_node_field(self, nmg_field, rmg_field, Lcnodelist, default_value = np.nan):
        """
        transfers the field value on each link to each rmg channel node assigned
        to the link in map_rmg_channel_nodes_to_nmg_rmg_nodes()

        Returns
        -------
        None.
        """
        # add field to the rmg if not already present
        if rmg_field not in self._grid.at_node.keys(): # field not
            self._grid.at_node[rmg_field] = np.ones(self._grid.number_of_nodes)*default_value
        
        for i, link in enumerate(self._nmgrid.active_links):
            value = self._nmgrid.at_link[nmg_field][link]
            link_nodes =  Lcnodelist[i]
            self._grid.at_node[rmg_field][link_nodes] = value
            
            
    def transfer_rmg_channel_node_field_to_nmg_link_field(self, rmg_field, nmg_field, Lcnodelist, metric = 'mean', default_value = np.nan):
        """
        transfer the max, min, mean or median field value of the rmg 
        channel nodes that represent each nmg link to the nmg link field.
        """
        # add field to the nmg if not already present
        if nmg_field not in self._nmgrid.at_link.keys(): # field not
            self._nmgrid.at_link[nmg_field] = np.ones(self._nmgrid.number_of_links)*default_value
        
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
            