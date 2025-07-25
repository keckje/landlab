
from landlab import Component, FieldError
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy as sc


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
    """get the downstream (outlet) and upstream (inlet) nodes at a link from 
    flow director. Linknodes in network model grid (nodes_at_link) may not be 
    correct. Output from this function should be used as input for all 
    ChannelNetworkGridTool functions that require a linknodes
    
    Parameters
    ----------
    nmgrid : network model grid
    
    Returns
    -------
    linknodes : np.array
        for a nmgrid of n links, returns a nx2 np array, the ith row of the
        array is the [downstream node id, upstream node id] of the ith link
    """
    
    fd = FlowDirectorSteepest(nmgrid, "topographic__elevation")
    fd.run_one_step()
    upstream_node_id = []; downstream_node_id = []
    for i in range(nmgrid.number_of_links):
        upstream_node_id.append(fd.upstream_node_at_link()[i])
        downstream_node_id.append(fd.downstream_node_at_link()[i])
    # create np array and transpose, each row is [downstream node id, upstream node id]
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
    
    
def _dist_func(x1,x2,y1,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5 


def extract_channel_nodes(grid,Ct):
    """
    interpret which nodes of the DEM represent the channel network 
    (including colluvial channels)

    Parameters
    ----------
    grid : Landlab raster model grid
        raster model grid used to represent the landscape, must have a 
        "drainage_area" field, units in m^2
    Ct : float
        Ct is the average contributing area, in m^2, at which channels begin
        on the landscape.

    Returns
    -------
    ChannelNodes : np.array
        id numbers of all channel nodes. 
    xyDf_C : pd.DataFrame
        x and y coordinates of each channel node

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
    """
    interpret which nodes of the DEM represent the fluvial channel 
    network

    Parameters
    ----------
    grid : Landlab raster model grid
        raster model grid used to represent the landscape, must have a 
        "drainage_area" field, units in m^2
    FCt : float
        FCt is the average contributing area, in m^2, at which fluvial channels 
        (cascade and lower-gradient channels) begin on the landscape.

    Returns
    -------
    FluvialChannelNodes : np.array
        id numbers of all fluvial channel nodes.
    xyDf_FC : pd.DataFrame
        x and y coordinates of each fluvial channel node

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


def extract_terrace_nodes(grid,ChannelNodes,FluvialChannelNodes,TerraceWidth = 1):
    """
    determine which raster model grid nodes coincide with channel terraces,
    which presently are asssumed to be a fixed width (number of nodes) from
    the fluvial channel nodes

    Parameters
    ----------
    grid : Landlab raster model grid
        raster model grid used to represent the landscape
    ChannelNodes : np.array
        id numbers of all channel nodes. Output of extract_channel_nodes
    FluvialChannelNodes : np.array
        id numbers of all fluvial channel nodes. Output of extract_fluvial_channel_nodes
    TerraceWidth : int
        width of terrace around fluvial channel nodes, in number of nodes. Default
        is 1.

    Returns
    -------
    TerraceNodes : np.array
        id numbers of all terrace nodes.

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
    """determine the distance between to points, as represented by their x and
    y coordinates using Pythagorean theorem"""
    return (((x2-x1)**2)+((y2-y1)**2))**0.5


def min_distance_to_network(grid, nodeid, xyDf):
    """
    determine the distance from a node to the nearest channel node (which include
    the colluvial reaches)

    Parameters
    ----------
    grid : Landlab raster model grid
        raster model grid used to represent the landscape
    nodeid : int
        id of the node
    xyDf : pd.DataFrame
        x and y coordinates of each channel node

    Returns
    -------
    dist_to_channel_nodes.min() : float
        minimum distance to the channel nodes from the node

    """
    dist_to_channel_nodes = xyDf.apply(lambda row: dist_between_points(grid.node_x[nodeid],
                                                                       grid.node_y[nodeid], 
                                                                       row['x'],
                                                                       row['y']), axis=1)
    return dist_to_channel_nodes.min() 


def _remove_duplicates(nmgrid, linknodes, Lnodelist, Ldistlist, LlinkIDlist, Lxlist, Lylist):
    """
    if two links overlay the same node, the node is assigned to the link with 
    the that has the largest contributing area

    Parameters
    ----------
    nmgrid : Landlab network model grid
        network model grid used to represent the landscape
    linknodes : np.array
        for a nmgrid of n links, returns a nx2 np array, the ith row of the
        array is the [downstream node id, upstream node id] of the ith link
    Lnodelist : list of lists
        all nodes that coincide with each link for each link
    Ldistlist : list of lists
        distance from the upstream end of the link to a node for all nodes that
        coincide with each link for each link
    LlinkIDlist : TYPE
        DESCRIPTION.
    Lxlist : TYPE
        DESCRIPTION.
    Lylist : TYPE
        DESCRIPTION.

    Returns
    -------
    LlinkIDlist : TYPE
        DESCRIPTION.
    Lnodelist : TYPE
        DESCRIPTION.
    Lxlist : TYPE
        DESCRIPTION.
    Lylist : TYPE
        DESCRIPTION.
    Ldistlist : TYPE
        DESCRIPTION.
    Lxy : TYPE
        DESCRIPTION.

    """
     for link in range(len(linknodes)):
         for other_link in range(len(linknodes)):
             if link != other_link:
                 link_nodes = Lnodelist[link]
                 other_link_nodes = Lnodelist[other_link]
                 link_a = nmgrid.at_link['drainage_area'][link]
                 other_link_a = nmgrid.at_link['drainage_area'][other_link]
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

     Lxy = pd.DataFrame(np.array([_flatten_lol(LlinkIDlist),
                         _flatten_lol(Lnodelist),
                         _flatten_lol(Lxlist),
                         _flatten_lol(Lylist),
                         _flatten_lol(Ldistlist)]).T,
                        columns = ['linkID','node','x','y','dist'])
     Lxy['linkID'] = Lxy['linkID'].astype(int)
     Lxy['node'] = Lxy['node'].astype(int)
     
     return LlinkIDlist, Lnodelist, Lxlist, Lylist, Ldistlist, Lxy
  

def map_nmg_links_to_rmg_nodes(nmgrid, grid, linknodes, nmgx, nmgy, remove_duplicates = False):
    """
    map the network model grid to the coincident raster model grid nodes (i.e.,
    create the nmg rmg nodes). The nmg rmg nodes are defined based on the 
    the rmg node id and the coincident nmg link and distance downstream on that
    link. In Eroder, erosion at
    a channel or terrace is becomes a pulse of sediment and is inserted into
    the network model grid at the link # and downstream location of the closest
    rmg nmg node.(see _parcelDFmaker function in eroder). 
    The pulser utlity uses the dataframe to transfer the depostion 
    to the nmg at the corrisponding link # and downstream distance.
    
    this code assumes each row of linknodes is [downstream node, upstream node]

    Parameters
    ----------
    nmgrid : Landlab network model grid
        network model grid used to represent the landscape
    grid : Landlab raster model grid
        raster model grid used to represent the landscape
    linknodes : TYPE
        DESCRIPTION.
    nmgx : TYPE
        DESCRIPTION.
    nmgy : TYPE
        DESCRIPTION.
    remove_duplicates : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    LlinkIDlist : TYPE
        DESCRIPTION.
    Lnodelist : TYPE
        DESCRIPTION.
    Lxlist : TYPE
        DESCRIPTION.
    Lylist : TYPE
        DESCRIPTION.
    Ldistlist : TYPE
        DESCRIPTION.
    Lxy : TYPE
        DESCRIPTION.

    """
    Lnodelist = [] #
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
            node = _node_at_coords(grid,x,y)
            if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                nodelist.append(node[0])  
                distlist.append(dist[i])
                linkIDlist.append(linkID)
                xlist.append(grid.node_x[node[0]])
                ylist.append(grid.node_y[node[0]])
                xy = {'linkID':linkID,
                      'node':node[0],
                      'x':grid.node_x[node[0]],
                      'y':grid.node_y[node[0]], # add dist
                      'dist':dist[i]}
                Lxy.append(xy)
            
        
        
        Lnodelist.append(nodelist)
        Ldistlist.append(distlist)
        LlinkIDlist.append(linkIDlist)
        Lxlist.append(xlist)
        Lylist.append(ylist)
    Lxy = pd.DataFrame(Lxy)
    
    # remove duplicates by assiging duplicate node to link with largest mean contributing area.
    if remove_duplicates:
       
        LlinkIDlist, Lnodelist, Lxlist, Lylist, Ldistlist, Lxy = _remove_duplicates(nmgrid, 
                                                                                    linknodes, 
                                                                                    Lnodelist, 
                                                                                    Ldistlist, 
                                                                                    LlinkIDlist, 
                                                                                    Lxlist, 
                                                                                    Lylist)
        
    return (LlinkIDlist, Lnodelist, Lxlist, Lylist, Ldistlist, Lxy)

