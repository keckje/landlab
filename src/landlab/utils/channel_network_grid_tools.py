# TO DO
# 2 add test with remove tribs to map_rmg_nodes_to_nmg_links, pull request 2
# 3 tests for map_nmg1_links_to_nmg2_links, pull request 3
# 0 add upside down channel network to confest
# 1 cleanup pull request 1, add tests to coincident node function
# # WHEN pull request for CNGT is done, update MWRo, eroder, mapper, DHG


import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from landlab.components.flow_director.flow_director_steepest import FlowDirectorSteepest

"""
A collection of tools for mapping values (e.g., flow, shear stress) between
network model grids and a raster model grid representation of a channel network.
"""


def _create_lol(linkID, list_of_vals):
    """convert a list of values to a list of lists of values, where each sublist
    contains the values for each unique link id in list linkID"""
    list_of_lists = []
    for link in np.unique(linkID):
        mask = np.array(linkID) == link
        list_of_lists.append(np.array(list_of_vals)[mask].tolist())
    return list_of_lists


def _flatten_lol(lol):
    """Flatten a list of lists.

    Parameters
    ----------
    lol : list of list
        A list where each element is itself a list.

    Returns
    -------
    list
        A single list containing all elements from the sublists.
    """
    return list(itertools.chain.from_iterable(lol))


def get_link_nodes(nmgrid):
    """Get the downstream (head) and upstream (tail) nodes at a link from
    flow director. The network model grid nodes_at_link attribute may not be
    ordered according to flow direction. Output from this function should be
    used for all channel_network_grid_tools functions that require a link_nodes
    input

    Parameters
    ----------
    nmgrid : network model grid

    Returns
    -------
    link_nodes : np.array
        for a nmgrid of n links, returns a nx2 np array, the ith row of the
        array is the [downstream node id, upstream node id] of the ith link
    """

    fd = FlowDirectorSteepest(nmgrid, "topographic__elevation")
    fd.run_one_step()

    return np.column_stack(
        (fd.downstream_node_at_link(), fd.upstream_node_at_link())
    ).astype(int, copy=False)


def _link_to_points_and_dist(
    point_0: tuple[float, float],
    point_1: tuple[float, float],
    number_of_points: int = 1000,
):
    """Given two points defined by coordinates x0,y0 and x1,y1, define a series
    of points between them and the distance from point x0,y0 to each point.

    Parameters
    ----------
    point_0 : tuple of 2 floats
        point 0 coordinates x and y
    point_1 : tuple of 2 floats
        point 1 coordinates x and y
    number_of_points : int
        number of points to create along the reach. The default is 1000.

    Returns
    -------
    X : np array
        x coordinate of points
    Y : np array
        y coordinate of points
    dist : np array
        linear distance between points

    """
    x0 = point_0[0]
    y0 = point_0[1]
    x1 = point_1[0]
    y1 = point_1[1]
    X = np.linspace(x0, x1, number_of_points)
    Y = np.linspace(y0, y1, number_of_points)
    dist = np.hypot(X - x0, Y - y0)

    return X, Y, dist


def _dist_func(x0, x1, y0, y1):
    return np.hypot(x0 - x1, y0 - y1)


def extract_channel_nodes(grid, Ct):
    """interpret which nodes of the DEM represent the channel network as all nodes
    that have a drainage area >= to the average drainage area at which
    channels initiate in the DEM (Ct, based on field or remote sensing evidence).

    Use Ct = average drainage area at which colluvial channels to get the entire
    channel network.

    Use Ct = the drainage area at which cascade channels typically begin to get
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
    return np.flatnonzero(grid.at_node["drainage_area"] >= Ct)


def extract_terrace_nodes(grid, terrace_width, acn, fcn):
    """Determine which raster model grid nodes coincide with channel terraces,
    which presently are assumed to be a fixed width (number of nodes) from
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
    terrace_nodes : np array
        array of all node IDs included in the terrace

    """
    # round to int in case provided as float
    terrace_width = round(terrace_width)
    if terrace_width < 1:
        raise ValueError(f"terrace width must be 1 or greater ({terrace_width})")

    acn = np.asarray(acn, dtype=int)
    current_nodes = np.asarray(fcn, dtype=int)
    terrace_nodes = np.array([], dtype=int)

    for _ in range(terrace_width):
        adj_dn = grid.diagonal_adjacent_nodes_at_node[current_nodes].ravel()
        adj_n = grid.adjacent_nodes_at_node[current_nodes].ravel()

        neighbors = np.unique(np.concatenate((adj_n, adj_dn)))
        neighbors = neighbors[neighbors != -1]

        terrace_nodes = np.setdiff1d(neighbors, acn, assume_unique=True)

        current_nodes = terrace_nodes

    return terrace_nodes


def min_distance_to_network(grid, acn, node_id):
    """Determine the shortest distance (as the crow flies) from a node to the
    channel network and the closest channel node

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
    x0, y0 = grid.node_x[node_id], grid.node_y[node_id]
    x_acn, y_acn = grid.node_x[acn], grid.node_y[acn]

    dist = np.hypot(x_acn - x0, y_acn - y0)

    idx = np.argmin(dist)
    offset = dist[idx]
    mdn = acn[idx]

    return float(offset), int(mdn)


def choose_from_repeated(
    sorted_array: ArrayLike,
    choose: Literal["first", "last"] = "last",
) -> NDArray[np.bool_]:
    """Mark the first/last element of repeated values in a **sorted** 1-D array.

    Parameters
    ----------
    sorted_array : array_like
        Assumed sorted by the grouping key.
    choose : {'first','last'}, optional
        Whether to mark the first or last item of each run.

    Examples
    --------
    >>> array = [0, 0, 0, 2, 2, 5, 6, 6, 6, 6, 6]
    >>> is_last = choose_from_repeated(array, choose="last")
    >>> is_last.astype(int)
    array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1])
    """
    a = np.asarray(sorted_array).ravel()

    same_as_previous = np.zeros(a.size, dtype=bool)

    if a.size <= 1:
        return np.ones(a.size, dtype=bool)

    same_as_previous[1:] = a[1:] == a[:-1]
    if choose == "last":
        keep_mask = np.ones_like(same_as_previous)
        keep_mask[:-1] = ~same_as_previous[1:]
    elif choose == "first":
        keep_mask = ~same_as_previous
    else:
        raise ValueError(f"choose must be 'first' or 'last', got {choose!r}")

    return keep_mask


def choose_unique(
    values: ArrayLike,
    order_by: Sequence[ArrayLike] | None = None,
    choose: Literal["first", "last"] = "last",
) -> NDArray[np.intp]:
    """Find indices of unique values, selecting one representative if repeated.

    Examples
    --------
    >>> choose_unique([0, 1, 0, 0, 1], order_by=([10.0, 11.0, 12.0, 13.0, 14],))
    array([3, 4])

    >>> choose_unique([1, 0, 0, 1, 0], order_by=([10.0, 11.0, 12.0, 13.0, 14],))
    array([3, 4])
    """
    values = np.asarray(values).ravel()

    order_by = (
        () if order_by is None else tuple(np.asarray(key).ravel() for key in order_by)
    )

    if any(key.size != values.size for key in order_by):
        raise ValueError("All `order_by` arrays must match `values` length")

    sorted_rows = np.lexsort(order_by + (values,))
    
    # sorted_rows = np.lexsort([order_by[0], values])

    is_last = choose_from_repeated(values[sorted_rows], choose=choose)

    return np.sort(sorted_rows[is_last])


def map_nmg_links_to_rmg_coincident_nodes(
    grid, nmgrid, link_nodes, remove_duplicates=False
):
    """Map links of a network model grid to all coincident raster model grid
    nodes. Each coincident raster model grid node is defined in terms of its
    x and y coordinates, the link it is mapped to and distance downstream from
    the upstream (tail) end of the link.


    Parameters
    ----------
    grid : raster model grid
    nmgrid : network model grid
    link_nodes : np array
        For a nmgrid of n links, link_nodes is a nx2 np array, the ith row of the
        array is the [downstream node id, upstream node id] of the ith link
    remove_duplicates : bool
        if True, when two or more links are coincident with the same node,
        the node is assigned to the link with the largest drainage area. If False,
        the node is assigned to each coincident link. The default is False.

    Returns
    -------

    nmg_link_to_rmg_coincident_nodes_mapper: pandas dataframe
        each row of the dataframe lists the link ID, the coincident node ID, the
        x and y coordinates and the downstream distance of the coincident node
        and the drainage area of the link

    """
    Lxy = []  # list of all nodes and node attributes that coincide with the
    # network model grid links
    # loop through all links in network model grid to determine raster grid cells
    # coincident with each link and equivalent distance from upstream (tail) node
    for linkID, lknd in enumerate(link_nodes):  # for each link in network grid

        x0 = nmgrid.x_of_node[lknd[0]]  # x and y of downstream link node
        y0 = nmgrid.y_of_node[lknd[0]]
        x1 = nmgrid.x_of_node[lknd[1]]  # x and y of upstream link node
        y1 = nmgrid.y_of_node[lknd[1]]

        # x and y coordinates and downstream distance from the upstream (tail)
        # node for 1000 points generated from downstream node to upstream node
        X, Y, dist = _link_to_points_and_dist((x0, y0), (x1, y1), number_of_points=1000)
        dist = dist.max() - dist  # convert to distance from tail node
        nodelist = []  # list of nodes along link
        for i, y in enumerate(Y):
            x = X[i]
            node = grid.find_nearest_node((x, y))
            # if node not already in list, append - many points will be in same cell;
            # only need to list cell once
            if node not in nodelist:
                nodelist.append(node)
                xy = {
                    "linkID": linkID,
                    "coincident_node": node,
                    "x": grid.node_x[node],
                    "y": grid.node_y[node],
                    "dist": dist[i],
                    "drainage_area": nmgrid.at_link["drainage_area"][linkID],
                }
                Lxy.append(xy)
    df = pd.DataFrame(Lxy)

    # if remove_duplicates, remove duplicate node id from link with smaller
    # contributing area.
    if remove_duplicates:
        values = df["coincident_node"].to_numpy()
        area = df["drainage_area"].to_numpy()

        row = np.arange(len(df), dtype=np.int64)

        idx = choose_unique(values=values, order_by=[area], choose="last") # order_by=[area, -row]
        idx.sort()

        df = df.iloc[idx].reset_index(drop=True)

    return df


####PULL REQUEST 2


def _remove_small_tribs(rmg_nodes_to_nmg_links_mapper, nmg_link_to_rmg_coincident_nodes_mapper):
    """remove channel rmg nodes mapped to link that represent first order channels
    not represented by the network model grid"""
    for link in np.unique(nmg_link_to_rmg_coincident_nodes_mapper['linkID'].values):
        # first get the coincident rmg node that represents the inlet to the link
        mask1 = nmg_link_to_rmg_coincident_nodes_mapper['linkID'] == link 
        # coincident nodes listed in nmg_link_to_rmg_coincident_nodes_mapper are 
        # ordered from outlet to inlet, so last node is coincident with inlet
        # min_area_node = nmg_link_to_rmg_coincident_nodes_mapper['coincident_node'][mask1].iloc[-1] 
        # node with shortest downstream distance from inlet is the inlet node
        min_dist = nmg_link_to_rmg_coincident_nodes_mapper['dist'][mask1].min()
        mask2 = nmg_link_to_rmg_coincident_nodes_mapper['dist'] == min_dist
        min_area_node = nmg_link_to_rmg_coincident_nodes_mapper['coincident_node'][mask1][mask2].iloc[0]
        # now get thecontributing area of the rmg channel node mapped to the link 
        # inlet coincident node. 
        mask3 = rmg_nodes_to_nmg_links_mapper['coincident_node'] == min_area_node 
        min_area = rmg_nodes_to_nmg_links_mapper['node_drainage_area'][mask3].min() # drainage area of channel inlet node
        # finally, find any nodes assigned to link that have drainage area 
        # less than the inlet node and remove
        mask4 = (rmg_nodes_to_nmg_links_mapper['linkID'] == link) & (rmg_nodes_to_nmg_links_mapper['node_drainage_area']<min_area)
        rmg_nodes_to_nmg_links_mapper = rmg_nodes_to_nmg_links_mapper.drop(rmg_nodes_to_nmg_links_mapper.index[mask4].values)     
    return rmg_nodes_to_nmg_links_mapper



def map_rmg_nodes_to_nmg_links(grid, nmg_link_to_rmg_coincident_nodes_mapper, rmg_nodes, remove_small_tribs = True):
    """Map the nodes representing the channel location in a DEM to the closest
    network model grid location. Network model grid location is described in 
    terms of link id and distance down link, measured from the inlet (tail) node
    of the link.
    
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
        are not matched to a Link. If False, the ndoes representin small, first 
        order channels will be mapped to the same link as the much larger channel 
        they drain into.

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
        link['node_drainage_area'] = grid.at_node['drainage_area'][n] # add node drainage area to attributes
        link_.append(link)
        
    rmg_nodes_to_nmg_links_mapper  = pd.concat(link_)
    rmg_nodes_to_nmg_links_mapper['node'] = rmg_nodes
    # organize column order in mapper
    rmg_nodes_to_nmg_links_mapper = rmg_nodes_to_nmg_links_mapper[['node','linkID','dist','coincident_node','drainage_area','node_drainage_area']].reset_index(drop=True) 

    if remove_small_tribs:# check for small tributary nodes assigned to link and remove them 
        rmg_nodes_to_nmg_links_mapper = _remove_small_tribs(rmg_nodes_to_nmg_links_mapper, 
                                                            nmg_link_to_rmg_coincident_nodes_mapper)
  
    return rmg_nodes_to_nmg_links_mapper 


def create_df_of_link_points(nmgrid, nodes_at_link, number_of_points):
    """convert the network model grid to a point representation, with each link
    of the grid represented by a series of number_of_points points. Each point
    is described by x and y coordinates and the link that the point represents 
    

    Parameters
    ----------
    nmgrid : network model grid
    nodes_at_link : np.array
        for a nmgrid of n links, a nx2 np array, the ith row of the
        array lists the two nodes defining the endpoints of the the ith link. 
        Order of the nodes may not be consistent (i.e., the may be listed as 
                                                  head, tail or tail, head) 
    number_of_points : int
        Each link is converted to a series of number_of_point points.

    Returns
    -------
    pandas dataframe that lists the link ID and x and y coordinates of each point

    """
    X_ = np.array([])
    Y_ = np.array([])
    link_ = np.array([])
    for linkID, lknd in enumerate(nodes_at_link):  # for each link in nmgrid 1 1

        x0 = nmgrid.x_of_node[lknd[0]]  # x and y of downstream link node
        y0 = nmgrid.y_of_node[lknd[0]]
        x1 = nmgrid.x_of_node[lknd[1]]  # x and y of upstream link node
        y1 = nmgrid.y_of_node[lknd[1]]

        # convert link to a series of points
        X, Y, dist = _link_to_points_and_dist((x0, y0), (x1, y1), number_of_points)
        
        X_ = np.concatenate((X_,X))
        Y_ = np.concatenate((Y_,Y))
        link_ = np.concatenate((link_,(np.ones(len(X))*linkID).astype(int)))
        
    return pd.DataFrame(data = zip(link_, X_ ,Y_), columns = ['linkID','X','Y'])
  
        

# def map_nmg1_links_to_nmg2_links(nmgrid_1, nmgrid_2, number_of_points = 11):
#     """given two slighly different network model grids of the same channel network,
#     map links from one network model grid (nmgrid_1) to the closest links of the
#     other network model grid (nmgrid_2). If two or more links of nmgrid_2 are equally 
#     close to a link of nmgrid_1, the link with the largest drainage area is mapped 
#     to the nmgrid_1 link. 
    

#     Parameters
#     ----------
#     nmgrid_1 : network model grid
#         grid that values will be mapped to
#     nmgrid_2 : network model grid
#         grid that values will be mapped from
#     number_of_points : int
#         Each link is converted to a series of number_of_point points. The relative
#         distance of each link to the other is determined using the points.
#         The default is 11. Below 11, output may not match expected.

#     Returns
#     -------
#     link_mapper : dict
#         Keys are the id of all links in nmgrid_1. Values are the link IDs of nmgrid_2
#         that are mapped to each nmgrid_1 link.
#     """    
    

#     def distance_between_links(row, XY):
#         return _dist_func(row['X'], XY[0], row['Y'], XY[1])# ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5
    
#     # # get the head and tail nodes of each link
#     # linknodes_1 = get_link_nodes(nmgrid_1)
#     # linknodes_2 = get_link_nodes(nmgrid_2)
    
#     # convert the network model grid to a point representation, as described by
#     # the link ID, x and y value of each point 
#     nmgrid_1_link_points = create_df_of_link_points(nmgrid_1, nmgrid_1.nodes_at_link, number_of_points)
#     nmgrid_2_link_points = create_df_of_link_points(nmgrid_2, nmgrid_2.nodes_at_link, number_of_points)
    
#     # for each point of each link of nmgrid_1, find the closest nmgrid_2 point
#     # and link. nmgrid_2 link with highest number of points mapped to nmgrid_1
#     # link is mapped to the nmgrid_1 link.
#     link_mapper ={}
#     for linkID in np.arange(nmgrid_1.number_of_links):#, lknd in enumerate(linknodes_1):  # for each link in nmgrid 1

#         sublist = nmgrid_1_link_points[['X','Y']][nmgrid_1_link_points['linkID'] == linkID]
#         LinkL = [] # id of nmg2 link that is closest to nmg1 rmg node
#         for j in range(len(sublist)):
#             XY = [sublist.iloc[j]['X'], sublist.iloc[j]['Y']] 
#             distances = nmgrid_2_link_points.apply(lambda row: distance_between_links(row, XY),axis=1)# compute the distance from the nmgrid_1 point and all nmgrid_2 points
#             offset = distances.min() # find the minimum distance between the nmg1 point and all nmg2 points
#             mdl = nmgrid_2_link_points['linkID'][(distances == offset)].values[0].astype(int)# get the nmg2 link id with point at minimum distance from nmg1 point, if more than one, pick the first one
#             LinkL.append(mdl) 
#         Links = np.array(LinkL)
#         # number of times each nmgrid_2 point was closest to nmgrid_1 link 
#         count =  np.bincount(Links)  

#         # nmgrid_2 link with highest count is matched to nmg1 link 
#         # if only one nmgrid_2 link has highest count, that is the link
#         if (count == count.max()).sum() == 1: 
#             Link = np.argmax(count) 
#         else: # if two or more nmgrid_2 links have the hightest count, select the 
#             # one that drains the largest area 
#             links_with_same_count = np.arange(len(count))[count == count.max()]
#             DAs_ =nmgrid_2.at_link['drainage_area'][links_with_same_count]
#             Link = links_with_same_count[DAs_ == DAs_.max()][0] # to remove bracket
#         link_mapper[linkID] = Link
        
#     return link_mapper



# possible alternative that may be quicker, using scipy function below
def map_nmg1_links_to_nmg2_links(nmgrid_1, nmgrid_2, number_of_points = 11):
    """given two slighly different network model grids of the same channel network,
    map links from one network model grid (nmgrid_1) to the closest links of the
    other network model grid (nmgrid_2). If two or more links of nmgrid_2 are equally 
    close to a link of nmgrid_1, the link with the largest drainage area is mapped 
    to the nmgrid_1 link. 
    

    Parameters
    ----------
    nmgrid_1 : network model grid
        grid that values will be mapped to
    nmgrid_2 : network model grid
        grid that values will be mapped from
    number_of_points : int
        Each link is converted to a series of number_of_point points. The relative
        distance of each link to the other is determined using the points.
        The default is 11. Below 11, output may not match expected.

    Returns
    -------
    link_mapper : dict
        Keys are the id of all links in nmgrid_1. Values are the link IDs of nmgrid_2
        that are mapped to each nmgrid_1 link.
    """    
    

    def distance_between_links(row, XY):
        return _dist_func(row['X'], XY[0], row['Y'], XY[1])# ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5
    
    # convert the network model grid to a point representation, as described by
    # the link ID, x and y value of each point 
    nmgrid_1_link_points = create_df_of_link_points(nmgrid_1, nmgrid_1.nodes_at_link, number_of_points)
    nmg1_linkIDs = nmgrid_1_link_points['linkID'].astype(int).values
    
    nmgrid_2_link_points = create_df_of_link_points(nmgrid_2, nmgrid_2.nodes_at_link, number_of_points)
    nmg2_linkIDs = nmgrid_2_link_points['linkID'].astype(int).values
    # for each point of each link of nmgrid_1, find the closest nmgrid_2 point
    # and link. nmgrid_2 link with highest number of points closest to the 
    # nmgrid_1 link is mapped to the nmgrid_1 link.

    sublist1 = nmgrid_1_link_points[['X','Y']] # get points that represent nmgrid_1
    sublist2 = nmgrid_2_link_points[['X','Y']] # get points that represent nmgrid_2
    distance_matrix = cdist(sublist1, sublist2, metric='euclidean') # create the distance matrix, which lists the distance between all nmgrid_1 and nmgrid_2 points
    distance_matrix_nodiag = distance_matrix # fill the diagonal values with inf
    np.fill_diagonal(distance_matrix_nodiag, np.inf)
    closest_point_indices = np.argmin(distance_matrix_nodiag, axis=1) # find the minimum values
    linkID_array = np.tile(nmg2_linkIDs, (len(nmg1_linkIDs),1)) # create a matrix of the nmg 2 link ids
    nmg2_link_matrix = linkID_array [np.arange(len(nmg1_linkIDs)), closest_point_indices] # get the link id of the closest node
    

    # now count the number of times each nmgrid_2 point was closest to nmgrid_1 link
    link_mapper ={}
    for linkID_1 in nmg1_linkIDs:
        linkIDs_2 = nmg2_link_matrix[nmg1_linkIDs == linkID_1]
        count =  np.bincount(linkIDs_2) 
        # nmgrid_2 link with highest count is matched to nmg1 link 
        # if only one nmgrid_2 link has highest count, that is the link
        if (count == count.max()).sum() == 1: 
            linkID_2 = np.argmax(count) 
        else: # if two or more nmgrid_2 links have the hightest count, select the 
            # one that drains the largest area 
            links_with_same_count = np.arange(len(count))[count == count.max()]
            DAs_ =nmgrid_2.at_link['drainage_area'][links_with_same_count]
            linkID_2 = links_with_same_count[DAs_ == DAs_.max()][0] # to remove bracket
        link_mapper[linkID_1] = linkID_2
        
    return link_mapper

        


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
    cn_to_nmg_link_mapper = map_rmg_nodes_to_nmg_links(grid, nmg_link_to_rmg_coincident_nodes_mapper, acn)
    tn_to_nmg_link_mapper = map_rmg_nodes_to_nmg_links(grid, nmg_link_to_rmg_coincident_nodes_mapper, tn)
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
            