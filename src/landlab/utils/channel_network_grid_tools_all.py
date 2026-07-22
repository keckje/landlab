
import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
import landlab
import warnings
import matplotlib.pyplot as plt

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


def define_true_elements(grid, field_name, element, elements_that_are_true):
    """for a  model grid, creates the field "field_name, assigns
    elements elements_that_are_true as ones and all others as zero"""
    if isinstance(grid, landlab.grid.raster.RasterModelGrid):
        if element == 'node':
            grid.at_node[field_name] = np.zeros(grid.number_of_nodes).astype(int)
            grid.at_node[field_name][elements_that_are_true] = 1
        if element == 'link':
            grid.at_link[field_name] = np.zeros(grid.number_of_links).astype(int)
            grid.at_link[field_name][elements_that_are_true] = 1
    
        
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
    nodes (nodes whose associated cell intersect the link). Each coincident raster model
    grid node is defined in terms of its x and y coordinates, the link it is mapped to
    and the downstream distance of the node on the link. The downstream distance
    of the node on the link is defined as the distance from the upstream end (tail) of
    the link to the farthest-downstream edge of the node's cell.


    Parameters
    ----------
    grid : raster model grid
    nmgrid : network model grid
    link_nodes : np array
        head and tail node of each link
    remove_duplicates : bool
        if True, when two or more links are coincident with the same node,
        the node is assigned to the link with the largest drainage area. If False,
        the node is assigned to each coincident link. The default is False.

    Returns
    -------

    nmg_link_to_rmg_coincident_nodes_mapper: dict
        each key of the dictionary contains an np.array whose length is equal to the
        number of coincident nodes. Keys include link ID, coincident node ID,
        downstream distance of the coincident node, x coordinate of the coicident
        node, y coordinate of the coincident node and drainage area of the link.

    """
    linkIDs_list = []
    nodes_list = []
    Xs_list = []
    Ys_list = []
    dists_list = []
    link_drainage_areas_list = []
    for linkID, lknd in enumerate(link_nodes):
        # x and y of downstream (head) node of link
        x0 = nmgrid.x_of_node[lknd[0]]
        y0 = nmgrid.y_of_node[lknd[0]]
        # x and y of upstream (tail) node of link
        x1 = nmgrid.x_of_node[lknd[1]]
        y1 = nmgrid.y_of_node[lknd[1]]

        # get x and y coordinates and downstream distance from the upstream
        # node for 1000 points generated from downstream node to upstream node
        Xs, Ys, dists = _link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=1000
        )
        dists = dists.max() - dists  # convert to distance from tail node
        nodes = grid.find_nearest_node((Xs, Ys))
        # using the x and y coordinate of the first (most downstream) point
        # within the node's cell to represent the node location on the link
        mask = choose_from_repeated(nodes, choose="first")
        nodes = nodes[mask]

        linkIDs_list.append((np.ones(len(nodes)) * linkID).astype(int))
        nodes_list.append(nodes)
        Xs_list.append(grid.node_x[nodes])
        Ys_list.append(grid.node_y[nodes])
        dists_list.append(dists[mask])
        link_drainage_areas_list.append(
            (np.ones(len(nodes)) * nmgrid.at_link["drainage_area"][linkID]).astype(
                float
            )
        )

    nmg_link_to_rmg_coincident_nodes_mapper = {
        "linkID": np.concatenate(linkIDs_list),
        "coincident_node": np.concatenate(nodes_list),
        "x": np.concatenate(Xs_list),
        "y": np.concatenate(Ys_list),
        "coincident_node_downstream_dist": np.concatenate(dists_list),
        "link_drainage_area": np.concatenate(link_drainage_areas_list),
    }

    if remove_duplicates:
        # if more than one link are assigned to the same coincident node, which
        # can occur at stream junctions, retain the link that has the largest 
        # link_drainage_area and remove the others.
        values = nmg_link_to_rmg_coincident_nodes_mapper["coincident_node"]
        area = nmg_link_to_rmg_coincident_nodes_mapper["link_drainage_area"]
        idx = choose_unique(values=values, order_by=[area], choose="last") 
        idx.sort()
        for key in nmg_link_to_rmg_coincident_nodes_mapper.keys():

            nmg_link_to_rmg_coincident_nodes_mapper[key] = (
                nmg_link_to_rmg_coincident_nodes_mapper[key][idx]
            )

    return nmg_link_to_rmg_coincident_nodes_mapper


####PULL REQUEST 2


def map_rmg_nodes_to_nmg_links(
    grid,
    nmg_link_to_rmg_coincident_nodes_mapper,
    rmg_nodes,
    remove_small_trib_ratio=None,
):
    """Map the nodes representing the channel location in a DEM to the closest
    network model grid location. Network model grid location is described in
    terms of link id and distance down link, measured from the inlet node (tail)
    of the link.

    Parameters
    ----------
    grid : raster model grid
        needs to have node field "drainage_area"
    nmg_link_to_rmg_coincident_nodes_mapper : dictionary
        keys include the link ID, the coincident node ID, the downstream distance 
        of the coincident node, the x and y coordinates of the coincident node and 
        the drainage area of the link
    rmg_nodes : np.array
        an array of node ids to be mapped to the nmg links
    remove_small_trib_ratio : None or float
        If float, channel nodes whose contributing area is much less than the contributing
        area of the mapped link are ignored. Where "much less" is defined as being less 
        than the contributing area of the mapped link times the "remove_small_trib_ratio".
        If specified as None, then channel nodes whose contributing area is much less than 
        the contributing area of the mapped link are retained. Default value is None. 

    Returns
    -------
    rmg_nodes_to_nmg_links_mapper : dictionary
        keys include the node ID, the link ID the node has been
        mapped too, the closest nmg-link-coincident node ID, the drainage area
        of the link and the drainage area of the node

    """

    def dist_between_nmg_and_rmg_nodes(row, xc, yc):
        """distance between channel node and link node"""
        return _dist_func(xc, row["x"], yc, row["y"])

    link_ = []
    node_drainage_area_ = []
    linkID_ = []
    coincident_node_downstream_dist_ = []
    coincident_node_ = []
    link_drainage_area_ = []
    
    for n in rmg_nodes:  # for each rmg node
        xc = grid.node_x[n]
        yc = grid.node_y[n]
        
        dist = _dist_func(xc, nmg_link_to_rmg_coincident_nodes_mapper['x'],
                          yc, nmg_link_to_rmg_coincident_nodes_mapper['y'])
        
        
        # pick closest coincident node and corresponding link
        mask = dist == dist.min()
        linkID = nmg_link_to_rmg_coincident_nodes_mapper["linkID"][mask]
        coincident_node_downstream_dist = nmg_link_to_rmg_coincident_nodes_mapper["coincident_node_downstream_dist"][mask]
        coincident_node = nmg_link_to_rmg_coincident_nodes_mapper["coincident_node"][mask]
        link_drainage_area = nmg_link_to_rmg_coincident_nodes_mapper["link_drainage_area"][mask]        
        # if more than one (which can happen because the confluence between two
        # links overlay the same node, or nodes associated with same link upstream
        # and downstream of rmg node are same distance from rmg node), pick coincident node 
        # associated with link that has the largest contributing area. If coincident nodes associated with
        # the same link, pink the most downstream.
        
        # creating the boolean masks
        bigger_link_mask = link_drainage_area == link_drainage_area.max()
        downstream_dist_mask = coincident_node_downstream_dist[bigger_link_mask] == coincident_node_downstream_dist[bigger_link_mask].min()
        
        node_drainage_area_.append(grid.at_node["drainage_area"][n])  # add node drainage area to attributes
        linkID_.append(linkID[bigger_link_mask][downstream_dist_mask])
        coincident_node_downstream_dist_.append(coincident_node_downstream_dist[bigger_link_mask][downstream_dist_mask])
        coincident_node_.append(coincident_node[bigger_link_mask][downstream_dist_mask])
        link_drainage_area_.append(link_drainage_area[bigger_link_mask][downstream_dist_mask])


    rmg_nodes_to_nmg_links_mapper = {"node":rmg_nodes,
                "linkID":np.concatenate(np.array(linkID_)),
                "coincident_node":np.concatenate(np.array(coincident_node_)),
                "coincident_node_downstream_dist":np.concatenate(np.array(coincident_node_downstream_dist_)),
                "link_drainage_area":np.concatenate(np.array(link_drainage_area_)),
                "node_drainage_area":np.array(node_drainage_area_)}

    if (
        remove_small_trib_ratio
    ):  # check for small tributary nodes assigned to link and remove them
        rmg_nodes_to_nmg_links_mapper = _remove_small_tribs(
            rmg_nodes_to_nmg_links_mapper,
            nmg_link_to_rmg_coincident_nodes_mapper,
            remove_small_trib_ratio,
        )

    return rmg_nodes_to_nmg_links_mapper


def _remove_small_tribs(
    rmg_nodes_to_nmg_links_mapper,
    nmg_link_to_rmg_coincident_nodes_mapper,
    remove_small_trib_ratio):
    """remove rmg channel nodes that represent first order channels that flow into
    a mainstem channels and likely do not have an equivalent nmg link"""

    for link in np.unique(nmg_link_to_rmg_coincident_nodes_mapper["linkID"]):
        if link in rmg_nodes_to_nmg_links_mapper["linkID"]:
            # first get the coincident rmg node mapped to the link inlet
            # (i.e., the coincident node with shortest downstream distance from link inlet)
            mask1 = nmg_link_to_rmg_coincident_nodes_mapper["linkID"] == link
            min_dist = nmg_link_to_rmg_coincident_nodes_mapper[
                "coincident_node_downstream_dist"
            ][mask1].min()
            mask2 = (
                nmg_link_to_rmg_coincident_nodes_mapper[
                    "coincident_node_downstream_dist"
                ]
                == min_dist
            )
            inlet_coincident_node = nmg_link_to_rmg_coincident_nodes_mapper[
                "coincident_node"
            ][mask1*mask2]#[0] # zero index needed?
            # now get the contributing area of the rmg node mapped to the link
            # inlet coincident node (inlet_CA). The inlet_CA value will be used to 
            # screen small tributary nodes.
            mask3 = (
                rmg_nodes_to_nmg_links_mapper["coincident_node"]
                == inlet_coincident_node
            )
            inlet_CA_ = rmg_nodes_to_nmg_links_mapper["node_drainage_area"][
                mask3
            ]
            # But, if a small tributary node happens to also be mapped to the link inlet, there may be
            # more than one contributing area associated with the inlet   
            # if there is more than one, remove the contributing area that is much less than the link contributing area
            # Where "much less" is defined as being less than the contributing area to the link divided by the factor "remove_small_trib_ratio"
            if (
                len(inlet_CA_) > 1
            ):  
                mask4 = (
                    inlet_CA_
                    > rmg_nodes_to_nmg_links_mapper["link_drainage_area"][0] # 
                    * remove_small_trib_ratio
                )
                inlet_CA = inlet_CA_[mask4]
                # if one or more areas are NOT much less than the contributing area to the link, pick the smallest
                if len(inlet_CA) >= 1:
                    inlet_CA = inlet_CA.min()
                # Or if all areas are much less than the the contributing area to the link, pick the smallest
                elif len(inlet_CA) == 0:
                    inlet_CA = inlet_CA_.min()
            else:
                inlet_CA = inlet_CA_.min()

            # Any nodes that have a contributing area less than the inlet_CA are removed
            mask5 = (rmg_nodes_to_nmg_links_mapper["linkID"] == link) & (
                rmg_nodes_to_nmg_links_mapper["node_drainage_area"] < inlet_CA
            )

            rmg_nodes_to_nmg_links_mapper = {
                key: val[~mask5] for key, val in rmg_nodes_to_nmg_links_mapper.items()
            }

    return rmg_nodes_to_nmg_links_mapper


####PULL REQUEST 3


def create_dict_of_link_points(nmgrid, nodes_at_link, number_of_points):
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
    X_ = []
    Y_ = []
    link_ = []
    for linkID, lknd in enumerate(nodes_at_link):  # for each link in nmgrid 1 1

        x0 = nmgrid.x_of_node[lknd[0]]  # x and y of downstream link node
        y0 = nmgrid.y_of_node[lknd[0]]
        x1 = nmgrid.x_of_node[lknd[1]]  # x and y of upstream link node
        y1 = nmgrid.y_of_node[lknd[1]]

        # convert link to a series of points
        X, Y, dist = _link_to_points_and_dist((x0, y0), (x1, y1), number_of_points)
        
        X_.append(X)
        Y_.append(Y)
        link_.append((np.ones(len(X))*linkID).astype(int))
        
    return {'linkID':np.concatenate(link_), 'X':np.concatenate(X_), 'Y':np.concatenate(Y_)}


def convert_links_to_LineString(nmgrid, nodes_at_link):
    """ function for converting links to a list of line strings, attempted to 
    use line strings to find closest link to each link, but Shapely distance
    function uses the minimum distance, not the mean."""
    link_lines = []
    for linkID, lknd in enumerate(nodes_at_link):
        x0 = nmgrid.x_of_node[lknd[0]]  # x and y of downstream link node
        y0 = nmgrid.y_of_node[lknd[0]]
        x1 = nmgrid.x_of_node[lknd[1]]  # x and y of upstream link node
        y1 = nmgrid.y_of_node[lknd[1]]
        link_lines.append(LineString([(x0,y0), (x1,y1)]))
    return link_lines
        


def plot_nmgrids(nmgrid_1, nmgrid_2):
    """compare links and link ids of two network model grids in a plot"""
    def plot_nmgrid(nmgrid, line_color, alpha, fontsize, label):
        xnode = nmgrid.x_of_node; xlink = nmgrid.midpoint_of_link[:,0]
        ynode = nmgrid.y_of_node; ylink = nmgrid.midpoint_of_link[:,1]
        for link,val in enumerate(nmgrid.nodes_at_link):
            xv = xnode[val]; yv = ynode[val]         
            if link == 0:
                plt.plot(xv,yv, color = line_color, alpha = alpha, label=label)
            else:
                plt.plot(xv,yv, color = line_color, alpha = alpha, label =  '_nolegend_')
            plt.text(xlink[link], ylink[link], str(link), size=fontsize, color=line_color, alpha = alpha)
    
    plt.figure(figsize=(5,5))
    plot_nmgrid(nmgrid_1, line_color = 'red',alpha =1,fontsize = 12,label='nmgrid_1')
    plot_nmgrid(nmgrid_2, line_color = 'green',alpha = 0.37,fontsize = 20,label ='nmgrid_2')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend()
    plt.show()



def map_nmg1_links_to_nmg2_links(nmgrid_1, nmgrid_2, number_of_points=11, plot_grids = False):
    """given two slightly different network model grids of the same channel network,
    map each link from one network model grid (nmgrid_1) to the closest (based on
    the mean distance between links) link of the other network model grid (nmgrid_2). 
    If two or more links of nmgrid_2 are equally close to a link of nmgrid_1, the 
    link with the largest drainage area is mapped to the nmgrid_1 link


    Parameters
    ----------
    nmgrid_1 : network model grid
        grid that values will be mapped to
    nmgrid_2 : network model grid
        grid that values will be mapped from
    number_of_points : int
        Each link is converted to a series of number_of_point points. The relative
        distance of each link to the other is determined using these points.
        The default is 11. Below 11, mapping may not match expected.

    Returns
    -------
    link_mapper : dict
        Keys are the id of all links in nmgrid_1. Values are the link IDs of nmgrid_2
        that are mapped to each nmgrid_1 link.
        
        
    WARNING: In some situations this function may not map as expected. Set plot_grids
    to True and inspect results to check
    """
    warnings.warn("In some situations this function may not map as expected. Set plot_grids to True and inspect results to check")
    
    def distance_between_links(row, XY):
        return _dist_func(
            row["X"], XY[0], row["Y"], XY[1]
        )  # ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5

    # convert the network model grid to a point representation, as described by
    # the link ID, x and y value of each point
    # change this to np array, get rid of pandas operations
    nmgrid_1_link_points = create_dict_of_link_points(
        nmgrid_1, nmgrid_1.nodes_at_link, number_of_points
    )
    nmg1_linkIDs = nmgrid_1_link_points["linkID"]#.astype(int).values

    nmgrid_2_link_points = create_dict_of_link_points(
        nmgrid_2, nmgrid_2.nodes_at_link, number_of_points
    )
    nmg2_linkIDs = nmgrid_2_link_points["linkID"]#.astype(int).values
    # for each point of each link of nmgrid_1, find the closest nmgrid_2 point
    # and link. nmgrid_2 link with highest number of points closest to the
    # nmgrid_1 link is mapped to the nmgrid_1 link.

    sublist1 = np.array([nmgrid_1_link_points['X'],nmgrid_1_link_points['Y']]).T#nmgrid_1_link_points[["X", "Y"]]  # get points that represent nmgrid_1
    sublist2 = np.array([nmgrid_2_link_points['X'],nmgrid_2_link_points['Y']]).T#nmgrid_2_link_points[["X", "Y"]]  # get points that represent nmgrid_2
    distance_matrix = cdist(
        sublist1, sublist2, metric="euclidean"
    )  # create the distance matrix, which lists the distance between all nmgrid_1 and nmgrid_2 points
    distance_matrix_nodiag = distance_matrix  # fill the diagonal values with inf
    np.fill_diagonal(distance_matrix_nodiag, np.inf) # this step may be incorrect
    closest_point_indices = np.argmin(
        distance_matrix_nodiag, axis=1
    )  # find the minimum values
    linkID_array = np.tile(
        nmg2_linkIDs, (len(nmg1_linkIDs), 1)
    )  # create a matrix of the nmg 2 link ids
    nmg2_link_matrix = linkID_array[
        np.arange(len(nmg1_linkIDs)), closest_point_indices
    ]  # get the link id of the closest node

    # now count the number of times each nmgrid_2 point was closest to nmgrid_1 link
    link_mapper = {}
    for linkID_1 in nmg1_linkIDs:
        linkIDs_2 = nmg2_link_matrix[nmg1_linkIDs == linkID_1]
        count = np.bincount(linkIDs_2)
        # nmgrid_2 link with highest count is matched to nmgrid_1 link
        # if only one nmgrid_2 link has highest count, that is the link
        if (count == count.max()).sum() == 1:
            linkID_2 = np.argmax(count)
        else:  # if two or more nmgrid_2 links have the hightest count, select the
            # one that drains the largest area
            links_with_same_count = np.arange(len(count))[count == count.max()]
            DAs_ = nmgrid_2.at_link['drainage_area'][links_with_same_count]
            linkID_2 = links_with_same_count[DAs_ == DAs_.max()][0]  # to remove bracket
        link_mapper[linkID_1] = linkID_2
        
    if plot_grids:
        plot_nmgrids(nmgrid_1, nmgrid_2)

    return link_mapper
        

####PULL REQUEST 4


def map_rmg_channel_nodes_to_nmg_nodes(grid, nmgrid, acn):
    sublist1 = np.array([grid.node_x[acn], grid.node_y[acn]]).T#nmgrid_1_link_points[["X", "Y"]]  # get points that represent nmgrid_1
    sublist2 = np.array([nmgrid.node_x,nmgrid.node_y]).T#nmgrid_2_link_points[["X", "Y"]]  # get points that represent nmgrid_2
    # create the distance matrix, which lists the distance between all nmgrid_1 and nmgrid_2 points
    distance_matrix = cdist(
        sublist2, sublist1, metric="euclidean"
    )  
    # find the minimum values
    closest_point_indices = np.argmin(
        distance_matrix, axis=1
    )  
    # create a matrix of grid node IDs
    nodeID_array = np.tile(
        acn, (nmgrid.number_of_nodes, 1)
    )
    # get the ID of the closest node
    closest_node_IDs = nodeID_array[
        np.arange(len(sublist2)), closest_point_indices
    ]  

    # return as a dict with keys as nmgrid link ID, values as ID of closest rmg node 
    return dict(zip(nmgrid.nodes, closest_node_IDs))


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
        
        
####PULL REQUEST 5


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
        # link_nodes =  cn_to_nmg_link_mapper['coincident_node'][cn_to_nmg_link_mapper['linkID'] == link]
        link_nodes =  cn_to_nmg_link_mapper['node'][cn_to_nmg_link_mapper['linkID'] == link]
        grid.at_node[rmg_field][link_nodes] = value
        
        
def transfer_rmg_channel_node_field_to_nmg_link_field(grid, nmgrid, rmg_field, nmg_field, cn_to_nmg_link_mapper, metric = 'mean', default_value = np.nan):
    """updates the field value of the nmg links using the mean, max, minimum or 
    median value of the rmg nodes mapped to each link"""
    # add field to the nmg if not already present
    if nmg_field not in nmgrid.at_link.keys(): # field not
        nmgrid.at_link[nmg_field] = np.ones(nmgrid.number_of_links)*default_value
    
    for i, link in enumerate(nmgrid.active_links):
        #link_nodes =  cn_to_nmg_link_mapper['coincident_node'][cn_to_nmg_link_mapper['linkID'] == link]
        link_nodes =  cn_to_nmg_link_mapper['node'][cn_to_nmg_link_mapper['linkID'] == link]
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


def update_rmg_channel_location_and_mapping(grid, nmgrid, Ct, BFt, terrace_width, link_nodes, nmg_link_to_rmg_coincident_nodes_mapper, remove_small_trib_ratio = None):
    """if the DEM changes, this function remaps channel and terrace nodes and updates 
    the mappers"""
    
    acn = extract_channel_nodes(grid,Ct)
    fcn = extract_channel_nodes(grid,BFt)

    acn_to_nmg_links_mapper = map_rmg_nodes_to_nmg_links_new(grid, nmg_link_to_rmg_coincident_nodes_mapper, acn, remove_small_trib_ratio)
    fcn_to_nmg_links_mapper = map_rmg_nodes_to_nmg_links_new(grid, nmg_link_to_rmg_coincident_nodes_mapper, fcn, remove_small_trib_ratio)


    # remove small tributaries from the all channel node array
    # by checking for node contributing areas that are much less than the inlet node contributing area
    if remove_small_trib_ratio:
        acn = np.isin(acn, acn_to_nmg_links_mapper ['node'])
        fcn = np.isin(fcn, fcn_to_nmg_links_mapper ['node'])
        
    tn = extract_terrace_nodes(grid, terrace_width, acn, fcn)
    tn_to_nmg_links_mapper = map_rmg_nodes_to_nmg_links_new(grid, nmg_link_to_rmg_coincident_nodes_mapper, tn)
    nmg_node_to_cn_mapper = map_rmg_channel_nodes_to_nmg_nodes_new(grid, nmgrid, acn)
    
    return {'acn':acn,
            'fcn':fcn,
            'tn':tn,
            'nmg_link_to_rmg_coincident_nodes_mapper':nmg_link_to_rmg_coincident_nodes_mapper,
            'cn_to_nmg_links_mapper':acn_to_nmg_links_mapper,
            'tn_to_nmg_links_mapper':tn_to_nmg_links_mapper,            
            'nmg_node_to_cn':nmg_node_to_cn_mapper}
    

####PULL REQUEST 6


def get_upslope_nodes(grid):
    """
    Get all upslope contributing nodes using the flow__receiver_node and 
    flow__reciever_node_order fields from FlowAccumulator, as corrected by 
    DepressionFinderAndRouter.
    
    Parameters
    ----------
    grid : raster model grid
        Needs the node field "drainage_area", which describes the topographic drainage
        area upslope of each node in m^2
        
    Returns
    -------
    upslope_dictionary: dict
        keys are the channel nodes, values are the indicies of all nodes upslope of the channel node
    """
    # Initialize an empty list for every node in the grid
    all_upslope_nodes_dict = {node: [] for node in range(grid.number_of_nodes)}
    
    # Get the receiver array
    receivers = grid.at_node['flow__receiver_node']
    
    # Node field 'flow__upstream_node_order' lists nodes from outlet to ridge.
    # By reversing this array, we process the network from the ridges down to the outlets.
    top_down_order = reversed(grid.at_node['flow__upstream_node_order'])
    
    # Pass the lists downstream
    for node in top_down_order:
        receiver = receivers[node]
        
        # If the node flows into a different node (i.e., it is not a sink/outlet)
        if receiver != node:
            # The receiver inherits the current node.
            all_upslope_nodes_dict[receiver].append(node)
            # ...AND the receiver inherits everything already accumulated by the current node.
            all_upslope_nodes_dict[receiver].extend(all_upslope_nodes_dict[node])
            
    return all_upslope_nodes_dict


def floodplain_mapper(grid,
                      channel_initiation_DA = 5000,
                      BFD_parameters = [0.274, 0.24],
                      BFD_factor = 50,
                      ):
    """
    map the floodplain for each node by finding all channel nodes and then for each
    channel node, finding all non-channel nodes whose elevation is within some 
    multiple of the nodes elevation plus the bankfull flow depth scaled by a factor.
    
    Parameters
    ----------
    grid : raster model grid
        Needs the node field "drainage_area", which describes the topographic drainage
        area upslope of each node in m^2
    channel_initiation_DA : int
        Topographic drainage area at which channels initiate, in m^2. The default is 5000.
    BFD_parameters : list of length 2
        The coefficent and exponent to a hydraulic geometry description of how 
        bankfull flow dpeth varies with topographic drainage area The default is [0.274, 0.24].
    BFD_factor : float
        factor multiplied by the bankfull depth to define upper elevation of floodplain relative
        to the channel node. The default is 50.

    
    Returns
    -------
    fn : np array
        Indicies of all nodes that define the floodplain and channels. To get just
        the floodplain nodes, remove the channel nodes using fn[np.isin(fn, acn, invert =True)]
    fn_cn : np array
        Node id that floodplain is mapped to. If not counted as a floodplain node, 
        then fn_cn is simply the node indice (node id). Note, this output includes 
        the channel nodes which are also mapped to themselves.
    """
    # define the bankfull flow hydraulic geometry parameters
    hg_c = BFD_parameters[0]
    hg_e = BFD_parameters[1]
    
    # create the upslope dictionary, upslope nodes to all nodes in grid
    upslope_dictionary = get_upslope_nodes(grid)
    
    # get all channel nodes
    acn = gt.extract_channel_nodes(grid, channel_initiation_DA)
    
    # add a floodplain_channel_node field. nodes that are not channel or floodplain
    # initially, this is just the ids (index value) of each node
    fn_cn = grid.nodes.flatten()

    # sort channel nodes by drainage area
    cn_drainage_area = grid.at_node['drainage_area'][acn]
    sorted_indices = np.argsort(cn_drainage_area)[::-1]
    acn_sorted = acn[sorted_indices]
    
    # beginning at the most downstream reach and working upstream, for each channel
    # node, find all uplope nodes that have an elevation below the node elevation +
    # BFD_factor*bankfull depth and assign them the drainage
    fn = acn
    for node in acn_sorted:
        # get all upslope nodes to node
        CA = grid.at_node['drainage_area'][node] #m2 to km2
        el = grid.at_node['topographic__elevation'][node]
        
        # define the bankfull depth and scale by the BFD_factor
        threshold_depth = BFD_factor*hg_c*(CA/(1000**2))**hg_e
        up_nodes = np.array(upslope_dictionary[node])
        # upslope nodes within el_factor * the bankfull depth
        floodplain_mask = grid.at_node['topographic__elevation'][up_nodes] < el+threshold_depth 
        fn = np.unique(np.concatenate((fn,up_nodes[floodplain_mask])))
        fn_cn[up_nodes[floodplain_mask]] = node
        
    return fn, fn_cn

