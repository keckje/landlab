import os
import numpy as np
import scipy as sc
import pandas as pd

import matplotlib.pyplot as plt

from landlab import RasterModelGrid, Component
from landlab.components import SinkFillerBarnes, FlowAccumulator, FlowDirectorMFD
from landlab.components.mass_wasting_router import MassWastingRunout
from landlab import imshow_grid

from landlab import imshow_grid_at_node



class MassWastingRunoutProbabilitySingleLandslide(MassWastingRunoutProbabilityBase):
    """this component iteratively runs MassWastingRunout using a user provided landslide
    polygon and instantiated MassWastingRunout model set up for the landslide. User 
    has the option to allow landslide area to vary or be fixed
    
    author: Jeff Keck
    """
    
    def __init__(self):
        """
        

        Parameters
        ----------
        grid : TYPE
            DESCRIPTION.
        ls_polygon : TYPE
            DESCRIPTION.
        MWR : TYPE
            DESCRIPTION.
        parameter_cdf : TYPE
            DESCRIPTION.
        vary_landslide_area : TYPE, optional
            DESCRIPTION. The default is False.
        number_iterations : TYPE, optional
            DESCRIPTION. The default is 200.
        plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        super().__init__(grid)

    
    def _random_node(self):
        """generate a random node index"""
        nodes = self._lsnds
        ln = len(nodes)-1   
        i = np.round(np.random.uniform(0,ln)).astype(int)
        return nodes[i]
    
    @staticmethod
    def _random_area(amin,amax):
        """given a min and max area, generate a random area"""
        return np.random.uniform(amin,amax)
    
    
    def _uan(self, n):
        """get upslope, adjacent nodes"""
        adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],self._grid.diagonal_adjacent_nodes_at_node[n]))
        uan = adj_n[self._grid.at_node['flow__receiver_node'][n] == -1]
        return uan
    
     
    def _ls_maker(self,n):
        """get all adjacent, upslope nodes to an initial node"""
        amin = self.amin
        amax = len(self._lsnds)*self._grid.dx**2
        area_limit = self._random_area(amin,amax)
        dx  = self._grid.dx

        nL = [n]
        nc = 0
        ls_n = nL
        c=0
        while nc*dx*dx < area_limit and c<10: # while count (area) of ls nodes is less than randomly picked value
            nn = []
            for n in nL: # get all adjacent, upslop enodes
                nn.append(self._uan(n))
            nL = np.unique(np.hstack(nn))
            # append upslope adjacent nodes to list of landslide nodes
            ls_n = np.hstack([ls_n, nL])
            # only keep unique nodes
            ls_n = np.unique(np.hstack(ls_n))
            # only keep nodes in landslide
            ls_n = ls_n[np.isin(ls_n,self._lsnds)]
            nc = len(ls_n)
            c+=1
        return ls_n


    def _generate_ls(self):
        """generate the random node"""
        n = self._random_node()
        
        # create the landslide polygon      
        ls_n = self._ls_maker(n)
        
        # prepare array that will set all nodes to non-landlide nodes
        lsa = np.zeros(len(self._nodes))
        # now change some of that array to the randomly generated landslide nodes
        lsa[ls_n] = 1
        # set grid field 'mass__wasting_id'
        self._grid.at_node['mass__wasting_id'] = lsa

    
