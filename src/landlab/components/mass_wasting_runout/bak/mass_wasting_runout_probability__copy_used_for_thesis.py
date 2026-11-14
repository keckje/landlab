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



class MassWastingRunoutProbability(Component):
    """this component iteratively runs MassWastingRunout using a user provided landslide
    polygon(s) and instantiated MassWastingRunout model set up for the landslide. User 
    has the option to allow landslide area to vary or be fixed or can provide externally
    modeled landslide areas that vary each model iteration.
    
    author: Jeff Keck
    """
    
    def __init__(
        self,
        grid,
        MWR,
        parameter_cdf,
        number_iterations = 200,
        plot = False,
        method = 'fixed_size_landslide',
        min_landslide_area = 200,
        modeled_input = {},
        reset_fields = True,
        seed = None):
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
        method : TYPE, string
            DESCRIPTION. The default is 'fixed_size_landslide'. can also be: 'external_model' or 'variable_size_landslide'
        number_iterations : TYPE, optional
            DESCRIPTION. The default is 200.
        plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self._grid = grid
        self.MWR = MWR
        self._parameter_cdf = parameter_cdf
    
        # get node id of landslide / potentially unstable slope polygon
        self.ni = number_iterations
        self.plot = plot
        self._method = method
        self._reset_fields = reset_fields
        self._initial_soil_depth = self._grid.at_node['soil__thickness'].copy()
        self._topographic__initial_elevation = self._grid.at_node['topographic__elevation'].copy()
        if self._grid.has_field('particle__diameter'):
            self._initial_particle_diameter = self._grid.at_node['particle__diameter'].copy()
        self._maker(seed)
        
        
        # determine _method, if needed, load modeled landslides and associated fields
        if self._method == 'external_model':
            if 'mass__wasting_id' in modeled_input:
                if modeled_input['mass__wasting_id'].shape[0] == len(self._grid.core_nodes):
                    self._modeled_input = modeled_input
                elif modeled_input['mass__wasting_id'].shape[0] == self._grid.number_of_nodes:                        
                    for key in modeled_input:
                        modeled_input[key] = modeled_input[key][self._grid.core_nodes]
                    self._modeled_input = modeled_input
                else:
                    msg = "externally modeled input is wrong shape"
                    raise ValueError(msg) 
            else:
                msg = "externally modeled input must at the very least have a mass__wasting_id key"
                raise ValueError(msg) 

        elif (self._method == 'fixed_size_landslide') or (self._method == 'variable_size_landslide'):
            self._nodes = np.hstack(self._grid.nodes)
            self._lsnds = self._nodes[self._grid.at_node['mass__wasting_id'] == 1]            
            self.amin = min_landslide_area                
            
            
        else:
            msg = "-{}- not a run option".format(self._method)
            raise ValueError(msg)             
            

    def _maker(self,seed):
        """prepares the np random generator"""        
        self.maker = np.random.RandomState(seed=seed)


    @staticmethod
    def _intp(x,y,x1): 
        f = sc.interpolate.interp1d(x,y)   
        y1 = f(x1)
        return y1
    
    
    def _generate_val(self,q,vals):
        """generate a random quantile"""
        q_ = self.maker.uniform(q.min(), q.max(),1)[0]
        # return equivalent parameter value
        return self._intp(x = q,y = vals,x1 = q_)
    
    
    def _multidirectionflowdirector(self):
        """
        removes rmg fields created by d8 flow director, runs multidirectionflowdirector
        """
        self._grid.delete_field(loc = 'node', name = 'flow__sink_flag')
        self._grid.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
        self._grid.delete_field(loc = 'node', name = 'flow__receiver_node')
        self._grid.delete_field(loc = 'node', name = 'topographic__steepest_slope')
        # run flow director, add slope and receiving node fields
        fd = FlowDirectorMFD(self._grid, diagonals=True,
                              partition_method = 'slope')
        fd.run_one_step()
    
    
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


    def _read_ls(self,i):
        """read in externally modeled landslide(s) and associated grid fields"""
        for key in self._modeled_input:
            self._grid.at_node[key][self._grid.core_nodes] = self._modeled_input[key][:,i]


    def _monte_carlo_runout(self):
        """run MWR ni times, each time using a different slpc and qsc value,
        sampled from the empirical distribution developed from the calibrator"""
        self.para_dict = {}
        self.ls_dict = {}
        self.runout_dict = {}      
        for i in range(self.ni):
            
                # reset soil depth, grain size and topography
            if self._reset_fields:
                self._grid.at_node['topographic__elevation'] = self._topographic__initial_elevation.copy()
                self._grid.at_node['soil__thickness'] = self._initial_soil_depth.copy()
            if self._grid.has_field('particle__diameter'):
                self._grid.at_node['particle__diameter'] = self._initial_particle_diameter.copy()
            
            if self._method == 'variable_size_landslide':
                self._generate_ls()          
            elif self._method == 'external_model':
                self._read_ls(i)
            # else, use mass_wasting_id used to instantiate MWR
    
            print("sending them down the hill, iteration {}".format(i))
            # reset flow director for new topography
            self._multidirectionflowdirector() 
    
            # update slpc
            q = self._parameter_cdf['slpc'][0]
            vals = self._parameter_cdf['slpc'][1]
            self.MWR.slpc =self._generate_val(q,vals)
            
            # update qsc
            q = self._parameter_cdf['SD'][0]
            vals = self._parameter_cdf['SD'][1]
            self.MWR.SD = self._generate_val(q,vals)
    
            self.MWR.run_one_step(dt = i)
            
            self.para_dict[i] = {'slpc':self.MWR.slpc, 'qsc':self.MWR.SD, 'alpha':self.MWR.cs}
            self.ls_dict[i] = self._grid.at_node['mass__wasting_id']
            self.runout_dict[i] = self._grid.at_node['topographic__elevation'] - self._topographic__initial_elevation
            
            print(i)        
    
    
            # potentially unstable slope
            if self.plot:
                plt.figure()
                LLT.plot_node_field_with_shaded_dem(mg,field = 'mass__wasting_id', fontsize = 10)
                plt.show()
            
    
                mg.at_node['dem_dif'] = dem_dif
                plt.figure()
                LLT.plot_node_field_with_shaded_dem(mg,field = 'dem_dif', fontsize = 10)
                plt.clim(-1,1)
                plt.show()


    def _runout_probability(self):
        runout_summary = pd.DataFrame.from_dict(self.runout_dict,orient = 'index')
        self.sampled_parameters = pd.DataFrame.from_dict(self.para_dict, orient = 'index')
        
        runout_summary_c = runout_summary.copy()
        runout_summary_s = runout_summary.copy()
        runout_summary_d = runout_summary.copy()
        
        runout_summary_c[np.abs(runout_summary_c)>0] = 1 
        runout_summary_s[runout_summary_s>=0] = 0; runout_summary_s[runout_summary_s<0] = 1
        depth_threshold = 1
        runout_summary_d[runout_summary_d<=depth_threshold] = 0; runout_summary_d[runout_summary_d>depth_threshold] = 1

        p_c = runout_summary_c.sum(axis=0)/self.ni
        p_s = runout_summary_s.sum(axis=0)/self.ni
        p_d = runout_summary_d.sum(axis=0)/self.ni

        self._grid.at_node['probability__of_runout'] = p_c
        self._grid.at_node['probability__of_erosion'] = p_s
        self._grid.at_node['probability__of_deposition'] = p_d


    def run_one_step(self):
        """determine probability of runout, erosion and deposition"""
        self._monte_carlo_runout()
        self._runout_probability()
        

def generate_cdf(results, parameter, labloc = [0.75,0.75], fs = 14):
    """Generates a histogram of the sampled parameter values. From the histogram,
    creates an empirical cdf of the parameter value. Returns the cdf quantile and
    parameter values.
    
    TODO: change so that results is a 1D np array

    Parameters
    ----------
    results : pandas dataframe
        table from calibrator that summarizes calibration run and includes the sampled parameter space
    parameter : str
        'slpc' or  'SD', which are Sc and qsc in the paper

    Returns
    -------
    two np arrays
        (1) quantile values of the empirical cdf and (2) assocciated parameter value of each quantile value 

    """

    col = 'candidate_value_'+parameter

    def lbl(col): 
        lb = {'candidate_value_SD':'$qs_c$', 'candidate_value_slpc':'$S_c$'}
        return lb[col]

    fig, ax = plt.subplots(figsize = (1.4,.9))
    n = results[col].shape[0]
    counts, edges, plot = plt.hist(results[col],bins = int(n**0.5),color = 'k',alpha = 0.8)
    plt.grid(alpha = 0.5)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=3)
    plt.text(edges.max()*labloc[0], counts.max()*labloc[1],lbl(col))
    plt.tick_params(axis = 'both', which = 'major', labelsize = fs-4)
    # if save:
    #     plt.savefig("histogram_"+col+".png", dpi = 300, bbox_inches='tight')
    
    plt.figure(figsize = (3,2))
    # sum the histogram bins to get the cdf, using the bin center
    bin_cntr = (edges[1:]+edges[0:-1])/2
    bin_cnt = counts.cumsum() 
    cp = bin_cnt/counts.sum()
    plt.plot(bin_cntr, cp, color = 'k', alpha = 0.8, linewidth = 1)
    plt.grid(alpha = 0.5)
    plt.xlabel(parameter)
    plt.ylabel('P(x)')
    # if save:
    #     plt.savefig(wdir+"cdfd_"+col+svnm+".png", dpi = 300, bbox_inches='tight')
    vals = bin_cntr
    q = cp
    return q, vals