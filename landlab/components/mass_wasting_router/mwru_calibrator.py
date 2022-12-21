
#TODO add determine_alpha function
# add constraint that selected alpha can not result in a scour depth greater than qsc
# run for Capitol forest sites

# setup
import os
import numpy as np
from scipy.stats import norm
import pandas as pd

import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import SinkFillerBarnes, FlowAccumulator, FlowDirectorMFD
from landlab.components.mass_wasting_router import MassWastingRunout
from landlab import imshow_grid

from landlab import imshow_grid_at_node

class MWRu_calibrator():
    """an adaptive Markov Chain Monte Carlo calibration utitlity for calibrating
    MassWastingRunout to observed landslide runout and/or scour and deposition

    author: Jeff Keck
    """

    def __init__(self,
                 MassWastingRunout,
                 params,
                 profile_calib_dict,
                 prior_distribution = "uniform",
                 method = 'both',
                 omega_metric = "runout",
                 RMSE_metric = "Vd",
                 jump_size = 0.2,
                 N_cycles = 10,
                 plot_tf = True,
                 seed = None,
                 alpha_min = 0.1,# 0.23 #0.1
                 alpha_max = 0.5,# 0.44#0.5
                 phi_minus = 0.9,
                 phi_plus = 1.1):
        """
        Parameters
        ----------
        MassWastingRunout: instantiated mass wasting runout that contains a
            raster model grid that contains the fields:
                topographic__elevation
                topographic__steepest_slope
                dem_dif_o

        params : list of lists
            each list is the min, max and opimal parameter value. Used to define
            the parameter space used by the MCMC sampler

        profile_calib_dict : dictonary, optional
            dictionary of parameters needed to evaluate calibration based on
            volumetric deposition versus elevation along the channel. Keys in the
            dicitonary include the following parameters:
                ----------
                el_l : float
                    lower limit of elevation range included in analysis.
                el_h : float
                    upper limit of elevation range included in analysis.
                mg : raster model grid
                    has fields topographic__elevation, topographic__steepest_slope and dem_dif_o and dem_dif_m
                channel_nodes : np.array
                    np.array of node ids of all nodes in channel, output from channel profiler
                channel_distance : np.array
                    np.array of distance at each node from outlet, output from channel profiler
                cL : float
                    length of grid cell for modeled grid
        prior_distribution: string
            can be "uniform" or "normal"

        method: string
            can be "RMSE", "omega" or "both"

        omega_metric: string
            can be "runout", "deposition" or "scour". Default is "runout"

        RMSE_metric: string
            can be 'Vu','Vd','dV','Vt'. Default is "Vd"

        prior_distribution: string
            can be "uniform" or "normal". used to determine liklihood of selected
            parameter value in the MCMC algorithm

        jump_size: float
            standard deviation of jump size for MCMC algorithm expressed as ratio of
            the jump size to the range between the minimum and maximum parameter values
            default: jump_size = 0.05

        N_cycles: int
            Number of iterations between updates to the jump size based on MCMC acceptance ratio
        """

        assert (profile_calib_dict is not None
                ),"must provide parameters for either profile or planemetric evaluation"

        self.MWRu = MassWastingRunout
        self.mg = MassWastingRunout.grid
        self.params = params
        self.pcd = profile_calib_dict
        self.prior_distribution = prior_distribution
        self.method = method
        self.omega_metric = omega_metric
        self.RMSE_metric = RMSE_metric
        self.jump_size = jump_size
        self.N_cycles = N_cycles
        self.initial_soil_depth = self.mg.at_node['soil__thickness'].copy()
        if self.mg.has_field("particle__diameter", at="node"):
            self.initial_particle_diameter = self.mg.at_node["particle__diameter"].copy()
        self.plot_tf = plot_tf
        self.dem_dif_m_dict ={} # for dedegging
        self._maker(seed)
        self.jstracking = [] # for tracking jumps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.phi_minus = phi_minus
        self.phi_plus = phi_plus


    def __call__(self, max_number_of_runs = 50):
        """instantiate the class"""
        if (self.method == "both") or (self.method == "RMSE"):
            self.mbLdf_o = self._channel_profile_deposition("observed")
        self._MCMC_sampler(max_number_of_runs)

    
    def _maker(self,seed):
        """prepares the np random generator"""
        
        self.maker = np.random.RandomState(seed=seed)


    def _simulation(self):
        """run the model, determine the cumulative modeled deposition along the
        channel centerline"""
        # reset the dem to the initial dem and soil to initial soil depth
        self.MWRu.grid.at_node['topographic__elevation'] = self.MWRu.grid.at_node['topographic__initial_elevation'].copy()
        self._update_topographic_slope()
        self.MWRu.grid.at_node['energy__elevation'] = self.MWRu.grid.at_node['topographic__elevation'].copy()
        self.MWRu.grid.at_node['soil__thickness'] = self.initial_soil_depth.copy()
        self.mg.at_node['disturbance_map'] = np.full(self.mg.number_of_nodes, False)
        if self.mg.has_field("particle__diameter", at="node"):
            self.mg.at_node["particle__diameter"] = self.initial_particle_diameter.copy()
        
            
        # run the model
        self.MWRu.run_one_step(dt = 0)
        # create the modeldiff_m field
        diff = self.mg.at_node['topographic__elevation'] - self.mg.at_node['topographic__initial_elevation']
        self.mg.at_node['dem_dif_m'] = diff
        
        self.dem_dif_m_dict[self.it] = self.mg.at_node['dem_dif_m']
        if self.plot_tf == True:
            plt.figure('iteration'+str(self.it))
            imshow_grid(self.mg,"dem_dif_m",cmap = 'RdBu_r')
            plt.title("it:{}, slpc:{}, SD:{}, alpha:{}".format(self.it, self.MWRu.slpc, self.MWRu.SD, self.MWRu.cs ))
            plt.clim(-1,1)
            plt.show()
            
            

    def _update_topographic_slope(self):
        """updates the topographic__slope and flow directions fields using the 
        topographic__elevation field"""
        fd = FlowDirectorMFD(self.mg, surface="topographic__elevation", diagonals=True,
                partition_method = self.MWRu.routing_partition_method)
        fd.run_one_step()

    def _channel_profile_deposition(self, datatype):
        """determines deposition patterns along the channel profile:
        """
        mg = self.mg
        
        el_l = self.pcd['el_l']
        el_h = self.pcd['el_h']
        channel_nodes = self.pcd['channel_nodes']
        channel_distance = self.pcd['channel_distance']
        node_slope = mg.at_node['topographic__steepest_slope']
        cL = self.pcd['cL']
       
        def cv_mass_change_v2(cn,dem,demd,el_l,el_h,dA,node_slope,channel_distance):

            el = dem[cn]
            # cumulative downstream deposition and upstream scour
            # dp: change > 0; Mask dem to get matching array of elevation
            dp = demd[demd>0]; dem_dp = dem[demd>0]
            # sc: change < 0; Mask dem to get matching array of elevation
            sc = demd[demd<0]; dem_sc = dem[demd<0]

            # sum masked dp and and sc
            # deposition below elevation
            dpe = np.nansum(dp[(dem_dp<el)&(dem_dp>=el_l)])
            # scour above elevation
            sce = np.nansum(sc[(dem_sc>el)&(dem_sc<=el_h)])
            # multiply by cell area to get volume
            Vt = dpe*dA
            dV = sce*dA

            # cumulative volumetric change in the upstream and downstream directions
            dd = np.nansum(demd[(dem<=el) & (dem>=el_l)])
            du = np.nansum(demd[(dem>=el) & (dem<=el_h)])
            # multiply by cell area to get volume
            Vd = dd*dA
            Vu = du*dA

            # get channel characteristics
            # same, nut get node distance
            cd = channel_distance[channel_nodes == cn]
            # node slope
            cns = node_slope[cn]
            return Vu, Vd, dV, Vt, cns, cn, cd, el
        
        def cv_mass_change(el,dem,demd,el_l,el_h,dA,channel_nodes,node_slope,channel_distance):
            """may need this approach, because node order can cause problems, 
            but most recent profile extraction did not have problem, need to double check"""

            # cumulative downstream deposition and upstream scour
            # dp: change > 0; Mask dem to get matching array of elevation
            dp = demd[demd>0]; dem_dp = dem[demd>0]
            # sc: change < 0; Mask dem to get matching array of elevation
            sc = demd[demd<0]; dem_sc = dem[demd<0]

            # sum masked dp and and sc
            # deposition below elevation
            dpe = np.nansum(dp[(dem_dp<el)&(dem_dp>=el_l)])
            # scour above elevation
            sce = np.nansum(sc[(dem_sc>el)&(dem_sc<=el_h)])
            # multiply by cell area to get volume
            Vt = dpe*dA
            dV = sce*dA

            # cumulative volumetric change in the upstream and downstream directions
            dd = np.nansum(demd[(dem<=el) & (dem>=el_l)])
            du = np.nansum(demd[(dem>=el) & (dem<=el_h)])
            # multiply by cell area to get volume
            Vd = dd*dA
            Vu = du*dA

            # get channel characteristics
            cne = dem[channel_nodes]; cne_d = cne-el # cne closest to zero is node
            # channel node at elevation closeset to el, use mask to finde
            cn = channel_nodes[np.abs(cne_d) == min(np.abs(cne_d))].astype(int)
            # same, nut get node distance
            cd = channel_distance[np.abs(cne_d) == min(np.abs(cne_d))].astype(int)
            # node slope
            cns = node_slope[cn]
            return Vu, Vd, dV, Vt, cns[0], cn[0], cd[0], el

        dem_m = mg.at_node['topographic__elevation']
        if datatype == "modeled":
            demd = mg.at_node['dem_dif_m']
            dem = mg.at_node['topographic__elevation'] 
        elif datatype == "observed":
            demd = mg.at_node['dem_dif_o']
            dem = mg.at_node['topographic__initial_elevation']+demd

        mbL = []
        # for el in np.linspace(el_l,el_h,100):
        #     mbL.append(cv_mass_change(el, dem, demd,el_l,el_h,cL**2,channel_nodes, node_slope, channel_distance))
        for cn in channel_nodes:
            mbL.append(cv_mass_change_v2(cn,dem,demd,el_l,el_h,cL**2,node_slope,channel_distance))    
        mbLdf = pd.DataFrame(mbL)
        mbLdf.columns = ['Vu','Vd','dV','Vt', 'slope', 'node','distance','elevation']
        return mbLdf


    def _omegaT(self, metric = 'runout'):
        """ determines intersection, over estimated area and underestimated area of
        modeled debris flow deposition and the the calibration metric OmegaT following
        Heiser et al. (2017)
        """
        c = 1
        n_a = self.mg.nodes.reshape(self.mg.shape[0]*self.mg.shape[1]) # all nodes
        na = self.mg.dx*self.mg.dy
        if metric == 'runout':
            n_o =  n_a[np.abs(self.mg.at_node['dem_dif_o']) > 0] # get nodes with scour or deposit
            n_m = n_a[np.abs(self.mg.at_node['dem_dif_m']) > 0]
        elif metric == 'deposition':
            n_o =  n_a[self.mg.at_node['dem_dif_o'] > 0] # get nodes with scour or deposit
            n_m = n_a[self.mg.at_node['dem_dif_m'] > 0]
        elif metric == 'scour':
            n_o =  n_a[self.mg.at_node['dem_dif_o'] < 0] # get nodes with scour or deposit
            n_m = n_a[self.mg.at_node['dem_dif_m'] < 0]
        self.a_o = n_o*na
        self.a_m = n_m*na
        n_x =  n_o[np.isin(n_o,n_m)]#np.unique(np.concatenate([n_o,n_m])) # intersection nodes
        n_u = n_o[~np.isin(n_o,n_m)] # underestimate
        n_o = n_m[~np.isin(n_m,n_o)] # overestimate
        
        X = len(n_x)*na
        U = len(n_u)*na*c
        O = len(n_o)*na*c
        T = X+U+O
        omegaT = X/T-U/T-O/T
        return omegaT



    def _RMSEomegaT(self, metric = 'runout'):
        """ determines intersection, over estimated area and underestimated area of
        modeled debris flow deposition and the the calibration metric OmegaT following
        Heiser et al. (2017)
        """
        c1 = 10; c2 = 2
        n_a = self.mg.nodes.reshape(self.mg.shape[0]*self.mg.shape[1]) # all nodes
        na = self.mg.dx*self.mg.dy
        if metric == 'runout':
            n_o =  n_a[np.abs(self.mg.at_node['dem_dif_o']) > 0] # get nodes with scour or deposit
            n_m = n_a[np.abs(self.mg.at_node['dem_dif_m']) > 0]
        elif metric == 'deposition':
            n_o =  n_a[self.mg.at_node['dem_dif_o'] > 0] # get nodes with scour or deposit
            n_m = n_a[self.mg.at_node['dem_dif_m'] > 0]
        elif metric == 'scour':
            n_o =  n_a[self.mg.at_node['dem_dif_o'] < 0] # get nodes with scour or deposit
            n_m = n_a[self.mg.at_node['dem_dif_m'] < 0]
        self.a_o = n_o*na
        self.a_m = n_m*na
        n_x =  n_o[np.isin(n_o,n_m)]#np.unique(np.concatenate([n_o,n_m])) # intersection nodes
        n_u = n_o[~np.isin(n_o,n_m)] # underestimate
        n_o = n_m[~np.isin(n_m,n_o)] # overestimate
        A_x = len(n_x)*self.mg.dx*self.mg.dy
        A_u = len(n_u)*self.mg.dx*self.mg.dy
        A_o = len(n_o)*self.mg.dx*self.mg.dy
        
        observed_ = self.mg.at_node['dem_dif_o']
        mask =  np.abs(observed_)>0 
        modeled_ = self.mg.at_node['dem_dif_m']       
        # mask_m =  np.abs(modeled_)<=0 
        # modeled_[mask_m] = np.abs(modeled_).max()
        
        modeled = modeled_[mask]
        observed = observed_[mask]
        X = A_x*self._RMSE(observed, modeled)
        # if X != 0:
        #     X = 1/X
        modeled = modeled_[n_u]
        observed = observed_[n_u]
        U = A_u*self._RMSE(observed, modeled)
        # if U != 0:
        #     U = 1/(U*c)
        modeled = modeled_[n_o]
        observed = observed_[n_o]        
        O = A_o*self._RMSE(observed, modeled)
        # if O != 0:
        #     O = 1/(O*c)
        T = X+U*c1+O*c2
        # T = X/T+(U*c)/T+(O*c)/T
        RMSEomegaT =1/T# X/T-U/T-O/T+1 ##
        return RMSEomegaT


    def _RMSEomegaTv2(self, metric = 'runout'):
         """ Computes the mean and max RSE over the intersection area. Metric is equal to 
         the mean RSE * the intersection area + the max RSE * the over and under prediction
         areas
         """
         c = 1
         n_a = self.mg.nodes.reshape(self.mg.shape[0]*self.mg.shape[1]) # all nodes
         na = self.mg.dx*self.mg.dy
         n_o =  n_a[np.abs(self.mg.at_node['dem_dif_o']) > 0] # get nodes with scour or deposit
         n_m = n_a[np.abs(self.mg.at_node['dem_dif_m']) > 0]
  
         self.a_o = n_o*na
         self.a_m = n_m*na
         n_x =  n_o[np.isin(n_o,n_m)]#np.unique(np.concatenate([n_o,n_m])) # intersection nodes
         n_u = n_o[~np.isin(n_o,n_m)] # underestimate
         n_o = n_m[~np.isin(n_m,n_o)] # overestimate
         A_x = len(n_x)*self.mg.dx*self.mg.dy
         A_u = len(n_u)*self.mg.dx*self.mg.dy
         A_o = len(n_o)*self.mg.dx*self.mg.dy
         
         observed_ = self.mg.at_node['dem_dif_o']
         mask =  np.abs(observed_)>0 
         modeled_ = self.mg.at_node['dem_dif_m']       
         # mask_m =  np.abs(modeled_)<=0 
         # modeled_[mask_m] = np.abs(modeled_).max()
         
         modeled = modeled_[mask]
         observed = observed_[mask]
         X = A_x*self._RMSE(observed, modeled)
  
         U = A_u*self._RMxSE(observed, modeled)
     
         O = A_o*self._RMxSE(observed, modeled)
  
         T = X+U*c+O*c
         # T = X/T+(U*c)/T+(O*c)/T
         RMSEomegaT =1/T# X/T-U/T-O/T+1 ##
         return RMSEomegaT

    def _RMSE_Vd(self):
        c = 2
        observed = self.mbLdf_o[self.RMSE_metric]; 
        modeled = self.mbLdf_m[self.RMSE_metric]
        modeled[modeled == 0] = -1*np.abs((observed-modeled).mean()*c) 
        RMSE_Vd = self._RMSE(observed, modeled)
        
        nm = 'V_rms, iteration'+str(self.it)
        if self.plot_tf == True:
            plt.figure(nm)
            plt.plot(self.mbLdf_o['distance'], observed, label = 'observed')
            plt.plot(self.mbLdf_m['distance'], modeled, label = 'modeled')
            plt.title(nm)
            plt.legend()
            plt.show()
        
        return RMSE_Vd

    def _RMSE(self, observed, modeled):
        """computes the root mean square error (RMSE) between two difference 
        datasets
        """
        if modeled.size == 0:
            modeled = np.array([0])
            observed = np.array([0])
        RMSE = (((observed-modeled)**2).mean())
        return RMSE
    

    def _RMxSE(self, observed, modeled):
        """computes the root mean square error (RMSE) between two difference 
        datasets
        """
        if modeled.size == 0:
            modeled = np.array([0])
            observed = np.array([0])
        RSEmax = (((observed-modeled)**2).max())**0.5
        return RSEmax

    
    def _deposition_thickness_error(self, metric = 'max'):
        """computes the deposition thickness error (DTE)"""
        if metric == 'max':
            h_o = self.mg.at_node['dem_dif_o'].max()
            h_m = self.mg.at_node['dem_dif_m'].max()
        if metric == 'mean':
            h_o = (self.mg.at_node['dem_dif_o'][self.mg.at_node['dem_dif_o']>0].mean())/self.a_o
            h_m = (self.mg.at_node['dem_dif_m'][self.mg.at_node['dem_dif_m']>0].mean())/self.a_m            

        DTE = 1/(np.exp(np.abs(h_o-h_m)))
        return DTE

    def _prior_probability(self, value, key):
        """get prior liklihood of parameter value"""
        min_val = self.params[key][0]
        max_val = self.params[key][1]
        if self.prior_distribution == "uniform":
            prior = 1 # uniform distribution, all values equally likely.
        if self.prior_distribution == "normal":
            mean = (min_val+high_val)/2
            # assume min and max values equivalent to 4 standard deviations from the mean
            sd = (max_val-min_val)/8
            prior = norm.pdf(value, mean,sd)
        return prior


    def _candidate_value(self, selected_value, key):
        """determine the candidate parameter value as a random value from
        a normal distribution with mean equal to the presently selected value and
        standard deviation equal to the jump size"""
        min_val = self.params[key][0]
        max_val = self.params[key][1]
        
        # get the jump size
        jump_size = self.jump_size*(max_val-min_val)
        pass_ = False
        # candidate value cant be outside of the max and min values
        while pass_ is False:
            candidate_value = self.maker.normal(selected_value, jump_size)
            if (candidate_value < min_val) or (candidate_value > max_val):
                pass_ = False
            else:
                pass_ = True
        return candidate_value, jump_size


    def _adjust_jump_size(self, acceptance_ratio):
        """following LeCoz et al., 2014, adjust jump variance based on acceptance
        ratio"""

        if acceptance_ratio < self.alpha_min:
            factor = self.phi_minus**0.5
        elif acceptance_ratio > self.alpha_max:
            factor = self.phi_plus**0.5
        else:
            factor = 1
        self.jump_size = self.jump_size*factor
        self.jstracking.append(self.jump_size)


    def _MCMC_sampler(self, number_of_runs):
        """
        Markov-Chain-Monte-Carlo sampler
        Adjusts parameter values included in params dicitonary.
        Landslide average thickness (t_avg) adjusted at landslide with id = 1.
        If other landslide ids, thickness at those landslides will not be adjusted.
        """
        self.LHList = []
        ar = [] # list for tracking acceptance ratio
        # dictionaries to store trial values
        selected_value = {}
        candidate_value = {}
        prior = {}
        jump_size = {}
        for i in range(number_of_runs):
            if i == 0:
                for key in self.params:
                    selected_value[key] = self.params[key][2]
            # select a candidate valye for the jump
            for key in self.params:
                candidate_value[key], jump_size[key] = self._candidate_value(selected_value[key], key)
            # liklihood of parameter given the min and max values

            prior_t = 1
            for key in self.params:
                prior_ = self._prior_probability(candidate_value[key], key)
                prior[key] = prior_
                prior_t = prior_t*prior_ # likelihood of all parameter values

            # update instance parameter values
            for key in self.params:
                if key == 'SD':
                    self.MWRu.SD = candidate_value[key]
                if key == 'cs':
                    self.MWRu.cs = candidate_value[key]
                if key == 'slpc':
                    self.MWRu.slpc = candidate_value[key] # slpc is a list
                if key == "t_avg":
                    # adjust thickness of landslide with id = 1
                    self.MWRu._grid.at_node['soil__thickness'][self.MWRu._grid.at_node['mass__wasting_id'] == 1] = candidate_value[key]
            
            # run simulation with updated parameter
            self.it = i
            self._simulation()
            if self.method == "omega":
                omegaT = 1+self._omegaT(metric = "runout")
                candidate_posterior = prior_t*omegaT
            elif self.method == "RMSE":
                self.mbLdf_m = self._channel_profile_deposition("modeled")
                # determine RMSE metric
                observed = self.mbLdf_o[self.RMSE_metric]; modeled = self.mbLdf_m[self.RMSE_metric]
                RMSE_Vd = self._RMSE(observed, modeled)

                observed = self.mg.at_node['dem_dif_o'][self.mbLdf_o['node']] 
                modeled = self.mg.at_node['dem_dif_m'][self.mbLdf_m['node']]
                RMSE_pf = self._RMSE(observed, modeled)
                
                observed = self.mg.at_node['dem_dif_o'] 
                modeled = self.mg.at_node['dem_dif_m']
                RMSE_map = self._RMSE(observed, modeled)
                # determine psoterior likilhood: product of RMSE, omegaT and prior liklihood
                candidate_posterior = prior_t*(1/RMSE_Vd)*(1/RMSE_pf)*(1/RMSE_map)
            elif self.method == "both":
                # get modeled deposition profile
                self.mbLdf_m = self._channel_profile_deposition("modeled")
                # determine RMSE metric
                # observed = self.mbLdf_o[self.RMSE_metric]; modeled = self.mbLdf_m[self.RMSE_metric]
                # RMSE_Vd = self._RMSE(observed, modeled)
                RMSE_Vd = self._RMSE_Vd()

                
                observed = self.mg.at_node['dem_dif_o'][self.mbLdf_o['node']] 
                modeled = self.mg.at_node['dem_dif_m'][self.mbLdf_m['node']]
                RMSE_pf = self._RMSE(observed, modeled)
                observed = self.mg.at_node['dem_dif_o'] 
                modeled = self.mg.at_node['dem_dif_m']
                RMSE_map = self._RMSE(observed, modeled)
                # determine deposition overlap metric, omegaT
                omegaT = 1+self._omegaT(metric = self.omega_metric)
                # determine the difference in thickness
                RMSEomegaT = self._RMSEomegaT(metric = self.omega_metric)
                DTE = self._deposition_thickness_error()
                # determine psoterior likilhood: product of RMSE, omegaT and prior liklihood
                candidate_posterior = prior_t*(1/RMSE_Vd)*omegaT*RMSEomegaT#*(1/RMSE_pf)*(1/RMSE_map)#*DTE
                # candidate_posterior = (1/RMSE_map)
            # decide to jump or not to jump
            if i == 0:
                acceptance_ratio = 1 # always accept the first candidate vlue
            else:
                acceptance_ratio = min(1, candidate_posterior/selected_posterior)
           # pick a random number between 0 and 1 assuming a uniform distribution
           # if number less than acceptance ratio, go with new parameter value.
           # if larger than ratio go with old parameter value
           # for first jump, probability willl always be less than or equal to
           # acceptance ratio (1)
            rv = self.maker.uniform(0,1,1)
            if rv < acceptance_ratio:
                selected_posterior = candidate_posterior
                for key in self.params:
                    selected_value[key] = candidate_value[key]
                msg = 'jumped to new value'; ar.append(1)
            else :
                selected_value = selected_value
                msg = 'staying put'; ar.append(0)

            p_table = []
            p_nms = []
            for key in self.params:
                p_table = p_table+[jump_size[key], candidate_value[key],selected_value[key]]
                p_nms = p_nms+['jump_size_'+key, 'candidate_value_'+key, 'selected_value_'+key]
            if self.method == "omega":
                self.LHList.append([i, self.MWRu.c, prior_t, omegaT,candidate_posterior,acceptance_ratio, rv, msg, selected_posterior]+p_table)
            elif self.method == "RMSE":
                self.LHList.append([i, self.MWRu.c, prior_t, 1/RMSE_Vd,1/RMSE_pf,1/RMSE_map,candidate_posterior,acceptance_ratio, rv, msg, selected_posterior]+p_table)
            elif self.method == "both":
                self.LHList.append([i, self.MWRu.c, prior_t, 1/RMSE_Vd,1/RMSE_pf,1/RMSE_map,DTE,RMSEomegaT,omegaT,candidate_posterior,acceptance_ratio, rv, msg, selected_posterior]+p_table)

            # adjust jump size every N_cycles
            if i%self.N_cycles == 0:
                mean_acceptance_ratio = np.array(ar).mean()
                self._adjust_jump_size(mean_acceptance_ratio)
                ar = [] # reset acceptance ratio tracking list

            print('MCMC iteration: {}, likelihood:{}, acceptance ratio:{}, random value:{},{}'.format(
                              i, np.round(candidate_posterior, decimals = 5),
                              np.round(acceptance_ratio, decimals = 3),
                              np.round(rv, decimals = 3), 
                              msg))

        self.LHvals = pd.DataFrame(self.LHList)
        if self.method == "omega":
            self.LHvals.columns = ['iteration', 'model interations', 'prior', 'omegaT', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms
        elif self.method == "RMSE":
            self.LHvals.columns = ['iteration', 'model interations', 'prior', '1/RMSE','1/RMSE p','1/RMSE m', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms
        elif self.method == "both":
            self.LHvals.columns = ['iteration', 'model interations', 'prior', '1/RMSE','1/RMSE p','1/RMSE m', 'DTE', 'RMSEomegaT', 'omegaT', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms

        self.calibration_values = self.LHvals[self.LHvals['selected_posterior'] == self.LHvals['selected_posterior'].max()] # {'SD': selected_value_SD, 'cs': selected_value_cs}


def profile_distance(mg, xsd):
    """small function to get distance between profile nodes. Nodes must be ordered
    from downstream to upstream (lowest to highest)"""

    x = mg.node_x[xsd]
    y = mg.node_y[xsd]

    def path(x,y):
        return(((x[0]-x[1])**2+(y[0]-y[1])**2)**.5)
    dist = [0]
    for i in range(len(x)-1):
        x_ = x[i:i+2]
        y_ = y[i:i+2]
        dist.append(dist[i]+path(x_,y_))
    return np.array(dist)


def profile_plot(mg, xsd, ef = 2, xlim = None, ylim = None, aspect = None, figsize = None, fs = 8):
    """function for plotting profile of observed pre and post-failure topography"""

    dist = profile_distance(mg, xsd)

    y_dem = mg.at_node['topographic__elevation'][xsd]
    y_demdf_o = y_dem+mg.at_node['dem_dif_o'][xsd]*ef

    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(2,5))
    plt.plot(dist,y_dem,'k-', alpha = .66, linewidth = 1)
    plt.plot(dist,y_demdf_o,'k--', alpha = 0.66, linewidth = 1, label = 'observed deposition and scour')
    plt.grid(alpha = 0.5)
    plt.legend(fontsize = fs)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fs)
    if aspect:
        ax.set_aspect(aspect)
    else:
        ax.set_aspect(2)
    if xlim:
        plt.xlim([xlim])
    if ylim:
        plt.ylim([ylim])


def view_profile_nodes(mg, xsd, field = 'dem_dif_o', clim = None):
    """function for plotting profile nodes in plan view, over any node field"""

    plt.figure(figsize = (5,5))
    imshow_grid_at_node(mg, field,cmap = 'RdBu_r')
    x_mn = mg.node_x[np.abs(mg.at_node['dem_dif_o']) > 0].min()
    x_mx = mg.node_x[np.abs(mg.at_node['dem_dif_o']) > 0].max()
    y_mn = mg.node_y[np.abs(mg.at_node['dem_dif_o']) > 0].min()
    y_mx = mg.node_y[np.abs(mg.at_node['dem_dif_o']) > 0].max()
    plt.xlim([x_mn, x_mx])
    plt.ylim([y_mn, y_mx])
    if clim:
        plt.clim(clim)
    plt.plot(mg.node_x[xsd],mg.node_y[xsd],'g.', alpha = 0.5, markersize = 3, label = 'profile nodes')
    plt.legend(fontsize = 7)
    plt.show()
