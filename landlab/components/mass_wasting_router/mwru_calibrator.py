# setup
import os
import numpy as np
from scipy.stats import norm
import pandas as pd

import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import SinkFillerBarnes, FlowAccumulator, FlowDirectorMFD
from landlab.components.mass_wasting_router.mass_wasting_runout import (MassWastingRunout,
                                                                       shear_stress_grains,
                                                                       shear_stress_static,
                                                                       erosion_rate,
                                                                       erosion_coef_k)
from landlab import imshow_grid

from landlab import imshow_grid_at_node

class MWRu_calibrator():
    """an adaptive Markov Chain Monte Carlo calibration utitlity for calibrating
    MassWastingRunout to observed landslide runout and/or scour and deposition

    author: Jeff Keck
    
    TODO:
    is _check_E_lessthan_lambda_times_qsc needed?
    save data for plots
    """

    def __init__(self,
                 MassWastingRunout,
                 params,
                 profile_calib_dict,
                 prior_distribution = "uniform",
                 method = 'both',
                 omega_metric = "runout",
                 RMSE_metric = "Vu",
                 jump_size = 0.2,
                 N_cycles = 10,
                 plot_tf = True,
                 seed = None,
                 alpha_min = 0.1,# 0.23 #0.1
                 alpha_max = 0.5,# 0.44#0.5
                 phi_minus = 0.9,
                 phi_plus = 1.1,
                 qsc_constraint = True):
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
                runout_profile_nodes : np.array
                    np.array of node id numbers that define profile on raster model grid
                runout_profile_distance : np.array
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

        self.MWR = MassWastingRunout
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
        self.qsc_constraint = qsc_constraint 


    def __call__(self, max_number_of_runs = 50):
        """instantiate the class"""
        if (self.method == "both") or (self.method == "RMSE"):
            # get profile of observed total flow
            self.mbLdf_o = self._channel_profile_deposition("observed")
            # compute total mobilized volume
            self.TMV = (-1*(self.mg.at_node['dem_dif_o'][self.mg.at_node['dem_dif_o']<0]).sum())*self.mg.dx*self.mg.dy
            # mean total flow
            self.Qtm = self.mbLdf_o[self.RMSE_metric].mean()
            
        self._MCMC_sampler(max_number_of_runs)

    
    def _maker(self,seed):
        """prepares the np random generator"""
        
        self.maker = np.random.RandomState(seed=seed)


    def _simulation(self):
        """run the model, determine the cumulative modeled deposition along the
        channel centerline"""
        # reset the dem to the initial dem and soil to initial soil depth
        self.MWR.grid.at_node['topographic__elevation'] = self.MWR.grid.at_node['topographic__initial_elevation'].copy()
        self._update_topographic_slope()
        self.MWR.grid.at_node['energy__elevation'] = self.MWR.grid.at_node['topographic__elevation'].copy()
        self.MWR.grid.at_node['soil__thickness'] = self.initial_soil_depth.copy()
        self.mg.at_node['disturbance_map'] = np.full(self.mg.number_of_nodes, False)
        if self.mg.has_field("particle__diameter", at="node"):
            self.mg.at_node["particle__diameter"] = self.initial_particle_diameter.copy()
        
            
        # run the model
        self.MWR.run_one_step(run_id = 0)
        # create the modeldiff_m field
        diff = self.mg.at_node['topographic__elevation'] - self.mg.at_node['topographic__initial_elevation']
        self.mg.at_node['dem_dif_m'] = diff
        
        self.dem_dif_m_dict[self.it] = self.mg.at_node['dem_dif_m']
        if self.plot_tf == True:
            plt.figure('iteration'+str(self.it))
            imshow_grid(self.mg,"dem_dif_m",cmap = 'RdBu_r')
            plt.title("it:{}, slpc:{}, qsc:{}, k:{}".format(self.it, self.MWR.slpc, self.MWR.qsc, self.MWR.k ))
            plt.clim(-1,1)
            plt.show()
                        

    def _update_topographic_slope(self):
        """updates the topographic__slope and flow directions fields using the 
        topographic__elevation field"""
        fd = FlowDirectorMFD(self.mg, surface="topographic__elevation", diagonals=True,
                partition_method = self.MWR.routing_partition_method)
        fd.run_one_step()


    def _channel_profile_deposition(self, datatype):
        """determines deposition patterns along the channel profile:
        """
        mg = self.mg
        dA = mg.dx*mg.dx
        el_l = self.pcd['el_l']
        el_h = self.pcd['el_h']
        
        runout_profile_nodes = self.pcd['runout_profile_nodes']
        
        # extract all nodes between lower and upper profile limits
        # use observed runout dem to get elevations
        demd_ = mg.at_node['dem_dif_o']
        dem_ = mg.at_node['topographic__initial_elevation']+demd_
        pel = dem_[runout_profile_nodes] # elevation of profile nodes 
        mask = (pel > el_l) & (pel < el_h)
        runout_profile_nodes = runout_profile_nodes[mask]
        runout_distance = self.pcd['runout_profile_distance'][mask]
        node_slope = mg.at_node['topographic__steepest_slope']
        cL = self.pcd['cL']
               
        def cv_mass_change(cn):

            el = dem[cn]

            # at point cn along the profile, produces 4 different metrics:

            # metric 1, Vt: cumulative aggradation downslope of point cn
            dp = demd[demd>0]; dem_dp = dem[demd>0]
            dpe = np.nansum(dp[(dem_dp<el)&(dem_dp>=el_l)])
            Vt = dpe*dA # multiply by cell area to get volume
            
            # metric 2, dV: cumulative scour upstream of point cn
            sc = demd[demd<0]; dem_sc = dem[demd<0]
            sce = np.nansum(sc[(dem_sc>el)&(dem_sc<=el_h)])
            dV = sce*dA # multiply by cell area to get volume

            # metric 3, Vd: cumulative erosion and aggradation downslope of point cn
            dd = np.nansum(demd[(dem<=el) & (dem>=el_l)])
            Vd = dd*dA
            
            # metric 4, Vu: cumulative flow volume past point cn
            #               equivalent to the cumulative erosion and aggradation 
            #               upslope of point cn * -1.
            du = np.nansum(demd[(dem>=el) & (dem<=el_h)])*-1 
            Vu = du*dA

            # get channel characteristics
            # same, nut get node distance
            rd = runout_distance[runout_profile_nodes == cn]
            # node slope
            cns = node_slope[cn]
            #return Vu, Vd, dV, Vt, cns, cn, cd, rd, el 
            return Vu, Vd, dV, Vt, cns, cn, rd, el

        dem_m = mg.at_node['topographic__elevation']
        if datatype == "modeled":
            demd = mg.at_node['dem_dif_m']
            dem = mg.at_node['topographic__elevation'] 
        elif datatype == "observed":
            demd = mg.at_node['dem_dif_o']
            dem = mg.at_node['topographic__initial_elevation']+demd

        mbL = []
        for cn in runout_profile_nodes:
            mbL.append(cv_mass_change(cn))
        mbLdf = pd.DataFrame(mbL)
        mbLdf.columns = ['Vu','Vd','dV','Vt', 'slope', 'node','runout_distance','elevation']
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
        omegaT = X/T-U/T-O/T+1
        return omegaT


    def _Vse(self, metric = 'runout'):
        """ determine the volumetric square error, normalized by the total mobilized
        volume.
        """
        
        c1 = 1; c2 = 1
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
        CA = self.mg.dx*self.mg.dy
        observed_ = self.mg.at_node['dem_dif_o']
        mask =  np.abs(observed_)>0 
        modeled_ = self.mg.at_node['dem_dif_m']       

        modeled = modeled_[mask]
        observed = observed_[mask]
        X = self._SE(observed, modeled)*CA**2
        modeled = modeled_[n_u]
        observed = observed_[n_u]
        U = self._SE(observed, modeled)*CA**2
        modeled = modeled_[n_o]
        observed = observed_[n_o]        
        O = self._SE(observed, modeled)*CA**2        
        Vse = (X+c1*U+c2*O)/(self.TMV**2)
        
        return Vse

    def _MSE_Qt(self):

        observed = self.mbLdf_o[self.RMSE_metric] 
        modeled = self.mbLdf_m[self.RMSE_metric]
        self.trial_qs_profiles[self.it] = modeled
        CA = self.mg.dx*self.mg.dy
        MSE_Qt = self._MSE(observed, modeled)/(self.Qtm**2)
        
        nm = 'V_rms, iteration'+str(self.it)
        if self.plot_tf == True:
            plt.figure(nm)
            plt.plot(self.mbLdf_o['runout_distance'], observed, label = 'observed')
            plt.plot(self.mbLdf_m['runout_distance'], modeled, label = 'modeled')
            plt.title(nm)
            plt.legend()
            plt.show()
        
        return MSE_Qt

    def _MSE(self, observed, modeled):
        """computes the root mean square error (RMSE) between two difference 
        datasets
        """
        if modeled.size == 0:
            modeled = np.array([0])
            observed = np.array([0])
        MSE = (((observed-modeled)**2).mean())
        return MSE   
    
    
    def _SE(self, observed, modeled):
        """computes the cumulative square error (RMSE) between two difference 
        datasets
        """
        if modeled.size == 0:
            modeled = np.array([0])
            observed = np.array([0])
            print("NO MODELED VALUES")
        SE = ((observed-modeled)**2).sum()
        return SE
    
    

    def _RMSE(self, observed, modeled):
        """computes the root mean square error (RMSE) between two difference 
        datasets
        """
        if modeled.size == 0:
            modeled = np.array([0])
            observed = np.array([0])
        RSEmax = (((observed-modeled)**2).mean())**0.5
        return RSEmax


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


    def _determine_erosion(self, value, solve_for = 'k'):
        """
        determine k using (36) or E_l using (35)  
        
        Parameters
        ----------
        ros : density of grains in runout [kg/m3]
        vs  : volumetric ratio of solids to matrix [m3/m3]
        h   : depth [m] - typical runout depth
        s   : slope [m/m] - average slope of erosion part of runout path
        eta : exponent of scour model (equation 7) - 
        E_l : average erosion rate per unit length of runout [m/m]
        dx  : cell width [m]
        slpc: average slope at which positive net deposition occurs
        Dp  : representative grain size [m]
    
        Returns
        -------
        k
    
        """
    
        rodf = self.MWR.vs*self.MWR.ros+(1-self.MWR.vs)*self.MWR.rof
        theta = np.arctan(self.MWR.s)
        
           
        if self.MWR.grain_shear:
            # use mean particle diameter in runout profile for representative grain size used to determine erosion rate
            Dp = self.MWR._grid.at_node['particle__diameter'][self.pcd['runout_profile_nodes']].mean()
            tau = shear_stress_grains(self.MWR.vs,
                                                   self.MWR.ros,
                                                   Dp,
                                                   self.MWR.h,
                                                   self.MWR.s,
                                                   self.MWR.g)
            
            # # use mean particle diameter in runout profile for representative grain size used to determine erosion rate
            # _Dp = self.MWR._grid.at_node['particle__diameter'][self.pcd['runout_profile_nodes']].mean()
            
            # print('grain-inertia')
            # # phi = np.arctan(self.MWR.slpc)
            # phi = np.arctan(0.32)
            
            # # inertial stresses
            # us = (self.MWR.g*self.MWR.h*self.MWR.s)**0.5
            # u = us*5.75*np.log10(self.MWR.h/_Dp)
            
            # dudz = u/self.MWR.h
            # Tcn = np.cos(theta)*self.MWR.vs*self.MWR.ros*(_Dp**2)*(dudz**2)
            # tau = Tcn*np.tan(phi)
        else:
            
            tau = shear_stress_static(self.MWR.vs,
                                      self.MWR.ros,
                                      self.MWR.rof,
                                      self.MWR.h,
                                      self.MWR.s,
                                      self.MWR.g)
            # print('quasi-static')
            # tau = rodf*self.MWR.g*self.MWR.h*(np.sin(theta))
    
        if solve_for == 'k':
            # k = value*self.mg.dx/(tau**self.MWR.eta)
            k = erosion_coef_k(value,
                               tau,
                               self.MWR.eta,
                               self.mg.dx)
            return_value = k
        elif solve_for == 'E_l':
            E_l = erosion_rate(value,tau,
                               self.MWR.eta,
                               self.mg.dx)
            # E_l = (value*tau**self.MWR.eta)/self.mg.dx
            return_value = E_l
        return return_value


    def _check_E_lessthan_lambda_times_qsc(self, candidate_value, jump_size, selected_value):
        """A check that average erosion depth (E) does not exceed flux constraint (E must be less than qsc*lambda)
        if average E>qsc, resample k until average E<(qsc*lambda) OR resample qsc until (qsc*lambda)>E"""
        # this may not be needed anymore
        equivalent_E = self._determine_erosion(self.MWR.k, solve_for = 'E_l')*self.mg.dx
        
        _lambda = 1 # when slpc is low, model is unstable when ~E>qsc.
        if self.MWR.slpc>=0.02: # when slpc is low (<0.01), model is unstable when ~E>(10*qsc)
            _lambda = 10
            
        if equivalent_E>self.MWR.qsc*_lambda:
            
            # if k is a calibration parameter, first apply constraint to k, since model is very sensitive to qsc
            if self.params.get('k'):
                # check if minimum k range is low enough
                equivalent_E_min = self._determine_erosion(self.params['k'][0], solve_for = 'E_l')*self.mg.dx
                if equivalent_E_min>self.MWR.qsc*_lambda:
                    msg = "minimum possible k value results in too much erosion"
                    raise ValueError(msg)                    
                else: # if low enough, randomly select an k value until the erosion equivalent is less than qsc
                    _pass = False
                    _i_ = 0
                    while _pass is False:
                        candidate_value['k'], jump_size['k'] = self._candidate_value(selected_value['k'], 'k')
                        self.MWR.k = candidate_value['k']
                        equivalent_E = self._determine_erosion(self.MWR.k, solve_for = 'E_l')*self.mg.dx
                        if equivalent_E < self.MWR.qsc*_lambda:
                            _pass = True
                            print('resampled, E<qsc')
                        _i_+=1; 
                        if _i_%1000 == 0:
                            print('after {} runs, all sampled k values are too large, decrease the lower range of k'.format(_i_))
            # if k is not a calibration parameter (k is fixed), then adjust qsc to meet constraint
            elif self.params.get('qsc'):
                # check if maximum qsi range is high enough
                equivalent_E = self._determine_erosion(self.MWR.k, solve_for = 'E_l')*self.mg.dx
                if equivalent_E > self.params['qsc'][1]*_lambda:
                    msg = "maximum possible qsc value is less than erosion caused by k value"
                    raise ValueError(msg)   
                else: # if high enough, randomly select a qsi value until that value exceeds the erosion equivalent of the k value
                    _pass = False
                    _i_ = 0
                    while _pass is False:
                        candidate_value['qsc'], jump_size['qsc'] = self._candidate_value(selected_value['qsc'], 'qsc')
                        self.MWR.qsc = candidate_value['qsc']
                        if equivalent_E < self.MWR.qsc*_lambda:
                            _pass = True
                            print('resampled, qsc>E')
                            _i_+=1; 
                            if _i_%1000 == 0:
                                print('after {} runs, all sampled qsc values are to small, increase the upper range of qsc'.format(_i_))
                
            else:
                msg = "minimum possible k value results in too much erosion"
                raise ValueError(msg)  
        else:
            print('E<qsc')

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
        self.trial_qs_profiles = {}
        self.trial_runout_maps = {}
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

            prior_t = 1  # THIS NEEDS TO GO AFTER _check_E_lessthan_lambda_times_qsc
            for key in self.params:
                prior_ = self._prior_probability(candidate_value[key], key)
                prior[key] = prior_
                prior_t = prior_t*prior_ # likelihood of all parameter values

            # update instance parameter values
            for key in self.params:
                if key == 'qsc':
                    self.MWR.qsc = candidate_value[key]
                if key == 'k':
                    self.MWR.k = candidate_value[key]
                if key == 'slpc':
                    self.MWR.slpc = candidate_value[key] # slpc is a list
                if key == "t_avg":
                    # adjust thickness of landslide with id = 1
                    self.MWR._grid.at_node['soil__thickness'][self.MWR._grid.at_node['mass__wasting_id'] == 1] = candidate_value[key]
            
            if self.qsc_constraint:
                self._check_E_lessthan_lambda_times_qsc(candidate_value, jump_size, selected_value)
            
            # run simulation with updated parameter
            self.it = i
            self._simulation()
            if self.method == "omega":
                omegaT = self._omegaT(metric = "runout")
                candidate_posterior = prior_t*omegaT
            elif self.method == "both":
                # get modeled deposition profile
                self.mbLdf_m = self._channel_profile_deposition("modeled")

                # MSE of Qt
                MSE_Qt = self._MSE_Qt()
                
                # omegaT
                omegaT = self._omegaT(metric = self.omega_metric)
                
                # volumetric square error
                Vse = self._Vse(metric = self.omega_metric)
                
                # DTE = self._deposition_thickness_error()
                
                # determine psoterior likilhood: product of RMSE, omegaT and prior liklihood
                candidate_posterior = prior_t*omegaT*(1/MSE_Qt)*(1/Vse)#*(1/RMSE_pf)*(1/RMSE_map)#*DTE
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
                self.LHList.append([i, self.MWR.c, prior_t, omegaT,candidate_posterior,acceptance_ratio, rv, msg, selected_posterior]+p_table)
            elif self.method == "RMSE":
                self.LHList.append([i, self.MWR.c, prior_t, MSE_Qt,1/RMSE_pf,1/RMSE_map,candidate_posterior,acceptance_ratio, rv, msg, selected_posterior]+p_table)
            elif self.method == "both":
                # self.LHList.append([i, self.MWR.c, self.TMV, self.Qtm, prior_t, omegaT, MSE_Qt**0.5, Vse**0.5, RMSE_pf, RMSE_map, DTE, candidate_posterior, acceptance_ratio, rv, msg, selected_posterior]+p_table)
                self.LHList.append([i, self.MWR.c, self.TMV, self.Qtm, prior_t, omegaT, MSE_Qt**0.5, Vse**0.5, candidate_posterior, acceptance_ratio, rv, msg, selected_posterior]+p_table)

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
            self.LHvals.columns = ['iteration', 'model iterations', 'prior', 'omegaT', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms
        elif self.method == "RMSE":
            self.LHvals.columns = ['iteration', 'model iterations', 'prior', '1/RMSE','1/RMSE p','1/RMSE m', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms
        elif self.method == "both":
            # self.LHvals.columns = ['iteration', 'model iterations', 'total_mobilized_volume', 'obs_mean_total_flow',  'prior', 'omegaT','MSE_Qt^1/2','Vse^1/2', 'RMSE_pf', 'RMSE_map', 'DTE', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms
            self.LHvals.columns = ['iteration', 'model iterations', 'total_mobilized_volume', 'obs_mean_total_flow',  'prior', 'omegaT','MSE_Qt^1/2','Vse^1/2', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms

        self.calibration_values = self.LHvals[self.LHvals['selected_posterior'] == self.LHvals['selected_posterior'].max()] # {'qsc': selected_value_SD, 'k': selected_value_cs}


def plot_node_field_with_shaded_dem(mg, field, save_name= None, plot_name = None, figsize = (12,9.5), 
                                    cmap = 'terrain', fontsize = 12, alpha = 0.5, cbr = None,  norm = None, allow_colorbar = True,
                                    var_name = None, var_units = None):
    if plot_name is None:
        plt.figure(field,figsize= figsize)
    else:
        plt.figure(plot_name,figsize= figsize)   
    imshow_grid_at_node(mg, 'hillshade', cmap='Greys',
                      grid_units=('coordinates', 'coordinates'),
                      shrink=0.75, var_name=None, var_units=None,output=None,allow_colorbar=False,color_for_closed= 'white')
    fig = imshow_grid_at_node(mg, field, cmap= cmap,
                      grid_units=('coordinates', 'coordinates'),
                      shrink=0.75, var_name=var_name, var_units=var_units,alpha = alpha,output=None,
                      color_for_closed= None, color_for_background = None,
                      norm = norm,allow_colorbar=allow_colorbar)
    
    plt.xlim([mg.x_of_node[mg.core_nodes].min()-20, mg.x_of_node[mg.core_nodes].max()+20])
    plt.ylim([mg.y_of_node[mg.core_nodes].min()-20, mg.y_of_node[mg.core_nodes].max()+20])
    
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.title(plot_name)
    if cbr is None:
        r_values = mg.at_node[field][mg.core_nodes]
        plt.clim(r_values.min(), r_values.max())
    else:
        
        plt.clim(cbr[0], cbr[1])

    if save_name is not None:
        plt.savefig(save_name+'.png', dpi = 300, bbox_inches='tight')

def define_profile_nodes(mg):
    """begininng from above the mass wasting source area, map the runout profile. 
    Mass wasting runout flow volume will be computed at each node along the runoutprofile"""
    
    # create to plot from which profile will be mapped
    plot_node_field_with_shaded_dem(mg,field = 'drainage_area', fontsize = 10,
                                    cmap = 'YlGnBu',alpha = .35,figsize = (12,12),
                                    allow_colorbar = False)
    lsn = np.hstack(mg.nodes)[mg.at_node['mass__wasting_id']>0]
    plt.scatter(mg.node_x[lsn], mg.node_y[lsn], 
           marker = '.', color = 'k',alpha = 1, label = 'initial mass wasting area nodes')
    
    
    def title_message(s):
        print(s)
        plt.title(s, fontsize=12)
        plt.draw()
    
    title_message('   Beginning from the upslope edge of the landslide crown or just upstream of the initial mass wasting\n \
        material, map the centerline of the potential runout path. Use interactive plot to zoom and scroll \n \
        (right click to zoom out). Move mouse curser, press "space" to add point, "delete" to remove point \n \
        and "enter" to finish.')
                  
    pts = np.asarray(plt.ginput(n=0, timeout=0, 
                                mouse_add=None, 
                                mouse_pop=None, 
                                mouse_stop=None))
    plt.plot(pts[:, 0], pts[:, 1], 'w--', lw=1)
    plt.plot(pts[:, 0], pts[:, 1], 'rx', lw=2)
    # grid node coordinates, translated to origin of 0,0
    gridx = mg.node_x#-grid.node_x[0] 
    gridy = mg.node_y#-grid.node_y[0]
    # extent of each cell in grid        
    ndxe = gridx+mg.dx/2
    ndxw = gridx-mg.dx/2
    ndyn = gridy+mg.dy/2
    ndys = gridy-mg.dy/2
    nodes = mg.nodes.reshape(mg.shape[0]*mg.shape[1],1)
    
    def _coordinates_to_rmg_nodes(pts):
        '''convert a list of coordinates that describe the map (x and y) location 
        of a profile to the coincident raster model grid nodes located along that profile
        '''
        Lnodelist = [] # list of lists of all nodes that coincide with each link
        Ldistlist = []
        Lxy= [] # list of all nodes the coincide with the profile
        k = 0 # node number
        cdist = 0 # cummulative distance along profile
        while k < len(pts)-1:
            x0 = pts[k][0] #x and y of downstream link node
            y0 = pts[k][1]
            x1 = pts[k+1][0] #x and y of upstream link node
            y1 = pts[k+1][1]
            # create 1000 points along domain of link
            X = np.linspace(x0,x1,1000)
            Xs = X-x0 # change begin value to zero
            # determine distance from upstream node to each point
            # y value of points
            if Xs.max() ==0: # if a vertical link (x is constant)
                vals = np.linspace(y0,y1,1000)
                dist = vals-y0 # distance along link, from downstream end upstream
                dist = dist.max()-dist #distance from updtream to downstream
            else: # all their lines
                vals = y0+(y1-y0)/(x1-x0)*(Xs)
                dist = ((vals-y0)**2+Xs**2)**.5
            # match points along link (vals) with grid cells that coincide with link
            nodelist = [] # list of nodes along link
            distlist = [] # list of distance along link corresponding to node
            for i,v in enumerate(vals):
                x = X[i]
                mask = (ndyn>=v) & (ndys<=v) & (ndxe>=x) & (ndxw<=x)  # mask - use multiple boolean tests to find cell that contains point on link
                node = nodes[mask] # use mask to extract node value
                if node.shape[0] > 1:
                    node = np.array([node[0]])
                # create list of all nodes that coincide with linke
                if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                    nodelist.append(node[0][0])
                    distlist.append(dist[i]+cdist)
                    xy = {'linkID':k,
                        'node':node[0][0],
                          'x':gridx[node[0][0]],
                          'y':gridy[node[0][0]],
                          'dist':dist[i]+cdist}
                    Lxy.append(xy)
            if k+1 == len(pts)-1:
                Lnodelist.append(nodelist)
                Ldistlist.append(distlist)
            else:
                Lnodelist.append(nodelist[:-1])
                Ldistlist.append(distlist[:-1])
            pnodes = np.hstack(Lnodelist)
            pnodedist = np.hstack(Ldistlist)
            k+=1
            cdist = cdist + dist.max()
        return (pnodes, pnodedist, Lxy)
    pnodes, pnodedist, Lxy = _coordinates_to_rmg_nodes(pts)
    
    plt.plot(mg.node_x[pnodes],mg.node_y[pnodes],'g.', alpha = 0.5, markersize = 3, label = 'profile nodes')
    plt.legend()

    return pnodes, pnodedist, Lxy 


def profile_plot(mg, pnodes, pnodedist, ef = 2, xlim = None, ylim = None, aspect = None, figsize = None, fs = 8):
    """function for plotting profile of observed pre- and post-failure topography"""
    y_dem = mg.at_node['topographic__elevation'][pnodes]
    y_demdf_o = y_dem+mg.at_node['dem_dif_o'][pnodes]*ef

    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(3,5))
    plt.plot(pnodedist,y_dem,'k-', alpha = .66, linewidth = 1, label = 'pre-runout-DEM')
    plt.plot(pnodedist,y_demdf_o,'k--', alpha = 0.66, linewidth = 1, label = 'post-runout-DEM')
    plt.grid(alpha = 0.5)
    plt.legend(fontsize = fs)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fs)
    plt.xlabel('horizontal distance, [m]',fontsize = fs)
    plt.ylabel('vertical distance, [m]',fontsize = fs)
    plt.title('observed pre- and post- runout DEM', fontsize = fs)
    if aspect:
        ax.set_aspect(aspect)
    else:
        ax.set_aspect(2)
    if xlim:
        plt.xlim([xlim])
    if ylim:
        plt.ylim([ylim])

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