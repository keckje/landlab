# setup
import os
import numpy as np
from scipy.stats import norm
import pandas as pd

import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import SinkFillerBarnes, FlowAccumulator, FlowDirectorMFD
from landlab.components.mass_wasting_runout.mass_wasting_runout import (MassWastingRunout,
                                                                       shear_stress_grains,
                                                                       shear_stress_static,
                                                                       erosion_rate,
                                                                       erosion_coef_k)
from landlab import imshow_grid

from landlab import imshow_grid_at_node

class MWRu_calibrator():
    """an adaptive Markov Chain Monte Carlo calibration utility for calibrating
    MassWastingRunout to observed landslide runout and/or scour and deposition
    
    TODO:   See other beyesian calibration apporaches that use more than two variables
            to figure out ways for visualizing results.
            
            Add an option for jumping in one parameter direction at a time rather
            than all parameters as presently implemented
            
            Make jump_size input a vector, if len(jump_size) == 1, all variables use same value
            otherwise, len(jump_size) == len(# of parameters) and each parameter is adjusted
                                             using its own jump size
            
            acceptance rate for each parameter is tracked and used to adjust the jumpsize of each parameter
            
    
    
    
    is _check_E_lessthan_lambda_times_qsc needed -  yes, but not as critical. Keeping
    will allow the user to exclude watery-like runout behavior
    save data for plots
    

    """

    def __init__(self,
                 MassWastingRunout,
                 MCMC_adjusted_parameters,
                 el_l,
                 el_h,
                 runout_profile_nodes,
                 runout_profile_distance,
                 prior_distribution = "uniform",
                 calibration_method = 'extent_and_sediment',
                 extent_metric = "entire_runout_extent",
                 profile_metric = "Qs",
                 jump_size = 0.2,
                 N_cycles = 10,
                 plot_tf = True,
                 seed = None,
                 alpha_min = 0.1,# 0.23 #0.1
                 alpha_max = 0.5,# 0.44#0.5
                 phi_minus = 0.9,
                 phi_plus = 1.1,
                 qsc_constraint = True,
                 show_progress = False):
        
        """Instantiate a MWR_calibrator class
        
        Parameters
        ----------
        MassWastingRunout: instantiated mass wasting runout that contains a
            raster model grid that contains the fields:
                topographic__elevation
                topographic__steepest_slope
                DoD_o

        MCMC_adjusted_parameters : dictionary
            Defines the parameter space sampled by the MCMC algorithm.
            Each key is one of the adjustable parameters (presently qsc, k, slpc and t_avg)
            The value for each key is a list of the min, max and optimal parameter value, in that order. 

        el_l : float
            lower limit of elevation range included in analysis.
            
        el_h : float
            upper limit of elevation range included in analysis.
            
        runout_profile_nodes : np.array
            np.array of node id numbers that define profile on raster model grid
            
        runout_profile_distance : np.array
            np.array of distance at each node from outlet, output from channel profiler
                    
        prior_distribution: string
            can be "uniform" or "normal"

        calibration_method: string
            can be "extent_only" or "extent_and_sediment"

        extent_metric: string
            can be "entire_runout_extent", "deposition_extent_only" or "scour_extent_only". Default is "entire_runout_extent"

        profile_metric: string
            can be 'Qs','EA_d','E_u' or 'A_d'. Default is "Qs"

        jump_size: float
            standard deviation of jump size for MCMC algorithm expressed as ratio of
            the jump size to the range between the minimum and maximum parameter values
            default: jump_size = 0.2

        N_cycles: int
            Number of iterations between updates to the jump size based on MCMC acceptance ratio
            
        """

        self.MWR = MassWastingRunout
        self.mg = MassWastingRunout.grid
        self.params = MCMC_adjusted_parameters
        self.el_l = el_l
        self.el_h = el_h
        self.runout_profile_nodes = runout_profile_nodes
        self.runout_profile_distance = runout_profile_distance
        prior_distribution = str(prior_distribution).lower()
        if prior_distribution not in {"uniform", "normal"}:
            raise ValueError(f"Unsupported distribution: {prior_distribution}")
        else:
            self.prior_distribution = prior_distribution
        calibration_method = str(calibration_method).lower()
        if calibration_method not in {"extent_only", "extent_and_sediment"}:
            raise ValueError(f"Unsupported calibration method: {calibration_method}")
        else:
            self.calibration_method = calibration_method
        extent_metric = str(extent_metric).lower()
        if extent_metric not in {"entire_runout_extent", "deposition_extent_only", "scour_extent_only"}:
            raise ValueError(f"Unsupported extent metric: {extent_metric}")
        else:
            self.extent_metric = extent_metric
        if profile_metric not in {"Qs","EA_d","E_u","A_d"}:
            raise ValueError(f"Unsupported profile metric: {profile_metric}")
        else:
            self.profile_metric = profile_metric
        self.jump_size = jump_size
        self.N_cycles = N_cycles
        self.initial_soil_depth = self.mg.at_node['soil__thickness'].copy()
        if self.mg.has_field("particle__diameter", at="node"):
            self.initial_particle_diameter = self.mg.at_node["particle__diameter"].copy()
        self.plot_tf = plot_tf
        self.DoD_m_dict ={}
        self._maker(seed)
        self.jstracking = [] # for tracking jumps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.phi_minus = phi_minus
        self.phi_plus = phi_plus
        self.qsc_constraint = qsc_constraint 
        self.show_progress = show_progress
        # determine the typical shear stress value given field observed flow characteristics
        self._determine_typical_shear_stress()


    def __call__(self, max_number_of_runs = 50):
        """run the calibrator max_number_runs interations"""
        if self.calibration_method == "extent_and_sediment":
            # get profile of observed total flow
            self.mbLdf_o = self._channel_profile_deposition("observed")
            # compute total mobilized volume
            self.TMV = (-1*(self.mg.at_node['DoD_o'][self.mg.at_node['DoD_o']<0]).sum())*self.mg.dx*self.mg.dy
            # mean total flow
            self.profile_metric_mean = self.mbLdf_o[self.profile_metric].mean()

            
        self._MCMC_sampler(max_number_of_runs)

    
    def _maker(self,seed):
        """prepares the np random generator"""       
        self.maker = np.random.RandomState(seed=seed)


    def _simulation(self):
        """reset mg to initial conditions, run the MWR, save the DoD"""
        
        # reset the dem to the initial dem and soil to initial soil depth
        self.MWR.grid.at_node['topographic__elevation'] = self.MWR.grid.at_node['topographic__initial_elevation'].copy()
        self._update_topographic_slope()
        self.MWR.grid.at_node['energy__elevation'] = self.MWR.grid.at_node['topographic__elevation'].copy()
        self.MWR.grid.at_node['soil__thickness'] = self.initial_soil_depth.copy()
        self.mg.at_node['disturbance_map'] = np.full(self.mg.number_of_nodes, False)
        if self.mg.has_field("particle__diameter", at="node"):
            self.mg.at_node["particle__diameter"] = self.initial_particle_diameter.copy()
        
            
        # run the model
        self.MWR.run_one_step()
        
        # create the modeldiff_m (modeled DoD) field
        diff = self.mg.at_node['topographic__elevation'] - self.mg.at_node['topographic__initial_elevation']
        self.mg.at_node['DoD_m'] = diff
        
        # save the DoD
        self.DoD_m_dict[self.it] = self.mg.at_node['DoD_m']
        
        if self.plot_tf == True:
            plt.figure('iteration'+str(self.it))
            imshow_grid(self.mg,"DoD_m",cmap = 'RdBu_r')
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
        """characterizes deposition patterns along the channel profile using four different metrics:
        """
        mg = self.mg
        dA = mg.dx*mg.dx
        # extract all nodes between lower and upper profile limits
        # use pre-runout dem to get elevations
        dem = mg.at_node['topographic__initial_elevation']
        pel = dem[self.runout_profile_nodes] # elevation of profile nodes 
        mask = (pel >= self.el_l) & (pel <= self.el_h)
        runout_profile_nodes = self.runout_profile_nodes[mask]
        runout_profile_distance = self.runout_profile_distance[mask]
               

        if datatype == "modeled":
            demd = mg.at_node['DoD_m']

        elif datatype == "observed":
            demd = mg.at_node['DoD_o']
            
        # dem = mg.at_node['topographic__initial_elevation']
        mbL = []
        for i, cn in enumerate(runout_profile_nodes):
            el = dem[cn]
            rd = runout_profile_distance[i]
            # metric 1, A_d: cumulative aggradation downslope of point cn
            A_d = self._downslope_A(dem, demd, el)
            # metric 2, E_u: cumulative scour upstream of point cn
            E_u = self._upslope_E(dem, demd, el)
            # metric 3, EA_d: cumulative erosion and aggradation downslope of point cn
            EA_d = self._downslope_A_and_E(dem, demd, el)
            # metric 4, Qs: cumulative flow volume past point cn equivalent to the cumulative erosion 
            # and aggradation upslope of point cn * -1., described by eq. 27"""
            Qs = self._Qs(dem, demd, el)
            mbL.append((Qs, EA_d, E_u, A_d, cn, rd, el))
        mbLdf = pd.DataFrame(mbL)
        mbLdf.columns = ['Qs','EA_d','E_u','A_d','node','runout_profile_distance','elevation']
        return mbLdf

    def _downslope_A(self, dem, demd, el):
        """profile metric 1, A_d: cumulative aggradation downslope of point cn"""
        dA = demd[demd>0]; dem_dA = dem[demd>0] # extract part of DEM and DoD with change
        dAs = np.nansum(dA[(dem_dA<=el)&(dem_dA>=self.el_l)]) # sum change in elevation range
        A_d = dAs*self.mg.dx*self.mg.dx # multiply by cell area to get volume  
        return A_d
    
    def _upslope_E(self, dem, demd, el):
        """profile metric 2, E_u: cumulative scour upslope (up-elevation) of point cn"""
        dE = demd[demd<0]; dem_dE = dem[demd<0]
        dEs = np.nansum(dE[(dem_dE>=el)&(dem_dE<=self.el_h)])
        E_u = dEs*self.mg.dx*self.mg.dx   
        return E_u
    
    def _downslope_A_and_E(self, dem, demd, el):
        """profile metric 3, EA_d: cumulative erosion and aggradation downslope of point cn"""
        dAE = demd[np.abs(demd)>0]; dem_dAE = dem[np.abs(demd)>0]
        dAEsd = np.nansum(dAE[(dem_dAE<=el) & (dem_dAE>=self.el_l)])
        EA_d = dAEsd*self.mg.dx*self.mg.dx 
        return EA_d
    
    def _Qs(self, dem, demd, el):
        """profile metric 4, Qs: cumulative flow volume past point cn equivalent to the cumulative erosion 
        and aggradation upslope of point cn * -1., equivalent to Qs, eq. 27"""
        dAE = demd[np.abs(demd)>0]; dem_dAE = dem[np.abs(demd)>0]
        dAEsu = np.nansum(dAE[(dem_dAE>=el) & (dem_dAE<=self.el_h)])*-1  
        Qs = dAEsu*self.mg.dx*self.mg.dx
        return Qs
    
    def _omegaT(self, metric = 'entire_runout_extent'):
        """ determines intersection, over estimated area and underestimated area of
        modeled debris flow deposition and the the calibration metric OmegaT following
        Heiser et al. (2017)
        """
        n_a = self.mg.nodes.reshape(self.mg.shape[0]*self.mg.shape[1]) # all nodes
        if metric == 'entire_runout_extent':
            n_o =  n_a[np.abs(self.mg.at_node['DoD_o']) > 0] # get both nodes aggradation and erosion nodes
            n_m = n_a[np.abs(self.mg.at_node['DoD_m']) > 0]
        elif metric == 'deposition_extent_only':
            n_o =  n_a[self.mg.at_node['DoD_o'] > 0] # get aggradation nodes
            n_m = n_a[self.mg.at_node['DoD_m'] > 0]
        elif metric == 'scour_extent_only':
            n_o =  n_a[self.mg.at_node['DoD_o'] < 0] # get erosion nodes
            n_m = n_a[self.mg.at_node['DoD_m'] < 0]
        self.a_o = n_o
        self.a_m = n_m
        n_x =  n_o[np.isin(n_o,n_m)] # intersection nodes
        n_u = n_o[~np.isin(n_o,n_m)] # underestimate
        n_o = n_m[~np.isin(n_m,n_o)] # overestimate
        
        X = len(n_x)
        U = len(n_u)
        O = len(n_o)
        T = X+U+O
        
        omegaT = X/T-U/T-O/T+1
        return omegaT


    def _SE_DoD(self, metric = 'entire_runout_extent'):
        """ determine the volumetric square error of the modeled DoD, normalized 
        by the total mobilized volume.
        """
        c1 = 1; c2 = 1
        n_a = self.mg.nodes.reshape(self.mg.shape[0]*self.mg.shape[1]) # all nodes
        na = self.mg.dx*self.mg.dy
        if metric == 'entire_runout_extent':
            n_o =  n_a[np.abs(self.mg.at_node['DoD_o']) > 0] # get nodes with scour or deposit
            n_m = n_a[np.abs(self.mg.at_node['DoD_m']) > 0]
        elif metric == 'deposition_extent_only':
            n_o =  n_a[self.mg.at_node['DoD_o'] > 0] # get nodes with scour or deposit
            n_m = n_a[self.mg.at_node['DoD_m'] > 0]
        elif metric == 'scour_extent_only':
            n_o =  n_a[self.mg.at_node['DoD_o'] < 0] # get nodes with scour or deposit
            n_m = n_a[self.mg.at_node['DoD_m'] < 0]
        self.a_o = n_o*na
        self.a_m = n_m*na
        n_x =  n_o[np.isin(n_o,n_m)] # intersection nodes
        n_u = n_o[~np.isin(n_o,n_m)] # underestimate
        n_o = n_m[~np.isin(n_m,n_o)] # overestimate
        A_x = len(n_x)*self.mg.dx*self.mg.dy
        A_u = len(n_u)*self.mg.dx*self.mg.dy
        A_o = len(n_o)*self.mg.dx*self.mg.dy
        CA = self.mg.dx*self.mg.dy
        observed_ = self.mg.at_node['DoD_o']
        # mask =  np.abs(observed_)>0 
        modeled_ = self.mg.at_node['DoD_m']       

        modeled = modeled_[n_x] # modeled_[mask]
        observed = observed_[n_x] # observed_[mask]
        X = self._SE(observed, modeled)*CA**2
        modeled = modeled_[n_u]
        observed = observed_[n_u]
        U = self._SE(observed, modeled)*CA**2
        modeled = modeled_[n_o]
        observed = observed_[n_o]        
        O = self._SE(observed, modeled)*CA**2        
        SE_DoD = (X+c1*U+c2*O)/(self.TMV**2)        
        return SE_DoD


    def _MSE_Qs(self):
        """computes the MSE of modeled Qt"""
        observed = self.mbLdf_o[self.profile_metric] 
        modeled = self.mbLdf_m[self.profile_metric]
        self.trial_Qs_profiles[self.it] = modeled # save profile
        CA = self.mg.dx*self.mg.dy
        MSE_Qs = self._MSE(observed, modeled)/(self.profile_metric_mean**2) # equation 29
        
        nm = 'Qs, m^3, iteration'+str(self.it)
        if self.plot_tf == True:
            plt.figure(nm)
            plt.plot(self.mbLdf_o['runout_profile_distance'], observed, label = 'observed')
            plt.plot(self.mbLdf_m['runout_profile_distance'], modeled, label = 'modeled')
            plt.title(nm)
            plt.legend()
            plt.show()  
        return MSE_Qs


    def _MSE(self, observed, modeled):
        """computes the mean square error (MSE) between two difference 
        datasets
        """
        if modeled.size == 0:
            modeled = np.array([0])
            observed = np.array([0])
        MSE = (((observed-modeled)**2).mean())
        return MSE   
    
    
    def _SE(self, observed, modeled):
        """computes the cumulative square error (SE) between two difference 
        datasets
        """
        if modeled.size == 0:
            modeled = np.array([0])
            observed = np.array([0])
        SE = ((observed-modeled)**2).sum()
        return SE
    

    def _prior_probability(self, value, min_val, max_val):
        """get prior liklihood of parameter value"""
        if self.prior_distribution == "uniform":
            prior = 1 # uniform distribution, all values equally likely, assign arbitrary weight of 1
        if self.prior_distribution == "normal":
            mean = (min_val+max_val)/2
            # assume min and max values equivalent to 4 standard deviations from the mean
            sd = (max_val-min_val)/8
            prior = norm.pdf(value, mean, sd)
        return prior


    def _determine_typical_shear_stress(self):
        """determine a typical debris flow basal shear stress value, used for 
        estimating an upper limit of the erosion coefficient and lower limit of
        threhosld flux qsc
        """
    
        rodf = self.MWR.vs*self.MWR.ros+(1-self.MWR.vs)*self.MWR.rof
        theta = np.arctan(self.MWR.s)
        
        if self.MWR.grain_shear:
            # use mean particle diameter in runout profile for representative grain size used to determine erosion rate
            Dp = self.MWR._grid.at_node['particle__diameter'][self.runout_profile_nodes].mean()
            self.tau = shear_stress_grains(self.MWR.vs,
                                      self.MWR.ros,
                                      Dp,
                                      self.MWR.h,
                                      self.MWR.s,
                                      self.MWR.g)
        else:
            self.tau = shear_stress_static(self.MWR.vs,
                                      self.MWR.ros,
                                      self.MWR.rof,
                                      self.MWR.h,
                                      self.MWR.s,
                                      self.MWR.g)


    def _candidate_value(self, selected_value, key):
        """determine the candidate parameter value as a random value from
        a normal distribution with mean equal to the presently selected value and
        standard deviation equal to the jump size, following MCMC sampler approach
        implemented in LeCoz et al., 2014. See BaRatin training materials,
        MCMCRegression.xlsx"""   
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



    def _check_E_lessthan_lambda_times_qsc(self, candidate_values, jump_sizes, selected_values):
        """A check that average erosion depth (E) does not exceed the flux constraint times some coefficient 
        (E must be less than qsc*lambda). If average E>qsc*lambda, if k is a calibration parameter, resample k until 
        average E<(qsc*lambda). If k is not a calibration parameter but qsc is, resample qsc until (qsc*lambda)>E"""
        equivalent_E = erosion_rate(self.MWR.k, self.tau, self.MWR.f, self.mg.dx)*self.mg.dx # rate times cell width to get depth
        _lambda = 1 
        if self.MWR.slpc>=0.02: # when slpc is high (>0.02), model is unstable when ~E>(10*qsc)
            _lambda = 10
        print('equivalent_E: {}, qsc: {}'.format(equivalent_E, self.MWR.qsc))   
        if equivalent_E>self.MWR.qsc*_lambda: # candidate values have already been transfered to MWR at this point
            
            # if k is a calibration parameter, first apply constraint to k, since model is very sensitive to qsc
            if self.params.get('k'):
                # check if k calibration range is low enough
                equivalent_E_min = erosion_rate(self.params['k'][2]-4*self.jump_size*(self.params['k'][1]-self.params['k'][0]),self.tau,self.MWR.f,self.mg.dx)*self.mg.dx # if erosion from k-four standard deviations (jumpsize)) greater than qsc, then the lower range needs to be decreased
                print('equivalent_E min: {}, qsc: {}'.format(equivalent_E_min, self.MWR.qsc))
                if equivalent_E_min>self.MWR.qsc*_lambda:
                    msg = "initial erosion coefficient k results in erosion depth>>qsc, decrease initial k and/or decrase it's min and max values"
                    raise ValueError(msg)                    
                else: # if low enough, randomly select an k value until the erosion equivalent is less than qsc
                    _pass = False
                    _i_ = 0
                    while _pass is False:  # repeatedly jump from the selected parameters until candidate_equivalent_E < qsc*_lambda
                        candidate_values['k'], jump_sizes['k'] = self._candidate_value(selected_values['k'], 'k')
                        candidate_equivalent_E = erosion_rate(candidate_values['k'],self.tau,self.MWR.f,self.mg.dx)*self.mg.dx
                        if candidate_equivalent_E < self.MWR.qsc*_lambda:
                            _pass = True
                            self.MWR.k = candidate_values['k']
                            if self.show_progress:
                                print('iteration {}, resampled, E<qsc'.format(self.it))
                            
                        _i_+=1; 
                        if _i_%1000 == 0:
                            print('after {} runs, all sampled k values are too large, decrease the lower range of k'.format(_i_))
            # if k is not a calibration parameter (k is fixed), then adjust qsc to meet constraint
            elif self.params.get('qsc'): # candidate values have already been transfered to MWR at this point
                # check if qsc calibration range is high enough
                if equivalent_E > self.params['qsc'][2]+4*self.jump_size*(self.params['qsc'][1]-self.params['qsc'][0])*_lambda: # if erosion is less than lambda*(qsc+four standard deviations (jumpsize)), then the upper range needs to be increased
                    msg = "maximum possible qsc value is less than erosion caused by k value"
                    raise ValueError(msg)   
                else: # if high enough, randomly select a qsi value until that value exceeds the erosion equivalent of the k value
                    _pass = False
                    _i_ = 0
                    while _pass is False: # repeatedly jump from the selected parameters until candidate_equivalent_E < qsc*_lambda
                        candidate_values['qsc'], jump_sizes['qsc'] = self._candidate_value(selected_values['qsc'], 'qsc')
                        if equivalent_E < candidate_values['qsc']*_lambda:
                            _pass = True
                            self.MWR.qsc = candidate_values['qsc']
                            if self.show_progress:
                                print('iteration {}, resampled, qsc>E'.format(self.it))
                        _i_+=1; 
                        if _i_%1000 == 0:
                            print('after {} runs, all sampled qsc values are to small, increase the upper range of qsc'.format(_i_))
                
            else:
                msg = "the erosion coeficient k causes erosion that exceeds qsc, reduce k or increase qsc to avoid model instability"
                raise ValueError(msg)  
            
        return candidate_values, jump_sizes
    
    


    def _adjust_jump_size(self, acceptance_ratio):
        """following LeCoz et al., 2014 (see the MCMC tutorials), adjust jump 
        variance based on acceptance ratio"""

        if acceptance_ratio < self.alpha_min:
            factor = self.phi_minus**0.5
        elif acceptance_ratio > self.alpha_max:
            factor = self.phi_plus**0.5
        else:
            factor = 1
        self.jump_size = self.jump_size*factor
        self.jstracking.append(self.jump_size)
        
    
    def _update_MWR_parameter_values(self, candidate_values):
        """update MWR parameter values with candidate values"""
        for key in self.params:
            if key == 'qsc':
                self.MWR.qsc = candidate_values[key]
            elif key == 'k':
                self.MWR.k = candidate_values[key]
            elif key == 'slpc':
                self.MWR.slpc = candidate_values[key] # slpc is a list
            elif key == "t_avg":
                # adjust thickness of landslide with id = 1
                self.MWR._grid.at_node['soil__thickness'][self.MWR._grid.at_node['mass__wasting_id'] == 1] = candidate_values[key]
            else:
                msg = "{} is not an adjustible parameter, correct the MCMC_adjusted_parameters input".format(key)
                raise ValueError(msg)  


    def _compute_candidate_value_prior_likelihood(self, candidate_values):
        """compute the prior likelihood of the candidate values as the 
        product of the prior probability of each parameter value"""
        prior_t = 1
        for key in self.params:
            min_value = self.params[key][0]
            max_value = self.params[key][1]
            value = candidate_values[key]
            prior_ = self._prior_probability(value, min_value, max_value)
            prior_t = prior_t*prior_ # likelihood (product of likelihoods) of all parameter values            
        return prior_t    
    
    
    def _determine_posterior_likelihood(self, prior_t):
        """determine the posterior likelihood of the candidate parameter values"""
        if self.calibration_method == "extent_only":
            omegaT = self._omegaT(metric = "entire_runout_extent")
            candidate_posterior = prior_t*omegaT
        elif self.calibration_method == "extent_and_sediment":
            # get modeled deposition profile
            self.mbLdf_m = self._channel_profile_deposition("modeled")

            # mean square error of the sediment transport profile (MSE_Qs)
            MSE_Qs = self._MSE_Qs()
            
            # omegaT
            omegaT = self._omegaT(metric = self.extent_metric)
            
            # square error of the modeled DoD (SE_DoD)
            SE_DoD = self._SE_DoD(metric = self.extent_metric)
            
            # determine posterior likilhood: product of prior liklihood, omegaT, Qt 
            candidate_posterior = prior_t*omegaT*(1/MSE_Qs)*(1/SE_DoD)
        return candidate_posterior, omegaT, MSE_Qs, SE_DoD
    
    
    def _jump_or_stay(self, candidate_values, candidate_posterior, selected_values):
        """given the candidate posterior, decide to use the candidate or stay
        at the presently selected value"""
        if self.it == 0:
            acceptance_ratio = 1 # always accept the first candidate vlue
        else:# run MWR
        # example_square_MWRu.run_one_step()


            acceptance_ratio = min(1, candidate_posterior/self.selected_posterior)
        # pick a random number between 0 and 1 assuming a uniform distribution
        # if number less than acceptance ratio, go with new parameter value.
        # if larger than ratio go with old parameter value
        # for first jump, probability will always be less than or equal to
        # acceptance ratio (1)
        rv = self.maker.uniform(0,1,1)
        if rv < acceptance_ratio:
            self.selected_posterior = candidate_posterior
            for key in self.params:
                selected_values[key] = candidate_values[key]
            msg = 'jumped to the candidate value'; self.ar.append(1)
        else :
            msg = 'staying at the selected value'; self.ar.append(0)
        return selected_values, acceptance_ratio, rv, msg
                

    def _MCMC_sampler(self, number_of_runs):
        """
        Markov-Chain-Monte-Carlo sampler
        Adjusts parameter values included in params dicitonary.
        Landslide average thickness (t_avg) adjusted at landslide with id = 1.
        If other landslide ids, thickness at those landslides will not be adjusted.
        """
        self.MCMC_stats_list = []
        self.trial_Qs_profiles = {}
        self.trial_runout_maps = {}
        self.ar = [] # list for tracking acceptance ratio
        # dictionaries to store trial values
        selected_values = {}
        candidate_values = {}
        # prior = {}
        jump_sizes = {}
        for self.it in range(number_of_runs):
            
            # get candidate parameter values
            # if first iteration, used the provided optimal parameter value as the first selected value
            if self.it == 0:
                for key in self.params:
                    selected_values[key] = self.params[key][2]
                    
            # for all other iterations, select a new candidate parameter value
            for key in self.params:
                candidate_values[key], jump_sizes[key] = self._candidate_value(selected_values[key], key)

            # update MWR parameter values
            self._update_MWR_parameter_values(candidate_values)

            # a check that erosion E doesn't exceed qsc for the candidate parameters
            # if k or qsc are a calibration parameter, will adjust k or qsc util E<qsc and update MWR parameter values
            if self.qsc_constraint:
                candidate_values, jump_sizes = self._check_E_lessthan_lambda_times_qsc(candidate_values, jump_sizes, selected_values)
            
            # likelihood of each candidate parameter value given the min and max values
            prior_t = self._compute_candidate_value_prior_likelihood(candidate_values)
 
            # run simulation with updated candidate parameters
            self._simulation()
            
            # determine candidate parameter posterior likelihood value (posterior pdf value) 
            candidate_posterior, omegaT, MSE_Qs, SE_DoD = self._determine_posterior_likelihood(prior_t)
            
            # decide to jump or not to jump from the presently selected parameter set
            selected_values, acceptance_ratio, rv, msg = self._jump_or_stay(candidate_values, candidate_posterior, selected_values)
            
            # save statistics of each MCMC iteration
            p_table = []
            p_nms = []
            for key in self.params:
                # the statistics saved depend on which parameter values were adjusted
                p_table = p_table+[jump_sizes[key], candidate_values[key], selected_values[key]]
                p_nms = p_nms+['jump_size_'+key, 'candidate_value_'+key, 'selected_value_'+key]
            if self.calibration_method == "extent_only":
                self.MCMC_stats_list.append([self.it, self.MWR.c, prior_t, omegaT,candidate_posterior, acceptance_ratio, rv, msg, self.selected_posterior]+p_table)
            elif self.calibration_method == "extent_and_sediment":
                self.MCMC_stats_list.append([self.it, self.MWR.c, self.TMV, self.profile_metric_mean, prior_t, omegaT, MSE_Qs**0.5, SE_DoD**0.5, candidate_posterior, acceptance_ratio, rv, msg, self.selected_posterior]+p_table)

            # adjust jump size for next iteration if it has been N_cycles iterations since the last adjustment
            if self.it%self.N_cycles == 0:
                mean_acceptance_ratio = np.array(self.ar).mean()
                self._adjust_jump_size(mean_acceptance_ratio)
                self.ar = [] # reset acceptance ratio tracking list
            
            # print progress to screen
            if self.show_progress and self.it%50 == 0:        
                print('MCMC iteration: {}'.format(self.it))
        
        # after number_of_runs iterations, organize MCMC statistics into a single dataframe
        self.MCMC_stats = pd.DataFrame(self.MCMC_stats_list)
        if self.calibration_method == "extent_only":
            self.MCMC_stats.columns = ['iteration', 'model iterations', 'prior', 'omegaT', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms
        elif self.calibration_method == "extent_and_sediment":
            self.MCMC_stats.columns = ['iteration', 'model iterations', 'total_mobilized_volume', 'obs_mean_total_flow',  'prior', 'omegaT','MSE_Qs^1/2','SE_DoD^1/2', 'candidate_posterior', 'acceptance_ratio', 'random value', 'msg', 'selected_posterior']+p_nms

        # get parameter set that results in highest posterior value
        self.calibration_values = self.MCMC_stats[self.MCMC_stats['selected_posterior'] == self.MCMC_stats['selected_posterior'].max()] 


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
    y_demdf_o = y_dem+mg.at_node['DoD_o'][pnodes]*ef

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