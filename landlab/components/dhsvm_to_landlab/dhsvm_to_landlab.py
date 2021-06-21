import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

import scipy as sc
from scipy.stats import moment as lm
from scipy.special import gamma

import xarray as xr

from landlab.plot import graph

import os as os

from landlab.utils.grid_t_tools import GridTTools

class DHSVMtoLandlab(GridTTools):

    """
    this component 
    
    
    TODO width and depth approximations are also updated based on parker 2007
         storm duration generator
    
    Parameters
    ----------
    path : 
    StreamFlowOnly : time series of flow at each DHSVM link
    grid : network model grid representation of DHSVM channel network 

    TO DO: change to run on grid slope and d50 each time step
            hydraulic geometry relation for width and depth updates with each time step - 90%
            
    
    # option 1
    iterate over full DHSVM time series, if DHSVM is less than threshold flow,
    NST is not run, ts advances. When DHSVM greater than threshold flow, NST
    is run with hydrualic condtions computed by calling the class instance and
    grid fields are computed.
    
    # option 2
    same as above but hydraulic condtions are determined assuming fixed channel
    condtions and read from self.DataArray to update grid fields.
    

    Examples
    --------
    >>> from landlab.utils.parcels import DhsvmToNST

    >>> streamflowonly = 'D:/UW_PhD/dhsvm/projects/b694/output/Streamflow_B694_V1_2011to2015.Only'
    >>> grid = example_network_model_grid 
    

    """

    def __init__(self,
                  grid = None,
                  nmgrid = None,
                  parcels = None,
                  nmgrid_d = None, 
                  DHSVM_flow = None,
                  DHSVM_dtw_dict = None,
                  precip = None,
                  method = 'stochastic',
                  time_aggregation = '24h',
                  begin_year = 2000,
                  end_year = 2050,
                  metric='max',
                  bankfull_flow_RI = 1.5,
                  fluvial_sediment_RI = 0.25,
                  hillslope_sediment_RI = 2.5,
                  tao_dict = None
                  ):        



        # call __init__ from parent classes
        super().__init__(grid, nmgrid)

        # network model grid
        self.nmgrid = nmgrid

        # initial parcels
        self.parcels = parcels

        # network model grid version of the dhsvm channel network
        self.nmgrid_d = nmgrid_d

        # hydrology inputs
        self._streamflowonly = DHSVM_flow/3600 # convert to m3/s
        
        # soil water
        self.appended_maps_name = DHSVM_dtw_dict['appended_maps_name']
        self.rpm = DHSVM_dtw_dict['rows_per_map']
        self.map_dates = DHSVM_dtw_dict['map_dates']
        self.nd = self.map_dates.shape[0]  #number of days(maps)

        
        # stochastic or time_series
        self._method = method
        
        # begin and end year of stochastic model run
        self.begin_year = begin_year
        self.end_year = end_year
        
        # bankfull flow return interval [yrs]
        self.Qbf_ri = bankfull_flow_RI
        
        # flow return interval [yrs] at which bedload transport occurs
        self.sed_ri = fluvial_sediment_RI
        
        # daily precipitation event interval above which landslides occur
        self.ls_ri = hillslope_sediment_RI
        
        # time aggregation
        self.tag = time_aggregation
        
        # aggregate precip time series
        if precip is None:
            self.precip = precip # no precip
        else:
            # resample precip data to daily
            self.precip = precip.resample(self.tag).sum().fillna(method='ffill')
            
            # determine probabiliity of event magnitude on a fluvial_sediment_RI basis
            # (rather than annul basis)
            self.PDS = self.partial_duration_series(self.precip, self.sed_ri)
            self.AMS = self.partial_duration_series(self.precip, 1)
            
            # determine probability distribution for event magnitude for all events >= to fluvial_sediment_RI 
            # fit distribution to partial duration series ( may include return intervals less than 1; return interval is not [yrs])
            self.P_Fx, self.P_x1, self.P_T, self.P_Q_ri  = self.fit_probability_distribution(AMS=self.PDS['value'], dist = 'LP3')
            # fit distribution to annual maximum series, used to determine return interval in year
            self.Pam_Fx, self.Pam_x1, self.Pam_T, self.Pam_Q_ri  = self.fit_probability_distribution(AMS=self.AMS['value'], dist = 'LP3')
        
       
        # aggregate flow time series
        if metric == 'max':
            self._streamflowonly_ag = self._streamflowonly.resample(self.tag).max().fillna(method='ffill') #convert sub  hourly obs to hourly

        elif metric == 'mean':
            self._streamflowonly_ag = self._streamflowonly.resample(self.tag).mean().fillna(method='ffill') #convert sub  hourly obs to hourly
    
        elif metric == 'min':
            self._streamflowonly_ag = self._streamflowonly.resample(self.tag).min().fillna(method='ffill') #convert sub  hourly obs to hourly

        elif (type(metric) is float) or (type(metric) is int):
            self._streamflowonly_ag = self._streamflowonly.groupby(pd.Grouper(freq='d')).quantile(metric,interpolation = 'nearest')
        
        # tau_dict
        if tao_dict is None:
            self.tao_dict = {'gHG':[0.15,-0.08], 'wHG':[2.205, 0.38], 'dHG':[0.274, 0.24]}

        
        # properties of raster model grid added as class variables in GridTTools
        
        # determine raster mg nodes that correspond to landlab network mg
        linknodes = self.nmgrid.nodes_at_link
        active_links = self.nmgrid.active_links
        nmgx = self.nmgrid.x_of_node
        nmgy = self.nmgrid.y_of_node

        out = self._LinktoNodes(linknodes, active_links, nmgx, nmgy)
        
        self.Lnodelist = out[0]
        self.Ldistlist = out[1]
        self.xyDf = pd.DataFrame(out[2])    
               
        # determine raster mg nodes that correspond to dhsvm network mg
        linknodes = self.nmgrid_d.nodes_at_link
        active_links = self.nmgrid_d.active_links
        nmgx = self.nmgrid_d.x_of_node
        nmgy = self.nmgrid_d.y_of_node

        out = self._LinktoNodes(linknodes, active_links, nmgx, nmgy)

        self.Lnodelist_d = out[0]
        self.Ldistlist_d = out[1]
        self.xyDf_d = pd.DataFrame(out[2])           
        
        ## define bedload and debris flow channel nodes       
        ## channel
        self._ChannelNodes() 
        
        # determine dhsvm network mg links that correspond to the landlab network mg
        self._DHSVM_network_to_NMG_Mapper()
        
        # create reduced size streamflow.only file with index according to nmg links
        self.streamflowonly_nmg()
        
        # determine dhsvm network mg links that correspond to the landlab network mg
        self._DHSVM_network_to_RMG_Mapper()    
        
        # compute partial duration series and bankful flow rate for each nmgrid_d link
        # used by the nmgrid        
        self.RI_flows()
        
        
        if self._method == 'stochastic':
            
            
            # approximate cdf of flow at each link
            self._flow_at_link_cdf()
            
            # approximate cdf of thickness of saturated zone at each cell 
            self._saturated_zone_thickness_cdf()
            
            # set initial depth to water table
            q_init = 0.5
            self.get_flow_at_links(q_init)
                   
            self._variable_channel_tau()
            
            self.get_depth_to_water_table_at_node(q_init)
            
            self._storm_dates_emperical_cdf()
            
            self._storm_date_maker()
            
            # initialize time
            self._time_idx = 0 # index
            self._time = self.storm_dates[0]  # duration of model run (hours, excludes time between time steps)
                 
        
    def __call__(self, ts=None):
        
        '''
        computes hydraulc condtions at time ts given the slope and d50 of
        an evolving network model nmgrid. D50 is computed from parcels in each link.
        slope is read from nmgrid, which is updated by NST. hydraulic condtions are 
        saved as attributes of the nmgrid.
        

        
        ts = dateime, time stamp
        
        output is a 1d np array of flow depth at each link of the landlab network mg
        
        '''

        
        # define Qlinks
        self.Qlinks = self._streamflowonly_ag
        
        self._constant_channel_tau()
        
        
    def _storm_date_maker(self):
        '''
        creates a date time index for the randomdly generated storm events
        that can be used for tracking changes in sediment dynamics with time

        Returns
        -------
        None.

        '''
        
        yrs = [0]
        c=0
        while yrs[c]<(self.end_year-self.begin_year+1):
            dt = np.random.normal(self.sed_ri, self.sed_ri/6,1)[0]
            if dt<0: # change in time can only be positive
                dt =0.05
            yrs.append(yrs[c]+dt)
            c+=1
        yrs = np.array(yrs)
        
        
        YEAR = self.begin_year+np.floor(yrs)
        
        doy = []
        for i,y in enumerate(YEAR):
            q_i = np.random.uniform(0,1,1)
            doy.append(int(self.intp(self.date_quantile, self.date_day, q_i)))

        doy = pd.DataFrame(np.array(doy))
        doy.columns = ['day-of-year']
        doyd = pd.to_datetime(doy['day-of-year'], format='%j').dt.strftime('%m-%d')
        
        dates = []
        for i,d in enumerate(doyd):
            date = str(int(YEAR[i]))+'-'+d
            dates.append(date)
        
        Tx = pd.to_datetime(np.sort(pd.to_datetime(dates)))
        # add random bit of time to each date to make sure no duplicates
        # Tx_new = []
        # for c,v in enumerate(Tx):
        #     Tx_new.append(Tx[c] + pd.to_timedelta(np.random.uniform(0,1,1),'h')[0])
            
        # Tx_new = pd.Series(Tx_new)
        # Tx = pd.to_datetime(Tx_new.values)
        
        self.storm_dates = Tx
                # time

        
        
        
    def _storm_dates_emperical_cdf(self):
        '''
        creates emperical cdf from partial duration series of precipitation events

        Returns
        -------
        None.

        '''
        pds_doy = self.PDS.index.dayofyear #convert PDS dates to day-of-year equivalent
        fig, ax = plt.subplots(figsize=(8, 4))
        n, bins, patches = plt.hist(pds_doy, 15, density=True, histtype='step', # emperical cdf
                           cumulative=True, label='Empirical')
        plt.xlabel('day of year')
        plt.ylabel('quantile')
        
        self.date_quantile = np.concatenate((np.array([0]),n))
        self.date_day = bins
        
  
    
    def _flow_at_link_cdf(self):
        
        # fit distribution to each nmg link
        self.Q_l_dist = {}
        for c, Link in enumerate(self.LinkMapper.keys()):
            
            # i_d = self.LinkMapper[Link]
            # # get the dhsvm network link id 'arcid' of nmgrid_d link i
            # dhsvmLinkID = self.nmgrid_d.at_link['arcid'][i_d]
            Q_l = self._streamflowonly_ag[Link]
            
            PDS_Q_l = self.partial_duration_series(Q_l, self.sed_ri)
            Q_l_Fx, Q_l_x1, Q_l_T, Q_l_Q_ri  = self.fit_probability_distribution(AMS=PDS_Q_l['value'], dist = 'LP3')
            self.Q_l_dist[Link] = [Q_l_Fx, Q_l_x1, Q_l_T, Q_l_Q_ri]
            if c%3 == 0:
                print('distribution fit to partial duration series of peak flows at link '+ str(Link))       
                
    
    def _saturated_zone_thickness_cdf(self):
        
        # open .asc version of DHSVM depth to soil water mapped output
        f = open(self.appended_maps_name, 'r')
        M = np.genfromtxt(f)
        f.close()
                
        # extract each map from appended map, convert nd row array of core nodes
        
        # NOTE!!!: DHSVM mapped outputs are oriented the same as the map i.e., top, 
        # bottom, right left sides of ascii file are north, south, west and east sides 
        # of the the DHSVM model domain. To convert to a Landlab 1-d array, need to 
        # flip output so that the south-west corner node is listed first
        
        dtw_l_an = [] # depth to water table list, 
        st_l_an = [] # saturated zone thickness
        c=0
        for i in range(1,self.nd+1):
            dtw_d = M[c:c+self.rpm,:] # map, DHSVM orientation
            dtw_l = np.hstack(np.flipud(dtw_d))[self._grid.core_nodes] # map, converted to landlab orientation, keep only core nodes
            st_l = self._grid.at_node['soil__thickness'][self._grid.core_nodes]-dtw_l # thickness of saturated zone
            c = c+self.rpm
            
            dtw_l_an.append(dtw_l)
            st_l_an.append(st_l)
            
        self.dtw_l_an = np.array(dtw_l_an)
        self.st_l_an = np.array(st_l_an)
        
        
        # compute cdf for each cell
        dtw_low_cv_nodes = [] # save node ids where water table changes little
        st_cdf = {} # dict to save saturated zone thickness cdf parameters for each node
        print_figs = False
        for c, n in enumerate(self._grid.core_nodes):
         
            
            pds = pd.Series(data = self.st_l_an[:,c], index = self.map_dates)
        
            Fx, x1, T, Q_ri = self.fit_probability_distribution(pds,dist = 'LN', print_figs = print_figs)
            st_cdf[n] = [Fx, x1]
            if Fx.std()/Fx.mean() < .05:
                dtw_low_cv_nodes.append(n)
            
            print_figs = False
            if c%2000 == 0:
                print('distribution fit to partial duration series of peak saturated zone thickness for '+ \
                      str(np.round((c/self.ncn)*100))+'% of core nodes' )
                print_figs = True
                
        self.st_cdf = st_cdf
        self.dtw_low_cv_nodes = dtw_low_cv_nodes
        
    
    def _variable_channel_tau(self):

            Q_ = []
            q_ = []
            d_ = []
            T_ = []
            U_ = []
            Uo_ = []
            nt_ = []               
            no_ = []
            Teff_ = []               
            Teffr_ = []
            deff_ = []
            
            # determine hydraulic conditions for each link in nmgrid
            # not because some nmgrid links may be mapped to the same nmgrid_d link
            # some nmgrid links will have the same flow rate
            
            for Link in self.LinkMapper.keys(): # for each link in nmgrid
                                     
                Q,q,d,T,U,Uo,nt,no,Teff,Teffr,deff = self.channel_hydraulics(Link)
                
                Q_.append(Q)
                q_.append(q)
                d_.append(d)
                T_.append(T)
                U_.append(U)
                Uo_.append(Uo)
                nt_.append(nt)                
                no_.append(no)
                Teff_.append(Teff)               
                Teffr_.append(Teffr)
                deff_.append(deff)
                
            # update nmgrid link fields for time ts
            self.nmgrid.at_link["flow"] = np.array(Q_)
            self.nmgrid.at_link["unit_flow"] = np.array(q_)                
            self.nmgrid.at_link["flow_depth"] = np.array(d_)
            self.nmgrid.at_link['total_stress'] = np.array(T_)
            self.nmgrid.at_link['mean_velocity'] = np.array(U_)
            self.nmgrid.at_link['mean_grain_velocity'] = np.array(Uo_)            
            self.nmgrid.at_link['total_roughness'] = np.array(nt_)
            self.nmgrid.at_link['grain_roughness'] = np.array(no_) 
            self.nmgrid.at_link['effective_stress'] = np.array(Teff_)             
            self.nmgrid.at_link['effective_stress_b'] = np.array(Teffr_)  
            self.nmgrid.at_link['effective_flow_depth'] = np.array(deff_)         
                                                                                
    
    def _constant_channel_tau(self):
        '''
        computes a time series of hydraulic conditions using the hydrograph
        at each link assuming channel slope and grain size are constant with time

        tao_dict = {'gHG':[0.15,-0.08], 'wHG':[2.205, 0.38], 'dHG':[0.274, 0.24]}                 
        '''
     
        # CHANGE TO: self.ComputeLinkHydraulics(Q = Q_) 
         
         
        Q_dict = {}
        q_dict = {}
        d_dict = {}
        T_dict = {}
        U_dict = {}
        Uo_dict = {}
        nt_dict = {}                
        no_dict = {}
        Teff_dict = {}               
        Teffr_dict = {}
        deff_dict = {}
        

        for Link in self.LinkMapper.keys(): # for each link in nmgrid
                                            

            Q,q,d,T,U,Uo,nt,no,Teff,Teffr,deff = self.channel_hydraulics(Link)


            Q_dict[Link] = Q
            q_dict[Link] = q            
            d_dict[Link] = d
            T_dict[Link] = T
            U_dict[Link] = U
            Uo_dict[Link] = Uo
            nt_dict[Link] = nt                
            no_dict[Link] = no
            Teff_dict[Link] = Teff               
            Teffr_dict[Link] = Teffr
            deff_dict[Link] = deff

            print(Link)                
        
        # convert to dataframe, transpose so that each row 
        # is timeseries for a single link
        Q_df = pd.DataFrame.from_dict(Q_dict, orient='columns').T
        q_df = pd.DataFrame.from_dict(q_dict, orient='columns').T
        d_df = pd.DataFrame.from_dict(d_dict, orient='columns').T
        T_df = pd.DataFrame.from_dict(T_dict, orient='columns').T
        U_df = pd.DataFrame.from_dict(U_dict, orient='columns').T
        Uo_df = pd.DataFrame.from_dict(Uo_dict, orient='columns').T
        nt_df = pd.DataFrame.from_dict(nt_dict, orient='columns').T
        no_df = pd.DataFrame.from_dict(no_dict, orient='columns').T
        Teff_df = pd.DataFrame.from_dict(Teff_dict, orient='columns').T            
        Teffr_df = pd.DataFrame.from_dict(Teffr_dict, orient='columns').T
        deff_df = pd.DataFrame.from_dict(deff_dict, orient='columns').T
        
        # convert to dataarray
        Q_da = xr.DataArray(Q_df,name = 'flow',dims = ['link','date'],
                            attrs = {'units':'m2/s'})        
        q_da = xr.DataArray(q_df,name = 'unit_flow',dims = ['link','date'],
                            attrs = {'units':'m2/s'})
        d_da = xr.DataArray(d_df,name = 'flow_depth',dims = ['link','date'],
                            attrs = {'units':'m'})
        T_da = xr.DataArray(T_df,name = 'total_stress',dims = ['link','date'],
                            attrs = {'units':'Pa'})
        U_da = xr.DataArray(U_df,name = 'mean_velocity',dims = ['link','date'],
                            attrs = {'units':'m/s'})  
        Uo_da = xr.DataArray(Uo_df,name = 'mean_virtual_velocity',dims = ['link','date'],
                            attrs = {'units':'m/s'})
        nt_da = xr.DataArray(nt_df,name = 'total_roughness',dims = ['link','date'],
                            attrs = {'units':'s/m(1/3)'})
        no_da = xr.DataArray(no_df,name = 'grain_roughness',dims = ['link','date'],
                            attrs = {'units':'s/m(1/3)'})
        Teff_da = xr.DataArray(Teff_df,name = 'effective_stress_a',dims = ['link','date'],
                            attrs = {'units':'Pa'})
        Teffr_da = xr.DataArray(Teffr_df,name = 'effective_stress_b',dims = ['link','date'],
                            attrs = {'units':'Pa'})
        deff_da = xr.DataArray(deff_df,name = 'effective_flow_depth',dims = ['link','date'],
                            attrs = {'units':'m'})              
        
        # combine dataarrays into one dataset, save as instance variable                      
        self.dataset = xr.merge([Q_da, q_da, d_da, T_da, U_da, Uo_da, nt_da, no_da, 
                                 Teff_da, Teffr_da, deff_da])
        
    
    def channel_hydraulics(self, Link):
        
        # # FLOW PARAMETERS
        # # i_d is equivalent link in dhsvm nmgrid
        # i_d = self.LinkMapper[Link]
        
        # # get the dhsvm network link id 'arcid' of nmgrid_d link i
        # dhsvmLinkID = self.nmgrid_d.at_link['arcid'][i_d]
        
        # # get link attributes from nmgrid grid       
        # Q = self.Qlinks[str(dhsvmLinkID)] # flow at link for time ts 
        # get flow rate at Link
        Q = self.Qlinks.loc[Link] # .loc call index by index name, not indici
        
        # look up bankfull flow and partial duration series
        Qb = self.RIflows[Link] # Q1.2 [m3/s]
        
        # look up reach attribute
        CA = self.nmgrid.at_link['drainage_area'][Link]/1e6 # m2 to km2
        S = self.nmgrid.at_link['channel_slope'][Link]              

        # determine link D50
        # parcel grain size in link i
        ct = self.parcels.dataset.time[-1]
        cur_parc = self.parcels.dataset['D'].sel(time = ct).values
        msk = self.parcels.dataset['element_id'].sel(time = ct) == Link
        # msk = np.where(self.parcels.dataset['element_id']==Link)
        Dp = pd.DataFrame(cur_parc[msk])#np.where(self.parcels.dataset['element_id']==Link)].values.flatten())
        
        # determine 50th percentile value
        D50 = Dp.quantile(.5).values #diam. parcels              
                        
        # grain size distribution based on d50 and assumed 
        # log normal distribution (Kondolf and Piegay 2006)
        Gdist = pd.DataFrame(np.random.lognormal(np.log(D50),0.5,size=1000))            
        D65 = Gdist.quantile(0.65).values[0]                
        D84 = Gdist.quantile(0.84).values[0]             
        
        # # TODO: update coefficent and exponent of width and depth HG based on d50 following Parker et al 2007                       
        wb = self.tao_dict['wHG'][0]*CA**self.tao_dict['wHG'][1]  # bankfull width [m]
        db = self.tao_dict['dHG'][0]*CA**self.tao_dict['dHG'][1]  # bankfull depth [m]

        # wb =  self.nmgrid.at_link['channel_width'][Link]     
        # db =  self.nmgrid.at_link['channel_depth'][Link]     
                    
        # approximate base channel width assuming 1:1 channel walls
        b =  wb - 2*db
        if b<0:
            b = 0
                    
        # depth is normal depth in trapezoidal channel geometry - very slow
        # d = depth_trapezoid(Q, S = S, b = b, m1 = 1, m2 = 1, n = 0.05)               
        d = db*(Q/Qb)**0.3 # approximation for flow depth               
        q = Q/(b+d/2)               
            
        # roughness and effective stress

        U,Uo,nt,no,T,Teff,Teffr,deff = self.flow_resistance_RR(q,d,S,D65,D84)
        
        return (Q, q, d, T, U, Uo, nt, no, Teff, Teffr, deff)
    
        
    def RI_flows(self):
        '''
        runs annual_maximum_series OR partial_duration_series for each link 
        in nmgrid and creates the RIflows dictionary, which has the bankfull 
        flow magnitude and partial duration series of each link

        Returns
        -------
        None.

        '''
            
        RIflows = {}
        for Link in self.LinkMapper.keys():
                    # i_d is equivalent link in dhsvm nmgrid
            # print(Link)
            # i_d = self.LinkMapper[Link]
            
            # # get the dhsvm network link id 'arcid' of nmgrid_d link i
            # dhsvmLinkID = self.nmgrid_d.at_link['arcid'][i_d]
            # print(dhsvmLinkID)
            # get link attributes from nmgrid grid
            
            # Qts = self._streamflowonly_ag[str(dhsvmLinkID)]# flow at link for time ts 
            Qts = self._streamflowonly_ag[Link]
            
            PDS = self.partial_duration_series(Qts)

            x = PDS['RI [yr]']                
            y = PDS['value']

            f = sc.interpolate.interp1d(x,y)
            
            RIflows[Link]  = f(self.Qbf_ri)
        
        self.RIflows = RIflows
        

    def streamflowonly_nmg(self):        
        # create reduced size streamflow.only file
        # move this function to a script as a pre-processing step to reduce memory needs

        nmg_streamflow_dict = {}
        for Link in self.LinkMapper.keys():
             i_d = self.LinkMapper[Link]
                     
             # get the dhsvm network link id 'arcid' of nmgrid_d link i
             dhsvmLinkID = self.nmgrid_d.at_link['arcid'][i_d]
             
             # nmg_streamflow_dict[str(dhsvmLinkID)] = self._streamflowonly_ag[str(dhsvmLinkID)]
             nmg_streamflow_dict[Link] = self._streamflowonly_ag[str(dhsvmLinkID)]
         
        self._streamflowonly_ag = pd.DataFrame.from_dict(nmg_streamflow_dict)
    
    
    def get_flow_at_links(self, q_i):
        '''
        
    
        Parameters
        ----------
        q_i : float
            quantile of cdf used to determine a flow rate
    
        Returns
        -------
        Q_i float
            flow rate for the given quantile
    
        '''
        Qlink_dict = {}
        for Link in self.LinkMapper.keys():     
            dst = self.Q_l_dist[Link]
            Fx = dst[0]; x1 = dst[1]; 
            Q_i = self.intp(Fx,x1,q_i, message = '#######Q#### at link '+str(Link))
            Qlink_dict[Link] = Q_i
           
        self.Qlinks = pd.DataFrame.from_dict(Qlink_dict, orient = 'index')
        
        
    def get_depth_to_water_table_at_node(self, q_i):

        dtw_field = (np.ones(self._grid.at_node['soil__thickness'].shape[0])*np.nan).astype(float)
        st = []
        dtw = []
        for c,n in enumerate(self._grid.core_nodes):
            Fx = self.st_cdf[n][0]; x1 = self.st_cdf[n][1]
            st_i = self.intp(Fx,x1,q_i)#, message = '#######DTW#### at node '+str(n))
            dtw_i = self._grid.at_node['soil__thickness'][n]-st_i
            st.append(st_i)
            dtw.append(dtw_i)
        
        dtw_cn = np.array(dtw) # depth to water at core nodes
        
        dtw_field[self._grid.core_nodes] = dtw_cn
        
        #updates depth__to_soil_water field
        self._grid.add_field('node', 'depth__to_water_table', dtw_field, clobber = True)

                

    def run_one_step(self, ts = None):
            
        
        if self._time_idx < len(self.storm_dates):
        
            if ts is None:
               
                # pick a random quantile (probability of non-exceedance) for all
                # events larger than a fluvial_sediment_RI event
                self.q_i = np.random.uniform(self.P_Fx.min(),self.P_Fx.max(),1)[0]
                
                # get the P rate for quantile q_i
                self.P_i = self.intp(self.P_Fx, self.P_x1, self.q_i, message = None)
                
                # get the return interval for the P [yrs]:
                try:
                    self.q_yrs_i = self.intp(self.Pam_x1, self.Pam_Fx, self.P_i, message = None)
                    self.P_ri = 1/(1-self.q_yrs_i)
                except: # may be below the 1 year event
                    self.P_ri = 0.5
                
                # get flow rate at each link for quantile q_i
                
                # create self.Qlinks, single flow value at each link
                self.get_flow_at_links(self.q_i)
                       
                self._variable_channel_tau()
                
                if self.P_ri > self.ls_ri: # update depth to water table if this is true, updating water table is slow
                    self.get_depth_to_water_table_at_node(self.q_yrs_i)
        
            else:
 
                self.Qlinks = self._streamflowonly_ag.loc[ts] # flow rate in all links at time ts
    
                self._variable_channel_tau()
                
                
                

            
            # date of storm
            self._time = self.storm_dates[self._time_idx] 
            
            # time between storms
            self._dtime = self.storm_dates[self._time_idx]-self.storm_dates[self._time_idx-1]        
        
            # update time       
            self._time_idx +=1 # index        
        
        else:
            msg = "end of time series"
            raise ValueError(msg)  

    
    def annual_maximum_series(self, time_series, plotting_position = 'Weibull'):
        '''
            
        Parameters
        ----------
        time_series : pd series
            time series of data from which annual maximum series will be computed
            index is pd.DateTime
        
        plotting_position : string
            plotting position forumula. Default is 'Weibull'
    
        Returns
        -------
        AMS : pandas dataframe
            a dataframe of the annual maximum series magnitude and return interval [yrs]
    
        '''
        
    
        
        ranks = {}
        RIp = {}
        MagpS ={}
        pds_l = {}
        Qri_l = {}
        
        time_series.name = 'value'
        time_series_a = time_series.resample('1Y').max() # resample to annual maximum value
        AMS = time_series_a.sort_values(ascending=False).to_frame() #sort large to small
        
        n = AMS.shape[0]
        ranks = np.array(range(1,n+1))
        
        # plottting position (exceedance probability if ordered large to small)
        if plotting_position == 'Cunnane':
            PP = (ranks-.4)/(n+.2) #compute plotting position 
        elif plotting_position == 'Weibull':
            PP = ranks/(n+1)
        elif plotting_position == 'Blom':
            PP = (ranks - (3/8))/(n+(1/4))
        else:
            PP = ranks/(n+1)
          
        EP = PP 
    
        T = 1/PP # return interval
        
        AMS['RI [yrs]']=T
        
        return AMS
    
    def partial_duration_series(self, time_series, RI_cutoff = 1, plotting_position = 'Weibull',sep=30):
        '''
        Uses a peaks-over-threshold approach to create a partial duration series
        from a time series of hydrologic data. 
        
        The PDS is truncated to exclude all events with RI < 1 
        
        Based on methods described in:
                Malamud B.D. and Turcotte D.L., 2006, The Applicability of power-law 
                frequency statstics to floods, Journal of Hydrology, v.322, p.168-180
    
        Parameters
        ----------
        time_series : pd series
            time series of data from which annual maximum series will be computed
            index is pd.DateTime
    
        RI_cutoff : pd series
             low limit of event return interval included in partial duration series       
        
        plotting_position : string
            plotting position forumula. Default is 'Weibull'
            
        sep = integer
            time separation [days] used to extract independent events
    
        Returns
        -------
        PDS : pandas dataframe
            a dataframe of the partial duration series magnitude and return interval [yr]
    
        '''
        
        ranks = {}
        RIp = {}
        MagpS ={}
        pds_l = {}
        Qri_l = {}
    
    
        Qdt = time_series.copy() # make copy so as not to write over original
    
        
        ll = Qdt.shape[0]
        c=1       
        pds = {}
        mq = Qdt.max()
        pds[Qdt[Qdt==mq].index[0]] = mq
        
        while  mq>0:
        
            Rf = Qdt[Qdt==mq].index+ datetime.timedelta(days=sep)
            Rp = Qdt[Qdt==mq].index- datetime.timedelta(days=sep)
            mask = (Qdt.index > Rp[0]) & (Qdt.index <= Rf[0])
            Qdt.loc[mask]=0
        
            mq = Qdt.max() 
    
            if mq >0: # last value of algorith is 0, dont add zero to pds
                pds[Qdt[Qdt==mq].index[0]] = mq
    
        pds_l = pds
        
        PDS = pd.DataFrame.from_dict(pds_l, orient='index')
        PDS.columns = ['value']
            
        Yrs = max(Qdt.index.year)-min(Qdt.index.year)+1 # number of years in time series
    
        n = PDS.shape[0]            
        ranks = np.array(range(1,n+1)) 
        
        # plottting position (exceedance probability if ordered large to small)
        if plotting_position == 'Cunnane':
            PP = (ranks-.4)/(Yrs+.2) #compute plotting position 
        elif plotting_position == 'Weibull':
            PP = ranks/(Yrs+1)
        elif plotting_position == 'Blom':
            PP = (ranks - (3/8))/(Yrs+(1/4))
        else:
            PP = ranks/(n+1)
        
        EP = PP
        T = 1/PP                        
        PDS['RI [yr]'] = T                    
        PDS = PDS[PDS['RI [yr]'] > RI_cutoff] # truncate to include all values >= to RI_cutoff
        
        return PDS

    
    def fit_probability_distribution(self, AMS, dist = 'LN', 
                       RI=[1.5], plotting_position = 'Weibull',
                       print_figs = False):
        '''
        Fits distribution to annual maximum series or partial duration series of
        data using methods and formulas described in: 
       
             Maidment, 1992, Handbook of Hydrology, Chapter 
             
        NOTE on interpreation of the fit distribution: 
        IF the distribution is fit to an annual maximum series, the resultant pdf 
        gives the liklihood of a given magnitude occuring during a single year. 
        
        If the distribution is fit to a partial duration series that lists magnitudes
        less than a return interval of 1, then the resultant pdf gives the likelihood 
        of a given magnitude occuring during any event larger than the minimum return 
        interval included in the partial duration series. 
        e.g., if the partial duration series includes events as small as the 0.25 year 
        event,then the fit distribtuion gives the likelihood of a given 
        flow magnitude during any 0.25 year and larger storm. 
           
        Parameters
        ----------
        AMS : pd series, index is pd.datetime
            annual maximum series (or partial duration series)
        dist : string
            can be 'LN' or 'LP3'.  default is 'LN'
            
            type of distribution fit to AMS. Can be either: lognormal or 
            log-Pearson type III.
            
            NOTE: distributions may not fit data
        
        RI : list of float
            flow magnitude is returned for all return interval values listed in RI 
            return interval values must be greater than 1
            The default is [1.5,2,5,25,50,100].
        
        plotting_position : string
            plotting position forumula. Default is 'Cunnane'
        
        Returns
        -------
        Fx : np.array
            cdf quantile
        x1 : np.array
            cdf value
        T : np.array
            quantile equivalent return interval [yr]
        Q_ri : dictionary
            key is each value in RI, value is magnitude
    
    
        '''
        # compute moments
        
        mn = AMS.values.mean() # m1
        vr = AMS.var() # m2
        cs = sc.stats.skew(AMS.values) 
        m2 = lm(AMS.values,moment = 2)
        m3 = lm(AMS.values,moment = 3)
        m4 = lm(AMS.values,moment = 4)
        lskew = m3
        L3 = m3*m2 #third Lmoment
        lkurt = m4
        L4 = m4*m2 #fourth Lmoment    
    
        
        if dist == 'LN':
        
            mu = mn
            sigma = vr**0.5
    
            mu_lognormal = np.log((mu ** 2)/ np.sqrt(sigma ** 2 + mu ** 2))
            
            sigma_lognormal = np.sqrt(np.log((sigma ** 2) / (mu ** 2) + 1))
            s = np.random.lognormal(mu_lognormal, sigma_lognormal, 10000)
            
            x1 = np.linspace(s.min(), s.max(), 10000)
            
            #for comparison to emperical estimate using plotting position
            x2 = np.sort(AMS.values, axis=0) # values in AMS, sorted small to large (pp is quantile)
        
            X = [x1,x2]
        
            fx = {}
            for i,x in enumerate(X):
                fx[i] = (np.exp(-(np.log(x) - mu_lognormal)**2 / (2 * sigma_lognormal**2)) \
                 / (x * sigma_lognormal * np.sqrt(2 * np.pi)))
                
            Fx = sc.integrate.cumtrapz(fx[0], x1, initial=0)
                 
        
        if dist == 'LP3':
            # natural log of flow data is pearson type 3 distributed
            x = np.log(AMS.values)
            
            mn = x.mean() #compute first three moments of log(data)
            vr = x.var()
            cs = sc.stats.skew(x)
            
            #parameters
            alphap = 4/cs**2#18.20
            betap = 2/((vr**0.5)*cs)
            xi = mn-alphap/betap
            
            E3 = np.exp(3*xi)*(betap/(betap-3))**alphap #18.33
            E2 = np.exp(2*xi)*(betap/(betap-2))**alphap
            muq = np.exp(1*xi)*(betap/(betap-1))**alphap
            vrq= np.exp(2*xi)*(((betap/(betap-2))**alphap)-(betap/(betap-1))**(2*alphap))
            csq = (E3-3*muq*E2+2*muq**3)/((vrq**.5)**3)
            
            alphapq = 4/csq**2
            alphapq = alphapq
            
            betapq = 2/((vrq**.5)*csq)
            betapq=betapq
            
            xiq = muq-alphapq/betapq
            xiq=xiq
            
            if betap <0:
                if 1.5*max(AMS) < np.exp(xi):
                    x1=np.linspace(1,1.5*max(AMS),num = 10000)
                else:
                    x1=np.linspace(1,np.exp(xi),num = 10000) #table 18.2.1
            if betap >0:
                x1=np.linspace(np.exp(xi),max(AMS)*1.5,num = 10000) 
             
           
            #for comparison to emperical estimate using plotting position
            x2 = np.sort(AMS, axis=0) # values in AMS, sorted small to large (pp is quantile)
            
            X = [x1,x2]
            
            fx = {}
            for i,x in enumerate(X):
                
                fp = (1/x)*np.absolute(betap)*(betap*(np.log(x)-xi))**(alphap-1)
                fp[np.isnan(fp)] = 0 # replace nan with 0
                sp = np.exp(-betap*(np.log(x)-xi))/((gamma(alphap)))
                sp[np.isnan(sp)] = 0
                fx[i] = fp*sp
            
            #cdf created by summing area under pdf
            Fx = sc.integrate.cumtrapz(fx[0], x1, initial=0)
            
        
       
        if print_figs:
            # normalized histogram of data with parameterized pdf
            fig, ax=plt.subplots(1,1,figsize=(6,6))
            plt.hist((AMS), bins='auto')  # arguments are passed to np.histogram
            plt.plot(x1, fx[0],'b-',markersize=8,linewidth=3)
            plt.title("normalized histogram and fit distribution")
            plt.xlim(0,1.5*max(AMS))
            plt.show()
                
    
            fig, ax=plt.subplots(1,1,figsize=(6,6))
            plt.title('cdf equivalent of fit distribution')
            plt.plot(x1,Fx) #examine cdf
            plt.show()
        
    
        #summary plot
        n = AMS.shape[0]
        ranks = np.array(range(1,n+1))
        #pp = (ranks-.4)/(n+0.2) #Cunnane
        # plottting position (exceedance probability if ordered large to small)
        if plotting_position == 'Cunnane':
            PP = (ranks-.4)/(n+.2) #compute plotting position 
        elif plotting_position == 'Weibull':
            PP = ranks/(n+1)
        elif plotting_position == 'Blom':
            PP = (ranks - (3/8))/(n+(1/4))
        else:
            PP = ranks/(n+1)
        
        T = 1/(1-Fx)
        
        x = T
        y = x1
        
        f = sc.interpolate.interp1d(x,y) 
        
        ri = RI
        Q_ri = {}
        
        for i in ri:
            Q_ri[i] = f(i)
        
        if print_figs:
            fig, ax=plt.subplots(1,1,figsize=(12,6))
            for i,v in enumerate(ri):
                
               plt.plot([-.5,1.5],[Q_ri[v],Q_ri[v]],'k--',linewidth = 1+5/len(ri)*i,alpha = 1-1/len(ri)*i, label = str(ri[i]))
            
            plt.plot(Fx,x1,'r',label = 'fitted distribution');
            plt.plot(PP,x2,'.', label ='emperical estimate');
            plt.ylim([min(x2)*0.5,max([Q_ri[v],x2.max()])*1.05])
            plt.xlim([-0.02,1.02])
            ax.legend(loc = 'upper center')
            plt.show()
    
        return Fx, x1, T, Q_ri 
                
    
    def depth_trapezoid(self, Q, S, b, m1 = 1, m2 = 1, n = 0.05):
        
        '''
        A main channel = A_t(b,m1,m2,y)
        P main channel = P_t(b,m1,m2,y)
        '''
    
        def A_t(b,m1,m2,y):
            '''
            Area of trapezoid, use for all trapezoids in compound channel
            '''
            A = (y/2)*(b+b+y*(m1+m2))
            return A
        
        def P_t(b,m1,m2,y):
            '''
            Wetted perimeter of trapezoid, below flood plains
            '''
            P = b+y*((1+m1**2)**(1/2)+(1+m2**2)**(1/2))
            return P
                
        def h(y):
            return np.abs(Q-(1/n)*A_t(b,m1,m2,y)*((A_t(b,m1,m2,y)/P_t(b,m1,m2,y))**(2/3))*S**(1/2))
    
        r = sc.optimize.minimize_scalar(h,method='bounded',bounds = [.1,10])
        
        y = r.x
        Rh = A_t(b,m1,m2,y)/P_t(b,m1,m2,y) 
        return (y,Rh)
            
                        
    def flow_resistance_Ferguson(self, D84,Rh):
        '''
        Computes total Manning's roughness and Darcy Weisbach friction factor 
        as a function of relative depth (D/D84) using equation 20 and equation 1 
        of Ferguson, 2007
        
        INPUT
        D84 - intermediate axis length [m] of 84th percentile grain size [m]
        D - flow depth at location of grain [m]
        Rh - hydraulic radius or average flow depth [m]
        
        OUTPUT
        (mannings roughness, darcy Weisbach friction factor)
        
        '''
        a1 = 7.5
        a2 = 2.36 
        f_b = 8*((((a1**2)+(a2**2)*(Rh/D84)**(5/3))**0.5)/(a1*a2*Rh/D84))**2 #eq 20
        n_b = ((Rh**(1/6))*f_b**0.5)/((8*9.81)**0.5) #eq 1
        return (n_b,f_b)            
            
    
    def flow_resistance_RR(self, q,d,S,D65,D84):
        '''
        Computes Manning roughness using Rickenmann and Recking, 2011 form of
        the Ferguson, 2007 flow resistance equation
    
        Parameters
        ----------
        q : float, np.array or pd.series
            discharge per unit width [m2/s] - average discharge per width channel
        d : float
            depth [m].
        S : float
            channel slope [m/m].
        D65 : float
            65th percentile grain diameter [m].
        D84 : float
            84th percentile grain diameter [m].
    
        Returns
        -------
        U : float, np.array or pd.series
            average velocity of flow [m/s].
        Uo : float, np.array or pd.series
            average virtual velocity of flow [m/s].
        nt : float, np.array or pd.series
            average total flow resistance, n [s/m(1/3)].
        no : float, np.array or pd.series
            base (grain) resistance, n [s/m(1/3)]
        T : float, np.array or pd.series
            total stress resisting flow (acting on channel) [Pa]
        Teff : float, np.array or pd.series
            effective stress on grains [Pa], method 1.
        Teffr : float, np.array or pd.series
            effective stress on grains [Pa], method 2.
        deff : float
            equivalent depth of effective flow [m]
    
        '''
        g = 9.81 # m/s2
        ro = 1000 # kg/m3
        # q = Q/(b+d)
        # compute q**
        qss = q/((g*S*D84**3)**0.5)
        # compute U**
        Uss = (1.443*qss**0.60)*(1+(qss/43.78)**0.8214)**-0.2435
        # compute U
        U = Uss*(g*S*D84)**0.5
        # compute d
        # d = q/U
        # compute Uo
        Uo = 3.70*(q**0.4)*(g**0.3)*(S**0.3)*(D84**-0.1)
        # compute ftot, ntot
        ft = (8*g*q*S)/(U**3)#(8*g*S)/((U**2)*(q**2))
        # nt = ((S**0.5)*(d**(2/3)))/(((8*g*d*S)**0.5)/(ft**0.5)) # Darcy-Weisbach 
        nt = (((q**0.4)*(S**0.3))/U)**(5/3) # mannings wide channel approximation  
        # compute fo, no
        fo = (8*g*q*S)/(Uo**3)#(8*g*S)/((Uo**2)*(q**2))
        # no = ((S**0.5)*(d**(2/3)))/(((8*g*d*S)**0.5)/(fo**0.5)) 
        no = (((q**0.4)*(S**0.3))/Uo)**(5/3)             
        # compute Teff using method 1: Wilcock, 2001
        Teff = 0.052*ro*((9.81*S*D65)**0.25)*(U**1.5) # effective stress on grains               
        # compute Teff using method 2: Erkan ratio
        T = (ro)*9.81*d*S #total stress acting on flow [N/m2]
        Teffr = T*(no/nt)**1.5
    
        deff = d*(Teff/T) # equivalent depth of effective flow  
        
        return (U,Uo,nt,no,T,Teff,Teffr,deff)
    
    
    




