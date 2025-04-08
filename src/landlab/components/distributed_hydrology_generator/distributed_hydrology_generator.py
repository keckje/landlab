
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy as sc
from scipy.stats import moment as lm
from scipy.special import gamma

import xarray as xr
from landlab import Component, FieldError
from landlab.plot import graph
from landlab.io import read_esri_ascii
from landlab.utils.channel_network_grid_tools import ChannelNetworkToolsMapper

class DistributedHydrologyGenerator(Component):

    """Generate a time series of flow rates and/or soil water conditions using
    the output from an external distribute hydrology model
    
    The DistributedHydrologyGenerator(DHG) component takes the raw modeled flow 
    and depth from a distributed hydrlogy model. From the modeled flow, it 
    parameterizes a probability distribution function (pdf) of flow rates at each
    link in a network model grid representation of the channel network.
    From the mapped depth to soil water file, it parameterizes a pdf of the depth 
    to soil water at each node in the raster model grid representaiton of the watershed.
    
    The external model channel network and grid do not need to match the landlab model exactly.
    Mapping functions determine which DHSVM network model grid links match
    with the network model grid links and which DHSVM grid cells match the landlab
    raster model grid cells. NOTE: to map between raster model grids, the Topmodel
    wetness index for both the Landlab grid and the DHSVM grid is needed. For the
    external model grid (which is assumed to be coarser), use a minimum contributing 
    cell number of 1 cell.
    
    The run one step function randomly picks a storm intensity (return interval) and 
    storm date and updates the raster model grid depth to water table field and/or 
    network model grid flow depth fields. The return interval of the flow and 
    soil-water condition at each link and/or grid cell is assumed uniform accross 
    the basin.
    
    NOTE: untested changes on flow may not work, last working draft is dhsvm_to_landlab_g.py
    TO DO: remove network mapping functions, turn into a utility
    TO DO: remove all functionality related to _load_dtw_different_grids
    TO DO: need to change storm date maker so that the minimum duration between storms is a parameter
    and if all created storms are within the parameter, it doesn't get stuck in the while loop
    author: Jeff Keck
    """

    def __init__(self,
                  grid = None,
                  nmgrid = None,
                  parcels = None,
                  nmgrid_d = None, 
                  DHSVM_flow = None,
                  DHSVM_dtw_dict = None, # dtw maps must include at least as many maps as years in hydro time series.
                  DHSVM_dtw_mapping_dict = None,
                  method = 'storm_generator',
                  flow_aggregation = '24h',
                  begin_year = 2000,
                  end_year = 2010,
                  flow_aggregation_metric='max',
                  flow_representative_reach = 0,
                  bankfull_flow_RI = 1.5,
                  fluvial_sediment_RI = 0.25,
                  hillslope_sediment_RI = 2.5,
                  tao_dict = None,
                  Ct = 5000,
                  BCt = 100000,
                  seed = None,
                  gtm = None
                  ):        

        # run component init
        super().__init__(grid)

        # determine run option based on user input
        # if both a nmg and rmg provided, will be option 1 or 2
        if (nmgrid != None) and (grid != None):
            
            # option 1 and 2 update flow, updating flow requires ChannelNetworkToolsMapper
            if gtm != None:
                self.gtm = gtm
            else:
                self.gtm = ChannelNetworkToolsMapper(grid = grid, nmgrid = nmgrid, Ct = Ct,BCt = BCt)
            # option 1, a dtw dictionary is provided and flow and depth to water table will be updated
            if DHSVM_dtw_dict != None:
                self.opt = 1
            # option 2, a dtw dictionary was not provided and only flow will be updated
            elif DHSVM_dtw_dict == None:
                self.opt = 2                
        
        # option 3, only a grid was provided and only depth to water table will be updated
        elif grid != None:
            # only depth to water table
            self.opt = 3
        else:
            raise ValueError("a network model grid and raster model grid or a" \
                             "raster model grid are required to run DHG")


        # check for network model grid inputs
        if nmgrid != None:
            if nmgrid_d != None:
                self.nmgrid_d = nmgrid_d
            else:
                raise ValueError("network model grid representation of the dhsvm" \
                                 " network required for translating dhsvm flow to" \
                                 " landlab model network model grid")   
            if DHSVM_flow is not None:
                self._streamflowonly = DHSVM_flow/3600 # convert to m3/s
            else:
                raise ValueError("DHSVM streamflow.only pandas datafram not provided")
        
        # check for depth to soil water maps
        if DHSVM_dtw_dict:
            self.appended_maps = DHSVM_dtw_dict['appended_maps']
            self.map_dates = pd.to_datetime(DHSVM_dtw_dict['map_dates'])
            self.nd = len(self.map_dates)  #number of days(maps)

            if DHSVM_dtw_mapping_dict:
                self.grid_d = DHSVM_dtw_mapping_dict['DHSVM_grid']
                self.x_trans = DHSVM_dtw_mapping_dict['x_trans']
                self.y_trans = DHSVM_dtw_mapping_dict['y_trans']
                self.f = DHSVM_dtw_mapping_dict['f']
            else:
                self.grid_d = None
                # val = input("DHSVM grid matches landlab grid? (y/n):")
                # if val != 'y':
                #     raise ValueError("DHSVM mapping dictionary needed to convert"\
                #                      " DHSVM grid values to landlab grid values")    
        else:
            val = input("depth to water table maps not provided, continue? (y/n):")
            if val != 'y':
                raise ValueError("DHSVM depth to water table maps not provided")



        if 'topographic__elevation' in grid.at_node:
            self.dem = grid.at_node['topographic__elevation']
        else:
            raise FieldError(
                'A topography is required as a component input!')
        
        #   flow receiver node
        if 'flow__receiver_node' in grid.at_node:
            self.frnode = grid.at_node['flow__receiver_node']
        else:
            raise FieldError(
                'A flow__receiver_node field is required as a component input!')  

        
        
        # set the seed of the np random generator
        self._maker(seed)
        
        # initial parcels
        if parcels != None:
            self.parcels = parcels
        else:
            self.parcels = None
        
         # flow aggregation metric
        self.flow_metric = flow_aggregation_metric
        
        # representative reach
        self.rep_reach = flow_representative_reach
        
        # bankfull flow return interval [yrs]
        self.Qbf_ri = bankfull_flow_RI
        
        # flow return interval [yrs] at which bedload transport occurs
        self.sed_ri = fluvial_sediment_RI
                               
        # stochastic or time_series
        self._method = method
        
        # begin and end year of stochastic model run
        self.begin_year = begin_year
        self.end_year = end_year
                
        # Event return interval above which landslides occur
        self.ls_ri = hillslope_sediment_RI
    
        # if self.ls_ri < 1: #TODO change this to minimum return interval in dtw time series
        #     raise ValueError("frequency of landslide storm events must be >= 1")
        
        
        # time aggregation
        self.tag = flow_aggregation

        # tau_dict
        if tao_dict != None:
            self.tao_dict = tao_dict
        else:
            self.tao_dict = {'gHG':[0.15,-0.08], 'wHG':[2.205, 0.38], 'dHG':[0.274, 0.24]}       


        ### NM Grid characteristics
        if nmgrid is not None:
            self._nmgrid = nmgrid

        ### Channel extraction parameters

        self.Ct = Ct # Channel initiation threshold [m2]   
        self.BCt = BCt # CA threshold for channels that typically transport bedload [m2] 

        
        # Run options:                          
        # (1) FLOW and DTW 
        if (nmgrid != None) and (DHSVM_dtw_dict != None):
            self._prep_flow()
            self._prep_depth_to_watertable()
        
        # (2) FLOW only
        elif nmgrid != None:     
            self._prep_flow()
        
        # (3) DTW only
        elif DHSVM_dtw_dict: 
            self._prep_depth_to_watertable()


        # initilize storm generator
        if self._method == 'storm_generator':
            
            # create list of storm dates
            self._storm_dates_emperical_cdf()
            
            self._storm_date_maker()
            

            # initialize time
            self._time_idx = 0 # index
            self._time = self.storm_dates[0]  # duration of model run (hours, excludes time between time steps)

        elif self._method == 'time_series':
            self.storm_dates = self.map_dates
            print('this method not implemented yet')
            
        elif self._method == 'mean_and_standard_deviation':
            print('added mean and standard deviation fields to the grid')
        else:
            msg = ("'{}' not an accepted method").format(self._method)
            
            
                             
    def _prep_flow(self):
        """Prepare DHG for generating flow values at each link"""
        
        # map raster model grid cells to network model grid and dhsvm network
        # properties of raster model grid added as class variables in GridTTools      
        # determine raster mg nodes that correspond to landlab network mg

        
        if not hasattr(self,"xyDf"):
            # determine raster mg nodes that correspond to nmg links 
            linknodes = self._nmgrid.nodes_at_link # make this an internal part of CNT
            active_links = self._nmgrid.active_links
            nmgx = self._nmgrid.x_of_node
            nmgy = self._nmgrid.y_of_node            
        
            out = self.gtm.map_nmg_links_to_rmg_nodes(linknodes, active_links, nmgx, nmgy)
            
            self.Lnodelist = out[0]
            self.Ldistlist = out[1]
            self.xyDf = pd.DataFrame(out[2])    
    


            # determine raster mg nodes that correspond to dhsvm links
        if not hasattr(self,"xyDf_d"):
            linknodes = self.nmgrid_d.nodes_at_link # make this an internal part of CNT
            active_links = self.nmgrid_d.active_links
            nmgx = self.nmgrid_d.x_of_node
            nmgy = self.nmgrid_d.y_of_node
        
            out = self.gtm.map_nmg_links_to_rmg_nodes(linknodes, active_links, nmgx, nmgy)
    
            self.Lnodelist_d = out[0]
            self.Ldistlist_d = out[1]
            self.xyDf_d = pd.DataFrame(out[2])           
        
        ## define bedload and debris flow channel nodes       
        ## channel
        # self._ChannelNodes()
        if not hasattr(self.gtm,"ChannelNodes"):
            self.gtm.extract_channel_nodes(self.Ct,self.BCt)
    
        # map dhsvm network model grid to landlab network model grid and prepare
        # time series of flow at each landlab network model grid link
    
        # determine dhsvm network mg links that correspond to the landlab network mg
        # self._map_nmg1_links_to_nmg2_links()
        self.LinkMapper, self.LinkMapL = self.gtm.map_nmg1_links_to_nmg2_links(self.Lnodelist,self.xyDf_d)

        # aggregate flow time series
        self._resample_flow()

        # create reduced size streamflow.only file with index according to nmg links
        self.streamflowonly_nmg()
        
        # determine dhsvm network mg links that correspond to the landlab network mg
        self.gtm.map_nmg_links_to_rmg_channel_nodes(self.xyDf_d)   # not needed? Use
        
        # compute partial duration series and bankful flow rate for each nmgrid_d link
        # used by the nmgrid        
        self.RI_flows()        
    
        # parameterize pdf to flow at each link
        self._flow_at_link_cdf()
        
        # parameterize pdf to flow at a representative reach 
        # (used to represent basin average hydrologic condition)
        self._rep_reach_flow_at_link_cdf()
        
    
    def _resample_flow(self):
        
        if self.tag == 'water_year':
            water_year = (self._streamflowonly_ag.index.month >= 10) + self._streamflowonly_ag.index.year
            self._streamflowonly_ag['water_year'] = water_year
            if self.flow_metric == 'max':
                self._streamflowonly_ag.groupby('water_year').max()
            if self.flow_metric == 'mean':
                self._streamflowonly_ag.groupby('water_year').mean() 
            if self.flow_metric == 'min':
                self._streamflowonly_ag.groupby('water_year').min() 
            # set water index to datetime
            self._streamflowonly_ag.index = pd.to_datetime(dat2.index, format ='%Y')
    
        else:       
            if self.flow_metric == 'max':
                self._streamflowonly_ag = self._streamflowonly.resample(self.tag).max().fillna(method='ffill') #convert sub  hourly obs to hourly
    
            elif self.flow_metric == 'mean':
                self._streamflowonly_ag = self._streamflowonly.resample(self.tag).mean().fillna(method='ffill') #convert sub  hourly obs to hourly
        
            elif self.flow_metric == 'min':
                self._streamflowonly_ag = self._streamflowonly.resample(self.tag).min().fillna(method='ffill') #convert sub  hourly obs to hourly

    def _prep_depth_to_watertable(self):
        """Prepare DHG for generating depth to soil water values at
        each node"""
        # load depth to water table maps into DHG as a 2-D
        # np array
        
        # initially set DHSVM grid wetness index as None
        self.lambda_coarse_wa = None
        
        # rows per map, used to iterate through appended map .asc file
        self.rpm = self._grid.shape[0]
        
        # range of quantile files used to define cdf at each raster model
        # grid node
        self.quant = np.arange(0,1.01,0.01)
        
        if self.grid_d: # if dhsvm gridding scheme does not match landlab
            self._load_dtw_different_grids()
        else: # dhsvm and landlab gridding schemes are the same
            self._load_dtw()
        
        # determine the minimum return interval of the smallest relative
        # wetness event
        self._dtw_min_ri = self._min_return_interval(dates = self.map_dates)       
       
        # parameterize pdf of basin mean relative wetness from maps
        self._mean_relative_wetness_cdf()
        
        # parameterize pdf to saturated zone thickness at each node
        self._saturated_zone_thickness_cdf()


    def hydrograph_to_channel_hydraulics(self):
        """Converts time series of flow at each link to a time series of hydraulic conditions
        including flow velocity, depth and effective depth. This method is called
        before running a model on the network model grid.  
        """
        
        # define Qlinks
        self.Qlinks = self._streamflowonly_ag
        
        self._constant_channel_tau()
        
    
    def _maker(self,seed):
        """prepares the np random generator"""
        
        self.maker = np.random.RandomState(seed=seed)

    
    def _storm_date_maker(self):
        """Creates a date time index for the randomdly generated storm events
        that can be used for tracking changes in sediment dynamics with time.
        only works if map_dates include the day and month of the event.
        
        """
        
        # first determine the number of events in each year
        yrs = [0]
        c=0
        while yrs[c]<(self.end_year-self.begin_year+1):
            dt = self.maker.normal(self.sed_ri, self.sed_ri/2,1)[0]
            if dt<0: # change in time can only be positive
                dt =0.05
            yrs.append(yrs[c]+dt)
            c+=1
        yrs = np.array(yrs)
        YEAR = self.begin_year+np.floor(yrs)

        # next assign a month and day to each event by randomly sampling from
        # an emperical cdf of the day of year large events occur        
        doy = []
        for i,y in enumerate(YEAR):
            # inital random date
            q_i = self.maker.uniform(0,1,1)
            jd = int(self.intp(self.date_quantile, self.date_day, q_i)) # julian day
            
            # make sure date at least 30 days apart from all other dates
            if i >0:
                # while date is within 30 days of any other date, keep resampling a date.
                while (np.abs(YEAR[0:i]*365+np.array(doy[0:i])-
                            (YEAR[i]*365+jd))<30).all():
                    q_i = self.maker.uniform(0,1,1)
                    jd = int(self.intp(self.date_quantile, self.date_day, q_i))
                    
     
            doy.append(jd)
            

        doy = pd.DataFrame(np.array(doy))
        doy.columns = ['day-of-year']
        doyd = pd.to_datetime(doy['day-of-year'], format='%j').dt.strftime('%m-%d')
        
        # combine the year and month and day of the storm
        dates = []
        for i,d in enumerate(doyd):
            date = str(int(YEAR[i]))+'-'+d
            dates.append(date)
        
        # dates in each year may not be in time sequential order
        # sort to make time sequential
        Tx = pd.to_datetime(np.sort(pd.to_datetime(dates)))
        
        #add random bit of time to each date to make sure no duplicates
        Tx_new = []
        for c,v in enumerate(Tx):
            Tx_new.append(Tx[c] + pd.to_timedelta(self.maker.uniform(0,1,1),'h')[0])
            
        Tx_new = pd.Series(Tx_new)
        Tx = pd.to_datetime(Tx_new.values)
        
        self.storm_dates = Tx
                # time
                
       
    def _storm_dates_emperical_cdf(self):
        """creates emperical cdf from partial duration series of precipitation events
        """
        try: 
            pds_doy = self.PDS.index.dayofyear #convert PDS dates to day-of-year equivalent
        except: # if run DTW only, then use the dates of the DTW maps
            pds_doy = self.PDS_s.index.dayofyear
            
        fig, ax = plt.subplots(figsize=(3, 3))
        n, bins, patches = plt.hist(pds_doy, 30, density=True, histtype='step', # emperical cdf
                           cumulative=True, label='Empirical')
        plt.xlabel('day of year')
        plt.ylabel('quantile')
        plt.xlim(bins.min()-1,bins.max())
        
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
            Q_l_Fx, Q_l_x1, Q_l_T, Q_l_Ty, Q_l_Q_ri  = self.fit_probability_distribution(AMS=PDS_Q_l['value'], dist = 'LP3')
            self.Q_l_dist[Link] = [Q_l_Fx, Q_l_x1, Q_l_T, Q_l_Ty, Q_l_Q_ri]
            
            if c%3 == 0:
                print('distribution fit to partial duration series of peak flows at link '+ str(Link))       


    def _rep_reach_flow_at_link_cdf(self):
        # determine probability distribution for flow event magnitude for all events >= to fluvial_sediment_RI 
        # use flow at location where frequency-magnitude relation is representative of hydrologic state of entire basin
        # fit distribution to partial duration series ( may include return intervals less than 1; return interval is not [yrs])
        Q_l = self._streamflowonly_ag[self.rep_reach]
        
        self.PDS = self.partial_duration_series(Q_l, self.sed_ri)
        # self.AMS = self.partial_duration_series(Q_l, 1)
        
        self.Q_Fx, self.Q_x1, self.Q_T, self.Q_Ty, self.u_Q_ri  = self.fit_probability_distribution(AMS=self.PDS['value'],
                                                                                         min_ri = self.PDS['RI [yr]'].min(), dist = 'LP3')
        # fit distribution to annual maximum series, used to determine return interval in year
        # self.Pam_Fx, self.Pam_x1, self.Pam_T, self.Pam_Q_ri  = self.fit_probability_distribution(AMS=self.AMS['value'], dist = 'LP3')
               
    
    def _load_dtw(self):
        """load raw depth to water table ascii file from DHSVM. 
        NOTE, DHSVM mapped outputs are binary. To convert to ascii, use myconvert.c. 
        See DHSVM documentation for use of myconvert.c
        If running with storm generator, file should be a partial duration series
        of the largest relative wetness (smallest depth to water table values)
        """
        
        if type(self.appended_maps) == str:
        # open .asc version of DHSVM depth to soil water mapped output
            f = open(self.appended_maps, 'r')
            M = np.genfromtxt(f)
            f.close()
                
        elif type(self.appended_maps) == xr.core.dataset.Dataset:
            M = self.appended_maps

        # extract each map from appended map, convert nd row array of core nodes
        
        # NOTE!!!: DHSVM mapped outputs are oriented the same as the map i.e., top, 
        # bottom, left, right sides of ascii file are north, south, west and east sides 
        # of the the DHSVM model domain. To convert to a Landlab 1-d array, need to 
        # flip output so that the south-west corner node is listed first
        
        dtw_l_an = [] # depth to water table list, 
        st_l_an = [] # saturated zone thickness list
        rw_l_an = [] # relative wetness list
        c=0
        for i in range(0,self.nd):
            # if an .asc file, extract each map sized chunk of the .asc file
            if type(self.appended_maps) == str: 
                dtw_d = M[c:c+self.rpm,:] # map, DHSVM orientation
                dtw_l = np.hstack(np.flipud(dtw_d))[self._grid.core_nodes] 
            # if a dataset, iterate over each time coordinate to acess data array
            elif type(self.appended_maps) == xr.core.dataset.Dataset:  
                dtw_d = np.squeeze(self.appended_maps.isel(time=[i])['wt'].values)
                dtw_l = np.hstack(dtw_d)[self._grid.core_nodes]
            st_l = self._grid.at_node['soil__thickness'][self._grid.core_nodes]-dtw_l # thickness of saturated zone
            rw_l = st_l/self._grid.at_node['soil__thickness'][self._grid.core_nodes]
            c = c+self.rpm
            
            dtw_l_an.append(dtw_l)
            st_l_an.append(st_l)
            rw_l_an.append(rw_l)
        
        # save list of maps as a 2-d np array. each row 'i' is a 1-d representation
        # of the map for date 'i'
        self.dtw_l_an = np.array(dtw_l_an) 
        self.st_l_an = np.array(st_l_an)
        self.rw_l_an = np.array(rw_l_an)
           

    def _mean_saturated_zone_thickness_cdf(self):
        """parammeterize a pdf to a partial duration series of the basin mean 
        saturated zone thickness - NOT USED"""
        
        self.PDS_s = pd.Series(data = self.dtw_l_an.mean(axis=1), index = pd.to_datetime(self.map_dates))
        self.Fx_s, self.x1_s, self.T_s, self.Ty_s, self.Q_ri_s = self.fit_probability_distribution(self.PDS_s,dist = 'LN', print_figs = print_figs)
                 

    def _mean_relative_wetness_cdf(self):
        """parammeterize a pdf to a partial duration series of the basin mean 
        relative wetness (staturated zone thickness / soil thickness)"""        

        self.PDS_s = pd.Series(data = self.rw_l_an.mean(axis=1), index = pd.to_datetime(self.map_dates))
        self.Fx_s, self.x1_s, self.T_s, self.Ty_s, self.Q_ri_s = self.fit_probability_distribution(self.PDS_s,dist = 'LN')
                 
   
    def _saturated_zone_thickness_cdf(self):
        """ at each core node, parameterize a cdf (pdf) to the partial duration series
        ( or annual maximum series) of maximum saturated thickness, solve the cdf at 
        1000 points along the domain of the cdf (0 to 1) and save as a row in an xarray 
        dataaray. Each row of the data array corrisponds to one of the core nodes.
        """
    
        # version 2, handle constant saturated thickness cells (during extreme events)
        def interpolate_q_val(x):
            scale = np.exp(x[0]); s = x[1]
            x1 = np.log(sc.stats.lognorm.ppf(self.quant,s,scale = scale))
            if s ==0: # if standard deviation is 0 (constant)
                x1 = np.ones(len(self.quant))*scale # cannont fit distribution, constant cdf.  
            return x1
        
        sat_thick = self.st_l_an
        mu_s = np.nanmean(self.st_l_an,axis=0) # nanmean(axis=0) # each column is a node
        ss = np.nanstd(self.st_l_an, axis = 0) # .std(axis=0)
        moments = np.array([mu_s,ss]).T
    
        st_cdf = [interpolate_q_val(x) for x in moments]  # only need x1, Fx is same for all    
    
        # change to DataArray
        self.st_cdf = xr.DataArray(data = st_cdf,
                        dims = ["node","quantile"],
                        coords = dict(
                                node=(["node"], self._grid.core_nodes ),
                                quantile=(["quantile"],self.quant)))

        mu_s_field = (np.ones(self._grid.at_node['soil__thickness'].shape[0])*np.nan).astype(float)
        mu_s_field[self._grid.core_nodes] = mu_s
        self._grid.add_field("node", "thickness__sat_zone_mean", mu_s_field, clobber = True)

        ss_field = (np.ones(self._grid.at_node["soil__thickness"].shape[0])*np.nan).astype(float)
        ss_field[self._grid.core_nodes] = ss
        self._grid.add_field("node", "thickness__sat_zone_stdev", ss_field, clobber = True)

    # interplate function used by other functions
    # grid tools also has this function
    def intp(self, x,y,x1,message = None): 
        '''ALL
        Parameters
        ----------
        x : list, np.array, pd.series of float or int
            x values used for interpolation
        y : list, np.array, pd.series of float or int
            x values used for interpolation
        x1 : float or int within domain of x
            interpolate at x1
    
        Returns
        -------
        float
            y1, interpolated value at x1
    
        '''  
        if x1 <= min(x):
            y1 = min(y)
            # print('TRIED TO INTERPOLATE AT '+str(x1)+' BUT MINIMUM INTERPOLATION RANGE IS: '+str(min(x)))
            if message:
                print(message)
        elif x1 >= max(x):
            y1 = max(y)
            # print('TRIED TO INTERPOLATE AT '+str(x1)+' BUT MAXIMUM INTERPOLATION RANGE IS: '+str(max(x)))
            if message:
                print(message)
        else:            
            f = sc.interpolate.interp1d(x,y)   
            y1 = f(x1)
        return y1
    
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
            self._nmgrid.at_link["flow"] = np.array(Q_)
            self._nmgrid.at_link["unit_flow"] = np.array(q_)                
            self._nmgrid.at_link["flow_depth"] = np.array(d_)
            self._nmgrid.at_link['total_stress'] = np.array(T_)
            self._nmgrid.at_link['mean_velocity'] = np.array(U_)
            self._nmgrid.at_link['mean_grain_velocity'] = np.array(Uo_)            
            self._nmgrid.at_link['total_roughness'] = np.array(nt_)
            self._nmgrid.at_link['grain_roughness'] = np.array(no_) 
            self._nmgrid.at_link['effective_stress'] = np.array(Teff_)             
            self._nmgrid.at_link['effective_stress_b'] = np.array(Teffr_)  
            self._nmgrid.at_link['effective_flow_depth'] = np.array(deff_)         
                                                                                
    
    def _constant_channel_tau(self):
        """
        computes a time series of hydraulic conditions using the hydrograph
        at each link assuming channel slope and grain size are constant with time

        tao_dict = {'gHG':[0.15,-0.08], 'wHG':[2.205, 0.38], 'dHG':[0.274, 0.24]}                 
        """
     
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
            
    def D50_parcels(self, Link):
        
        ct = self.parcels.dataset.time[-1]
        cur_parc = self.parcels.dataset['D'].sel(time = ct).values
        msk = self.parcels.dataset['element_id'].sel(time = ct) == Link
        # msk = np.where(self.parcels.dataset['element_id']==Link)
        Dp = pd.DataFrame(cur_parc[msk])#np.where(self.parcels.dataset['element_id']==Link)].values.flatten())
        
        # determine 50th percentile value
        D50 = Dp.quantile(.5).values #diam. parcels     
        return D50

    
    def D50_hydraulic_geometry(self,drainage_area):
        """ 
        Parameters
        ----------
        user_d50 : list
            TYPE: list of length 1 or 2
            list of length 1: value D50 of all links in the network
            list of length 2: the first value is the coefficient, the second
                value is the exponent of a hydraulic geomtry relation between D50 and 
                grid.at_link.drainage_area. 
    
        Raises
        ------
        ValueError
            if user_d50 is not a list or length 1 or 2, .
    
        Returns
        -------
        ndarray of float
            D50.
    
        """
        user_D50 = self.tao_dict['gHG']
        
        if (type(user_D50) == list) and (len(user_D50)<=2) and (len(user_D50)>0):
    
            if len(user_D50) == 2: # specified contributing area and d50 relation
                a = user_D50[0]
                n = user_D50[1]
                D50  = a*drainage_area**n
            if len(user_d50) == 1: # d50 is constance across basin
                D50 = np.full_like(element_id, user_D50[0], dtype=float)
        else:
            msg = "user defined D50 must be a list of length 1 or 2"
            raise ValueError(msg)
            
        return D50
    
    
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
        CA = self._nmgrid.at_link['drainage_area'][Link]/1e6 # m2 to km2
        S = self._nmgrid.at_link['channel_slope'][Link]              

        # determine link D50
        # parcel grain size in link i
        # replace these lines
        # ct = self.parcels.dataset.time[-1]
        # cur_parc = self.parcels.dataset['D'].sel(time = ct).values
        # msk = self.parcels.dataset['element_id'].sel(time = ct) == Link
        # Dp = pd.DataFrame(cur_parc[msk])#np.where(self.parcels.dataset['element_id']==Link)].values.flatten())
        
        # # determine 50th percentile value
        # D50 = Dp.quantile(.5).values #diam. parcels          
        
        if self.parcels:
            D50 = self.D50_parcels(Link)
        else:
            D50 = self.D50_hydraulic_geometry(CA)
                
        # grain size distribution based on d50 and assumed 
        # log normal distribution (Kondolf and Piegay 2006)
        Gdist = pd.DataFrame(self.maker.lognormal(np.log(D50),0.5,size=1000))            
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

        U,Uo,nt,no,T,Teff,Teffr,deff,Teff_s1,Teff_s2 = self.flow_resistance_RR(q,d,S,D65,D84)
        
        return (Q, q, d, T, U, Uo, nt, no, Teff, Teffr, deff)
    
        
    def RI_flows(self):
        """
        runs annual_maximum_series OR partial_duration_series for each link 
        in nmgrid and creates the RIflows dictionary, which has the bankfull 
        flow magnitude and partial duration series of each link

        Returns
        -------
        None.

        """
            
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
         
        self._streamflowonly_ag = pd.DataFrame.from_dict(nmg_streamflow_dict) # only includes the columns associated with the nmg
    
    
    def get_flow_at_links(self, q_i):
        """
        Parameters
        ----------
        q_i : float
            quantile of cdf used to determine a flow rate
    
        Returns
        -------
        Q_i float
            flow rate for the given quantile
    
        """
        Qlink_dict = {}
        for Link in self.LinkMapper.keys():     
            dst = self.Q_l_dist[Link]
            Fx = dst[0]; x1 = dst[1]; 
            Q_i = self.intp(Fx,x1,q_i, message = '#######Q#### at link '+str(Link))
            Qlink_dict[Link] = Q_i
           
        self.Qlinks = pd.DataFrame.from_dict(Qlink_dict, orient = 'index')
        
        
    def get_depth_to_water_table_at_node(self, q_i):
        """given the quantile value of a soil recharge event, look up the corrisponding
        depth to water table at each node, update the depth__to_water_table field"""
        
        def interp1d_np(data, x, xi):
            return np.interp(xi, x, data)
    
        #use xr.apply_ufunc to quickly look up quanntile value
        # #%%time 
        st = xr.apply_ufunc(
            interp1d_np,  # first the function
            self.st_cdf,  # now arguments in the order expected by 'interp1_np'
            self.st_cdf.coords['quantile'],
            q_i,
            input_core_dims = [["quantile"],["quantile"],[]],
            exclude_dims=set(("quantile",)),
            vectorize = True
        ) 
      
        # soild depth at core nodes
        soild = self._grid.at_node['soil__thickness'][self._grid.core_nodes]
        
        # apply constraints
        st[st<0]  = 0 # saturated thickness cannot be less than zero
        st[st>soild] = soild[st>soild] # saturated thickness cannot be greater than soil thickness
        
        # convert saturated thickness to depth to water table
        dtw = soild-st.values
        
        # # apply constraints
        # # depth to water table cannot be less than zero
        # dtw[dtw<0]  = 0
    
        # # depth to water table cannot be greater than soil thickness
        # dtw[dtw>soild] = soild[dtw>soild]
        
        # update
        #self._grid.at_node['depth__to_water_table'][self._grid.core_nodes] = dtw
        
        dtw_field = (np.ones(self._grid.at_node['soil__thickness'].shape[0])*np.nan).astype(float)
        dtw_field[self._grid.core_nodes] = dtw
        
        self._grid.add_field('node', 'saturated__thickness', dtw_field, clobber = True)
        self._grid.add_field('node', 'depth__to_water_table', dtw_field, clobber = True)


    
    def _run_one_step_flow(self, ts = None):
        """updates the flow depth field of the network model grid"""

        if ts is None:
            self.q_i = self.maker.uniform(self.Q_Fx.min(), self.Q_Fx.max(),1)[0]
            
            # get the flow rate at representative link and equivalent 
            # event return interval cdf of flow rates to get flow
            self.Q_i = self.intp(self.Q_Fx, self.Q_x1, self.q_i, message = None)
            
            # cdf of flow rate in terms of annual return interval
            self.Q_ri = self.intp(self.Q_Fx, self.Q_Ty, self.q_i, message = None)
                            
            ## GET FLOW RATES
            # get flow value at each link for quantile self.q_i (self.Qlinks)
            self.get_flow_at_links(self.q_i)
            # convert to flow depth and effective flow depth       
            self._variable_channel_tau()

        else:
 
            self.Qlinks = self._streamflowonly_ag.loc[ts] # flow rate in all links at time ts

            self._variable_channel_tau()        

        
    def _run_one_step_DTW(self, ts = None, DTW_only = True):
        """updates the depth to soil water field of the raster model grid"""
        
        # convert return interval to an annual quantile value if 
        # return interval > 1 yr. Note: sub-annual return intervals 
        # can not be converted to an annual quantile value
        if ts is None:
            
            # if only updating the depth to water of the raster model grid
            if DTW_only is True:
                print('DTW_only is True')
                self.q_i = self.maker.uniform(self.Fx_s.min(), self.Fx_s.max(),1)[0]
    
                # get the flow rate at representative link and equivalent 
                # event return interval cdf of flow rates to get flow
                self.s_i = self.intp(self.Fx_s, self.x1_s, self.q_i, message = None)
                
                # cdf of flow rate in terms of annual return interval (this is event
                # return interval if dtw maps are same length as number of years)
                self.s_ri = self.intp(self.Fx_s, self.Ty_s, self.q_i, message = None)
    
                # if annual return interval greater than return interval of smallest
                # storm included in depth to water table partial duration series
                # and larger than user specified minimum return period to update
                # grid
                if (self.s_ri > self._dtw_min_ri) & (self.s_ri > self.ls_ri):
                    self.q_yrs_i = 1-1/self.s_ri
                    self.get_depth_to_water_table_at_node(self.q_yrs_i)
            
            else:

                    self.q_yrs_i = 1-1/self.Q_ri                
                    self.get_depth_to_water_table_at_node(self.q_yrs_i)
            
            
    def run_one_step(self, ts = None):
        
        """computes hydraulc condtions at time ts given the slope and d50 of
        an evolving network model nmgrid. D50 is computed from parcels in each link.
        slope is read from nmgrid, which is updated by NST. hydraulic condtions are 
        saved as attributes of the nmgrid.
        ts = dateime, time stamp
        
        output is a 1d np array of flow depth at each link of the landlab network mg
        
        """                   
        if self._time_idx <= len(self.storm_dates):
        

               
            # Run options:                          
            # (1) FLOW and DTW 
            if self.opt == 1:
                self._run_one_step_flow(ts)
                self._run_one_step_DTW(ts, DTW_only = False)
            
            # (2) FLOW only
            elif self.opt == 2:     
                self._run_one_step_flow(ts)
            
            # (3) DTW only
            elif self.opt == 3: 
                self._run_one_step_DTW(ts)  
                    
                
            # UPDATE TIME FOR NEXT ITERATION
            if self._time_idx < len(self.storm_dates): 
                
                # date of storm
                self._time = self.storm_dates[self._time_idx] 
                
                # time between storms
                self._dtime = self.storm_dates[self._time_idx]-self.storm_dates[self._time_idx-1]        
            
                # update time       
                self._time_idx +=1 # index        
        
        
        else:
            msg = "end of time series"
            raise ValueError(msg)  


    def _min_return_interval(self, dates, plotting_position = 'Weibull'):
        """given the dates of a partial duration series, returns the minimum
        annual return period"""
        dates = pd.to_datetime(dates)
        Yrs = max(dates.year)-min(dates.year)+1 # number of years in time series
    
        n = dates.shape[0]            
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
        
        T = 1/PP                        
        return min(T)        
    
    def partial_duration_series(self, time_series, RI_cutoff = 1, plotting_position = 'Weibull',sep=30):
        """
        Uses a peaks-over-threshold approach to create a partial duration series
        from a time series of hydrologic data. 
        
        The PDS is truncated to exclude all events with RI < RI_cutoff 
        
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
    
        """
        
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

    
    def fit_probability_distribution(self, AMS, min_ri = 1, dist = 'LN', 
                       RI=[1.5], plotting_position = 'Weibull',
                       print_figs = False):
        """
        TODO: use scipy functions lognorm.ppf(quantile,s= std, scale = exp(mean))
        and pearson3.ppf(quantile, skew, scale = exp(mean))
        apply to an array or dataarray, whichever faster
        
        Fits distribution to annual maximum series or partial duration series of
        data using built in numpy function and methods described in: 
       
             Maidment, 1992, Handbook of Hydrology, Chapter 18
             
        NOTE on interpreation of the fit distribution: 
        
        IF the distribution is fit to an annual maximum series, the resultant pdf 
        gives the annual liklihood, the likelihood of a given magnitude occuring 
        during a single year. 
        
        If the distribution is fit to a partial duration series that lists magnitudes
        less than a return interval of 1, then the resultant pdf gives the event likelihood, 
        the likelihood of a given magnitude occuring during any event larger than the 
        minimum return interval included in the partial duration series. 
        e.g., if the partial duration series includes is fit to events as small as 
        the 0.25 year event,then the fit distribtuion gives the likelihood of a given 
        flow magnitude during any 0.25 year and larger storm. 
        
        To convert the event return period to an annual return period, divide the 
        event return period by the minimum return interval included in the partial
        duration series (min_ri)
           
        Parameters
        ----------
        AMS : pd series, index is pd.datetime
            annual maximum series (or partial duration series)
        min_ri: float
            minimum return period included in the partial duration series. 
            If annual maximum series, this valueis 1.
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
            cdf quantile (domain of cdf, 0 to 1)
        x1 : np.array
            cdf value
        T : np.array
            quantile value converted to an event return interval [event] 
        Ty : np.array
            annual return interval equivalent to quantile value [years]
        Q_ri : dictionary
            key is each value in RI, value is magnitude        
        """
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
            s = self.maker.lognormal(mu_lognormal, sigma_lognormal, 10000)
            
            x1 = np.linspace(s.min(), s.max(), 10000)
            
            #for comparison to emperical estimate using plotting position
            x2 = np.sort(AMS.values, axis=0) # values in AMS, sorted small to large (pp is quantile)
        
            X = [x1,x2]
        
            fx = {}
            for i,x in enumerate(X):
                fx[i] = (np.exp(-(np.log(x) - mu_lognormal)**2 / (2 * sigma_lognormal**2)) \
                 / (x * sigma_lognormal * np.sqrt(2 * np.pi)))
                
            Fx = sc.integrate.cumulative_trapezoid(fx[0], x1, initial=0)
                 
        
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
            
            # cdf created by summing area under pdf
            Fx = sc.integrate.cumulative_trapezoid(fx[0], x1, initial=0)
            
        
       
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
        
        T = 1/(1-Fx) # return interval [event] from non-exceedance probability
        
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
    
        # if events with a return interval less than 1 year included in partial
        # duration series, convert return period per minimum event to annual return period
        
        if min_ri < 1:
            Ty = T/(1/min_ri) # return period [yrs]
        else:
            Ty = T
    
        return Fx, x1, T, Ty, Q_ri 
                
    
    def depth_trapezoid(self, Q, S, b, m1 = 1, m2 = 1, n = 0.05):
        
        """
        A main channel = A_t(b,m1,m2,y)
        P main channel = P_t(b,m1,m2,y)
        """
    
        def A_t(b,m1,m2,y):
            """
            Area of trapezoid, use for all trapezoids in compound channel
            """
            A = (y/2)*(b+b+y*(m1+m2))
            return A
        
        def P_t(b,m1,m2,y):
            """
            Wetted perimeter of trapezoid, below flood plains
            """
            P = b+y*((1+m1**2)**(1/2)+(1+m2**2)**(1/2))
            return P
                
        def h(y):
            return np.abs(Q-(1/n)*A_t(b,m1,m2,y)*((A_t(b,m1,m2,y)/P_t(b,m1,m2,y))**(2/3))*S**(1/2))
    
        r = sc.optimize.minimize_scalar(h,method='bounded',bounds = [.1,10])
        
        y = r.x
        Rh = A_t(b,m1,m2,y)/P_t(b,m1,m2,y) 
        return (y,Rh)
            
       
    
    def flow_resistance_RR(self, q,d,S,D65,D84):
        """
        # TODO: reduce number of returned values
        
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
            
                
        TODO width and depth approximations are updated based on parker 2007
   
        """
        g = 9.81 # m/s2
        ro = 1000 # kg/m3
        
        # depth slope product for total stress
        T = (ro)*9.81*d*S #total stress acting on flow [N/m2]
        
        # compute q**, RR2011 eq 11
        qss = q/((g*S*D84**3)**0.5)
        # compute U**, RR2011 eq 22
        Uss = (1.443*qss**0.60)*(1+(qss/43.78)**0.8214)**-0.2435
        # compute U, RR2011 eq 12
        U = Uss*(g*S*D84)**0.5
        # compute d
        # d = q/U
        # compute Uo, RR2011 eq 20A, U** replaced with eq 12, q** replaced with eq 11
        # rearrange to solve for Uo, velocity equivalent to the grain roughness
        # see notes
        Uo = 3.70*(q**0.4)*(g**0.3)*(S**0.3)*(D84**-0.1)
        # compute total friction factor and roughness ftot, ntot
        ft = (8*g*q*S)/(U**3) # chezy formula, eq 1

        # nt = (((q**0.4)*(S**0.3))/U)**(5/3) # mannings wide channel approximation  
        nt = ((S**0.5)*(d**(2/3)))/(((8*g*d*S)**0.5)/(ft**0.5)) # Darcy-Weisbach         
        # compute grain friction factor and roughness, fo, no
        fo = (8*g*q*S)/(Uo**3) # chezy formula, eq 1

        # no = (((q**0.4)*(S**0.3))/Uo)**(5/3) # mannings wide channel approximation  
        no = ((S**0.5)*(d**(2/3)))/(((8*g*d*S)**0.5)/(fo**0.5)) # n from friction factor, RR2011 eq 1              
        
        # grain roughness approach
        Teff_e = T*(no/nt)**1.5 # Istanbulluoglu, 2003
        
        # grain roughness and velocity approach
        Teff_w = 0.052*ro*((9.81*S*D65)**0.25)*(U**1.5) # Wilcock 2001             
        
        # reduced slope approach
        Seff_s1 = S*((fo/ft)**0.5)**1.5 # Schneider et al., 2015, Rickenmann 2012 eq 28.12
        Seff_s2 = S*((U/Uo)**1.5)**1.5 # using RR2011 eq. 27a for (fo/ft)**0.5        
        # depth slope product using reduced slope
        Teff_s1 = (ro)*9.81*d*Seff_s1
        Teff_s2 =  (ro)*9.81*d*Seff_s2       
    
        deff = d*(Teff_s1/T) # equivalent depth of effective flow  
        
        return (U,Uo,nt,no,T,Teff_w,Teff_e,deff,Teff_s1, Teff_s2)
    
    
    




