
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy as sc
from scipy.stats import moment as lm
from scipy.special import gamma

import xarray as xr
from landlab.plot import graph

from landlab.utils.grid_t_tools_a import GridTTools

class DHSVMtoLandlab(GridTTools):

    """this component takes the raw modeled flow and depth to soil water output 
    from DHSVM. From the modeled flow, it parameterizes a probability distribution 
    function (pdf) of flow rates at each link in a network model grid representation
    of the channel network. From the mapped depth to soil water file, it parameterizes
    a pdf of the depth to soil water at each node in the raster model grid 
    representaiton of the watershed.
    
    The dhsvm channel network and grid do not need to exactly match the landlab model.
    Mapping functions convert determine which DHSVM network model grid links match
    with the network model grid links and which DHSVM grid cells match the landlab
    raster model grid cells.
    
    The run one step function randomly picks a storm return interval and updates
    the raster model grid depth to water table field and network model grid
    flow depth fields assuming a uniform hydrologic frequency accross
    the basin.

    THe minimum return interval of the randomly selected storm intensities are specified
    by the user.
    
    
    TODO width and depth approximations are updated based on parker 2007
         storm duration generator
         D50 is approximated based on contributing area, parcels are optional
    
    
    author: Jeff Keck
    """

    def __init__(self,
                  grid = None,
                  nmgrid = None,
                  parcels = None,
                  nmgrid_d = None, 
                  DHSVM_flow = None,
                  DHSVM_dtw_dict = None, # dtw maps must include at least as many maps as years in hydro time series.
                  DHSVM_dtw_mapping_dict = None
                  precip = None,
                  method = 'storm_generator',
                  flow_aggregation = '24h',
                  begin_year = 2000,
                  end_year = 2010,
                  flow_aggregation_metric='max',
                  flow_representative_reach = 0,
                  bankfull_flow_RI = 1.5,
                  fluvial_sediment_RI = 0.25,
                  hillslope_sediment_RI = 2.5,
                  tao_dict = None
                  ):        


        # call __init__ from parent classes
        if (nmgrid != None) and (grid != None):
            super().__init__(grid, nmgrid)
            if DHSVM_dtw_dict != None:
                self.opt = 1
            elif DHSVM_dtw_dict == None:
                self.opt = 2                
        elif grid != None:
            super().__init__(grid)
            self.opt = 3
        else:
            raise ValueError("a network model grid and raster model grid or a" \
                             "raster model grid are required to run DHSVMtoLandlab")


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
            self.appended_maps_name = DHSVM_dtw_dict['appended_maps_name']
            self.rpm = DHSVM_dtw_dict['rows_per_map']
            self.map_dates = DHSVM_dtw_dict['map_dates']
            self.nd = self.map_dates.shape[0]  #number of days(maps)

            if DHSVM_dtw_mapping_dict:
                self.DHSVM_dem = DHSVM_dtw_mapping_dict['DHSVM_dem']
                self.x_trans = DHSVM_dtw_mapping_dict['x_trans']
                self.y_trans = DHSVM_dtw_mapping_dict['y_trans']
                self.DHSVM_lambda = DHSVM_dtw_mapping_dict['DHSVM_lambda']
                self.landlab_lambda = DHSVM_dtw_mapping_dict['landlab_lambda']
                self.f = DHSVM_dtw_mapping_dict['f']
            else:
                val = input("DHSVM grid matches landlab grid? (y/n):")
                if val != 'y':
                    raise ValueError("DHSVM mapping dictionary needed to convert"\
                                     " DHSVM grid values to landlab grid values")    
        else:
            self.rpm = 0
            val = input("depth to water table maps not provided, continue? (y/n):")
            if val != 'y':
                raise ValueError("DHSVM depth to water table maps not provided")
        
        
        
        # initial parcels
        if parcels != None:
            self.parcels = parcels
        else:
            self.parcels = None
               
        # stochastic or time_series
        self._method = method
        
        # begin and end year of stochastic model run
        self.begin_year = begin_year
        self.end_year = end_year
        
        # flow aggregation metric
        self.flow_metric = flow_aggregation_metric
        
        # representative reach
        self.rep_reach = flow_representative_reach
        
        # bankfull flow return interval [yrs]
        self.Qbf_ri = bankfull_flow_RI
        
        # flow return interval [yrs] at which bedload transport occurs
        self.sed_ri = fluvial_sediment_RI
        
        # daily precipitation event interval above which landslides occur
        self.ls_ri = hillslope_sediment_RI
    
        if self.ls_ri < 1: #TODO change this to minimum return interval in dtw time series
            raise ValueError("frequency of landslide storm events must be >= 1")
        
        # time aggregation
        self.tag = flow_aggregation

        # tau_dict
        if tao_dict != None:
            self.tao_dict = tao_dict
        else:
            self.tao_dict = {'gHG':[0.15,-0.08], 'wHG':[2.205, 0.38], 'dHG':[0.274, 0.24]}       
        
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

            
                             
    def _prep_flow(self):
        """Prepare DHSVMtoLanlab for generating flow values at each link"""
        
        # map raster model grid cells to network model grid and dhsvm network
        # properties of raster model grid added as class variables in GridTTools      
        # determine raster mg nodes that correspond to landlab network mg
        linknodes = self._nmgrid.nodes_at_link
        active_links = self._nmgrid.active_links
        nmgx = self._nmgrid.x_of_node
        nmgy = self._nmgrid.y_of_node

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
    
        # map dhsvm network model grid to landlab network model grid and prepare
        # time series of flow at each landlab network model grid link
    
        # determine dhsvm network mg links that correspond to the landlab network mg
        self._DHSVM_network_to_NMG_Mapper()
        
        # aggregate flow time series
        if self.flow_metric == 'max':
            self._streamflowonly_ag = self._streamflowonly.resample(self.tag).max().fillna(method='ffill') #convert sub  hourly obs to hourly

        elif self.flow_metric == 'mean':
            self._streamflowonly_ag = self._streamflowonly.resample(self.tag).mean().fillna(method='ffill') #convert sub  hourly obs to hourly
    
        elif self.flow_metric == 'min':
            self._streamflowonly_ag = self._streamflowonly.resample(self.tag).min().fillna(method='ffill') #convert sub  hourly obs to hourly

        elif (type(metric) is float) or (type(metric) is int):
            self._streamflowonly_ag = self._streamflowonly.groupby(pd.Grouper(freq='d')).quantile(metric,interpolation = 'nearest')
                
        # create reduced size streamflow.only file with index according to nmg links
        self.streamflowonly_nmg()
        
        # determine dhsvm network mg links that correspond to the landlab network mg
        self._DHSVM_network_to_RMG_Mapper()    
        
        # compute partial duration series and bankful flow rate for each nmgrid_d link
        # used by the nmgrid        
        self.RI_flows()        
    
        # parameterize pdf to flow at each link
        self._flow_at_link_cdf()
        
        # parameterize pdf to flow at a representative reach 
        # (used to represent basin average hydrologic condition)
        self._rep_reach_flow_at_link_cdf()
        

    def _prep_depth_to_watertable(self):
        """Prepare DHSVMtoLanlab for generating depth to soil water values at
        each node"""
        # load depth to water table maps into DHSVMtoLandlab as a 2-D
        # np array
        
        if DHSVM_dtw_mapping_dict: # if dhsmv gridding scheme does not match landlab
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
        """
        converts time series of flow at each link to a time series of hydraulic conditions
        including flow velocity, depth and effective depth. This method is called
        before running a model on the network model grid. The network model grid
        model then iterates through each time stamp of the channel hydraulics 
        """
        
        # define Qlinks
        self.Qlinks = self._streamflowonly_ag
        
        self._constant_channel_tau()
        
        
    def _storm_date_maker(self):
        """
        creates a date time index for the randomdly generated storm events
        that can be used for tracking changes in sediment dynamics with time
        """
        
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
        """creates emperical cdf from partial duration series of precipitation events
        """
        
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
        
        # open .asc version of DHSVM depth to soil water mapped output
        f = open(self.appended_maps_name, 'r')
        M = np.genfromtxt(f)
        f.close()
                
        # extract each map from appended map, convert nd row array of core nodes
        
        # NOTE!!!: DHSVM mapped outputs are oriented the same as the map i.e., top, 
        # bottom, left, right sides of ascii file are north, south, west and east sides 
        # of the the DHSVM model domain. To convert to a Landlab 1-d array, need to 
        # flip output so that the south-west corner node is listed first
        
        dtw_l_an = [] # depth to water table list, 
        st_l_an = [] # saturated zone thickness list
        rw_l_an = [] # relative wetness list
        c=0
        for i in range(1,self.nd+1):
            dtw_d = M[c:c+self.rpm,:] # map, DHSVM orientation
            dtw_l = np.hstack(np.flipud(dtw_d))[self._grid.core_nodes] # map, converted to landlab orientation, keep only core nodes
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
       
    
    def DHSVM_to_RMG_cell_mapper(self):
        
        
        # first create a dataframe of the southwest (x1,y1) and northeast (x2,y2)
        # corners of all grid cells in the landlab grid
        self.df = pd.DataFrame(zip(self.ndxw,self.ndys,self.ndxe,self.ndyn),columns = ['x1','y1','x2','y2'])
        
        # required inputs:
        # self variables: DHSVM_dem, x_trans, y_trans
        # coarse DEM
        self.grid_d, z_ = read_esri_ascii('D:/UW_PhD/PreeventsProject/NooksackGroupProject/dem_mapping/test_with_b694/b694dff33msc.asc', name='topographic__elevation')
        
        # x and y translation to apply to the coarse grid
        self.x_trans = -50
        self.y_trans = -50
                
        ## function
        # first establishes coarse grid topology and applies translation
        dx_d = self.grid_d.dx #width of cell
        dy_d = self.grid_d.dy #height of cell
        
        # receivers = frnode #receiver nodes (node that receives runoff from node) = frnode #receiver nodes (node that receives runoff from node)
        
        # nodes, reshaped in into m*n,1 array like other mg fields
        self.nodes_d = self.grid_d.nodes.reshape(self.grid_d.shape[0]*self.grid_d.shape[1],1)
        self.rnodes_d = self.grid_d.nodes.reshape(self.grid_d.shape[0]*self.grid_d.shape[1]) #nodes in single column array
     
        gridx_d = self.grid_d.node_x+self.x_trans 
        gridy_d = self.grid_d.node_y+self.y_trans
        
        # determine coordinates of southwest and northeast corners  
        ndxe_d = gridx_d+dx_d/2
        ndxw_d = gridx_d-dx_d/2
        ndyn_d = gridy_d+dy_d/2
        ndys_d = gridy_d-dy_d/2
        
        
        # convert coordinantes into a dataframe
        DF = pd.DataFrame(zip(ndxw_d,ndys_d,ndxe_d,ndyn_d),columns = ['X1','Y1','X2','Y2'])
    
    
    
    
        def area_weights_overlapping_cells_df(row):
            """determines area of overlap between two rectangles"""

            DF['x1'] = row['x1']; DF['y1'] = row['y1']; DF['x2'] = row['x2']; DF['y2'] = row['y2'];
            dx = DF[['x2','X2']].min(axis=1)-DF[['x1','X1']].max(axis=1)
            dy = DF[['y2','Y2']].min(axis=1)-DF[['y1','Y1']].max(axis=1)
            mask = (dx>0) & (dy>0)
            overlapping_cells = rnodes_d[mask]
            DFa = DF[mask]
            overlapping_cells_area = dx[mask]*dy[mask]

            return overlapping_cells, overlapping_cells_area
   
   
        self.cells_areas = self.df.apply(area_weights_overlapping_cells_df, axis =1)



    def dhsvm_dtw_to_landlab_dtw(self, DHSVM_dtw_map):
        """given a map of dhsvm dtw values on a coarser griding schem, 
        interpoltes dtw at the landlab gridding scheme"""
    
        # self variables: DHSVM_lambda, landlab_lambda, f
        # note: minimum ca of 1 cell (rather than 0 for a single cell) used to determine lambda
        # coarse lambda values
        (mg1, lambda_coarse) = read_esri_ascii(self.DHSVM_lambda, name='lambda')
        self.grid_d.add_field('node',  'lambda', lambda_coarse)
        # fine lambda values
        (mg2, lambda_fine) = read_esri_ascii(self.landlab_lambda, name='lambda')
        self._grid.add_field('node', 'lambda',lambda_fine)
        # topomodel f value
        f = 2
    
        # value = cells_areas[nm]
        def weighted_average_value_df(val_i, coarse_grid_values):
            """given the cells and percent weight of each cell, determines the weighted
            average value at a different cell"""
            # computes weighted average lambda value of a cell
            cells = val_i[0]
            weights  = val_i[1].values
            values = coarse_grid_values[cells]
            wtvalue = sum(values*(weights/100))
            return wtvalue
        
        lambda_fine = cells_areas.apply(weighted_average_value_df, args = (lambda_coarse,))
    
    
        dtw_fine = DHSVM_dtw_map + (mg.at_node['LambdaCoarse']-mg.at_node['LambdaFine'])/f    
    
        return dtw_fine


    
    def _load_dtw_different_grids(self):
        """if dhsvm grid does not match the landlab grid, use this function to
        load the dtw maps into landlab.
        
        This function converts a coarse grid of modeled hydrology to a fine grid
        useing the topmodel lambda approximation
        """
                # open .asc version of DHSVM depth to soil water mapped output
        f = open(self.appended_maps_name, 'r')
        M = np.genfromtxt(f)
        f.close()
                
        # first determine overlapping dhsvm grid cells for each landlab grid cell
        # for large grids this is slow
        self.DHSVM_to_RMG_cell_mapper()
        
        
        dtw_l_an = [] # depth to water table list, 
        st_l_an = [] # saturated zone thickness list
        rw_l_an = [] # relative wetness list
        c=0
        for i in range(1,self.nd+1):
            dtw_d = M[c:c+self.rpm,:] # map, DHSVM orientation
            dtw_l_coarse = np.hstack(np.flipud(dtw_d))[self._grid.core_nodes] # map, converted to landlab orientation, keep only core nodes
            # convert dhsvm dtw map to landlab grid resolution 
            dtw_l = self.dhsvm_dtw_to_landlab_dtw(dtw_l_coarse)
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
        saturated zone thickness"""
        
        self.PDS_s = pd.Series(data = self.dtw_l_an.mean(axis=1), index = self.map_dates)
        self.Fx_s, self.x1_s, self.T_s, self.Ty_s, self.Q_ri_s = self.fit_probability_distribution(self.PDS_s,dist = 'LN', print_figs = print_figs)
                 

    def _mean_relative_wetness_cdf(self):
        """parammeterize a pdf to a partial duration series of the basin mean 
        relative wetness (staturated zone thickness / soil thickness)"""        

        self.PDS_s = pd.Series(data = self.rw_l_an.mean(axis=1), index = self.map_dates)
        self.Fx_s, self.x1_s, self.T_s, self.Ty_s, self.Q_ri_s = self.fit_probability_distribution(self.PDS_s,dist = 'LN')
                 

    
    def _saturated_zone_thickness_cdf(self, print_figs = False):
        """generate a cdf of the saturated water table thickness at each node"""
              
        # compute cdf for each cell
        dtw_low_cv_nodes = [] # save node ids where water table changes little
        st_cdf = {} # dict to save saturated zone thickness cdf parameters for each node
        for c, n in enumerate(self._grid.core_nodes):

            # time series of saturated zone thickness at node 'c' is column 'c'
            # from the 2-d array st_l_an
            pds = pd.Series(data = self.st_l_an[:,c], index = self.map_dates)
        
            Fx, x1, T, Ty, Q_ri = self.fit_probability_distribution(pds,dist = 'LN', print_figs = print_figs)
            st_cdf[n] = [Fx, x1]
            
            
            if Fx.std()/Fx.mean() < .05:
                dtw_low_cv_nodes.append(n)
            
            print_figs = False
            if c%2000 == 0:
                print('distribution fit to partial duration series of peak saturated zone thickness for '+ \
                      str(np.round((c/self.ncn)*100))+'% of core nodes' )
                # print_figs = True
                
        self.st_cdf = st_cdf
        self.dtw_low_cv_nodes = dtw_low_cv_nodes

        
    def _DHSVM_network_to_NMG_Mapper(self):    
        
        '''DtoL
        determine the closest dhsvm nmg (nmg_d) link to each link in the landlab
        nmg. Note, the landlab nmg id of the dhsmv network is used, not the DHSVM
        network id (arcID) To translate the nmg_d link id to the DHSVM network id: 
            self.nmg_d.at_link['arcid'][i], where i is the nmg_d link id.
        '''
        
        #compute distance between deposit and all network cells
        def Distance(row):
            return ((row['x']-XY[0])**2+(row['y']-XY[1])**2)**.5
        
        # for each node equivalent of each nmg link, find the closest nmg_d node
        # and record the equivalent nmg_d link id 
        
        LinkMapper ={}
        LinkMapL = {}
        for i, sublist in enumerate(self.Lnodelist):# for each list of link nodes
            LinkL = []
            for j in sublist: # for each node associated with the nmg link
                XY = [self.gridx[j], self.gridy[j]] # get x and y coordinate of node
                nmg_d_dist = self.xyDf_d.apply(Distance,axis=1) # compute the distance to all dhsvm nodes
                offset = nmg_d_dist.min() # find the minimum distance
                mdn = self.xyDf_d['linkID'].values[(nmg_d_dist == offset).values][0]# link id of minimum distance node
                LinkL.append(mdn)
            LinkMapL[i] = LinkL
            LinkMapper[i] = np.argmax(np.bincount(np.array(LinkL))) # dhsvm link that has most link ids in link list
            print('nmg link '+ str(i)+' mapped to equivalent DHSVM link')
            
        # return (Mapper, MapL)
        self.LinkMapper = LinkMapper
        self.LinkMapL = LinkMapL

    
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
        '''   
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
    
        '''
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
        """at each nodelook up saturated zone thickness, convert to depth to 
        water table. After cycling through all nodes, depth__to_water_table field
        of raster model grid is updated"""

        dtw_field = (np.ones(self._grid.at_node['soil__thickness'].shape[0])*np.nan).astype(float)
        st = [] # saturated thickness list
        dtw = [] # deppth to water table list
        # for each core node
        for c,n in enumerate(self._grid.core_nodes):
            # look up domain and range of saturated thickness cdf for node 'c'
            Fx = self.st_cdf[n][0]; x1 = self.st_cdf[n][1]
            # get saturated thickness for quantile 'q_i'
            st_i = self.intp(Fx,x1,q_i)#, message = '#######DTW#### at node '+str(n))
            # convert saturated thickness to depth to soil water
            dtw_i = self._grid.at_node['soil__thickness'][n]-st_i
            
            st.append(st_i)
            dtw.append(dtw_i)
        
        dtw_cn = np.array(dtw) # depth to water at core nodes
        
        dtw_field[self._grid.core_nodes] = dtw_cn
        
        #updates depth__to_soil_water field
        self._grid.add_field('node', 'depth__to_water_table', dtw_field, clobber = True)

    
    def _run_one_step_flow(self, ts = None):
        ## DETERMINE QUANTILE VALUE OF BASIN HYDROLOGIC STATE
        # pick a random quantile (probability of non-exceedance) that 
        # corrisponds to the event non-exceedance probability of the  
        # flow magnitude 
        # (NOTE: not the annual non-exceedance probability if
        # flow rates used to parameterize basin pdf included < 1 yr RI
        # flow rates)
        if ts is None:
            self.q_i = np.random.uniform(self.Q_Fx.min(), self.Q_Fx.max(),1)[0]
            
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
        """TODO write so that flow return interval is used if available or DTW
        return interval if not available"""
        
        # convert return interval to to an annual quantile value if 
        # return interval > 1 yr. Note: sub-annual return intervals 
        # can not be converted to an annual quantile value
        if ts is None:
            
            # if only updating the depth to water of the raster model grid
            if DTW_only is True:
                self.q_i = np.random.uniform(self.Fx_s.min(), self.Fx_s.max(),1)[0]
    
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
        
        '''computes hydraulc condtions at time ts given the slope and d50 of
        an evolving network model nmgrid. D50 is computed from parcels in each link.
        slope is read from nmgrid, which is updated by NST. hydraulic condtions are 
        saved as attributes of the nmgrid.
        ts = dateime, time stamp
        
        output is a 1d np array of flow depth at each link of the landlab network mg
        
        '''                    
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
                    
                # ## DETERMINE QUANTILE VALUE OF BASIN HYDROLOGIC STATE
                # # pick a random quantile (probability of non-exceedance) that 
                # # corrisponds to the event non-exceedance probability of the  
                # # flow magnitude 
                # # (NOTE: not the annual non-exceedance probability if
                # # flow rates used to parameterize basin pdf included < 1 yr RI
                # # flow rates)
                
                # self.q_i = np.random.uniform(self.Q_Fx.min(), self.Q_Fx.max(),1)[0]
                
                # # get the flow rate at representative link and equivalent 
                # # event return interval cdf of flow rates to get flow
                # self.Q_i = self.intp(self.Q_Fx, self.Q_x1, self.q_i, message = None)
                
                # # cdf of flow rate in terms of annual return interval
                # self.Q_ri = self.intp(self.Q_Fx, self.Q_Ty, self.q_i, message = None)
                                
                # ## GET FLOW RATES
                # # get flow value at each link for quantile self.q_i (self.Qlinks)
                # self.get_flow_at_links(self.q_i)
                # # convert to flow depth and effective flow depth       
                # self._variable_channel_tau()
                
                # ## GET DEPTH TO SOIL WATER
                # # if depth to water tablesupdate depth to water table if this is true, 
                # # updating water table is slow                    
                # if self.rpm >0:

                #     # convert return interval to to an annual quantile value if 
                #     # return interval > 1 yr. Note: sub-annual return intervals 
                #     # can not be converted to an annual quantile value
                #     if self.Q_ri > 1:
                #         self.q_yrs_i = 1-1/self.Q_ri                    

                #     if self.Q_ri > self.ls_ri: 
                #         self.get_depth_to_water_table_at_node(self.q_yrs_i)
        

                
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

    def _min_return_interval(self, dates, plotting_position = 'Weibull'):
        print(dates)
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
        '''
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

    
    def fit_probability_distribution(self, AMS, min_ri = 1, dist = 'LN', 
                       RI=[1.5], plotting_position = 'Weibull',
                       print_figs = False):
        '''
        Fits distribution to annual maximum series or partial duration series of
        data using methods and formulas described in: 
       
             Maidment, 1992, Handbook of Hydrology, Chapter 
             
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
        event return period by (1 / min_ri)
           
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
            
            # cdf created by summing area under pdf
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
    
    
    




