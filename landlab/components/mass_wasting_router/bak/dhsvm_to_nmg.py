# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:45:26 2020

@author: keckj
"""
#%% SETUP WORKSPACE
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.interpolate as interpolate
import xarray as xr

from landlab.plot import graph



#%%  
class DHSVMtoNMG:

    """
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
    
    
    TO DO: Add pairing function that identifies which DHSVM links overlap with the nmg links. Use pairing to create

    """

    def __init__(self, DHSVMflow, nmg, nmg_d, mg, parcels, timedelta = '24h', method='max', taudict = None, precip = None):

        # DHSVMflow = pd.read_csv(streamflowonly, delim_whitespace=True,
        #                         parse_dates=True, index_col='DATE')

        # initial flow
        self._streamflowonly = DHSVMflow/3600
        
        # network model grid
        self.nmg = nmg
        
        self.nmg_d = nmg_d
        
        # initial parcels
        self.parcels = parcels
        
        td = timedelta
        
        if precip is None:
            self.precip = precip
        else:
            self.precip = precip.resample(td).sum().fillna(method='ffill')
        
        # if aggregate, the aggregation method
        
        if method == 'max':
            self._streamflowonly_ag = self._streamflowonly.resample(td).max().fillna(method='ffill') #convert sub  hourly obs to hourly
            # if precip:
            #     self.precip = precip.resample(td).max().fillna(method='ffill')
        elif method == 'mean':
            self._streamflowonly_ag = self._streamflowonly.resample(td).mean().fillna(method='ffill') #convert sub  hourly obs to hourly
            # if precip:
            #     self.precip = precip.resample(td).mean().fillna(method='ffill')        
        elif method == 'min':
            self._streamflowonly_ag = self._streamflowonly.resample(td).min().fillna(method='ffill') #convert sub  hourly obs to hourly
            # if precip:
            #     self.precip = precip.resample(td).min().fillna(method='ffill') 
        elif (type(method) is float) or (type(method) is int):
            self._streamflowonly_ag = self._streamflowonly.groupby(pd.Grouper(freq='d')).quantile(method,interpolation = 'nearest')
            # if precip:
            #     self.precip = precip.groupby(pd.Grouper(freq='d')).quantile(method,interpolation = 'nearest')
        
        if taudict is None:
            self.taodict = {'gHG':[0.15,-0.08], 'wHG':[2.205, 0.38], 'dHG':[0.274, 0.24]}


        # properties of raster model grid used to creat the nmg
        
        ### Grid characteristics
        grid = mg
        self.gr = grid.shape[0] #number of rows
        self.gc = grid.shape[1] #number of columns
        self.dx = grid.dx #width of cell
        self.dy = grid.dy #height of cell

        # grid node coordinates, translated to origin of 0,0
        self.gridx = grid.node_x#-grid.node_x[0] 
        self.gridy = grid.node_y#-grid.node_y[0]
        
        # extent of each cell in grid        
        self.ndxe = self.gridx+self.dx/2
        self.ndxw = self.gridx-self.dx/2
        self.ndyn = self.gridy+self.dy/2
        self.ndys = self.gridy-self.dy/2

        self.nodes = grid.nodes.reshape(grid.shape[0]*grid.shape[1],1)
        
        # determine raster mg nodes that correspond to landlab network mg
        self._LinktoNodes(NetType = 'nmg')
        
        # determine raster mg nodes that correspond to dhsvm network mg
        self._LinktoNodes(NetType = 'dhsvm')        
        
        # determine dhsvm network mg links that correspond to the landlab network mg
        self._Mapper()
        
        # compute partial duration series and bankful flow rate for each nmg_d link
        # used by the nmg        
        self._RIflows()
        
    def __call__(self, ts=None):
        
        '''
        computes hydraulc condtions at time ts given the slope and d50 of
        an evolving network model nmg. D50 is computed from parcels in each link.
        slope is read from nmg, which is updated by NST. hydraulic condtions are 
        saved as attributes of the nmg.
        
        TODO width and depth approximations are also updated based on parker 2007
        
        ts = dateime, time stamp
        
        output is a 1d np array of flow depth at each link of the landlab network mg
        
        '''
        ro = 1000 # density water [kg/m3] # make this a class variable
        g = 9.81 # acc. of gracity [m/s2]
        
        
        if ts is None:
            
            self._ConstantChannelTau(self, taodict)
        
        else:
            
                
            Qlinks = self._streamflowonly_ag.loc[ts] # flow rate in all links at time ts
            
            # linksInNMG = np.unique(np.array(list(self.Mapper.values()))) # DHSVM links in the network model grid
            
            
            Q_ = []
            d_ = []
            T_ = []
            U_ = []
            Uo_ = []
            nt_ = []               
            no_ = []
            Teff_ = []               
            Teffr_ = []
            deff_ = []
            
            # determine hydraulic conditions for each link in nmg
            # not because some nmg links may be mapped to the same nmg_d link
            # some nmg links will have the same flow rate
            
            for Link in self.Mapper.keys(): # for each link in nmg
                                     
                # FLOW PARAMETERS
                # i_d is equivalent link in dhsvm nmg
                i_d = self.Mapper[Link]
                
                # get the dhsvm network link id 'arcid' of nmg_d link i
                dhsvmLinkID = self.nmg_d.at_link['arcid'][i_d]
                
                # get link attributes from nmg grid
                
                Q = Qlinks[str(dhsvmLinkID)] # flow at link for time ts 
                
                # look up bankfull flow and partial duration series
                Qb, pds = self.RIflows[Link] # Q1.2 [m3/s]
                
                # look up reach attribute
                CA = self.nmg.at_link['drainage_area'][Link]/1e6 # m2 to km2
                S = self.nmg.at_link['channel_slope'][Link]              
    
                
    
                # determine link D50
                # parcel grain size in link i
                Dp = pd.DataFrame(self.parcels.dataset['D'][np.where(self.parcels.dataset['element_id']==Link)].values.flatten())
                
                # determine 50th percentile value
                D50 = Dp.quantile(.5).values #diam. parcels              
                                
                # grain size distribution based on d50 and assumed 
                # log normal distribution (Kondolf and Piegay 2006)
                Gdist = pd.DataFrame(np.random.lognormal(np.log(D50),0.5,size=1000))            
                D65 = Gdist.quantile(0.65).values[0]                
                D84 = Gdist.quantile(0.84).values[0]             
                
                # # TO DO: update coefficent and exponent of width and depth HG based on d50 following Parker et al 2007                       
                wb = self.taodict['wHG'][0]*CA**self.taodict['wHG'][1]  # bankfull width [m]
                db = self.taodict['dHG'][0]*CA**self.taodict['dHG'][1]  # bankfull depth [m]
    
                # wb =  self.nmg.at_link['channel_width'][Link]     
                # db =  self.nmg.at_link['channel_depth'][Link]     
                            
                # approximate base channel width assuming 1:1 channel walls
                b =  wb - 2*db
                if b<0:
                    b = 0
                            

                # depth is normal depth in trapezoidal channel geometry - very slow
                # d = Depth_Trapezoid(Q, S = S, b = b, m1 = 1, m2 = 1, n = 0.05)               
                d = db*(Q/Qb)**0.3 # approximation for flow depth               
                q = Q/(b+d/2)               
            
                
                # roughness and effective stress
    
                U,Uo,nt,no,T,Teff,Teffr,deff = flow_resistance_RR(q,d,S,D65,D84)
    
                
                Q_.append(Q)
                d_.append(d)
                T_.append(T)
                U_.append(U)
                Uo_.append(Uo)
                nt_.append(nt)                
                no_.append(no)
                Teff_.append(Teff)               
                Teffr_.append(Teffr)
                deff_.append(deff)
                
            # update nmg link fields for time ts
            self.nmg.at_link["flow"] = np.array(Q_)
            self.nmg.at_link["flow_depth"] = np.array(d_)
            self.nmg.at_link['total_stress'] = np.array(T_)
            self.nmg.at_link['mean_velocity'] = np.array(U_)
            self.nmg.at_link['mean_grain_velocity'] = np.array(Uo_)            
            self.nmg.at_link['total_roughness'] = np.array(nt_)
            self.nmg.at_link['grain_roughness'] = np.array(no_) 
            self.nmg.at_link['effective_stress'] = np.array(Teff_)             
            self.nmg.at_link['effective_stress_b'] = np.array(Teffr_)  
            self.nmg.at_link['effective_depth'] = np.array(deff_)         
    
    def _LinktoNodes(self, NetType = 'nmg'):
        '''
        #convert links to coincident nodes
            #loop through all links in network grid to determine raster grid cells that coincide with each link
            #and equivalent distance from upstream node on link
        '''
        
        def LinktoNodes_code(linknodes, active_links, nmgx, nmgy):
            Lnodelist = [] #list of lists of all nodes that coincide with each link
            Ldistlist = [] #list of lists of all nodes that coincide with each link
            xdDFlist = []
            Lxy= [] #list of all nodes the coincide with the network links
                  
            for k,lk in enumerate(linknodes) : #for each link in network grid
                
                linkID = active_links[k] #link id (indicie of link in nmg fields)
                
                lknd = lk #link node numbers
                
                x0 = nmgx[lknd[0]] #x and y of downstream link node
                y0 = nmgy[lknd[0]]
                x1 = nmgx[lknd[1]] #x and y of upstream link node
                y1 = nmgy[lknd[1]]
                
                #create 1000 points along domain of link
                X = np.linspace(x0,x1,1000)
                Xs = X-x0 #change begin value to zero
                
                #determine distance from upstream node to each point
                #y value of points
                if Xs.max() ==0: #if a vertical link (x is constant)
                    vals = np.linspace(y0,y1,1000)
                    dist = vals-y0 #distance along link, from downstream end upstream
                    dist = dist.max()-dist #distance from updtream to downstream
                else: #all their lines
                    vals = y0+(y1-y0)/(x1-x0)*(Xs)
                    dist = ((vals-y0)**2+Xs**2)**.5
                    dist = dist.max()-dist #distance from updtream to downstream
                
                    
              
               #match points along link (vals) with grid cells that coincide with link
                
                nodelist = [] #list of nodes along link
                distlist = [] #list of distance along link corresponding to node
                # print(vals)
                for i,v in enumerate(vals):
                    
                    x = X[i]
                    
                    mask = (self.ndyn>=v) & (self.ndys<=v) & (self.ndxe>=x) & (self.ndxw<=x)  #mask - use multiple boolian tests to find cell that contains point on link
                    
                    node = self.nodes[mask] #use mask to extract node value
                    # print(node)
                    if node.shape[0] > 1:
                        node = np.array([node[0]])
                    # print(node)
                    # create list of all nodes that coincide with linke
                    if node not in nodelist: #if node not already in list, append - many points will be in same cell; only need to list cell once
                        nodelist.append(node[0][0])  
                        distlist.append(dist[i])
                        xy = {'linkID':linkID,
                            'node':node[0][0],
                              'x':self.gridx[node[0][0]],
                              'y':self.gridy[node[0][0]]}
                        Lxy.append(xy)
                
                Lnodelist.append(nodelist)
                Ldistlist.append(distlist)
                
            return (Lnodelist, Ldistlist, Lxy)

        if NetType == 'nmg':
            linknodes = self.nmg.nodes_at_link
            active_links = self.nmg.active_links
            nmgx = self.nmg.x_of_node
            nmgy = self.nmg.y_of_node
    
            out = LinktoNodes_code(linknodes, active_links, nmgx, nmgy)
            
            self.Lnodelist = out[0]
            self.Ldistlist = out[1]
            self.xyDf = pd.DataFrame(out[2])    
        
        elif NetType == 'dhsvm':
        
            linknodes = self.nmg_d.nodes_at_link
            active_links = self.nmg_d.active_links
            nmgx = self.nmg_d.x_of_node
            nmgy = self.nmg_d.y_of_node
    
            out = LinktoNodes_code(linknodes, active_links, nmgx, nmgy)

            self.Lnodelist_d = out[0]
            self.Ldistlist_d = out[1]
            self.xyDf_d = pd.DataFrame(out[2])             
     



    def _Mapper(self):    
        
        '''
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
        
        Mapper ={}
        MapL = {}
        for i, sublist in enumerate(self.Lnodelist):# for each list of link nodes
            LinkL = []
            for j in sublist: # for each node associated with the nmg link
                XY = [self.gridx[j], self.gridy[j]] # get x and y coordinate of node
                nmg_d_dist = self.xyDf_d.apply(Distance,axis=1) # compute the distance to all dhsvm nodes
                offset = nmg_d_dist.min() # find the minimum distance
                mdn = self.xyDf_d['linkID'].values[(nmg_d_dist == offset).values][0]# link id of minimum distance node
                LinkL.append(mdn)
            MapL[i] = LinkL
            Mapper[i] = np.argmax(np.bincount(np.array(LinkL))) # dhsvm link that has most link ids in link list
            print('nmg link '+ str(i)+' mapped')
            
        self.Mapper = Mapper
        self.MapL = MapL
        
        print('         before calling this DHSVMtoNST instance:')
        print('please check dhsvm nmg to landlab nmg mapping is correct')
        print('if not correct, mannual edit the mapper dictionary .Mapper')
            
                ## side by side 
        plt.figure(figsize=(25,20))
        ## Plot nodes + links
        plt.subplot(1,2,1)
        graph.plot_nodes(self.nmg,with_id=False,markersize=4)
        graph.plot_links(self.nmg)
        plt.title("landlab nmg Links")
        plt.xlim([self.nmg_d.x_of_node.min(),self.nmg_d.x_of_node.max()])
        plt.ylim([self.nmg_d.y_of_node.min(),self.nmg_d.y_of_node.max()])
        
        ## Plot nodes + links
        plt.subplot(1,2,2)
        graph.plot_nodes(self.nmg_d,with_id=False,markersize=4)
        graph.plot_links(self.nmg_d)
        plt.title("DHSVM nmg Links")
        plt.xlim([self.nmg_d.x_of_node.min(),self.nmg_d.x_of_node.max()])
        plt.ylim([self.nmg_d.y_of_node.min(),self.nmg_d.y_of_node.max()])
        plt.show()
        
        # plot overlapping
        ## Plot nodes + links
        plt.figure(figsize=(25,35))
        graph.plot_links(self.nmg, color = 'red')
        graph.plot_links(self.nmg_d, color = 'blue')
        plt.title("DHSVM nmg Links")
        plt.xlim([self.nmg_d.x_of_node.min(),self.nmg_d.x_of_node.max()])
        plt.ylim([self.nmg_d.y_of_node.min(),self.nmg_d.y_of_node.max()])
        plt.show()

    
    
    def _ConstantChannelTau(self, taodict):
        '''
        computes a time series of hydraulic conditions using the hydrograph
        at each link assuming channel slope and grain size are constant with time

        taodict = {'gHG':[0.15,-0.08], 'wHG':[2.205, 0.38], 'dHG':[0.274, 0.24]}                 
        '''

        # store flow as xarray dataset
        Q_df = self._streamflowonly_ag.T
        Q_da = xr.DataArray(Q_df,name = 'flow',dims = ['link','date'],
                    attrs = {'units':'m3/s'})      

    
        
        # compute daily mean flow depth, flow velocity and effective shear stress
        
        # note link id in dataframe different order than link order of each
        # np.array of link attributes
        # iterate over each link instead of broadcast to compute depth
        # and effective shear stress
        
        ro = 1000 # density water [kg/m3]
        g = 9.81 # acc. of gracity [m/s2]
     
        q_dict = {}
        d_dict = {}
        T_dict = {}
        U_dict = {}
        Uo_dict = {}
        nt_dict = {}                
        no_dict = {}
        Teff_dict = {}               
        Teffr_dict = {}
        
 ###
        for Link in self.Mapper.keys(): # for each link in nmg
                                            
            # FLOW PARAMETERS
            # i_d is equivalent link in dhsvm nmg
            i_d = self.Mapper[Link]
            
            # get the dhsvm network link id 'arcid' of nmg_d link i
            dhsvmLinkID = self.nmg_d.at_link['arcid'][i_d]
            
            # get link attributes from nmg grid
            
            Q = self._streamflowonly_ag[str(i)] # entire time series of flow at link
            
            # look up bankfull flow and partial duration series
            Qb, pds = self.RIflow(Link) # Q1.2 [m3/s]
            
            # look up reach attribute
            CA = self.nmg.at_link['drainage_area'][Link]
            S = self.nmg.at_link['channel_slope'][Link]              

            

            # determine link D50
            # parcel grain size in link i
            Dp = pd.DataFrame(self.parcels.dataset['D'][np.where(self.parcels.dataset['element_id']==Link)].values.flatten())
            
            # determine 50th percentile value
            D50 = Dp.quantile(.5).values #diam. parcels              
                            
            # grain size distribution based on d50 and assumed 
            # log normal distribution (Kondolf and Piegay 2006)
            Gdist = pd.DataFrame(np.random.lognormal(np.log(D50),0.5,size=1000))            
            D65 = Gdist.quantile(0.65).values[0]                
            D84 = Gdist.quantile(0.84).values[0]             
            
            # # TO DO: update coefficent and exponent of width and depth HG based on d50 following Parker et al 2007                       
            # wb = self.taodict['wHG'][0]*CA**self.taodict['wHG'][1]  # bankfull width [m]
            # db = self.taodict['dHG'][0]*CA**self.taodict['dHG'][1]  # bankfull depth [m]

            wb =  self.nmg.at_link['channel_width'][Link]     
            db =  self.nmg.at_link['channel_depth'][Link]     
                        
            # approximate base channel width assuming 1:1 channel walls
            b =  wb - 2*db
                        
            #  TIME SERIES
            # depth is normal depth in trapezoidal channel geometry - very slow
            # d = Depth_Trapezoid(Q, S = S, b = b, m1 = 1, m2 = 1, n = 0.05)               
            d = db*(Q/Qb)**0.3 # approximation for flow depth               
            q = Q/(b+d/2)               

            # total stress
            T = ro*g*d*S               
            
            # roughness and effective stress

            U,Uo,nt,no,T,Teff,Teffr,deff = flow_resistance_RR(q,d,S,D65,D84)


###
            q_dict[i] = q            
            d_dict[i] = d
            T_dict[i] = T
            U_dict[i] = U
            Uo_dict[i] = Uo
            nt_dict[i] = nt                
            no_dict[i] = no
            Teff_dict[i] = Teff               
            Teffr_dict[i] = Teffr

            print(i)                
        
        # convert to dataframe, transpose so that each row 
        # is timeseries for a single link
        q_df = pd.DataFrame.from_dict(q_dict, orient='columns').T
        d_df = pd.DataFrame.from_dict(d_dict, orient='columns').T
        T_df = pd.DataFrame.from_dict(T_dict, orient='columns').T
        U_df = pd.DataFrame.from_dict(U_dict, orient='columns').T
        Uo_df = pd.DataFrame.from_dict(Uo_dict, orient='columns').T
        nt_df = pd.DataFrame.from_dict(nt_dict, orient='columns').T
        no_df = pd.DataFrame.from_dict(no_dict, orient='columns').T
        Teff_df = pd.DataFrame.from_dict(Teff_dict, orient='columns').T            
        Teffr_df = pd.DataFrame.from_dict(Teffr_dict, orient='columns').T    
        
        # convert to dataarray
        q_da = xr.DataArray(q_df,name = 'Unitflow',dims = ['link','date'],
                            attrs = {'units':'m2/s'})
        d_da = xr.DataArray(d_df,name = 'Depth',dims = ['link','date'],
                            attrs = {'units':'m'})
        T_da = xr.DataArray(T_df,name = 'TotalStress',dims = ['link','date'],
                            attrs = {'units':'Pa'})
        U_da = xr.DataArray(U_df,name = 'MeanVelocity',dims = ['link','date'],
                            attrs = {'units':'m/s'})  
        Uo_da = xr.DataArray(Uo_df,name = 'MeanVirtualVelocity',dims = ['link','date'],
                            attrs = {'units':'m/s'})
        nt_da = xr.DataArray(nt_df,name = 'TotalRoughness',dims = ['link','date'],
                            attrs = {'units':'s/m(1/3)'})
        no_da = xr.DataArray(no_df,name = 'GrainRoughness',dims = ['link','date'],
                            attrs = {'units':'s/m(1/3)'})
        Teff_da = xr.DataArray(Teff_df,name = 'EffectiveStress_a',dims = ['link','date'],
                            attrs = {'units':'Pa'})
        Teffr_da = xr.DataArray(Teffr_df,name = 'EffectiveStress_b',dims = ['link','date'],
                            attrs = {'units':'Pa'})             
        
        # combine dataarrays into one dataset, save as instance variable                      
        self.DataSet = xr.merge([Q_da, q_da, d_da, T_da, U_da, Uo_da, nt_da, no_da, 
                                 Teff_da, Teffr_da])
        
        
    def _RIflows(self):
        '''
        runs RIflow for each link in nmg
        
        create the RIflows dictionary, which has the bankfull flow magnitude
        and partial duration series of each link

        Returns
        -------
        None.

        '''
            
        RIflows = {}
        for Link in self.Mapper.keys():
                    # i_d is equivalent link in dhsvm nmg
            print(Link)
            i_d = self.Mapper[Link]
            
            # get the dhsvm network link id 'arcid' of nmg_d link i
            dhsvmLinkID = self.nmg_d.at_link['arcid'][i_d]
            print(dhsvmLinkID)
            # get link attributes from nmg grid
            
            Qts = self._streamflowonly_ag[str(dhsvmLinkID)]# flow at link for time ts 
            
            
            RIflows[Link]  = self._RIflow(Qts)
        
        self.RIflows = RIflows
    
    def _RIflow(self,Qts):
    
        '''
        Given a time sereies of flow, returns the flow magnitude of a specified
        return interval flow
    
        #CLEAN THIS UP
    
        Parameters
        ----------
        Q : pandas series
            time series of flow
        RI : float or int
            DESCRIPTION. The default is 1.2. Return interval [yrs] of flow event
        sep : int
            DESCRIPTION. The default is 30. Days removed before and after a peak
            to create the partial duration series
    
        Returns
        -------
        Qri
    
        '''
        RI=1.5
        sep=30    
        #sep=30 #separation between storm events to be considered independent - flow
        #sep =7 #precipitation
        
        RIp = {}
        pds_l = {}
        Qri_l = {}
    
        Qdt = Qts.copy(deep=True)
      
        ll = Qdt.shape[0]
        c=1       
        pds = {}
        mq = Qdt.max()
        pds[Qdt[Qdt==mq].index[0]] = mq
        
        while  mq>0:
        
            Rf = Qdt[Qdt==mq].index + datetime.timedelta(days=sep)
            Rp = Qdt[Qdt==mq].index - datetime.timedelta(days=sep)
            mask = (Qdt.index > Rp[0]) & (Qdt.index <= Rf[0])
            Qdt.loc[mask]=0
        
            mq = Qdt.max() 
            
            if mq >0: # last value of algorith is 0, dont add zero to pds
                pds[Qdt[Qdt==mq].index[0]] = mq
    
    
        pds_l = pds
        
        pds_d = pd.DataFrame.from_dict(pds_l, orient='index')

        Yrs = max(Qts.index.year)-min(Qts.index.year)+1
        n=pds_d.shape[0]            
        ranks = np.array(range(1,n+1)) 
        #compute plotting position or exceedance probability (probablity of equal to or larger) if largest to smallest. NOTE: quantile if smallest to largest
        PP = (ranks-.4)/(Yrs+.2) 
        #exceedance probability, eqaul to plotting position if ordered large to small
        EP = PP
        T = 1/PP                        
        pds_d['T'] = T                    
        x = T
        y = pds_d[0]
        
        RIp = pds_d
        
        f = interpolate.interp1d(x,y)
        
        #determine RI flow
        Qri = {}
        try:
            Qri = f(RI)
        except:
            Qri = np.nan 
        return Qri, RIp #flow rate of RI event



def Depth_Trapezoid(Q, S, b, m1 = 1, m2 = 1, n = 0.05):
    
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

    r = scipy.optimize.minimize_scalar(h,method='bounded',bounds = [.1,10])
    
    y = r.x
    Rh = A_t(b,m1,m2,y)/P_t(b,m1,m2,y) 
    return (y,Rh)
        
                    
def flow_resistance_Ferguson(D84,Rh):
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
        

def flow_resistance_RR(q,d,S,D65,D84):
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
    