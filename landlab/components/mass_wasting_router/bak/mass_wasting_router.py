# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:36:00 2019

@author: keckj

    TODO: erode channel node elevations using flow rate at grid cell
          record datetime of each timestep
          add parameter for using different router and option to have no terrace cells (for models with larger grid cells)
          clean up doc strings and comments
          tests (review Datacamp class), submittal to LANDLAB
          debris flow erosion model that accounts for available regolith thickness
          

    WISH LIST  
    ADD entrainment model - See Frank et al., 2015, user parameterizes based on literature OR results of model runs in RAMMS
        first draft done
       
        

"""

import os as os
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

from landlab import Component, FieldError
from landlab.components import (FlowDirectorMFD, FlowAccumulator, DepressionFinderAndRouter)
from landlab import imshow_grid, imshow_grid_at_node

from landlab.utils.grid_t_tools import GridTTools


class MassWastingRouter(GridTTools):
    
    '''

    Parameters
    ----------
    grid: ModelGrid 
        Landlab Model Grid object with node fields: topographic__elevation, flow_reciever node, masswasting_nodes
    nmgrid: NetworkModelGrid
        Landlab Network Model Grid object

    
    -future Parameters-
    movement_type: string
        defines model used for clumping
    material_type: string
        defines model used for clumping

    
    Construction:
    
        MWR = MassWastingRouter(grid, nmgrid)
              
    '''
    

    
    _name = 'MassWastingRouter'
    
    _unit_agnostic = False
    
    _version = 1.0
    
    _info = {}
    
    # TO DO: move below to _info 
    _input_var_names = (
        'topographic__elevation', 'flow__receiver_node','MW__probability'
    )
    
    _output_var_names = (
    'high__MW_probability','mass__wasting_clumps'
    )
    
    _var_units = {
    'topographic__elevation': 'm',
    'flow__receiver_node': 'node number',
    'MW__probability': 'annual probability of occurance',
    'high__MW_probability': 'binary',    
    'mass__wasting_clumps': 'number assigned to mass wasting clump',
    }
    
    _var_mapping = {
    'topographic__elevation': 'node',
    'flow__receiver_node': 'node',
    'MW__probability': 'node',
    'high__MW_probability':'node',
    'mass__wasting_clumps': 'node'
    }
    
    _var_doc = {
    'topographic__elevation':
        'elevation of the ground surface relative to some datum',
    'flow__receiver_node': 'node that receives flow from each node',
    'MW__probability':
        'annual probability of shallow landslide occurance',
    'high__MW_probability':'nodes input into mass wasting clumping component',
    'mass__wasting_clumps': 
        'unstable grid cells that are assumed to fail as one, single mass'
    }
        
#%%    
    
    def __init__(
            self, 
            grid,
            nmgrid,
            Ct = 5000,
            BCt = 100000,
            MW_to_channel_threshold = 50,
            PartOfChannel_buffer = 10, # may not need this parameter
            TerraceWidth = 1,
            FluvialErosionRate = [[0.03,-0.43], [0.01,-0.43]],
            probability_threshold = 0.75,
            min_mw_cells = 1,
            min_depth = 0.2, # minimum parcel depth, parcels smaller than this are aggregated into larger parcels
            **kwds):
        
        """
        
        Parameters
        ----------
        grid:ModelGrid 
            Landlab ModelGrid object
        nmgrid:network model grid
            Landlab Network Model Grid object
        movement_type: string
            defines model used for clumping
        material_type: string
            defines model used for clumping
        
        ###PARAMETERS TO ADD###
            deposits: datarecord with fields for deposit remaining volume, upstream extent (link, position on link), 
            downstream extent, coefficient for exponential decay curve that determines rates material is released, flow rate
            that causes release (effective flow approximation), time since deposition 
        
            entrainment parameters
            
            runout parameters
            
            velocity parameters
        
        """
        
        # call __init__ from parent classes
        super().__init__(grid, nmgrid, Ct, BCt, MW_to_channel_threshold, 
                         PartOfChannel_buffer, TerraceWidth)
               

        #   landslide probability
        if 'MW__probability' in grid.at_node:
            self.mwprob = grid.at_node['MW__probability']
        else:
            raise FieldError(
                'A MW__probability field is required as a component input!')
        
        #   high probability mass wasting cells
        if 'high__MW_probability' in grid.at_node:
            self.hmwprob = grid.at_node['high__MW_probability']
        else:
            self.hmwprob = grid.add_zeros('node',
                                        'high__MW_probability') 
       
        
        #   mass wasting clumps
        if 'mass__wasting_clumps' in grid.at_node:
            self.mwclump = grid.at_node['mass__wasting_clumps']
        else:
            self.mwclump = grid.add_zeros('node',
                                        'mass__wasting_clumps')
            
        #   soil thickness
        if 'soil__thickness' in grid.at_node:
            self.Soil_h = grid.at_node['soil__thickness']
        else:
            raise FieldError(
                'A soil__thickness field is required as a component input!')


        # years since disturbance
        if 'years__since_disturbance' in grid.at_node:
            self.years_since_disturbance = self._grid.at_node['years__since_disturbance'] 
        else:
            self.years_since_disturbance = 25*np.ones(self.dem.shape) #
   
    
        self._grid.add_field('topographic__initial_elevation',
                        self._grid.at_node['topographic__elevation'],
                        at='node',
                        copy = True,clobber=True)

        # prep MWR 
        # time
        self._time_idx = 0 # index
        self._time = 0.0 # duration of model run (hours, excludes time between time steps)
        # TODO need to keep track of actual time (add difference between date of each iteration)

        ### clumping                  
        self.MW_to_C_threshold = MW_to_channel_threshold # maximum distance [m] from channel for downslope clumping                            
        self.probability_threshold = probability_threshold # probability of landslide threshold      
        self.min_mw_cells = min_mw_cells # minimum number of cells to be a mass wasting clump
        
        ### runout method
        self._method = 'ScourAndDeposition'

        ### fluvial erosion            
        self.C_a = FluvialErosionRate[0][0]
        self.C_b = FluvialErosionRate[0][1]
        self.T_a = FluvialErosionRate[1][0]
        self.T_b = FluvialErosionRate[1][1]
        
        self.ErRate = 0 # debris flow scour depth for simple runout model
        
        self.min_d = min_depth
        
        # dictionaries of variables for each iteration, for plotting
        self.DFcells_dict = {} # dictionary of all computational cells during the debris flow routing algorithm
        self.parcelDF_dict = {} # dictionary of the parcel dataframe
        self.LS_df_dict = {}
        self.LSclump_dict = {}        
        self.FED_all = {}
        self.FENodes_all = {}     
        
        #### Define the raster model grid representation of the network model grid
        out = self._LinktoNodes(linknodes = self.linknodes, 
                                active_links = self._nmgrid.active_links,
                                nmgx = self.nmgridx, nmgy = self.nmgridy)

        self.Lnodelist = out[0]
        self.Ldistlist = out[1]
        self.xyDf = pd.DataFrame(out[2])
                        
        ## define bedload and debris flow channel nodes       
        ## channel
        self._ChannelNodes()        
        ## terrace
        self._TerraceNodes()
        ## define fluvial erosion rates of channel and terrace nodes (no fluvial erosion on hillslopes)
        self._DefineErosionRates()
        
        
        ### create initial values
        self._extractLSCells() # initial high landslide probability grid cells             
        self.parcelDF = pd.DataFrame([]) # initial parcel DF          
        self.dem_initial = self._grid.at_node['topographic__initial_elevation'].copy() # set initial elevation
        self.dem_previous_time_step = self._grid.at_node['topographic__initial_elevation'].copy() # set previous time step elevation
        self.dem_dz_cumulative = self.dem - self.dem_initial
        self.dem_mw_dzdt = self.dem_dz_cumulative # initial cells of deposition and scour - none, TODO, make this an optional input        
        self.DistNodes = np.array([])

    
    def _DefineErosionRates(self):
        '''
        defines the coefficient and exponent of a negative power function that
        predicts fluvial erosion rate per storm [L/storm] as a funciton of time 
        since the last disturbance. 
        
        function is defined for all channel nodes, including the debris flow channel
        nodes. 
        
        Erosion rate is specified as the parameters of a negative power function       
        that predicts fluvial erosion rate [m/storm] relative to time since 
        disturbance, indpendent of the magnitude of the flow event.


        Returns
        -------
        None.

        '''
        coefL = []
        
        # coeficients of fluvial erosion/storm as a function of time

        
        for i in self.rnodes:
            if i in self.TerraceNodes:
                coefL.append(np.array([self.T_a, self.T_b]))
            elif i in self.ChannelNodes:
                coefL.append(np.array([self.C_a, self.C_b]))        
            else:
                coefL.append(np.array([0, 0]))
                
        
        self.FluvialErosionRate = np.array(coefL,dtype = 'float')   

         
    
    def _extractLSCells(self):
        '''
        extract areas that exceed annual probability threshold
        '''
        a = self.mwprob 
        
        type(a) # np array
        
        # keep as (nxr)x1 array, convert to pd dataframe
        a_df = pd.DataFrame(a)
                    
        mask = a_df > self.probability_threshold
        
        # boolean list of which grid cells are landslides
        self._this_timesteps_landslides = mask.values

        
        a_th = a_df[mask] #Make all values less than threshold NaN
        
        a_thg = a_th.values
        
        #add new field to grid that contains all cells with annual probability greater than threshold
        self.hmwprob = a_thg #consider removing this instance variable
        
        self.grid.at_node['high__MW_probability'] = self.hmwprob
                   
        #create mass wasting unit list           
        self.LS_cells = self.nodes[mask]
        # print(self.LS_cells)            


    
    def aLScell(self,cl):
        """
        returns boolian list of adjacent cells that are also mass wasting cells 
        """
        cl_m = []
        for c,j in enumerate(cl): #for each adjacent and not divergent cell, check if a mass wasting cell  
            
            tv1 = self.LS_cells[self.LS_cells==j] #check if cell number in mass wasting cell list
            
            if len(tv1) != 0: #if mass wasting cell (is included in LS_cells), add to into mass wasting unit LSd[i]
                
                cl_m.append(True)
            else:
                cl_m.append(False)
    
        return cl_m
    
    
    
    def GroupAdjacentLScells(self,lsc):
        '''
        returns list of cells to clump (adjacent, not divergent) with input cell (lsc)
        '''
        
        
        #(1) determine adjacent cells
        Adj = self.AdjCells(lsc)
        # print('adjcells')
        ac = Adj[0] #grid cell numbers of adjacent cells
        acn = Adj[1] #adjacent cell numbers based on local number schem
        
        #(2) mask for list of adject cells: keep only ones that are not divergent
        ac_m = self.NotDivergent(lsc,ac,acn) 
        # print('notdivergent')
        andc = list(np.array(ac)[ac_m]) #adjacent not divergent cells
        
        #(3) check if mass wasting cells
        andc_m = self.aLScell(andc) 
        # print('checkmasswastingcell')
        #final list of cells 
        group = [lsc] 
        
        group.extend(list(np.array(andc )[andc_m]))#adjacent, notdivergent and landslide cells
        
    
        
        return group
        
    
    def _MassWastingExtent(self):
        '''
       
        TODO: look into replaceing pandas dataframe and pd.merge(inner) with np.in1d(array1,array2) 
            and pd.erge(outer) call with np.unique(), get rid of dataframe operations
        

        '''
        
        #single cell clump
        groupALS_l = [] 
        for i, v in enumerate(self.LS_cells):
            
            group = self.GroupAdjacentLScells(v)
            if len(group) >= self.min_mw_cells: # at least this number of cells to be a clump
                groupALS_l.append(group) # list of all single cell clumps
        
        # print('groupadjacentcells')
        
        groupALS_lc = groupALS_l*1  #copy of list, to keep original
        self.groupALS_lc = groupALS_lc
        
        #create the mass wasting clump
        LSclump = {} # final mass wasting clump dicitonary
        
        ll = len(groupALS_lc)
        c=0
        while ll > 0:
            
            # first: clump single cell clumps into one clump if they share cells
            
            # use first cell clump in list to searchps
            df1 = pd.DataFrame({'cell':list(groupALS_lc[0])})            
            i =1
            while i < ll: # for each cell clump in groupALS_lc
                #print(i)
                df2 = pd.DataFrame({'cell':list(groupALS_lc[i])}) # set cell list i as df2
                # check if df1 and df2 share cells (if no shared cells, dfm.shape = 0)  
                dfm = pd.merge(df1,df2, on = 'cell', how= "inner") 
            
                if dfm.shape[0] > 0: # if shared cells
                    groupALS_lc.remove(groupALS_lc[i]) # cell list i is removed from groupALS_lc
                    ll = len(groupALS_lc)
                    # cell list df2 is combined with df1, dupilicate cells are list onely once,
                    # and the combined list becomes df1 using pd.merge method
                    df1 = pd.merge(df1,df2, on = 'cell', how= "outer") 
                
                i+=1
            
            # second, add downslope cells to mass wasting clump
            # for each cell in df1
                # determine downslope flow paths (cellsD)
            # determine minimum downslope distance to channel network 
            distL = []
            cellsD = {}
            for cell in df1['cell'].values:
                dist, cells = self._downslopecells(cell)
                distL.append(dist)
                cellsD[cell] = cells
            
            md = np.array(distL).min() # minimum distance to downstream channel

            # if md < MW_to_C_threshold, clump downslope cells         
            if md <= self.MW_to_C_threshold: 
            # downhill gridcells fail with landslide cells (added to clump)
                for cell in list(cellsD.keys()): # for each list of downstream nodes
                    
                    # TODO may need to remove any ls cells that are in downslope path
                    # lscls  = self.aLScell(cellsD[cell])                    
                    # np.array(andc )[andc_m]
                    
                    # convert list to df
                    df2 = pd.DataFrame({'cell':list(cellsD[cell])})
                                                         
                    # use pd merge to add unique cell values
                    df1 = pd.merge(df1,df2, on = 'cell', how= "outer") 
            

            # from ids of all cells in clump df1, get attributes of each cell, organize into dataframe
            # save in dictionary

            dff = pd.DataFrame({'cell':list(df1.cell),
                                'x location':list(self._grid.node_x[df1.cell]),
                                'y location':list(self._grid.node_y[df1.cell]),
                                'elevation [m]':list(self._grid.at_node['topographic__elevation'][df1.cell]),
                                'soil thickness [m]':list(self._grid.at_node['soil__thickness'][df1.cell]),
                                'slope [m/m]':list(self._grid.at_node['topographic__slope'][df1.cell])    
                                })
            LSclump[c] = dff

            # subtract 90% of soil depth from dem, 
            self._grid.at_node['topographic__elevation'][df1.cell] = self._grid.at_node['topographic__elevation'][df1.cell] - 0.9*self._grid.at_node['soil__thickness'][df1.cell]
       
            
            # set soil thickness to 0 at landslide cells
            self._grid.at_node['soil__thickness'][df1.cell] =0      
            
            # once all lists that share cells are appended to clump df1                   
            # remove initial cell list from list
            groupALS_lc.remove(groupALS_lc[0])    
            
            #update length of remaining 
            ll = len(groupALS_lc)

            c=c+1
        
    
        # summarize characteristics of each mass wasting unit in a dataframe
        LS = OrderedDict({})
        cellarea = self._grid.area_of_cell[0]#100 #m2
        
        for key, df in LSclump.items():
            slp = df['slope [m/m]'].mean() #mean slope of cells
            st = df['soil thickness [m]'].mean() #mean soil thickness of cells
            rc = df[df['elevation [m]']==df['elevation [m]'].min()] # choose cell with lowest elevation as initiation point of slide
            rc = rc.iloc[0]
            vol = st*cellarea*df.shape[0]
            #rc = rc.set_index(pd.Index([0]))
            LS[key] = [rc['cell'], rc['x location'], rc['y location'], rc['elevation [m]'], st, slp, vol]
        
        try:
            LS_df = pd.DataFrame.from_dict(LS, orient = 'index', columns = LSclump[0].columns.to_list()+['vol [m^3]'])
        except:
            LS_df = pd.DataFrame([])
        
        #create mass__wasting_clumps field
        for c, nv in enumerate(LSclump.values()):
            self.mwclump[list(nv['cell'].values)] = c+1 #plus one so that first value is 1
        
        self.grid.at_node['mass__wasting_clumps'] = self.mwclump
        self.LS_df = LS_df
        self.LSclump = LSclump
        self.LS_df_dict[self._time_idx] = LS_df.copy()
        self.LSclump_dict[self._time_idx] = LSclump.copy()        
        # return groupALS_l,LSclump,LS_df #clumps around a single cell, clumps for each landslide, summary of landslide


   
    def _MassWastingRunout(self, DistT =2000 , SlopeT = .15, ):
        '''
        compute mass wasting unit runout from volume and location.
        Distance traveled computed using algorith described in Benda and Cundy, 1990
        Maximum distance computed using variation of Crominas, 1996
        
        TO DO: 
            remove material at landslide
            entrains, add volume to debris flow, removes elevation from DEM
        
        '''
        Locd = {}
        Disd = {}
        if self.LS_df.empty is True:
            Loc_df =pd.DataFrame([])
            Dis_df =pd.DataFrame([])
            Dist =pd.DataFrame([])
        else:
            
            for index, row in self.LS_df.iterrows():
                
                loc = []
                dist = []
                c = 0
                
                loc.append(int(row['cell']))
                dist.append((self.xdif[int(row['cell'])]**2+self.ydif[int(row['cell'])]**2)**.5)
                
               
                
                #list of conditionals
                cond = [lambda x: DistT>x[0] and SlopeT<x[1], 
                        lambda x: SlopeT<x,
                        lambda x: DistT>x]
                              
               
                flow = True                
                while flow == True:
                    slope  = self.grid.at_node['topographic__slope'][loc[c]]
                    # volslide = volslide+volslide*.1 # slide accretes
                    
                    # select conditional based on user input
                    if DistT and SlopeT:
                        conditional = cond[0]
                        cx = [sum(dist),slope]
                    elif SlopeT:
                        conditional = cond[1]
                        cx = slope
                    elif DistT:
                        conditional = cond[2]
                        cx = sum(dist)
                        
                    if conditional(cx):

                        loc.append(self.grid.at_node['flow__receiver_node'][loc[c]])

                        dist.append((self.xdif[self.grid.at_node['flow__receiver_node'][loc[c]]]**2+self.ydif[self.grid.at_node['flow__receiver_node'][loc[c]]]**2)**.5)
                        
                        # erode DEM
                        self.grid.at_node['topographic__elevation'][loc[c]] = self.grid.at_node['topographic__elevation'][loc[c]]-self.ErRate
                        c=c+1
                        if loc[-1] == loc[-2]: #check that runout is not stuck at same node # NEED TO DEBUG
                            break
                    else:
                        flow = False
                
                Locd[index] = loc
                Disd[index] = dist
            
            
            Loc_df = pd.DataFrame.from_dict(Locd, orient = 'index')
            Dis_df = pd.DataFrame.from_dict(Disd, orient = 'index')
            Dist = Dis_df.sum(axis=1)
        
        self.Locd = Locd
        self.Disd = Disd
        self.Dist = Dist
        
        # return Locd,Loc_df,Disd,Dis_df,Dist               
        # return Locd,Disd,Dist


    def _MassWastingScourAndDeposition(self):
        # print('self._time_idx is: '+str(self._time_idx))
        # save data for plots
        # if self._time_idx == 1:
        SavePlot = False
        # else:
        #     SavePlot = False
            
        self.DFcells = {}
        self.RunoutPlotInterval = 5
        
        # depostion
        slpc = 0.1# critical slope at which debris flow stops
        SV = 2 # material stops at cell when volume is below this
        
        # release parameters for landslide
        nps = 8 # number of pulses, list for each landslide
        nid = 5 # delay between pulses (iterations), list for each landslide
        
        # max erosion depth per perception coefficient
        dc = 0.02
        
        
        if self.LS_df.empty is True:
            print('no debris flows to route')
        else:
            
            ivL = self.LS_df['vol [m^3]'][0::2].values.astype('int')
            innL = self.LS_df['cell'][0::2].values.astype('int')       
        
            cL = {}
            
            # ivL and innL can be a list of values. For each value in list:
            for i,inn in enumerate(innL):
                # print(i)
                cL[i] = []
                
                # # release parameters for landslide
                # nps = 8 # number of pulses, list for each landslide
                # nid = 5 # delay between pulses (iterations), list for each landslide
                
                # set up initial landslide cell
                iv =ivL[i]/nps# initial volume (total volume/number of pulses) 
            
                
                # initial receiving nodes (cells) from landslide
                rn = self._grid.at_node.dataset['flow__receiver_node'].values[inn]
                self.rn = rn
                rni = rn[np.where(rn != -1)]
                
                # initial receiving proportions from landslide
                rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[inn]
                rp = rp[np.where(rp > 0)]
                rvi = rp*iv # volume sent to each receving cell
            
                # now loop through each receiving node, 
                # determine next set of recieving nodes
                # repeat until no more receiving nodes (material deposits)
                
                self.DFcells[inn] = []
                slpr = []
                c = 0
                c2=0
                arn = rni
                arv = rvi
                cs = 1.5e-5
                enL = []
                while len(arn)>0:# or c <300:
            
                    
                    # time to pulse conditional statement
                    # add pulse (fraction) of total landslide volume
                    # at interval "nid" until total number of pulses is less than total
                    # for landslide volume (nps*nid)
            
                    if ((c+1)%nid ==0) & (c2<=nps*nid):
                        arn = np.concatenate((arn,rni))
                        arv = np.concatenate((arv,rvi))
                        
                    arn_ns = np.array([])
                    arv_ns = np.array([])
                    # print('arn:'+str(arn))
                    
                    # for each unique cell in receiving node list arn
                    # print('arn')
                    # print(arn)
                    for n in np.unique(arn):
                        n = int(n)
                        
                        # slope at cell (use highes slope)
                        slpn = self._grid.at_node['topographic__steepest_slope'][n].max()
                        
                        # incoming volume: sum of all upslope volume inputs
                        vin = np.sum(arv[arn == n])
                        
                        # determine deposition volume following Campforts, et al., 2020            
                        # since using volume (rather than volume/dx), L is (1-(slpn/slpc)**2) 
                        # rather than mg.dx/(1-(slpn/slpc)**2)           
                        Lnum = np.max([(1-(slpn/slpc)**2),0])
                        
                        # deposition volume
                        df_depth = vin/(self._grid.dx*self._grid.dx) #df depth
                        
                        dpd = vin*Lnum # deposition volume
                        
                        # max erosion depth
                        # dmx = dc*self._grid.at_node['soil__thickness'][n]
                        
                        # determine erosion volume (function of slope)
                        # change this to be a function of preciption depth
                        # er = (dmx-dmx*Lnum)*self.dx*self.dy
                        
                        T_df = 1700*9.81*df_depth*slpn # shear stress from df
                        
                                    # max erosion depth equals regolith (soil) thickness
                        dmx = self._grid.at_node['soil__thickness'][n]
                        
                        # erosion depth: 
                        er = min(dmx, cs*T_df)
                        
                        enL.append(er)
                        
                        # erosion volume
                        ev = er*self._grid.dx*self._grid.dy
                        # volumetric balance at cell
                        
                        # determine volume sent to downslope cells
                        
                        # additional constraint to control debris flow behavoir
                        # if flux to a cell is below threshold, debris is forced to stop
                        # print('incoming volume')
                        # print(vin)
                        if vin <=SV:
                            dpd = vin # all volume that enter cell is deposited 
                            vo = 0 # debris stops, so volume out is 0
            
                            # determine change in cell height
                            deta = (dpd)/(self.dx*self.dy) # (deposition)/cell area
            
                        else:
                            vo = vin-dpd+ev # vol out = vol in - vol deposited + vol eroded
                            
                            # determine change in cell height
                            deta = (dpd-ev)/(self.dx*self.dy) # (deposition-erosion)/cell area
            
                        # print(deta)
                        # update raster model grid regolith thickness and dem elevation

                        # if deta larger than regolith thickness, deta equals regolith thickness (fresh bedrock is not eroded)
                        if self._grid.at_node['soil__thickness'][n]+deta <0:
                            deta = - self._grid.at_node['soil__thickness'][n]    
                        
                        # Regolith - difference between the fresh bedrock surface and the top surface of the dem
                        self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]+deta                         

                        # Topographic elevation - top surface of the dem
                        self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta
                                                       
                        # build list of receiving nodes and receiving volumes for next iteration        
                
                        # material stops at node if transport volume is 0 OR node is a 
                        # boundary node
                        if vo>0 and n not in self._grid.boundary_nodes: 
                            
                
                            th = 0
                            # receiving proportion of volume from cell n to each downslope cell
                            rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[n]
                            rp = rp[np.where(rp > th)]
                            # print(rp)        
                            
                            # receiving nodes (cells)
                            rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
                            rn = rn[np.where(rn != -1)]            
                            # rn = rn[np.where(rp > th)]
                            
                            # receiving volume
                            rv = rp*vo
                       
                            # store receiving nodes and volumes in temporary arrays
                            arn_ns =np.concatenate((arn_ns,rn), axis = 0) # next step receiving node list
                            arv_ns = np.concatenate((arv_ns,rv), axis = 0) # next steip receiving node incoming volume list
                            
                                    
                    # once all cells in iteration have been evaluated, temporary receiving
                    # node and node volume arrays become arrays for next iteration
                    arn = arn_ns
                    arv = arv_ns
                    
                    # update DEM slope 
                    # fd = FlowDirectorDINF(mg) # update slope
                    fd = FlowDirectorMFD(self._grid, diagonals=True,
                                    partition_method = 'slope')
                    fd.run_one_step()
                    
                                            
                    if c%self.RunoutPlotInterval == 0:
                        cL[i].append(c)
                        
                        if SavePlot:
                            try:
                                # save DEM difference (DEM for iteration C - initial DEM)
                                _ = self._grid.add_field( 'topographic__elevation_d_'+str(i)+'_'+str(c)+'_'+str(self._time_idx),
                                                self._grid.at_node['topographic__elevation']-self.dem_initial,
                                                at='node')
                    
                                self.DFcells[inn].append(arn.astype(int))
                            except:
                                print('maps already written')
                    c+=1
                    
                    # update pulse counter
                    c2+=nid
                    # print('COUnTER')
                    # print(c)
                    # print(c2)
               
        # determine change in dem caused by mass wasting since last time step (fluvial erosion is not counted)
        self.dem_mw_dzdt = self._grid.at_node['topographic__elevation'] - self.dem_previous_time_step
        print('max change dem')
        print(self.dem_mw_dzdt.max())
        self.DFcells_dict[self._time_idx] = self.DFcells
                   
    
    def _TimeSinceDisturbance(self, dt):
        '''
        Years since a cell was disturbed advance in time increases period of 
        time since the storm event and the last storm event (ts_n - ts_n-1)
        if landslides, time since disturbance in any cell that has a change 
        in elevation caused by the debris flows is set to 0 (fluvial erosion does not count)

        '''
        # all grid cells advance forward in time by amount dt
        self.years_since_disturbance+=dt
        
        if self._this_timesteps_landslides.any():   
            # all grid cells that have scour or erosion from landslide or debris flow set to zero   
            self.years_since_disturbance[self.dem_mw_dzdt != 0] = 13/365 # 13 days selected to match max observed single day sediment transport following disturbance
              
    
    def _FluvialErosion(self):
        '''
        determine pulse location based on the time since mw caused disturbance
        small changes in cell elevation are ignored
        
        pulse volume is determined as a function of the time since the mw
        disturbance
        
        this is run each time step
                        
        NOTE: percentage of debris flow or terrace that becomes a pulse 
        of sediment is determined independently of flow magnitude
        '''
        # change in dpeth less than this is ignored
        # dmin = 0.1
            
        # rnodes = self._grid.nodes.reshape(mg.shape[0]*mg.shape[1]) # reshame mg.nodes into 1d array
        

        
        if np.any(np.abs(self.dem_mw_dzdt) > 0): # if there are no disturbed cells
            
            
            if shear_stress_erosion:
                DistMask = self.dem_mw_dzdt > 0
        
        
            if time_dependent_erosion:
            # disturbance (dz>0) mask
            DistMask = np.abs(self.dem_mw_dzdt) > 0 # deposit/scour during last debris flow was greater than threhsold value. 
            self.NewDistNodes = self.rnodes[DistMask] # all node ids that have disturbance larger than minimum value
            
            #new deposit and scour cell ids are appeneded to list of disturbed cells
            self.DistNodes = np.unique(np.concatenate((self.DistNodes,self.rnodes[DistMask]))).astype(int)            
            FERateC  = self.FluvialErosionRate[self.DistNodes] # fluvial erosion rate coefficients 
            cnd_msk = FERateC[:,0] > 0 # channel nodes only mask, no ersion applied to hillslope nodes
            self.FERateC = FERateC[cnd_msk] # coefficients of erosion function for channel nodes
            
            self.FENodes = self.DistNodes[cnd_msk] 
            print('no disturbed cells')
            
        if len(self.FENodes)>0:
            
            FERateCs = np.stack(self.FERateC) # change format
            # determine erosion depth

            
            # time since distubrance [years]
            YSD = self.years_since_disturbance[self.FENodes]
            
            # fluvial erosion rate (depth for storm) = a*x**b, x is time since distubrance #TODO: apply this to terrace cells only
            FED = FERateCs[:,0]*YSD**FERateCs[:,1]
            
            # force cells that have not been disturbed longer than ___ years to have no erosion
            # FED[YSD>3] = 0

            # Where FED larger than regolith thickness, FED equals regolith thickness (fresh bedrock is not eroded)
            MxErMask = self._grid.at_node['soil__thickness'][self.FENodes]< FED
            FED[MxErMask] = self._grid.at_node['soil__thickness'][self.FENodes][MxErMask]
            
            # update regolith depth : 
            self._grid.at_node['soil__thickness'][self.FENodes] = self._grid.at_node['soil__thickness'][self.FENodes]-FED

            # update dem       
            self._grid.at_node['topographic__elevation'][self.FENodes] = self._grid.at_node['topographic__elevation'][self.FENodes]-FED
            

            # TODO make new erosion function for channel cells that erodes as a function of flow shear stress
            # FED
            
            # apply criteria erosion can't exceed maximum, i.e., regolith depth
            # FED[FED>FEDmax] = FEDmax[FED>FEDmax]
            
            # apply minimum depth conditional
            # FEDMask = np.abs(FED) > dmin # scour depth during storm must be greater than threhsold value. 
            # FED = FED[FEDMask]
            
            ###TODO - create parcel aggregator - combines erosion depth into a single parcel equal to the minium parcel size (depth)
            ### deposits parcels at node locationn closest to center of deposits
            
            
            self.FED_all[self._time_idx] = FED
            self.FENodes_all[self._time_idx] = self.FENodes
            
            FED_ag = []
            FENodes_ag = []
            c = 0 # advance through FED using counter c
            while c < len(FED):
                d = FED[c]
                if d < self.min_d:
                    dsum = d
                    ag_node_L = [] # list of nodes aggregated into one parcel
                    while dsum < self.min_d:
                        dsum = dsum+d # add depth of nodes
                        ag_node_L.append(self.FENodes[c])  #add node to list of nodes in parcel
                        c+=1
                        if c >= len(FED):
                            break
                else:
                    ag_node_L = [self.FENodes[c]] 
                    dsum = d
                    c+=1
                
                FED_ag.append(dsum) # cumulative depth in each aggregated parcel (will be greater than d_min)
                FENodes_ag.append(ag_node_L[-1]) # the last node in the list is designated as the deposition node
            
            self.FED = np.array(FED_ag)
            self.FENodes = np.array(FENodes_ag)


            self.FED = FED#np.array(FED_ag)
            self.FENodes = self.FENodes#np.array(FENodes_ag)
            
            
            # self.FEDMask = FEDMask
            # self.FENodes = self.FENodes[FEDMask]

            # store as class variable
            # self.FED = FED
            # self.FEDmax = FEDmax
            
            # convert depth to volume        
            self.FEV = FED*self._grid.dx*self._grid.dx
            
            # TODO SUBTRACT FED FROM DEM
            
            
        else:
            print('no disturbed cells to fluvially erode')
            self.FENodes = np.array([])
        

        # print(self.FEV)
        # print(self.FENodes)
        # return (FENodes, FEV)
        
    
    
    def _parcelDFmaker(self):
        '''
        from the list of cell locations of the pulse and the volume of the pulse, 
        convert to a dataframe of pulses (ParcelDF) that is the input for pulser
        '''
        
        def LDistanceRatio(row):
            '''
            # determine deposition location on reach - ratio of distance from 
              inlet to deposition location in link to length of link

            '''
            return row['link_downstream_distance [m]']/self.linklength[int(row['link_#'])] 
        
        if len(self.FENodes) == 0:
            FEDn = []
            parcelDF = pd.DataFrame([])
            self.parcelDF = parcelDF

        
        else:
            Lmwlink = []            
            for h, FEDn in enumerate(self.FENodes): #for each cell deposit
            
                depXY = [self._grid.node_x[FEDn],self._grid.node_y[FEDn]] #deposition location x and y coordinate
                
                #search cells of links to find closest link grid
            
        
                #compute distance between deposit and all network cells
                def Distance(row):
                    return ((row['x']-depXY[0])**2+(row['y']-depXY[1])**2)**.5
                
                nmg_dist = self.xyDf.apply(Distance,axis=1)
                
                offset = nmg_dist.min()
                mdn = self.xyDf[nmg_dist == offset] #minimum distance node
                
                
                #find link that contains raster grid cell
                
                search = mdn['node'].iloc[0] #node number, first value if more than one grid cell is min dist from debris flow
                for i, sublist in enumerate(self.Lnodelist): #for each list of nodes (corresponding to each link i) 
                    if search in sublist: #if node is in list, then 
                        link_n = sublist#Lnodelist[i]
                        en = link_n.index(search)
                        link_d = self.Ldistlist[i]
                        ld = link_d[en]
                        linkID = i
                        mwlink = OrderedDict({'mw_unit':h,'vol [m^3]':self.FEV[h],'raster_grid_cell_#':FEDn,'link_#':linkID,'link_cell_#':search,'raster_grid_to_link_offset [m]':offset,'link_downstream_distance [m]':ld})
                        Lmwlink.append(mwlink)
                        break #for now use the first link found - later, CHANGE this to use largest order channel
                    else:
                        if i ==  len(self.Lnodelist):
                            print(' DID NOT FIND A LINK NODE THAT MATCHES THE DEPOSIT NODE ')

                
            parcelDF = pd.DataFrame(Lmwlink)
            pLinkDistanceRatio = parcelDF.apply(LDistanceRatio,axis=1)
            pLinkDistanceRatio.name = 'link_downstream_distance'        
            self.parcelDF = pd.concat([parcelDF,pLinkDistanceRatio],axis=1)            
        
        
        
        self.parcelDF_dict[self._time_idx] = self.parcelDF.copy() # save a copy of each pulse


        
    def _multidirectionflowdirector(self):
        '''
        removes rmg fields created by d8 flow director, runs multidirectionflowdirector

        '''
        self._grid.delete_field(loc = 'node', name = 'flow__sink_flag')
        self._grid.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
        self._grid.delete_field(loc = 'node', name = 'flow__receiver_node')
        self._grid.delete_field(loc = 'node', name = 'topographic__steepest_slope')
        # run flow director, add slope and receiving node fields
        fd = FlowDirectorMFD(self._grid, diagonals=True,
                              partition_method = 'slope')        
        fd.run_one_step()

    def _d8flowdirector(self):
        '''
        removes fields created by multidirectionflowdirector, runs d8flow director

        '''
        self._grid.delete_field(loc = 'node', name = 'flow__sink_flag')
        self._grid.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
        self._grid.delete_field(loc = 'node', name = 'flow__receiver_node')
        self._grid.delete_field(loc = 'node', name = 'topographic__steepest_slope')
        try:
            self._grid.delete_field(loc = 'node', name = 'flow__receiver_proportions') # not needed?
        except:
            None
        
        # run flow director, add slope and receiving node fields
        # re-compute slope and flow directions
        # sfb = SinkFillerBarnes(self._grid,'topographic__elevation', method='D8',fill_flat = False, 
        #                        ignore_overfill = False)
        # sfb.run_one_step()
        
        fr = FlowAccumulator(self._grid,'topographic__elevation',flow_director='D8')
        fr.run_one_step()
        
        df_4 = DepressionFinderAndRouter(self._grid)
        df_4.map_depressions()

    def run_one_step(self, dt):
        """Run MassWastingRouter forward in time.

        When the MassWastingRouter runs forward in time the following
        steps occur:

            1. If there are landslides to be routed:
                a. Determines extent of each landslide
                b. Routes the landslide through the raster model grid dem and
                    determines the raster model grid deposition location
                c. Converts the raster model grid deposition deposition 
                    location to a network model grid deposition location
                   
            2. If there are landslide deposits to erode
                a. looks up time since deposition and remaining volume, determines number of parcels to release
                b. checks if volume is greater than number of parcels
                c. if parcels to be released, releases parcels at a random location within the length of the deposit
                d. updates the remaining volume and time since deposit


        Parameters
        ----------
        dt : float
            Duration of time to run the NetworkSedimentTransporter forward.


        """
        self.mwprob = self.grid.at_node['MW__probability'] #  update mw probability variable
        self.hmwprob = self.grid.at_node['high__MW_probability']  #update boolean landslide field
        self._extractLSCells()

        
        if self._method == 'simple':
            if self._this_timesteps_landslides.any():
            
                # determine extent of landslides
                self._MassWastingExtent()
                print('masswastingextent')            
                # determine runout pathout
                self._MassWastingRunout()
                print('masswastingrounout')
                # convert mass wasting deposit location and attributes to parcel attributes  
                self._parcelDFmaker()
                print('rasteroutputtolink')
        
            else:
                self.parcelDF = pd.DataFrame([])
                print('No landslides to route this time step, checking for terrace deposits')
        
        if self._method == 'ScourAndDeposition':

            if self._this_timesteps_landslides.any():

                # determine extent of landslides
                self._MassWastingExtent()
                print('masswastingextent')     
                # run multi-flow director needed for debris flow routing
                self._multidirectionflowdirector()   
                # route debris flows, update dem to account for scour and 
                # deposition caused by debris flow and landslide processes
                self._MassWastingScourAndDeposition()
                print('scour and deposition') 
            # determine time since each cell was disturbed
            self._TimeSinceDisturbance(dt)
            # fluvially erode any recently disturbed cells, create lists of 
            # cells and volume at each cell that enters the channel network
            self._FluvialErosion()
            print('fluvial erosion')
            # set dem as dem as dem representing the previous time timestep
            self.dem_previous_time_step = self._grid.at_node['topographic__elevation'].copy() # set previous time step elevation
            # compute the cumulative vertical change of each grid cell
            self.dem_dz_cumulative = self._grid.at_node['topographic__elevation'] - self.dem_initial
            # convert list of cells and volumes to a dataframe compatiable with
            # the sediment pulser utility
            self._parcelDFmaker()
                        
            # re-run d8flowdirector for clumping and distance computations
            self._d8flowdirector()
            print('reset flow directions to d8')
        
        self._time += dt  # cumulative modeling time (not time or time stamp)
        self._time_idx += 1  # update iteration index 
            
            # terrace eroder