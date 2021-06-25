import numpy as np
import pandas as pd
from collections import OrderedDict

from landlab import Component, FieldError
from landlab.components import (FlowDirectorMFD, FlowAccumulator, DepressionFinderAndRouter,FlowDirectorSteepest)
from landlab import imshow_grid, imshow_grid_at_node


from landlab.components.mass_wasting_router.mass_wasting_SED import MassWastingSED
from landlab.utils.grid_t_tools import GridTTools


class MassWastingRouter(GridTTools):

    '''a component that redistributes mass wasting derived sediment through a 
    watershed and determines what portion of and where the sediment enters 
    the channel network

    This component is designed to couple a mass-wasting model with a sediment
    transport model. It has been written using the LandslideProbability component 
    and NetworkSedimentTransporter component. Any other mass-wasting model
    or network scale sediment transport model can be coupled with the mass wasting
    router so lang as the input are formatted correctly.
    
    componenmt overview:
        
    Hillslope scale landslides are interpreted from the a raster model grid 
    "Landslide__Probability" field.

    Attributes of each landslide are summarized from raster model grid fields values
    of all cells in each landslide and used to approximate initial debris flow
    volume and grain size distribution.

    The landslide is routed through the watershed as a debris flow using a cellular
    automata debris flow model that scours, deposits and updates the raster model grid
    field "topographic__elevation".

    An emperical model for sediment delivery to the channel converts zones of
    deposition and scour to a dataframe of sediment location and volumes
    input into the fluvial network.

    This component is unit sensitive. Units specific to each required field are
    listed below.



    TODO: erode channel node elevations using flow rate at grid cell
          record datetime of each timestep
          add parameter for using different router and option to have no terrace cells (for models with larger grid cells)
          clean up doc strings and comments
          tests (review Datacamp class), submittal to LANDLAB
          debris flow erosion model that accounts for available regolith thickness

    WISH LIST
    ADD entrainment model - See Frank et al., 2015, user parameterizes based on literature OR results of model runs in RAMMS
        first draft done

    '''



    _name = 'MassWastingRouter'

    _unit_agnostic = False

    _version = 1.0

    _info = {}


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
            probability_threshold = 0.75,  # landslide mapping parameters
            min_mw_cells = 1,
            release_dict = {'number of pulses':8, # MassWastingSED parameters 
                            'iteration delay':5 },
            df_dict = {'critical slope':0.1, 
                       'minimum flux':3,
                       'scour coefficient':0.025},
            FluvialErosionRate = [[0.03,-0.43], [0.01,-0.43]], # Fluvial erosion rate parameters

            parcel_volume = 0.2, # minimum parcel depth, parcels smaller than this are aggregated into larger parcels
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

        # minimum parcel size (parcels smaller than this are aggregated into a single parcel this size)
        self.parcel_volume = parcel_volume

        # dictionaries of variables for each iteration, for plotting
        self.LS_df_dict = {} #  a single dataframe for each storm, each clumps is a row, columns are average attributes of all cells in the clump
        self.LSclump_dict = {} # a single dictionary for each storm, each key is a dataframe with attributes of all cells in the clump
        self.DF_DEM_dif_dict = {} # dem differening during DF for plots
        self.DFcells_dict = {} # all precipitions during the debris flow routing algorithm
        self.StormDEM_dict = {} # DEM following debris flow and landslides
        self.FED_all = {} # all fluvial erosion depths
        self.FENodes_all = {} # all fluvial erosion nodes
        self.parcelDF_dict = {} #  parcels dataframe, created from aggregated fluvial erosion nodes

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

        ## create the nmg to rmg node mapper
        self._NMG_node_to_RMG_node_mapper()

        ## define nmg node elevation based on rmg channel nodes
        self._update_NMG_nodes()


        ### create initial values
        self._extractLSCells() # initial high landslide probability grid cells
        self.parcelDF = pd.DataFrame([]) # initial parcel DF
        self.dem_initial = self._grid.at_node['topographic__initial_elevation'].copy() # set initial elevation
        self.dem_previous_time_step = self._grid.at_node['topographic__initial_elevation'].copy() # set previous time step elevation
        self.dem_dz_cumulative = self.dem - self.dem_initial
        self.dem_mw_dzdt = self.dem_dz_cumulative # initial cells of deposition and scour - none, TODO, make this an optional input
        self.DistNodes = np.array([])
        self._grid.at_node['mass__wasting_events'] = np.zeros(self.nodes.shape[0]).astype(int)
        self._grid.at_node['mass__wasting_volumes'] = np.zeros(self.nodes.shape[0])

        ### instantiate MassWastingSED
        self.DebrisFlows = MassWastingSED(self._grid,release_dict,df_dict)

        

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
        LSclump = {} # final mass wasting clump dicitonary,
        #oantains a dataframe listing attributes of all cells in each clump

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


        # LS_df  - a single dataframe, each clumps is a row, columns are average attributes of all cells in the clump
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
        self.LSclump_dict[self._time_idx] = LSclump.copy(); print('SAVED A CLUMP')
 
        # prepare MassWastingSED rmg inputs "mass__wasting_events" and "mass__wasting_volumes"

        mw_events = np.zeros(np.hstack(self._grid.nodes).shape[0])
        mw_events_volumes = np.zeros(np.hstack(self._grid.nodes).shape[0])
        

        ivL = self.LS_df['vol [m^3]'][0::1].values # initial volume list
        innL = self.LS_df['cell'][0::1].values.astype('int') # initial node number list        
        
        mw_events[innL] = 1
        mw_events_volumes[innL] = ivL 
        
        self._grid.at_node["mass__wasting_events"] = mw_events.astype(int)
        self._grid.at_node["mass__wasting_volumes"] = mw_events_volumes
        
        
        
        # return groupALS_l,LSclump,LS_df #clumps around a single cell, clumps for each landslide, summary of landslide



    # def _MassWastingScourAndDeposition(self):

    #     SavePlot = True


    #     self.DFcells = {}
    #     self.RunoutPlotInterval = 5



    #     # debris flow runout parameters

    #     # release parameters for landslide
    #     nps = 8 # number of pulses, list for each landslide
    #     nid = 5 # delay between pulses (iterations), list for each landslide


    #     # critical slope at which debris flow stops
    #     # higher reduces spread and runout distance
    #     slpc = 0.1

    #     # forced stop volume threshold
    #     # material stops at cell when volume is below this,
    #     # higher increases spread and runout distance
    #     SV = 3.3

    #     # entrainment coefficient
    #     # very sensitive to entrainment coefficient
    #     # higher causes higher entrainment rate, longer runout, larger spread
    #     cs = 3.9e-5



    #     if self.LS_df.empty is True:
    #         print('no debris flows to route')
    #     else:

    #         ivL = self.LS_df['vol [m^3]'][0::1].values.astype('int') # initial volume list
    #         innL = self.LS_df['cell'][0::1].values.astype('int') # initial node number list

    #         cL = {}
    #         dem_dif_L = []
    #         # ivL and innL can be a list of values. For each value in list:
    #         for i,inn in enumerate(innL):
    #             # print(i)
    #             cL[i] = []


    #             # set up initial landslide cell
    #             iv =ivL[i]/nps# initial volume (total volume/number of pulses)

    #             # initial receiving nodes (cells) from landslide
    #             rn = self._grid.at_node.dataset['flow__receiver_node'].values[inn]
    #             self.rn = rn
    #             rni = rn[np.where(rn != -1)]

    #             # initial receiving proportions from landslide
    #             rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[inn]
    #             rp = rp[np.where(rp > 0)]
    #             rvi = rp*iv # volume sent to each receving cell

    #             # now loop through each receiving node,
    #             # determine next set of recieving nodes
    #             # repeat until no more receiving nodes (material deposits)

    #             self.DFcells[inn] = []
    #             slpr = []
    #             c = 0
    #             c2 = 0
    #             arn = rni
    #             arv = rvi

    #             enL = []
    #             while len(arn)>0:# or c <300:


    #                 # time to pulse conditional statement
    #                 # add pulse (fraction) of total landslide volume
    #                 # at interval "nid" until total number of pulses is less than total
    #                 # for landslide volume (nps*nid)

    #                 if ((c+1)%nid ==0) & (c2<=nps*nid):
    #                     arn = np.concatenate((arn,rni))
    #                     arv = np.concatenate((arv,rvi))

    #                 arn_ns = np.array([])
    #                 arv_ns = np.array([])

    #                 # for each unique cell in receiving node list arn
    #                 for n in np.unique(arn):
    #                     n = int(n)

    #                     # slope at cell (use highes slope)
    #                     slpn = self._grid.at_node['topographic__steepest_slope'][n].max()

    #                     # incoming volume: sum of all upslope volume inputs
    #                     vin = np.sum(arv[arn == n])

    #                     # determine deposition volume following Campforts, et al., 2020
    #                     # since using volume (rather than volume/dx), L is (1-(slpn/slpc)**2)
    #                     # rather than mg.dx/(1-(slpn/slpc)**2)
    #                     Lnum = np.max([(1-(slpn/slpc)**2),0])

    #                     # deposition volume
    #                     df_depth = vin/(self._grid.dx*self._grid.dx) #df depth

    #                     dpd = vin*Lnum # deposition volume

    #                     # max erosion depth

    #                     # determine erosion volume (function of slope, shear stress)
    #                     T_df = 1700*9.81*df_depth*slpn # shear stress from df

    #                     # max erosion depth equals regolith (soil) thickness
    #                     dmx = self._grid.at_node['soil__thickness'][n]

    #                     # erosion depth:
    #                     er = min(dmx, cs*T_df)

    #                     # erosion volume
    #                     ev = er*self._grid.dx*self._grid.dy
    #                     # volumetric balance at cell

    #                     # determine volume sent to downslope cells

    #                     # additional constraint to control debris flow behavoir
    #                     # if flux*cell width sent to a cell is below threshold, debris is forced to stop
    #                     if vin <=SV:
    #                         dpd = vin # all volume that enter cell is deposited
    #                         vo = 0 # debris stops, so volume out is 0

    #                         # determine change in cell height
    #                         deta = (dpd)/(self.dx*self.dy) # (deposition)/cell area

    #                     else:
    #                         vo = vin-dpd+ev # vol out = vol in - vol deposited + vol eroded

    #                         # determine change in cell height
    #                         deta = (dpd-ev)/(self.dx*self.dy) # (deposition-erosion)/cell area


    #                     # update raster model grid regolith thickness and dem elevation
    #                     # if deta larger than regolith thickness, deta equals regolith thickness (fresh bedrock is not eroded)
    #                     if self._grid.at_node['soil__thickness'][n]+deta <0:
    #                         deta = - self._grid.at_node['soil__thickness'][n]

    #                     # Regolith - difference between the fresh bedrock surface and the top surface of the dem
    #                     self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]+deta

    #                     # Topographic elevation - top surface of the dem
    #                     self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta

    #                     # build list of receiving nodes and receiving volumes for next iteration

    #                     # material stops at node if transport volume is 0 OR node is a
    #                     # boundary node
    #                     if vo>0 and n not in self._grid.boundary_nodes:


    #                         th = 0
    #                         # receiving proportion of volume from cell n to each downslope cell
    #                         rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[n]
    #                         rp = rp[np.where(rp > th)]
    #                         # print(rp)

    #                         # receiving nodes (cells)
    #                         rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
    #                         rn = rn[np.where(rn != -1)]
    #                         # rn = rn[np.where(rp > th)]

    #                         # receiving volume
    #                         rv = rp*vo

    #                         # store receiving nodes and volumes in temporary arrays
    #                         arn_ns =np.concatenate((arn_ns,rn), axis = 0) # next step receiving node list
    #                         arv_ns = np.concatenate((arv_ns,rv), axis = 0) # next steip receiving node incoming volume list


    #                 # once all cells in iteration have been evaluated, temporary receiving
    #                 # node and node volume arrays become arrays for next iteration
    #                 arn = arn_ns
    #                 arv = arv_ns

    #                 # update DEM slope
    #                 # fd = FlowDirectorDINF(mg) # update slope
    #                 fd = FlowDirectorMFD(self._grid, diagonals=True,
    #                                 partition_method = 'slope')
    #                 fd.run_one_step()


    #                 # save dem difference and precipitions for illustration plots
    #                 if c%self.RunoutPlotInterval == 0:

    #                     cL[i].append(c)


    #                     # save precipitions
    #                     self.DFcells[inn].append(arn.astype(int))

    #                     # save dem differencing
    #                     dem_dif = self._grid.at_node['topographic__elevation']-self.dem_initial
    #                     # dem_dif = np.expand_dims(dem_dif, axis=0)

    #                     dem_dif_L.append(dem_dif)



    #                 c+=1


    #                 c2+=nid

    #         self.DF_DEM_dif_dict[self._time_idx] = dem_dif_L

    #         # all
    #         self.DFcells_dict[self._time_idx] = self.DFcells

    #         self.StormDEM_dict[self._time_idx] = self._grid.at_node['topographic__elevation'].copy()


    def _dem_dz(self):
        '''
        determine change in dem since the last time step

        Returns
        -------
        None.

        '''

        self.dem_mw_dzdt = self._grid.at_node['topographic__elevation'] - self.dem_previous_time_step
        # print('max change dem due to debris flows')
        # print(self.dem_mw_dzdt.max())


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


    def _FluvialErosion(self, erosion_model = 'time'):
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

        if erosion_model == 'shear_stress':
            if np.any(self.dem_mw_dzdt > 0):
                DistMask = self.dem_mw_dzdt > 0
                self.FENodes = self.rnodes[DistMask]
                self.FED = self.dem_mw_dzdt[DistMask] #change to this as maximum for shear stress based estimate
                self.FEV = self.FED*self._grid.dx*self._grid.dx

        if erosion_model == 'time':

            # if np.any(np.abs(self.dem_mw_dzdt) > 0): # if there are no disturbed cells
                # disturbance (dz>0) mask
            DistMask = np.abs(self.dem_mw_dzdt) > 0 # deposit/scour during last debris flow was greater than threhsold value.
            self.NewDistNodes = self.rnodes[DistMask] # all node ids that have disturbance larger than minimum value

            #new deposit and scour cell ids are appeneded to list of disturbed cells
            self.DistNodes = np.unique(np.concatenate((self.DistNodes,self.rnodes[DistMask]))).astype(int)
            FERateC  = self.FluvialErosionRate[self.DistNodes] # fluvial erosion rate coefficients
            cnd_msk = FERateC[:,0] > 0 # channel nodes only mask, no ersion applied to hillslope nodes
            self.FERateC = FERateC[cnd_msk] # coefficients of erosion function for channel nodes

            self.FENodes = self.DistNodes[cnd_msk]
            # print('no disturbed cells')

            if len(self.FENodes)>0:

                FERateCs = np.stack(self.FERateC) # change format
                # determine erosion depth


                # time since distubrance [years]
                self.YSD = self.years_since_disturbance[self.FENodes]

                # fluvial erosion rate (depth for storm) = a*x**b, x is time since distubrance #TODO: apply this to terrace cells only
                self.FED = FERateCs[:,0]*self.YSD**FERateCs[:,1]

                # force cells that have not been disturbed longer than ___ years to have no erosion
                # FED[YSD>3] = 0

                # Where FED larger than regolith thickness, FED equals regolith thickness (fresh bedrock is not eroded)
                MxErMask = self._grid.at_node['soil__thickness'][self.FENodes]< self.FED
                self.FED[MxErMask] = self._grid.at_node['soil__thickness'][self.FENodes][MxErMask]

                # update regolith depth :
                self._grid.at_node['soil__thickness'][self.FENodes] = self._grid.at_node['soil__thickness'][self.FENodes]-self.FED

                # update dem
                self._grid.at_node['topographic__elevation'][self.FENodes] = self._grid.at_node['topographic__elevation'][self.FENodes]-self.FED


                self.FED_all[self._time_idx] = self.FED
                self.FENodes_all[self._time_idx] = self.FENodes


                # convert depth to volume
                self.FEV = self.FED*self._grid.dx*self._grid.dx

            else:
                print('no disturbed cells to fluvially erode')
                self.FENodes = np.array([])


    def _parcelAggregator(self):
        '''
        reduces the number of parcels entered into a channel network by aggregating
        parcels into larger parcels

        TODO: make self.parcel_volume automatically determined based on a maxmum
        number of parcels per iteration

        Returns
        -------
        None.

        '''
        if len(self.FENodes)>0:
            # aggregates parcels into
            FEV_ag = []
            FENodes_ag = []
            c = 0 # advance through FED using counter c
            while c < len(self.FEV):
                v = self.FEV[c]
                if v < self.parcel_volume:
                    vsum = v
                    ag_node_L = [] # list of nodes aggregated into one parcel
                    while vsum < self.parcel_volume:
                        vsum = vsum+v # add depth of nodes
                        ag_node_L.append(self.FENodes[c])  #add node to list of nodes in parcel
                        c+=1
                        if c >= len(self.FEV):
                            break
                else:
                    ag_node_L = [self.FENodes[c]]
                    vsum = v
                    c+=1

                FEV_ag.append(vsum) # cumulative depth in each aggregated parcel (will be greater than d_min)
                FENodes_ag.append(ag_node_L[-1]) # the last node in the list is designated as the deposition node

            self.FEV = np.array(FEV_ag)
            self.FENodes = np.array(FENodes_ag)
        else:
            print('no disturbed cells to aggregate')



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



    def _update_NMG_nodes(self):
        '''
        updates the elevation of the nmg nodes based on the closest channel rmg node
        updates the link slopes based on the updated nmg node elevations

        Returns
        -------
        None.

        '''

        # update elevation
        for i, node in enumerate(self.nmg_nodes):
            RMG_node = self.NMGtoRMGnodeMapper[i]
            self._nmgrid.at_node['topographic__elevation'][i] = self._grid.at_node['topographic__elevation'][RMG_node]

        # update slope
        nmg_fd = FlowDirectorSteepest(self._nmgrid, "topographic__elevation")
        nmg_fd.run_one_step()

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
        # self.hmwprob = self.grid.at_node['high__MW_probability']  #update boolean landslide field
        self._extractLSCells()


        if self._method == 'simple':
            if self._this_timesteps_landslides.any():

                # determine extent of landslides
                self._MassWastingExtent()
                # print('masswastingextent')
                # determine runout pathout
                self._MassWastingRunout()
                # print('masswastingrounout')
                # convert mass wasting deposit location and attributes to parcel attributes
                self._parcelDFmaker()
                # print('rasteroutputtolink')

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
                # print('multiflowdirection')
                # route debris flows, update dem
                self.DebrisFlows.run_one_step(dt)
                
                self.DFcells_dict[self._time_idx] = DebrisFlows.DFcells
                self.StormDEM_dict[self._time_idx] = DebrisFlows._grid.at_node['topographic__elevation'].copy()                
                # print('scour and deposition')

                # subtract previous storm dem from this dem
                self._dem_dz()
                # print('dem differencing to determine mass-wasting deposition and scour zones')

                # update NMG node elevation
                self._update_NMG_nodes()
                # print('updated NMG node elevation')

                # reset landslide probability field
                self._grid.at_node['MW__probability'] = np.zeros(self._grid.at_node['topographic__elevation'].shape[0])

            # determine time since each cell was disturbed by mass wasting process
            self._TimeSinceDisturbance(dt)
            # print('determed time since last mass wasting disturbance')
            # fluvially erode any recently disturbed cells, create lists of
            # cells and volume at each cell that enters the channel network
            self._FluvialErosion()
            # print('fluvial erosion')

            self._parcelAggregator()
            # print('aggregating parcels')

            # set dem as dem as dem representing the previous time timestep
            self.dem_previous_time_step = self._grid.at_node['topographic__elevation'].copy() # set previous time step elevation

            # compute the cumulative vertical change of each grid cell
            self.dem_dz_cumulative = self._grid.at_node['topographic__elevation'] - self.dem_initial

            # convert list of cells and volumes to a dataframe compatiable with
            # the sediment pulser utility
            self._parcelDFmaker()

            # re-run d8flowdirector
            # (clumping and down-slope distance computations require d-8 flow direction)
            self._d8flowdirector()
            # print('reset flow directions to d8')



        self._time += dt  # cumulative modeling time (not time or time stamp)
        self._time_idx += 1  # update iteration index

