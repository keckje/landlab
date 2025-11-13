# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import OrderedDict

from landlab import RasterModelGrid
from landlab import Component, FieldError
from landlab.components import FlowAccumulator
from landlab.components import(FlowDirectorD8, 
                                FlowDirectorDINF, 
                                FlowDirectorMFD, 
                                FlowDirectorSteepest)

from landlab.utils.channel_network_grid_tools import ChannelNetworkToolsInterpretor


class LandslideMapper(Component):

    """ a component that maps hillslope scale landslides from a raster model grid
    of topographic elevation and landslide factor of saftey or landslide 
    probability and returns a table and raster model grid fields that indicate
    the location and summarize the attributes of each mapped landslide
    
    TODO:   1.look into replaceing pandas dataframe and pd.merge(inner) with np.in1d(array1,array2)
            and pd.erge(outer) call with np.unique(), get rid of dataframe operations
            2.look at replacing the NotDivergent function with the landlab convergence function
            3.rethink how to combine clumps and remove them from the initial clump list

    """

    _name = 'LandslideMapper'

    _unit_agnostic = False

    _version = 1.0

    
    _info = {
        
        'mass__wasting_potential': {            # slope stability metric that can be 'landslide probability' or "factor of safety"
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
            },
        
        'topographic__elevation': {            
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
            },
        
        'soil__thickness': {            
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "regolith (soil) thickness, measured perpendicular to the \
            land surface and includes all materials above the unweathered bedrock \
            surface, e.g., saprolite, colluvium, alluvium, glacial drift"
            },
        
        "flow__receiver_node": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
            },
        
        'topographic__slope': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
            
            },

        }
        
        
    def __init__(
        self,
        grid,
        nmgrid = None,
        Ct = 5000,
        BCt = 100000,
        MW_to_channel_threshold = 50,
        PartOfChannel_buffer = 10, # may not need this parameter            
        mass_wasting_threshold = 0.75,  # landslide mapping parameters
        threshold_type = "greater_than",
        min_mw_cells = 1,
        gti = None,
        ):

        
        # call __init__ from parent classes
        # super().__init__(grid, nmgrid, Ct, BCt)
        
        super().__init__(grid)

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


        if 'mass__wasting_potential' in grid.at_node:
            self.mwprob = grid.at_node['mass__wasting_potential']
        else:
            raise FieldError(
                'A mass__wasting_potential field is required as a component input!')

        #   mass wasting clumps
        if 'mass__wasting_clumps' in grid.at_node:
            self.mwclump = grid.at_node['mass__wasting_clumps']
        else:
            self.mwclump = grid.add_zeros('node',
                                        'mass__wasting_clumps')                                          

        # instantiate channel network grid tools or use provided instance
        if gti != None:
            self.gti = gti
            # check if ChannelNodes already created                   
        else:
            self.gti = ChannelNetworkToolsInterpretor(grid = grid,Ct = Ct,BCt = BCt)

        # if gtm != None:
        #     self.gtm = gtm
        # else:
        #     self.gtm = ChannelNetworkToolsMapper(grid = grid, nmgrid = nmgrid, Ct = Ct,BCt = BCt)


        if not hasattr(gti,"ChannelNodes"):
            self.gti.extract_channel_nodes(Ct,BCt)

        
        ### prep LandslideMapper
        self.MW_to_C_threshold = MW_to_channel_threshold # maximum distance [m] from channel for downslope clumping
        self.mass_wasting_threshold = mass_wasting_threshold # probability of landslide threshold
        self.threshold_type = threshold_type
        self.min_mw_cells = min_mw_cells # minimum number of cells to be a mass wasting clump
              
        
        # initial class variable values
        self._extractLSCells() # initial high landslide probability grid cells
        self.parcelDF = pd.DataFrame([]) # initial parcel DF    
        self.POCbuffer = PartOfChannel_buffer
        self.LS_df = pd.DataFrame([])
        self.LSclump = {} # initial empty


    def _extractLSCells(self):
        '''extracts the node id numbers of all nodes that have a MW__probabiliity
        larger than the user specified threshold
        '''
        a = self.mwprob

        type(a) # np array

        # keep as (nxr)x1 array, convert to pd dataframe
        # a_df = pd.DataFrame(a)
        if self.threshold_type == 'greater_than':
            mask = a > self.mass_wasting_threshold
        elif self.threshold_type == 'less_than':
            mask = a < self.mass_wasting_threshold
        else:
            raise FieldError(
                'not an accepted threshold type!')

        # boolean list of which grid cells are landslides
        self._this_timesteps_landslides = mask

        a_th = a[mask] #Make all values less than threshold NaN

        #add new field to grid that contains all cells with annual probability greater than threshold
        self.hmwprob = a_th #consider removing this instance variable

        # self.grid.at_node['high__MW_probability'] = self.hmwprob # change to mass__wasting_potential

        #create mass wasting unit list
        self.LS_cells = self.gti.nodes[mask].flatten() # could just use rnodes
        # print(self.LS_cells)

                
    def AdjCells(self,n):
            '''MWR                
            returns cell numbers of cells adjacent to cell (n) AND the flow direction indexes
            
            gr - number of rows in grid
            gc - number of columns in grid
            n - cell number other cells are adjacent to
            ac - node numbers
            acn - a
            
            #change to use mg.adjacent_nodes_at_node[] and mg.diagonal_adjacent_nodes_at_node[]
    
            '''
            gc = self.gti.gc
            gr = self.gti.gr
            
            u = n-gc
            ur = n-gc+1
            r = n+1
            br = n+gc+1
            b = n+gc
            bl = n+gc-1
            l = n-1
            ul = n-gc-1   
            
            
            if n == 0:                      #top left corner
                ac = [r,br,b]
                acn = [2,3,4]               #flow direction index
            elif n == gc-1:                 #top right corner
                ac = [b,bl,l]
                acn = [4,5,6]
            elif n == gr*gc-1:              #bottom right corner
                ac = [u,l,ul]  
                acn = [0,6,7]
            elif n == gr*gc-gc:             #bottom left corner
                ac = [u,ur,r]
                acn = [0,1,2]
            elif n<gc-1 and n>0:            #top side
                ac = [r,br,b,bl,l]
                acn = [2,3,4,5,6]
            elif n%gc == gc-1:              #right side
                ac = [u,b,bl,l,ul]
                acn = [0,4,5,6,7]
            elif n<gr*gc-1 and n>gr*gc-gc:  #bottom side
                ac = [u,ur,r,l,ul]
                acn = [0,1,2,6,7]
            elif n%gc == 0:                 #left side
                ac = [u,ur,r,br,b]
                acn = [0,1,2,3,4]
            else:                           #inside grid
                ac = [u,ur,r,br,b,bl,l,ul]
                acn = [0,1,2,3,4,5,6,7]
                
            return ac,acn #adjacent cell numbers and cell numbers for direction grid

                        
    def NotDivergent(self,n,ac,acn):
        '''takes mass wasting cell and adjacent cell numbers as input
        returns boolian list of cells that are not divergent (convergent or planar)
        
        xdif - change in x in slope direction to receiving cell
        ydif -  change in y in slope direciton to receiving cell
        n - int, mass wasting unit cell
        ac - list of int, grid number of adjacent cells
        acn - list of int, adjacent cell number
        '''
        
        xd = self.gti.xdif[n]
        yd = self.gti.ydif[n]
        
        '''       
        functions that check if cell is not divergent based on center cell direction
        
        Input to functions:
        x - adjacent cell number (ordered 0 to 7, with 0 north of center, clockwise)
        yda - change in y direction at adjacent cell to it's receiving cell
        xda - change in x direction at adjacent cell to it's receiving cell
            '''
    
        def pyzx(x,yda,xda): #positve y zero x  (slopes north)
    
            return {            
                0: True,
                1: (yda > 0 and xda == 0) or (yda > 0 and xda < 0) or (yda == 0 and xda < 0),    
                2: (yda > 0 and xda == 0) or (yda > 0 and xda < 0) or (yda == 0 and xda < 0),
                3: (yda > 0 and xda == 0) or (yda > 0 and xda < 0),
                4: (yda > 0 and xda == 0),
                5: (yda > 0 and xda == 0) or (yda > 0 and xda > 0),
                6: (yda > 0 and xda == 0) or (yda > 0 and xda > 0) or (yda == 0 and xda > 0),
                7: (yda > 0 and xda == 0) or (yda > 0 and xda > 0) or (yda == 0 and xda > 0),
                    
            }[x]
        
        def nyzx(x,yda,xda): #negative y, zero x (slopes south)
            
            return {
                0: (yda < 0 and xda == 0),
                1: (yda < 0 and xda == 0) or (yda < 0 and xda < 0),
                2: (yda < 0 and xda == 0) or (yda < 0 and xda < 0) or (yda == 0 and xda < 0),
                3: (yda < 0 and xda == 0) or (yda < 0 and xda < 0) or (yda == 0 and xda < 0),
                4: True,
                5: (yda < 0 and xda == 0) or (yda < 0 and xda > 0) or (yda == 0 and xda > 0),
                6: (yda < 0 and xda == 0) or (yda < 0 and xda > 0) or (yda == 0 and xda > 0),
                7: (yda < 0 and xda == 0) or (yda < 0 and xda > 0),
                
            }[x]
            
        def zypx(x,yda,xda): #zero y, positive x (slopes east)
            
            return {
                0: (yda == 0 and xda > 0) or (yda < 0 and xda > 0) or (yda < 0 and xda == 0),
                1: (yda == 0 and xda > 0) or (yda < 0 and xda > 0) or (yda < 0 and xda == 0),
                2: True,
                3: (yda == 0 and xda > 0) or (yda > 0 and xda > 0) or (yda > 0 and xda == 0),
                4: (yda == 0 and xda > 0) or (yda > 0 and xda > 0) or (yda > 0 and xda == 0),
                5: (yda == 0 and xda > 0) or (yda > 0 and xda > 0),
                6: (yda == 0 and xda > 0),
                7: (yda == 0 and xda > 0) or (yda < 0 and xda > 0),
                
            }[x]
        
        def zynx(x,yda,xda): #zero y, negative x (slopes west)
            
            return {
                0: (yda == 0 and xda < 0) or (yda < 0 and xda < 0) or (yda < 0 and xda == 0),
                1: (yda == 0 and xda < 0) or (yda < 0 and xda < 0),
                2: (yda == 0 and xda < 0),
                3: (yda == 0 and xda < 0) or (yda > 0 and xda < 0),
                4: (yda == 0 and xda < 0) or (yda > 0 and xda < 0) or (yda > 0 and xda == 0), 
                5: (yda == 0 and xda < 0) or (yda > 0 and xda < 0) or (yda > 0 and xda == 0),
                6: True,
                7: (yda == 0 and xda < 0) or (yda < 0 and xda < 0) or (yda < 0 and xda == 0),
                
            }[x]
        
        def pypx(x,yda,xda): #positive y, positive x (slopes northeast)
            
            return {
                0: (yda > 0 and xda > 0) or (yda == 0 and xda > 0) or (yda < 0 and xda > 0), 
                1: True,
                2: (yda > 0 and xda > 0) or (yda > 0 and xda == 0) or (yda > 0 and xda < 0),
                3: (yda > 0 and xda > 0) or (yda > 0 and xda == 0) or (yda > 0 and xda < 0), 
                4: (yda > 0 and xda > 0) or (yda > 0 and xda == 0), 
                5: (yda > 0 and xda > 0), 
                6: (yda > 0 and xda > 0) or (yda == 0 and xda > 0), 
                7: (yda > 0 and xda > 0) or (yda == 0 and xda > 0) or (yda < 0 and xda > 0), 
                
            }[x]
        
        def pynx(x,yda,xda): # positive y, negative x (slopes northwest)
            
            return {
                0: (yda > 0 and xda < 0) or (yda == 0 and xda < 0) or (yda < 0 and xda < 0), 
                1: (yda > 0 and xda < 0) or (yda == 0 and xda < 0) or (yda < 0 and xda < 0), 
                2: (yda > 0 and xda < 0) or (yda == 0 and xda < 0), 
                3: (yda > 0 and xda < 0),  
                4: (yda > 0 and xda < 0) or (yda > 0 and xda == 0), 
                5: (yda > 0 and xda < 0) or (yda > 0 and xda == 0) or (yda > 0 and xda > 0), 
                6: (yda > 0 and xda < 0) or (yda > 0 and xda == 0) or (yda > 0 and xda > 0), 
                7: True, 
                
            }[x]
        
        def nypx(x,yda,xda): #negative y, postive x (slopes southeast)
            
            return {
                0: (yda < 0 and xda > 0) or (yda < 0 and xda == 0),  
                1: (yda < 0 and xda > 0) or (yda < 0 and xda == 0) or (yda < 0 and xda < 0), 
                2: (yda < 0 and xda > 0) or (yda < 0 and xda == 0) or (yda < 0 and xda < 0), 
                3: True, 
                4: (yda < 0 and xda > 0) or (yda == 0 and xda > 0) or (yda > 0 and xda > 0),
                5: (yda < 0 and xda > 0) or (yda == 0 and xda > 0) or (yda > 0 and xda > 0), 
                6: (yda < 0 and xda > 0) or (yda == 0 and xda > 0), 
                7: (yda < 0 and xda > 0), 
                
            }[x]
        
        def nynx(x,yda,xda): #negative y, negative x (slopes southwest)
            
            return {
                0: (yda < 0 and xda < 0) or (yda < 0 and xda == 0), 
                1: (yda < 0 and xda < 0), 
                2: (yda < 0 and xda < 0) or (yda == 0 and xda < 0), 
                3: (yda < 0 and xda < 0) or (yda == 0 and xda < 0) or (yda > 0 and xda < 0), 
                4: (yda < 0 and xda < 0) or (yda == 0 and xda < 0) or (yda > 0 and xda < 0), 
                5: True, 
                6: (yda < 0 and xda < 0) or (yda < 0 and xda == 0) or (yda < 0 and xda > 0),
                7: (yda < 0 and xda < 0) or (yda < 0 and xda == 0) or (yda < 0 and xda > 0), 
                
            }[x]
        
            
        ydal = self.gti.ydif[ac]
        xdal = self.gti.xdif[ac]
        
        #create mask for adjacent cells using  check if not divergent (True) or divergent (false)
        if yd > 0 and xd == 0:
            ac_m = np.array(list(map(pyzx,acn,ydal,xdal)))
        elif yd < 0 and xd == 0:
            ac_m = np.array(list(map(nyzx,acn,ydal,xdal)))        
        elif yd == 0 and xd > 0:
            ac_m = np.array(list(map(zypx,acn,ydal,xdal)))
        elif yd == 0 and xd < 0:
            ac_m = np.array(list(map(zynx,acn,ydal,xdal)))
        elif yd > 0 and xd > 0:
            ac_m = np.array(list(map(pypx,acn,ydal,xdal)))
        elif yd > 0 and xd < 0:
            ac_m = np.array(list(map(pynx,acn,ydal,xdal)))
        elif yd < 0 and xd > 0:
            ac_m = np.array(list(map(nypx,acn,ydal,xdal)))
        elif yd < 0 and xd < 0:
            ac_m = np.array(list(map(nynx,acn,ydal,xdal)))
        else: # if yd == 0 and xd == 0, flat ground? # NEED TO DETERMINE WHEN THIS OCCURS
            print('WARNING, CELL '+str(n)+'HAS INDETERMINANT FLOW DIRECTION')
            ac_m = np.array(list(map(nynx,acn,ydal,xdal)))
        
        return ac_m

    
    def aLScell(self,cl):
        """returns boolian list of adjacent cells that are also mass wasting cells
        
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
        """returns list of cells to clump (adjacent, not divergent) with input cell (lsc)
        """


        #(1) determine adjacent cells
        Adj = self.AdjCells(lsc)
        # print('adjcells')
        ac = Adj[0] #grid cell numbers of adjacent cells
        acn = Adj[1] #adjacent cell numbers based on local number schem

        # print(ac)
        # print(acn)
        # print(lsc)
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

    
    def _downslopecells(self,StartCell): # add channel_nodes as an input
        '''MWR
        compute distance between a given cell and the nearest downslope channel 
        network cell. distance is computed to POCbuffer distance from channel network
        cell. Track which cells are downslope
        
        channel network is the colluvial channel network
        
        this function is used to determine the lower extent of a landslide.
        
        use this to determine 
        
        (1) check if clump meets distance threshold (is close enough to channel network)
          to causethe hillslope below to fail         
        
        (2) if it does, get list of all unique cells and add to clump
        
        '''

        loc = []
        dist = []
        c = 0
        
        loc.append(int(StartCell))
        dist.append((self.gti.xdif[int(StartCell)]**2+self.gti.ydif[int(StartCell)]**2)**.5)
                       
        flow = True
        
        while flow == True:
            
            slope  = self._grid.at_node['topographic__slope'][loc[c]]
            #compute distance between deposit and all debris flow network cells
            cdist, nc = self.gti.min_distance_to_network(loc[c],  ChType = 'debrisflow') # change this to gt.min_distance_to_network(self._grid, channel_nodes, loc[c])
            #TO DO need to change so that if distance to network is less than minimum, then stop
            
            
            #if loc[c] not in self.xyDF['node'].values: # channel network doesnt always match DEM
            # print(cdist)
            if cdist > self.POCbuffer: # downslope distance measured to POCbuffer from channel    
                loc.append(self._grid.at_node['flow__receiver_node'][loc[c]])
                dist.append((self.gti.xdif[self._grid.at_node['flow__receiver_node'][loc[c]]]**2+self.gti.ydif[self._grid.at_node['flow__receiver_node'][loc[c]]]**2)**.5)
                c=c+1
                if loc[-1] == loc[-2]: # check that runout is not stuck at same node # NEED TO DEBUG
                    break
            else:
                flow = False
        
        Dista = np.array(dist)
        Dist = Dista.sum()
        return (Dist,loc)


    def _MassWastingExtent(self):
        """maps hillslope scale landslides from a raster model grid fields of
        topographic elevation and landslide factor of saftey or landslide probability
        """

        # minimum number of cells criteria
        groupALS_l = []
        for i, v in enumerate(self.LS_cells):
            # print(v)
            group = self.GroupAdjacentLScells(v)
            if len(group) >= self.min_mw_cells: # at least this number of cells to be a clump
                groupALS_l.append(group) # list of all single cell clumps
        
        if len(groupALS_l) > 0: # if any landslide initial clumps, run the following:
            
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
                    dfm = pd.merge(df1,df2, on = 'cell', how= "inner") # if no cells shared, dfm is an empty dataframe
    
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
    
                # modify DEM and soil thickness in mapper?
                # subtract 90% of soil depth from dem,
                # self._grid.at_node['topographic__elevation'][df1.cell] = self._grid.at_node['topographic__elevation'][df1.cell] - 0.9*self._grid.at_node['soil__thickness'][df1.cell]
    
    
                # set soil thickness to 0 at landslide cells
                # self._grid.at_node['soil__thickness'][df1.cell] =0
    
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
            except: # high landslide potential cells may not meet clumping criteria
                LS_df = pd.DataFrame([])
    
            #create mass__wasting_clumps field
            for c, nv in enumerate(LSclump.values()):
                self.mwclump[list(nv['cell'].values)] = c+1 #plus one so that first value is 1
    
            self.grid.at_node['mass__wasting_clumps'] = self.mwclump
            self.LS_df = LS_df
            self.LSclump = LSclump
            # self.LS_df_dict[self._time_idx] = LS_df.copy()
            # self.LSclump_dict[self._time_idx] = LSclump.copy(); print('SAVED A CLUMP')
     
            # prepare raster model grid fields "mass__wasting_events", "mass__wasting_volumes"
            # and "mass__wasting_ids"
                
            mw_events = np.zeros(np.hstack(self._grid.nodes).shape[0])
            mw_events_volumes = np.zeros(np.hstack(self._grid.nodes).shape[0])
            mw_event_ids = np.zeros(np.hstack(self._grid.nodes).shape[0])
    
            ivL = self.LS_df['vol [m^3]'][0::1].values # initial volume list
            innL = self.LS_df['cell'][0::1].values.astype('int') # initial node number list        
            
            #ls clump
            mw_events[innL] = 1
            mw_events_volumes[innL] = ivL 
            
            self._grid.at_node["mass__wasting_events"] = mw_events.astype(int)
            self._grid.at_node["mass__wasting_volumes"] = mw_events_volumes
            
            mw_id = 1
            for key in self.LSclump:
                
                nds = self.LSclump[key]['cell'].values.astype(int)
                mw_event_ids[nds] = mw_id
                # mw_id+=1
                
            self._grid.at_node["mass__wasting_id"] = mw_event_ids.astype(int) # need to check for clumps, update mass__wasting_id with zeros if no clumps
        else:
            self._grid.at_node["mass__wasting_id"] =  np.zeros(self._grid.number_of_nodes).astype(int)

    def run_one_step(self):
        """map hillslope scale landslides from the grid of landslide potential
        values.

        """
        self.mwprob = self.grid.at_node['mass__wasting_potential'] #  update mw probability variable
        self._extractLSCells()


        if self._this_timesteps_landslides.any():
            print('new landslides')
            # determine extent of landslides
            self._MassWastingExtent()
            print('mapped landslides')
        else:
            print('no cells meet mass wasting threshold')
