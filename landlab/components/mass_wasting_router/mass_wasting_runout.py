# -*- coding: utf-8 -*-

import numpy as np
from landlab import RasterModelGrid
from landlab import Component, FieldError
from landlab.components import FlowAccumulator
from landlab.components import(FlowDirectorD8, 
                                FlowDirectorDINF, 
                                FlowDirectorMFD, 
                                FlowDirectorSteepest)



class MassWastingRunout(Component):
    
    '''a cellular automata mass wasting model that routes an initial mass wasting  
    volume (e.g., a landslide) through a watershed, determines Scour, Entrainment
    and Depostion depths and updates the DEM. 
        
    author: Jeff Keck
    '''
    
    
    _name = 'MassWastingRunout'
    
    _unit_agnostic = False
    
    _info = {
        
        'mass__wasting_events': {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "1 indicates mass wasting event, 0 is no event",
            },
        
        'mass__wasting_volumes': {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "initial mass wasting volumes",
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
        
        'flow__receiver_proportions': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of proportion of flow sent to each receiver.",
            },
        
        'topographic__steepest_slope': {
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
    release_dict,
    df_dict,
    save_df_dem = False, 
    run_id = 0, 
    itL = 1000):
        
    
        super().__init__(grid)
        # self.grid = grid
        self.release_dict = release_dict
        self.df_dict = df_dict
        self.save = save_df_dem
        self.run_id = run_id
        self.itL = itL
        self.df_evo_maps = {}
    
    

    """route an initial mass wasting volume through a watershed, determine Scour, 
    Entrainment and Depostion depths and update the dem
    
    
    Parameters
    ----------
    mg : landlab raster model grid
        raster model grid
    ivL : list of floats
        list of initial landslide volumes
    innL : list of int
        list of nodes where landslide volumes are released on the dem. 
        len(innL) = len(ivL)
    
    
    
    df_dict : dictionary
        a dictionary of parameters that control 
        the behavoir of the cellular-automata debris flow model formatted as follows:
                df_dict = {
                    'critical slope':0.07, 'minimum flux':0.3,
                    'scour coefficient':0.02}
            
            where: 
                critical slope: float
                    angle of repose of mass wasting material , L/L
                minimum-flux: float
                    minimum volumetric flux, i.e., volume/grid.dx per iteration, (L^2)/T.
                    flux below this threshold stops at the cell as a deposit
                scour coefficient: float
                    coefficient that converts the depth-slope approximation of
                    total shear stress from the debris flow to a entrainment and
                    scour depth
                    
                    
    
    
    release_dict : dictionary
        a dictionary of parameters that control the release of the mass wasting 
        material into the watershed            
                mw_release_dict = {
                    'number of pulses': 8, 
                    'iteration delay': 5
                    }   
                
               where:
                   number of pulses: int
                       number of pulses that are used to release the total mass
                       wasting volume
                   iteration delay: int
                       number of iterations between each pulse
                    
                

    
    save_df_dem : boolean
        Save topographic elevation after each model iteration?. This could creat up to 
        itterL number of maps for each debris flow. The default is False.
    itL : int
        maximum number of iterations the cellular-automata model runs before it
        is forced to stop. The default is 1000.

    
    Returns
    -------
    None.

    """
    


    def _scour_entrain_deposit(self):
        
        # convert map of mass wasting locations and volumes to lists
        mask = self._grid.at_node['mass__wasting_events'] == 1
        innL = np.hstack(self._grid.nodes)[mask]
        ivL = self._grid.at_node['mass__wasting_volumes'][mask]
        
        
        # release parameters for landslide
        nps = self.release_dict['number of pulses']
        nid = self.release_dict['iteration delay']    
        
        # critical slope at which debris flow stops
        # higher reduces spread and runout distance
        slpc = self.df_dict['critical slope']
        
        # forced stop volume threshold
        # material stops at cell when volume is below this,
        # higher increases spread and runout distance
        SV = self.df_dict['minimum flux']
    
        # entrainment coefficient
        # very sensitive to entrainment coefficient
        # higher causes higher entrainment rate, longer runout, larger spread
        cs = self.df_dict['scour coefficient']
    
    
        DFcells = {}            
        cL = {}
        
        # ivL and innL can be a list of values. For each value in list:
        for i,inn in enumerate(innL):
            # print(i)
            cL[i] = []
            self.df_evo_maps[i] = {}
    
    
            # set up initial landslide cell
            iv =ivL[i]/nps[i]# initial volume (total volume/number of pulses) 
            print(iv)
            
            # initial receiving nodes (cells) from landslide
            rn = self._grid.at_node.dataset['flow__receiver_node'].values[inn]
            rni = rn[np.where(rn != -1)]
            
            # initial receiving proportions from landslide
            rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[inn]
            rp = rp[np.where(rp > 0)]
            print(rp)
            rvi = rp*iv # volume sent to each receving cell
            
        
            # now loop through each receiving node, 
            # determine next set of recieving nodes
            # repeat until no more receiving nodes (material deposits)
            
            DFcells[inn] = []
            slpr = []
            c = 0
            c2=0
            arn = rni
            arv = rvi
            enL = []
            
            while len(arn)>0 and c < self.itL:
        
                # add pulse (fraction) of total landslide volume
                # at interval "nid" until total number of pulses is equal to total
                # for landslide volume (nps*nid)
        
                if ((c+1)%nid[i] ==0) & (c2<=nps[i]*nid[i]):
                    arn = np.concatenate((arn,rni))
                    arv = np.concatenate((arv,rvi))
                    
                arn_ns = np.array([])
                arv_ns = np.array([])
                df_dem = {} # dict to save array of node value for each iteration of flow
                
                # for each unique cell in receiving node list arn
                for n in np.unique(arn):
                    n = int(n)
                    
                    # slope at cell (use highest slope)
                    slpn = self._grid.at_node['topographic__steepest_slope'][n].max()
                    
                    # incoming volume: sum of all upslope volume inputs
                    vin = np.sum(arv[arn == n])
                    
                    # determine deposition volume following Campforts, et al., 2020            
                    # since using volume (rather than volume/dx), L is (1-(slpn/slpc)**2) 
                    # rather than mg.dx/(1-(slpn/slpc)**2)           
                    Lnum = np.max([(1-(slpn/slpc)**2),0])
                    
                    dpd = vin*Lnum # deposition volume
                    
                    # determine erosion depth
                    
                    # debris flow depth over cell    
                    df_depth = vin/(self._grid.dx*self._grid.dx) #df depth
                    
                    #determine erosion volume (function of slope)                   
                    # depth-slope product approximation of total shear stress on channel bed [kPa]
                    T_df = 1700*9.81*df_depth*slpn/1000
         
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
                    if vin <=SV:
                        dpd = vin # all volume that enter cell is deposited 
                        vo = 0 # debris stops, so volume out is 0
        
                        # determine change in cell height
                        deta = (dpd)/(self._grid.dx*self._grid.dy) # (deposition)/cell area
        
                    else:
                        vo = vin-dpd+ev # vol out = vol in - vol deposited + vol eroded
                        
                        # determine change in cell height
                        deta = (dpd-ev)/(self._grid.dx*self._grid.dy) # (deposition-erosion)/cell area
        
                    
                    # update raster model grid regolith thickness and dem
                    
                    
                    # if deta larger than regolith thickness, deta equals regolith thickness (fresh bedrock is not eroded)
                    if self._grid.at_node['soil__thickness'][n]+deta <0:
                        deta = - self._grid.at_node['soil__thickness'][n]           
                    
                    # Regolith - difference between the fresh bedrock surface and the top surface of the dem
                    self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]+deta 
        
                    # Topographic elevation - top surface of the dem
                    self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta
                    # print(mg.at_node['topographic__elevation'][n])
            
            
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
                
                
                if self.save:
                    cL[i].append(c)
                    
                    # save DEM 
                    # _ = self._grid.add_field( 'topographic__elevation_run_id_'+str(self.run_id)+str(i)+'_'+str(c),
                    #                 self._grid.at_node['topographic__elevation'].copy(),
                    #                 at='node')
                    
                    self.df_evo_maps[i][c] = self._grid.at_node['topographic__elevation'].copy()
                    # save precipitions
                    DFcells[inn].append(arn.astype(int))
    
                # update iteration counter
                c+=1
                
                # update pulse counter
                c2+=nid[i]
                
                if c%20 ==0:
                    print(c)            
            
        self.DFcells = DFcells



    def run_one_step(self, dt):
        """route the initial mass wasting volumes through the dem and update
        the dem based on the scour, entrainment and depostion depths at each 
        cell
        

        Parameters
        ----------
        dt : foat
            duration of storm, in seconds

        Returns
        -------
        None.

        """
        
        self.run_id = dt
        self._scour_entrain_deposit()
        