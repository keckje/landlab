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
    itL = 1000,
    opt1 = None,
    opt2 = False,
    opt3 = False,
    opt4 = False):
        
        super().__init__(grid)
        # self.grid = grid
        self.release_dict = release_dict
        self.df_dict = df_dict
        self.save = save_df_dem
        self.run_id = run_id
        self.itL = itL
        self.df_evo_maps = {}
        self.df_th_maps = {}
    
        # set run options
        self.opt1 = opt1
        self.opt2 = opt2
        self.opt3 = opt3
        self.opt4 = opt4
        
        
        # define initial topographic + mass wasting thickness topography
        self._grid.at_node['topographic__elevation_MW_surface'] = self._grid.at_node['topographic__elevation'].copy()
        self._grid.at_node['topographic__initial_elevation'] = self._grid.at_node['topographic__elevation'].copy()
    
    """route an initial mass wasting volume through a watershed, determine Scour, 
    Entrainment and Depostion depths and update the DEM
    
    
    Parameters
    ----------
    grid : landlab raster model grid
        raster model grid 

    release_dict : dictionary
        a dictionary of parameters that control the release of the mass wasting 
        material into the watershed. A pulse (fraction) of the total landslide volume
        is added at interval "nid" until the total number of pulses is equal to 
        the total landslide volume             
                mw_release_dict = {
                    'number of pulses': 2, 
                    'iteration delay': 2
                    }   
                
                where:
                    number of pulses: int
                        number of pulses that are used to release the total mass
                        wasting volume
                    iteration delay: int
                        number of iterations between each pulse
    
    df_dict : dictionary
        a dictionary of parameters that control the behavoir of the cellular-automata 
        debris flow model formatted as follows:
                df_dict = {
                    'critical slope':0.05, 'minimum flux':0.1,
                    'scour coefficient':0.03}
            
            where: 
                critical slope: float
                    angle of repose or friction coefficient of mass wasting material , L/L
                minimum-flux: float
                    minimum volumetric flux per unit contour width, i.e., 
                    volume/(grid.dx*grid.dx) per iteration, (L)/i.
                    flux below this threshold stops at the cell as a deposit
                scour coefficient: float [L/((M*L*T^-2)/L^2)]
                    coefficient that converts the depth-slope approximation of
                    total shear stress (kPa from the debris flow to a 
                    scour depth [m]
                        
    save : boolean
        Save topographic elevation of watershed after each model iteration? 
        The default is False.
    itL : int
        maximum number of iterations the cellular-automata model runs before it
        is forced to stop. The default is 1000.
    
    
    opt1: list
        list of length 1 for a basin uniform Sc value
        list of length 2 for hydraulic geometry defined Sc
        coefficient and exponent of a user defined function for mass wasting 
        crictical slope. e.g., [0.146,0.051] sets Sc<=0.1 at 
        contributing area > ~1100 m2
        
    opt2: boolean
        use topographic elevation + the thickness of the mass wasting material
        to route runout?
        default is False
    
    opt3: boolean
        If downslope cells are higher than mass wasting cell, use downslope 
        cell height to set minimum deposit depth?
        default is False
    
    opt4: boolean
         Allow settling after deposition during runout? Settling is applied each
         iteration of the model run after outflow from all cells have been computed
         default is False

    Returns
    -------
    None.
    
    """
    
    def run_one_step(self, dt):
        """route a list of mass wasting volumes through a DEM and update
        the DEM based on the scour, entrainment and depostion depths at each 
        cell        
        ----------
        Parameters
        dt : foat
            duration of storm, in seconds

        Returns
        -------
        None.

        """
        # set instance time stamp
        self.run_id = dt
            
        # convert map of mass wasting locations and volumes to lists
        mask = self._grid.at_node['mass__wasting_events'] == 1
        innL = np.hstack(self._grid.nodes)[mask]
        ivL = self._grid.at_node['mass__wasting_volumes'][mask]
                
        # release parameters for landslide
        nps = list(self.release_dict['number of pulses'])
        nid = list(self.release_dict['iteration delay'])
        
        # if nps is a single value (rather than list of values for each mass wasting event)
        if (len(nps) ==1) & (len(nps) < len(innL)):
            nps = np.ones(len(innL))*nps
        
        # if nid is a single value (rather than list of values for each mass wasting event)            
        if (len(nid) ==1) & (len(nid) < len(innL)):
            nid = np.ones(len(innL))*nid

        # material stops at cell when flux / cell width is below this,
        self.SD = self.df_dict['minimum flux']
        
        # entrainment coefficient
        self.cs = self.df_dict['scour coefficient']
                
        DFcells = {}            
        cL = {}
        # lists for tracking behavior of model
        self.enL = [] # entrainment depth / regolith depth
        self.dfdL = [] # incoming debris flow thickness (qsi)
        self.TdfL = [] #        
        #For each landslide in list:
        for i,inn in enumerate(innL):
        
            cL[i] = []
            self.df_evo_maps[i] = {}
        
            # set up initial landslide cell
            iv =ivL[i]/nps[i]# initial volume (total volume/number of pulses) 
            
            # initial receiving nodes (cells) from landslide
            rn = self._grid.at_node.dataset['flow__receiver_node'].values[inn]
            
            if type(rn) is not np.ndarray: # if only 1 cell, format int, change to np.array
                rn = np.array([rn])
            
            rni = rn[np.where(rn != -1)]
            
            # initial receiving proportions from landslide
            rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[inn]
            rp = rp[np.where(np.array(rp) > 0)]
        
            rvi = rp*iv # volume sent to each receving cell        
        
            # now loop through each receiving node, 
            # determine next set of recieving nodes
            # repeat until no more receiving nodes (material deposits)
            
            DFcells[inn] = []
            slpr = []
            c = 0
            c2= 0
            arn = rni 
            arv = rvi
            self.DEMdfD = {}
    
            while len(arn)>0 and c < self.itL:
              
                # release the initial landslide volume
                # if first iteration, receiving cell = initial receiving list
                # initial volume = volume/nps
                if c == 0: 
                    arn = rni; arv = rvi
                    c2+=1
                    
                # for following iterations, add initial volume/nps every nid iterations
                # until the volume has been added nps times
                elif nps[i]>1:
                    if ((c)%nid[i] == 0) & (c2<=nps[i]-1):
                        arn = np.concatenate((arn,rni))
                        arv = np.concatenate((arv,rvi))
                        # update pulse counter
                        c2+=1        
                
                arn_ns = np.array([])
                arv_ns = np.array([])
                # mwh = [] # list of debris flow thickness at each cell that has
                # debris flow for iteration c
                
                # for each unique cell in receiving node list arn
                arn_u = np.unique(arn) # unique arn list
                self.D_L = []

                # mass conintuity and next iteration precipitons
                detaL = []
                qsoL = []
                rnL = []
                for n in arn_u:
                    
                    n = int(n)        

                    deta, qso, rn = self._scour_entrain_deposit(n, arv, arn)
                    detaL.append(deta); qsoL.append(qso); rnL.append(rn)
                        
                    ## prepare receiving nodes for next iteration
                    
                    # material stops at node if flux / cell width is 0 OR node is a 
                    # boundary node
                    if qso>0 and n not in self._grid.boundary_nodes: 
                        
                        # receiving proportion of qso from cell n to each downslope cell
                        rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[n]
                        rp = rp[np.where(rp > 0)] # only downslope cells considered
                        
                        # receiving volume
                        vo = qso*self._grid.dx*self._grid.dy # convert qso to volume
                        rv = rp*vo
                   
                        # store receiving nodes and volumes in temporary arrays
                        arn_ns = np.concatenate((arn_ns,rn), axis = 0) # next step receiving node list
                        arv_ns = np.concatenate((arv_ns,rv), axis = 0) # next steip receiving node incoming volume list

                # update the dem
                for ii, n in enumerate(arn_u):
                    n = int(n)  
                    
                    # update dem
                    self._update_dem(n, ii, detaL, qsoL)
                


                if self.save:
                    cL[i].append(c)
                    self.df_evo_maps[i][c] = self._grid.at_node['topographic__elevation'].copy()
                    # save precipitions
                    DFcells[inn].append(arn)
        
                
                ## update slope fields after DEM has been updated
                # fd = FlowDirectorDINF(self._grid) # update slope including debris flow thickness                     
                fd = FlowDirectorMFD(self._grid, surface="topographic__elevation" ,diagonals=True,
                                partition_method = 'slope')
                fd.run_one_step()

                DEMf = self._grid.at_node['topographic__elevation'].copy()
                
                DEMdf_r = DEMf-self._grid.at_node['topographic__initial_elevation']
                       
                if self.opt4:

                    self.dif  = self._grid.at_node['topographic__elevation']-self._grid.at_node['topographic__initial_elevation']
                    
                    for ii, n in enumerate(arn_u):
                       
                        # deposit material settles
                        self._deposit_settles(ii, n)
                    
                if self.opt2:
     
                              
                    # update slope for next iteration using the mass wasting surface 
                    fd = FlowDirectorMFD(self._grid, surface="topographic__elevation_MW_surface", diagonals=True,
                            partition_method = 'slope')
                    fd.run_one_step()
                                        
                    # remove debris flow depth from cells this iteration (depth returns next iteration)
                    self._grid.at_node['topographic__elevation_MW_surface'][arn_u] = self._grid.at_node['topographic__elevation_MW_surface'][arn_u]-qsoL
                        
                DEMf = self._grid.at_node['topographic__elevation'].copy()
                
                DEMdf_rd = DEMf-self._grid.at_node['topographic__initial_elevation']
        
                self.DEMdfD[str(c)] = {'DEMdf_r':DEMdf_r.sum()*self._grid.dx*self._grid.dy,
                                  'DEMdf_rd':DEMdf_rd.sum()*self._grid.dx*self._grid.dy}     
              
                # once all cells in iteration have been evaluated, temporary receiving
                # node and node volume arrays become arrays for next iteration
                arn = arn_ns.astype(int)
                arn_u = np.unique(arn) # unique arn list
                arv = arv_ns                
        
                # update iteration counter
                c+=1
                
                if c%20 ==0:
                    print(c)            
            
        self.DFcells = DFcells


    def _scour_entrain_deposit(self, n, arv, arn):
        """ mass conservation at a grid cell: determines the erosion, deposition
        change in topographic elevation and flow out of a cell"""
        
        # get average elevation of downslope cells
        # receiving nodes (cells)
        rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
        rn = rn[np.where(rn != -1)]            
        # rn = rn[np.where(rp > th)]
           
        # slope at cell (use highest slope)
        slpn = self._grid.at_node['topographic__steepest_slope'][n].max()
        
        # incoming volume: sum of all upslope volume inputs
        vin = np.sum(arv[arn == n])
        
        # convert to flux/cell width
        qsi = vin/(self._grid.dx*self._grid.dx)
                            
        # additional constraint to control debris flow behavoir
        # if flux to a cell is below threshold, debris is forced to stop
        if qsi <=self.SD:
            D = qsi # all material that enters cell is deposited 
            qso = 0 # debris stops, so qso is 0
            E = 0 # no erosion
            # determine change in cell height
            deta = D # (deposition)/cell area
        else:
            ## deposition
            
            # determine deposition depth following Campforts, et al., 2020            
            # since using flux per unit contour width (rather than flux), 
            # L is (1-(slpn/slpc)**2) rather than dx/(1-(slpn/slpc)**2)         
            
            # critical slope
            if self.opt1:
                a = self.df_dict['critical slope'][0]
                b = self.df_dict['critical slope'][1]
                slpc = a*self._grid.at_node['drainage_area'][n]**b
            else:
                slpc = self.df_dict['critical slope'][0]
            
            Lnum = np.max([(1-(slpn/slpc)**2),0])
            
            if self.opt2:
                zo = self._grid.at_node['topographic__elevation'][rn].min()
                zi = self._grid.at_node['topographic__elevation'][n]
                if zi<zo and qsi>(zo-zi) and self.opt3:
                    # print('opt3')
                    # print(str(zo-zi+(qsi-(zo-zi))*Lnum)+' ,'+str(qsi*Lnum))
                    # D = min(zo-zi+(qsi-(zo-zi))*Lnum,qsi*Lnum)
                    D = zo-zi+(qsi-(zo-zi))*Lnum
                else:
                    D = qsi*Lnum # deposition depth
            else:
                D = qsi*Lnum
                
            ## erosion depth
            
            # debris flow depth over cell    
            df_depth = qsi #vin/(self._grid.dx*self._grid.dx) #df depth
            
            # depth-slope product approximation of total shear stress on channel bed [kPa]
            T_df = 1700*9.81*df_depth*slpn/1000
     
            # max erosion depth equals regolith (soil) thickness
            dmx = self._grid.at_node['soil__thickness'][n]
            
            # erosion depth: 
            E = min(dmx, self.cs*T_df)
            
            ## flow out
            qso = qsi-D+E
            
            ## change in cell elevation
            deta = D-E 

            # model behavior tracking
            self.enL.append(E/dmx)
            
            self.dfdL.append(df_depth)
            
            self.TdfL.append(T_df)
            
            
        # list of deposition depths at cells in iteration 
        self.D_L.append(D)            
            
        return deta, qso, rn
                

    def _update_dem(self,n, ii, detaL, qsoL):
        """updates the topographic elevation and soil thickness fields at a
        grid cell"""
        
        deta = detaL[ii]; qso = qsoL[ii]; #mwh = qso
        
        # Regolith - difference between the fresh bedrock surface and the top surface of the dem
        self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]+deta 
    
        # update raster model grid regolith thickness and dem
        if self.opt2:
            
            # topographic elevation - does not include thickness of moving debris flow
            self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta                    
            # keep list of debris flow depth
                                   
            # Topographic elevation MW surface - top surface of the dem + moving mass wasting material thickness
            self._grid.at_node['topographic__elevation_MW_surface'][n] = self._grid.at_node['topographic__elevation'].copy()[n]+qso
            
            # mass wasting depth list, removed after slope determined
            # mwh.append(qso)
        else:
  
            # Topographic elevation - top surface of the dem
            self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta
        
        # return mwh


    def _deposit_settles(self, ii, n):
        """ for each unique node in receiving node list, after entrainment, deposition 
        and change in node elevation have been determined, check that the height of the node 
        is not greater than permitted by angle of repose/friction angle as evaluated from 
        the lowest cell
        """
        if self.D_L[ii] >0:
            rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
            # slope to all receiving cells
            slpn = self._grid.at_node['topographic__steepest_slope'][n]
            
            # only consider downslope cells
            slpn = slpn[np.where(rn != -1)] 
            rn = rn[np.where(rn != -1)]  
                     
            # critical slope
            if self.opt1:
                a = self.df_dict['critical slope'][0]
                b = self.df_dict['critical slope'][1]
                slpc = a*self._grid.at_node['drainage_area'][n]**b
            else:
                slpc = self.df_dict['critical slope'][0]
            
            # only consider all cells that slope > Sc
            rn = rn[slpn>slpc]
            slpn = slpn[slpn>slpc]            

            rndif = self.dif[rn]
            slpn = slpn[abs(rndif)>0]  
            rn = rn[abs(rndif)>0]

            # if slope to downlsope nodes > Sc, adjust elevation of node n
            if len(rn)>=1:
                
                # destribute material to downslope nodesbased on weighted
                # average slope (same as multiflow direciton proportions, 
                # but here only determined for downslope nodes in which
                # S  > Sc )
                         
                sslp = sum(slpn)
                pp = slpn/sslp
                
                # determine the total flux / unit width sent to S > Sc downslope cells
                # mean downslope cell elevation
                zo = self._grid.at_node['topographic__elevation'][rn].min()
                
                # node n elevation
                zi = self._grid.at_node['topographic__elevation'][n]
                
                # height of node n using Sc*dx above mean downslope elevation
                slp_h = slpc*self._grid.dx             
            
                qso_s = (zi - (zo+slp_h))/2 # out going sediment depth
                
                if qso_s < 0: # no negative outflow
                    qso_s = 0
                
                if qso_s > self.D_L[ii]: # settlement out can not exceed deposit
                    qso_s= self.D_L[ii]
                
                qso_s_i = qso_s*pp # proportion sent to each receiving cell                
                
                # update the topographic elevation
                self._grid.at_node['topographic__elevation'][n]=self._grid.at_node['topographic__elevation'][n]-qso_s
                self._grid.at_node['topographic__elevation'][rn]=self._grid.at_node['topographic__elevation'][rn]+qso_s_i
                
                # update the soil thickness
                self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]-qso_s
                self._grid.at_node['soil__thickness'][rn] = self._grid.at_node['soil__thickness'][rn]+qso_s_i



        