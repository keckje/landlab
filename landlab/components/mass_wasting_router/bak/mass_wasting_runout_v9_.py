# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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
        
        'mass__wasting_id': {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "interger or float id of each mass wasting area is assigned \
                to all nodes representing the mass wasting area."
            },
        
        
        'topographic__elevation': {            
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "elevation of the ground surface",
            },
        
        'soil__thickness': {            
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
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
        'particle__diameter': {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "representative particle diameter at each node, this might \
            vary with underlying geology, contributing area or field observations"
            ,
            
            },

        }
    
    
    
    
    def __init__(
    self,
    grid,
    release_dict,
    mw_dict,
    save = False, 
    run_id = 0, 
    itL = 1000,
    routing_surface = "energy__elevation",
    settle_deposit = False):
        
        super().__init__(grid)
        # self.grid = grid
        self.release_dict = release_dict
        self.mw_dict = mw_dict
        self.save = save
        self.run_id = run_id
        self.itL = itL
        self.partition_method = "slope"#'square_root_of_slope' 
        self.routing_surface = routing_surface
        self.settle_deposit = settle_deposit
        self.VaryDp = self._grid.has_field('node', 'particle__diameter')
        
        if self.VaryDp:
            print(' running with spatially variable Dp ')

        # release parameters for landslide
        self.nps = list(self.release_dict['number of pulses'])
        self.nid = list(self.release_dict['iteration delay'])
       
        
        # set mass wasting parameters
        
        # material stops at cell when flux / cell width is below this,
        
        if len(self.mw_dict['critical slope'])>1:
            self.a = self.mw_dict['critical slope'][0]
            self.b = self.mw_dict['critical slope'][1]
        else:
            self.slpc = self.mw_dict['critical slope'][0]
        
        self.SD = self.mw_dict['minimum flux']
        
        self.cs = self.mw_dict['scour coefficient']
        
        if 'scour exponent' in self.mw_dict:
            self.eta = self.mw_dict['scour exponent']
        else:
            self.eta = 0.2
        
        if 'effective particle diameter' in self.mw_dict:
            self.Dp = self.mw_dict['effective particle diameter']
        else:
            self.Dp = 0.25   
            
        if 'vol solids concentration' in self.mw_dict:
            self.vs = self.mw_dict['vol solids concentration']
        else:
            self.vs = 0.6
        
        if 'density solids' in self.mw_dict:
            self.ros = self.mw_dict['density solids']
        else:
            self.ros = 2650
            
        if 'density fluid' in self.mw_dict:
            self.rof = self.mw_dict['density fluid']
        else:
            self.rof = 1000
        
        if 'gravity' in self.mw_dict:
            self.g = self.mw_dict['gravity'] 
        else:
            self.g = 9.81
        
        # density of debris flow mixture
        self.rodf = self.vs*self.ros+(1-self.vs)*self.rof
        
        
        # define initial topographic + mass wasting thickness topography
        self._grid.at_node['energy__elevation'] = self._grid.at_node['topographic__elevation'].copy()
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
    
    mw_dict : dictionary
        a dictionary of parameters that control the behavoir of the cellular-automata 
        debris flow model formatted as follows:
                mw_dict = {
                    'critical slope':0.05, 'minimum flux':0.1,
                    'scour coefficient':0.03}
            
            where: 
                critical slope: list of floats
                    critical slope (angle of repose if no cohesion) of mass wasting material , L/L
                            list of length 1 for a basin uniform Sc value
                            list of length 2 for hydraulic geometry defined Sc
                            coefficient and exponent of a user defined function for mass wasting 
                            crictical slope. e.g., [0.146,0.051] sets Sc<=0.1 at 
                            contributing area > ~1100 m2
                                        
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
    
  
    routing_surface: str
        "energy__elevation" to use the potential energy surface of the landscape to route
        "topographic__elevation" use the ground surface of the landscape to route

    
    
    settle_deposit: boolean
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
                   
        # mass wasting cells coded according to id
        mask = self._grid.at_node['mass__wasting_id'] > 0

        # list of unique mw_id
        self.mw_ids = np.unique(self._grid.at_node['mass__wasting_id'][mask])
        
        # innL is list of lists of nodes in each landslide
        
        innL = []
        for mw_id in self.mw_ids:
            ls_mask = self._grid.at_node['mass__wasting_id'] == mw_id
            # list of lists, node ids in each landslide
            innL.append(np.hstack(self._grid.nodes)[ls_mask])
        
        # if nps is a single value (rather than list of values for each mass wasting event)
        # this still may work, can be set to 1 pulse, no delay for landslide area failures
        if (len(self.nps) ==1) & (len(self.nps) < len(innL)):
            self.nps = np.ones(len(innL))*self.nps
        
        # if nid is a single value (rather than list of values for each mass wasting event)            
        if (len(self.nid) ==1) & (len(self.nid) < len(innL)):
            self.nid = np.ones(len(innL))*self.nid
        
        
        # data containers for saving model images and behavior statistics       
        cL = {} 
        self.df_evo_maps = {} # copy of dem after each routing iteration
        self.enL = [] # entrainment depth / regolith depth
        self.dfdL = [] # incoming debris flow thickness (qsi)
        self.TdfL = [] # basal shear stress       
        
        # data containers for tests
        self.pd_r = {}
        self.st_r = {}
        self.tss_r ={}
        self.frn_r = {}
        self.te_r = {}
        self.arv_r = {}
        self.arn_r = {}
        self.arpd_r = {}
        
        # For each mass wasting event in list:
        for mw_i,inn in enumerate(innL):
                         
            mw_id = self.mw_ids[mw_i]

            # prep data containers
            cL[mw_i] = []
            self.df_evo_maps[mw_i] = {}
            self.DEMdfD = {}

            # prep data containers for tests
            self.pd_r[mw_id] = []
            self.st_r[mw_id] = []
            self.tss_r[mw_id] = []
            self.frn_r[mw_id] = []
            self.te_r[mw_id] = []            
            self.arv_r[mw_id] = []
            self.arn_r[mw_id] = []
            self.arpd_r[mw_id] = []     

            
            # prepare initial mass wasting material (precipitons) for release 
            self._prep_initial_mass_wasting_material(inn, mw_i)
        
            # now loop through each receiving node in list rni, 
            # determine next set of recieving nodes
            # repeat until no more receiving nodes (material deposits)
                                        
            c = 0 # model iteration counter
            c_dr = 0 # mass wasting event delayed-release counter
              
            while len(self.arn)>0 and c < self.itL:
                # release the initial landslide volume
                # if first iteration, receiving cell = initial receiving list
                # initial volume = volume/nps
                if c == 0: 
                    c_dr+=1
                    
                # for following iterations, add initial volume/nps every nid iterations
                # until the volume has been added nps times
                elif self.nps[mw_i]>1:
                    if ((c)%self.nid[mw_i] == 0) & (c_dr<=self.nps[mw_i]-1):
                        self.arn = np.concatenate((self.arn,rni))
                        self.arv = np.concatenate((self.arv,rvi))
                        self.arpd = np.concatenate((self.arpd,rpdi))
                        # update pulse counter
                        c_dr+=1        
                
                # receiving node, volume and particle diameter temporary arrays
                # that become the arrays for the next model step (iteration)
                self.arn_ns = np.array([])
                self.arv_ns = np.array([])
                self.arpd_ns = np.array([])
                
                # for each unique cell in receiving node list self.arn
                arn_u = np.unique(self.arn).astype(int)  # unique arn list
                
                # determine the incoming volume and depth to each node in arn_u
                self.vqdat = self.vin_qsi(arn_u) ##
                       
                if self.routing_surface == "energy__elevation":
                    # update slope fields using the dem surface for scour,
                    # erosion and qso computations
                    self._update_topographic_slope()

                # determine scour, entrain, deposition and outflow depths and
                # the outflowing particle diameter, arranged in array nudat
                self.nudat = self._scour_entrain_deposit_updatePD()
                
                # update grid field: topographic__elevation with the values in 
                # nudat
                self._update_dem()

                ### update grid field: particle__diameter with the values in 
                # nudat
                if self.VaryDp:
                    self._update_channel_particle_diameter()
                
                ### update slope fields after DEM has been updated for settlement
                self._update_topographic_slope()                   
                       
                if self.settle_deposit:
                    self.dif  = self._grid.at_node['topographic__elevation']-self._grid.at_node['topographic__initial_elevation']
                    ### settle unrealistically tall mounds in the deposit                    
                    self._settle(arn_u)
                    
                if self.routing_surface == "energy__elevation":
                    # update the energy surface dem
                    self._update_E_dem() ##                                 
                    # update slope for routing using energy slope
                    self._update_energy_slope()
                else:
                    # update slope using routing using topographic slope
                    self._update_topographic_slope()                                    

                # once all cells in iteration have been evaluated, temporary receiving
                # node, node volume and node particle diameter arrays become arrays 
                # for next iteration
                self.arn = self.arn_ns.astype(int)
                self.arv = self.arv_ns #   
                self.arpd = self.arpd_ns
        

                if self.save:

                    cL[mw_i].append(c)
                    
                    DEMf = self._grid.at_node['topographic__elevation'].copy()
                    
                    DEMdf_r = DEMf-self._grid.at_node['topographic__initial_elevation']
            
                    self.DEMdfD[str(c)] = {'DEMdf_r':DEMdf_r.sum()*self._grid.dx*self._grid.dy}     
                    
                    ### save maps for video
                    self.df_evo_maps[mw_i][c] = self._grid.at_node['topographic__elevation'].copy()

                    # data for tests
                    self.pd_r[mw_id].append(self._grid.at_node['particle__diameter'].copy())
                    self.st_r[mw_id].append(self._grid.at_node['soil__thickness'].copy())
                    self.tss_r[mw_id].append(self._grid.at_node['topographic__steepest_slope'].copy())
                    self.frn_r[mw_id].append(self._grid.at_node['flow__receiver_node'].copy())
                    self.te_r[mw_id].append(self._grid.at_node['topographic__elevation'].copy())
                    self.arv_r[mw_id].append(self.arv)
                    self.arn_r[mw_id].append(self.arn)
                    self.arpd_r[mw_id].append(self.arpd)     


                # update iteration counter
                c+=1
                
                if c%20 ==0:
                    print(c)  


    def _prep_initial_mass_wasting_material(self, inn, mw_i):
        """from an initial source area (landslide) prepare the initial lists 
        of receiving nodes and incoming volumes and particle diameters per precipiton,
        remove the source material from the topographic elevation dem"""
        
        # lists of initial recieving node, volume and particle diameter
        rni = np.array([])
        rvi = np.array([])
        rpdi = np.array([])
        
        # order lowest to highest
        node_z = self._grid.at_node.dataset['topographic__elevation'][inn]
        zdf = pd.DataFrame({'nodes':inn,'z':node_z})
        zdf = zdf.sort_values('z')            
        
        for ci, ni in enumerate(zdf['nodes'].values):
             
            # get receiving nodes of node ni in mw index mw_i
            # print(ni)
            rn = self._grid.at_node.dataset['flow__receiver_node'].values[ni]
            rn = rn[np.where(rn != -1)]
            
            # receiving proportion of qso from cell n to each downslope cell
            rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[ni]
            rp = rp[np.where(rp > 0)] # only downslope cells considered
            
            # get volume (out) of node ni  
            vo = self._grid.at_node.dataset['soil__thickness'].values[ni]*self._grid.dx*self._grid.dy/self.nps[mw_i]# initial volume (total volume/number of pulses)   
            # divide into proportion going to each receiving node
            rv = rp*vo
            
            if self.VaryDp:
                # get initial mass wasting particle diameter (out) of node ni
                pd_out = self._grid.at_node.dataset['particle__diameter'].values[ni]
            else:
                pd_out = self.Dp
            # particle diameter to each recieving node
            rpd = np.ones(len(rv))*pd_out
       
            # append receiving node ids, volumes and particle diameters to initial lists
            rni = np.concatenate((rni,rn), axis = 0) 
            rvi = np.concatenate((rvi,rv), axis = 0) 
            rpdi = np.concatenate((rpdi,rpd), axis = 0) 
            
            # remove material from node
            self._grid.at_node.dataset['topographic__elevation'][ni] =  (                   
            self._grid.at_node.dataset['topographic__elevation'][ni] - 
            self._grid.at_node['soil__thickness'][ni])
            
            self._grid.at_node['soil__thickness'][ni] = 0
            
            
            # update slope to reflect node material removed from dem
            self._update_topographic_slope()

        self.arn = rni
        self.arv = rvi
        self.arpd = rpdi


    def vin_qsi(self,arn_u):
        """determine volume and depth of incoming material"""

        def VQ(n):           
            # total incoming volume
            vin = np.sum(self.arv[self.arn == n])
            # convert to flux/cell width
            qsi = vin/(self._grid.dx*self._grid.dx)
            return vin, qsi

        ll=np.array([VQ(n) for n in arn_u], dtype=object)     
        arn_ur = np.reshape(arn_u,(-1,1))
        vqdat = np.concatenate((arn_ur,ll),axis=1)
    
        return vqdat
        
    
    def _scour_entrain_deposit_updatePD(self):
        """ mass conservation at a grid cell: determines the erosion, deposition
        change in topographic elevation and flow out of a cell as a function of
        debris flow friction angle, particle diameter and the underlying DEM
        slope"""
        
        # list of deposition depths, used in settlement algorithm
        self.D_L = []
        def SEDU(vq_r):
            """function for iteratively determing scour, entrainment and
            deposition depths using node id to look up incoming flux and
            downslope nodes and slopes"""            

            n = vq_r[0]; vin = vq_r[1]; qsi = vq_r[2]
            
            # get average elevation of downslope cells
            # receiving nodes (cells)
            rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
            rn = rn[np.where(rn != -1)]            
               
            # slope at cell (use highest slope)
            slpn = self._grid.at_node['topographic__steepest_slope'][n].max()
            
            
            if len(self.mw_dict['critical slope'])>1: # if option 1, critical slope is not constant but depends on location
                self.slpc = self.a*self._grid.at_node['drainage_area'][n]**self.b
                        
            # incoming particle diameter (weighted average)
            pd_in = self._particle_diameter_in(n,vin) # move
                            
            # additional constraint to control debris flow behavoir
            # if flux to a cell is below threshold, debris is forced to stop
            if qsi <=self.SD:
                D = qsi # all material that enters cell is deposited 
                qso = 0 # debris stops, so qso is 0
                E = 0 # no erosion
                # determine change in cell height
                deta = D # (deposition)/cell area
                # node grain size remains the same                
            else:

                ### scour
                if self.VaryDp:   
                    opt = 2
                else:
                    opt = 1
                    
                E, pd_up = self._scour(n, qsi, slpn, opt = opt, pd_in = pd_in)                
                ###

                ### deposition
                # note: included entrained material
                D = self._deposit(qsi+E, slpn, rn, n)
                ###

                # entrainment is mass conservation at cell
                ## flow out
                qso = qsi-D+E
                
                # small qso are considered zero
                qso  = np.round(qso,decimals = 8)
                # if n == 522:
                #     print("qso-----{}".format(qsi))                              
                ## change in node elevation
                deta = D-E 
                              
                # material stops at node if flux / cell width is 0 OR node is a 
                # boundary node
                
                if qso>0 and n not in self._grid.boundary_nodes: 
                    
                    # receiving proportion of qso from cell n to each downslope cell
                    rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[n]
                    rp = rp[np.where(rp > 0)] # only downslope cells considered
                    
                    # receiving volume
                    vo = qso*self._grid.dx*self._grid.dy # convert qso to volume
                    rv = rp*vo
               
                    # particle diameter out (weighted average)
                    pd_out = self._particle_diameter_out(pd_up,pd_in,qsi,E,D)    
                    rpd = np.ones(len(rv))*pd_out
               
                    # store receiving nodes and volumes in temporary arrays
                    self.arn_ns = np.concatenate((self.arn_ns,rn), axis = 0) # next step receiving node list
                    self.arv_ns = np.concatenate((self.arv_ns,rv), axis = 0) # next step receiving node incoming volume list
                    self.arpd_ns = np.concatenate((self.arpd_ns,rpd), axis = 0) # next step receiving node incoming particle diameter list
     
                
                # model behavior tracking
                self.enL.append(E)
                
                self.dfdL.append(qsi)


            # updated node particle diameter (weighted average)
            n_pd = self._particle_diameter_node(n,pd_in,E,D)
            
            # list of deposition depths at cells in iteration 
            self.D_L.append(D)            
                
            return deta, qso, rn, n_pd
    
        
        # apply SEDU function to all unique nodes in arn (arn_u)
        # create nudat, an np.array of data for updating fields at each node
        # that can be applied using vector operations
        ll=np.array([SEDU(r) for r in self.vqdat],dtype=object)     
        arn_ur = np.reshape(self.vqdat[:,0],(-1,1))
        nudat = np.concatenate((arn_ur,ll),axis=1)
        # print(nudat)
        return nudat

    
    def _update_E_dem(self):
        """update energy__elevation"""
    
        n = self.vqdat[:,0].astype(int); qsi = self.vqdat[:,2]; 
        # Energy dem
        self._grid.at_node['energy__elevation'][n] = self._grid.at_node['topographic__elevation'].copy()[n]+qsi        


    def _update_energy_slope(self):
        """updates the topographic__slope and flow directions fields using the 
        energy__elevation field"""
        fd = FlowDirectorMFD(self._grid, surface="energy__elevation", diagonals=True,
                partition_method = self.partition_method)
        fd.run_one_step()

        
    def _update_dem(self):
        """updates the topographic elevation of the landscape dem and soil 
        thickness fields"""
               
        n = self.nudat[:,0].astype(int); deta = self.nudat[:,1]

        self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]+deta   
        # topographic dem
        self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta


    def _update_topographic_slope(self):
        """updates the topographic__slope and flow directions fields using the 
        topographic__elevation field"""
        fd = FlowDirectorMFD(self._grid, surface="topographic__elevation", diagonals=True,
                partition_method = self.partition_method)
        fd.run_one_step()

    
    def _update_channel_particle_diameter(self):
        """ for each unique node in receiving node list, update the grain size
        using the grain size determined in _scour_entrain_deposit
        """

        n = self.nudat[:,0].astype(int); new_node_pd = self.nudat[:,4]
        if np.isnan(np.sum(new_node_pd)):
            raise ValueError("particle diameter is {}".format(new_node_pd))
        
            
        self._grid.at_node['particle__diameter'][n] = new_node_pd 
                
         
    def _settle(self, arn_u):
        """ for each unique node in receiving node list, after entrainment, deposition 
        and change in node elevation have been determined, check that the height of the node 
        is not greater than permitted by angle of repose/critical slope as evaluated from 
        the lowest cell. Note slope is not updated in this function so that it can be applied
        simultaneously to all nodes that settle during the iteration. 
        """
        for ii, n in enumerate(arn_u): # for each node in the list, use the slope field, computed from the previous iteration, to compute settlment and settlment direction to adjacent cells
            if self.D_L[ii] >0:
                rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
                # slope to all receiving cells
                slpn = self._grid.at_node['topographic__steepest_slope'][n]
                
                # only consider downslope cells
                slpn = slpn[np.where(rn != -1)] 
                rn = rn[np.where(rn != -1)]  
                         
                # critical slope
                if len(self.mw_dict['critical slope'])>1:
                    self.slpc = self.a*self._grid.at_node['drainage_area'][n]**self.b

                
                # only consider all cells that slope > Sc
                rn = rn[slpn>self.slpc]
                slpn = slpn[slpn>self.slpc]            
    
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
                    slp_h = self.slpc*self._grid.dx             
                
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
    

    def _scour(self, n, depth, slope, opt = 1, pd_in = None):
        """determines the scour depth based on user selected method
        check for required inputs at beginning of class
        """
            

        # depth-slope product approximation of hydrostaic/quasi-static 
        # shear stress on channel bed [Pa]
        theta = np.arctan(slope) # convert tan(theta) to theta
        Dp = pd_in # mass wasting particle diameter is Dp
        opt = 1
        if opt ==1:
            # following Frank et al., 2015, approximate erosion depth as a linear
            # function of total stress under uniform flow conditions
            
            # erosion depth,:
            Tbs = self.rodf*self.g*depth*np.sin(theta)
            Ec = self.cs*(Tbs)**self.eta
        
        if opt == 2:
            # shear stress apprixmated as a power functino of inertial shear stress
            phi = np.arctan(self.slpc) # approximate phi [radians] from criticle slope [L/L]
            
            # inertial stresses
            us = (self.g*depth*slope)**0.5
            u = us*5.5*np.log10(depth/Dp)
            
            dudz = u/depth
            Tcn = np.cos(theta)*self.vs*self.ros*(Dp**2)*(dudz**2)
            Tbs = Tcn*np.tan(phi)

            Ec = (self.cs*Tbs**self.eta)        
       
        dmx = self._grid.at_node['soil__thickness'][n]
        
        # particle size of scoured material
        if opt == 2:
            pd_up = self._grid.at_node['particle__diameter'][n]
        else:
            pd_up = self.Dp
        
        E = min(dmx, Ec) # convert Tb to kPa

        # model behavior tracking
        self.TdfL.append(Tbs)
        
        return(E,pd_up)

                
    def _deposit(self,qsi,slpn,rn,n):
        """determine deposition depth following Campforts, et al., 2020            
        since using flux per unit contour width (rather than flux), 
        L is (1-(slpn/slpc)**2) rather than dx/(1-(slpn/slpc)**2) but limited
        by the debris flow material critical slope (angle of repose)"""
                
        Lnum = np.max([(1-(slpn/self.slpc)**2),0])
        
        DL = qsi*Lnum
        
        if self.routing_surface == "energy__elevation":
            
            slp_h = self.slpc*self._grid.dx
            
            
            adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
            self._grid.diagonal_adjacent_nodes_at_node[n]))
                            
            # elevation at node i
            zi = self._grid.at_node['topographic__elevation'][n]
            
            # energy at node i
            ei = qsi+zi
            
            # nodes below energy surface
            rn_e = adj_n[self._grid.at_node['energy__elevation'][adj_n]<ei]
                      
            
            zo = self._grid.at_node['energy__elevation'][rn_e].mean()
            
            
            if zo>zi:            

                Dc = min(0.5*qsi+(zo-zi+slp_h)/2,qsi)
            
            elif (zo<=zi) and ((zi-zo)<slp_h):
                
                Dc = min(0.5*qsi+(-zi+zo+slp_h)/2,qsi)
            else:
                
                Dc = DL
            
            if Dc <0:
                print("n {}, qsi{}, ei {}, DL {}, Dc {}".format(n,qsi,ei,DL,Dc))
                raise(ValueError)

            D = min(DL,Dc)
                
        else:
            D = DL
        
        return(D)
    
    
    def _particle_diameter_in(self,n,vin):
        """determine the weighted average particle diameter of the incoming
        flow"""       
        if (vin == 0):
            pd_in = 0
        elif (np.isnan(vin)) or (np.isinf(vin)):
            msg = "in-flowing volume is nan or inf"
            raise ValueError(msg)
        else:           
            pd_in = np.sum((self.arpd[self.arn == n])*(self.arv[self.arn == n])/vin)        
        return pd_in


    def _particle_diameter_node(self,n,pd_in,E,D):
        """determine the weighted average particle diameter of deposited +
        in-situ deposit"""

        if (D+self._grid.at_node['soil__thickness'][n]-E > 0):
            n_pd = (self._grid.at_node['particle__diameter'][n]* 
            (self._grid.at_node['soil__thickness'][n]-E)+ 
            pd_in*D)/(D+self._grid.at_node['soil__thickness'][n]-E)
        else:
            n_pd = 0
            
        if (n_pd <0) or (np.isnan(n_pd)) or (np.isinf(n_pd)):
            msg = "node particle diameter is negative, nan or inf"
            # print("n_pd{}, pd_in{}, E{}, D{}, n{}".format(n_pd, pd_in, E, D, n))
            raise ValueError(msg)        
        return n_pd

    
    def _particle_diameter_out(self,pd_up,pd_in,qsi,E,D):
        """determine the weighted average particle diameter of the outgoing
        flow"""
        
        pd_out = np.sum((pd_up*E+pd_in*(qsi-D))/(qsi-D+E))
        
        
        if (pd_out <=0) or (np.isnan(pd_out)) or (np.isinf(pd_out)):
            msg = "out-flowing particle diameter is zero, negative, nan or inf"
            # print("pd_up{}, pd_in{}, qsi{}, E{}, D{}".format(pd_up, pd_in, qsi, E, D))
            raise ValueError(msg)
        
        return pd_out