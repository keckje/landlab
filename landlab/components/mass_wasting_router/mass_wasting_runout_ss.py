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
    and Depostion depths, tracks grain size and updates the DEM. This model is
    intended for modeling the runout of individually mapped landslides and landslides
    inferred from a landslide hazard map.

    This version includes a forward-only rule that allows routing by surface slope.
    To visualize effect of turning rule on and off, using in thin-flume script.
    
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
    settle_deposit = False,
    deposition_rule = "critical_slope",
    veg_factor = 3,
    dist_to_full_flux_constraint = 0,
    deposit_style = 'downslope_deposit',
    anti_sloshing = False,
    sloshing_check_frequency = 20,
    effective_qsi = False):
        
        super().__init__(grid)

        self.release_dict = release_dict
        self.mw_dict = mw_dict
        self.save = save
        self.run_id = run_id
        self.itL = itL
        self.routing_partition_method = 'slope'#'square_root_of_slope'  #
        self.routing_surface = routing_surface
        self.settle_deposit = settle_deposit
        self.VaryDp = self._grid.has_field('node', 'particle__diameter')
        self.deposition_rule = deposition_rule
        self.veg_factor = veg_factor # increase SD by this factor for all undisturbed nodes
        self.dist_to_full_flux_constraint = dist_to_full_flux_constraint # make his larger than zero if material is stuck
        self._deposit_style = deposit_style # assume deposition below cell or not
        self.slosh_limit = 1
        self._anti_sloshing = anti_sloshing
        self._check_frequency = sloshing_check_frequency
        self.effecitve_qsi = effective_qsi
        self.momentum = True
        if self.VaryDp:
            print(' running with spatially variable Dp ')

        # release parameters for landslide
        self.nps = list(self.release_dict['number of pulses'])
        self.nid = list(self.release_dict['iteration delay'])
       

        if len(self.mw_dict['critical slope'])>1: # change this to check for list input if defined as func of CA
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
                
        if 'effective particle diameter' in self.mw_dict: # specify Dp
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
        
        if 'typical flow thickness, scour' in self.mw_dict:
            self.h = self.mw_dict['typical flow thickness, scour'] 
        else:
            self.h = 2

        if 'typical slope, scour' in self.mw_dict:
            self.s = self.mw_dict['typical slope, scour'] 
        else:
            self.s = 0.15
        if 'max observed flow depth' in self.mw_dict:
            self.qsi_max = self.mw_dict['max observed flow depth']
        else:
            self.qsi_max = None


        # density of debris flow mixture
        self.rodf = self.vs*self.ros+(1-self.vs)*self.rof
        # distance equivalent iteration
        self.d_it = int(self.dist_to_full_flux_constraint/self._grid.dx)
        
        # define initial topographic + mass wasting thickness topography
        self._grid.at_node['energy__elevation'] = self._grid.at_node['topographic__elevation'].copy()
        self._grid.at_node['topographic__initial_elevation'] = self._grid.at_node['topographic__elevation'].copy()
        
        # define the initial disturbance map
        # if the user included a disturbance map field then:
        if self._grid.has_field('disturbance_map'):
            self._grid.at_node['disturbance_map'] = self._grid.at_node['disturbance_map'].astype(bool)
        else: # otherwise define everthing as undisturbed
            self._grid.at_node['disturbance_map'] = np.full(self._grid.number_of_nodes, False)
    
   
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
                    'number of pulses': 1, 
                    'iteration delay': 1
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
        self.topo_evo_maps = {}
        self.enL = [] # entrainment depth / regolith depth
        self.DpL = [] # deposition depth
        self.dfdL = [] # incoming debris flow thickness (qsi)
        self.TdfL = [] # basal shear stress 
        self.slopeL = [] # slope
        self.velocityL = [] # velocity (if any)
        
        # data containers for tests
        self.pd_r = {}
        self.st_r = {}
        self.tss_r ={}
        self.frn_r = {}
        self.frp_r = {}
        self.arv_r = {}
        self.arn_r = {}
        self.arpd_r = {}
        self.arndn_r = {}
        
        # For each mass wasting event in list:
        for mw_i,inn in enumerate(innL):
                         
            mw_id = self.mw_ids[mw_i]
            self._lsvol = self._grid.at_node['soil__thickness'][inn].sum()*self._grid.dx*self._grid.dy # set volume
            
            if self.qsi_max == None:
                self.qsi_max = self._grid.at_node['soil__thickness'][inn].max()

            # prep data containers
            cL[mw_i] = []
            self.df_evo_maps[mw_i] = {}
            self.topo_evo_maps[mw_i] = {}
            self.DEMdfD = {}

            # prep data containers for tests
            self.pd_r[mw_id] = []
            self.st_r[mw_id] = []
            self.tss_r[mw_id] = []
            self.frn_r[mw_id] = []
            self.frp_r[mw_id] = []            
            self.arv_r[mw_id] = []
            self.arn_r[mw_id] = []
            self.arpd_r[mw_id] = []   
            self.arndn_r[mw_id] = []
            
            # prepare initial mass wasting material (precipitons) for release 
            self._prep_initial_mass_wasting_material(inn, mw_i)

            # save first set of data
            self.df_evo_maps[mw_i][0] = self._grid.at_node['energy__elevation'].copy()
            self.topo_evo_maps[mw_i][0] = self._grid.at_node['topographic__elevation'].copy()
            self.DEMdfD[0] = {'DEMdf_r':0}            
            if self.VaryDp:
                self.pd_r[mw_id].append(self._grid.at_node['particle__diameter'].copy())
            self.st_r[mw_id].append(self._grid.at_node['soil__thickness'].copy())
            self.tss_r[mw_id].append(self._grid.at_node['topographic__steepest_slope'].copy())
            self.frn_r[mw_id].append(self._grid.at_node['flow__receiver_node'].copy())
            self.frp_r[mw_id].append(self._grid.at_node['flow__receiver_proportions'].copy())
            self.arv_r[mw_id].append(self.arv)
            self.arn_r[mw_id].append(self.arn)
            self.arpd_r[mw_id].append(self.arpd)
            self.arndn_r[mw_id].append(self.arndn)
        
            # now loop through each receiving node in list rni, 
            # determine next set of recieving nodes
            # repeat until no more receiving nodes (material deposits)                
            c = 0 # model iteration counter
            c_dr = 0 # mass wasting event delayed-release counter
            self.sloshed = 0 # number of times model has sloshed
            
            while len(self.arn)>0 and c < self.itL:
                self.c = c
                if self.d_it == 0:
                    self.SD_v = self.SD
                else:
                    self.SD_v = self.SD*(min(c/self.d_it,1))
                # release the initial landslide volume
                # if first iteration, receiving cell = initial receiving list
                # initial volume = volume/nps
                if c == 0: 
                    c_dr+=1
                    
                # for following iterations, add initial volume/nps every nid iterations
                # until the volume has been added nps times
                elif self.nps[mw_i]>1:
                    if ((c)%self.nid[mw_i] == 0) & (c_dr<=self.nps[mw_i]-1):
                        self.arn = np.concatenate((self.arn, self.rni))
                        self.arv = np.concatenate((self.arv, self.rvi))
                        self.arpd = np.concatenate((self.arpd, self.rpdi))
                        # update pulse counter
                        c_dr+=1        
                
                # receiving node, volume and particle diameter temporary arrays
                # that become the arrays for the next model step (iteration)
                self.arndn_ns = np.array([])
                self.arn_ns = np.array([])
                self.arv_ns = np.array([])
                self.arpd_ns = np.array([])
                
                self.arnL = []
                self.arvL = []
                self.arpdL = []
                self.arndnL = []
                
                # for each unique cell in receiving node list self.arn
                arn_u = np.unique(self.arn).astype(int)  # unique arn list
                
                # determine the incoming volume and depth to each node in arn_u
                self.vqdat = self._vin_qsi(arn_u) ##
                
                # update energy elevation as node elevation plus incoming flow thickness
                # this happens even if using topographic__elevation to route so that 
                # the thickness of the debris flow is tracked for plotting
                self._update_E_dem()
                
                if self.routing_surface == "energy__elevation":                              
                    # if considering the energy surface in flow routing, 
                    # update the receiving nodes (self.rn) to represent the 
                    # the flow surface
                    self._update_energy_slope()   
                    self.rp = self._grid.at_node['flow__receiver_proportions'].copy() # not using self.rp, only need rn for constraint
                    self.rn = self._grid.at_node['flow__receiver_node'].copy()
                    # scour function uses the underlying topography. Grid fields need to
                    # be updated for before deposition or scour caused by material 
                    # after it enters node n
                    self._update_topographic_slope()

                # determine scour, entrain, deposition and outflow depths and
                # the outflowing particle diameter, arranged in array nudat
                self.nudat = self._scour_entrain_deposit_updatePD()                

                # determine the receiving nodes, volumes and grain size
                self._determine_rnp_attributes()                               
                
                # update grid field: topographic__elevation with the values in 
                # nudat. Do this after directing flow, because assume deposition 
                # does not impact flow direction
                self._update_dem()
               
                ### update topographic slope field for deposition to detemine where
                # settling will occur - move this innto settle_deposit
                               
                self._update_topographic_slope()  
                # set receiving nodes (self.rn) and proportions (self.rp) based on underlying topography                 
                self.rp = self._grid.at_node['flow__receiver_proportions'].copy()
                self.rn = self._grid.at_node['flow__receiver_node'].copy()
                
                ### update grid field: particle__diameter with the values in 
                # nudat
                if self.VaryDp:
                    self._update_channel_particle_diameter()
                              
                self.dif  = self._grid.at_node['topographic__elevation']-self._grid.at_node['topographic__initial_elevation']
                
                # not sure if settle_deposit is useful or needed
                if self.settle_deposit:
                    
                    ### settle unrealistically tall mounds in the deposit                    
                    self._settle(arn_u)
                    self._update_topographic_slope()
                    self.rp = self._grid.at_node['flow__receiver_proportions'].copy()
                    self.rn = self._grid.at_node['flow__receiver_node'].copy()

                # self.dif  = self._grid.at_node['topographic__elevation']-self._grid.at_node['topographic__initial_elevation']
                self._update_disturbance_map()  

                # once all cells in iteration have been evaluated, temporary receiving
                # node, node volume and node particle diameter arrays become arrays 
                # for next iteration
                self.arndn = self.arndn_ns.astype(int)
                self.arn = self.arn_ns.astype(int)
                self.arv = self.arv_ns #   
                self.arpd = self.arpd_ns

                if self.save:

                    cL[mw_i].append(c)
                    
                    DEMf = self._grid.at_node['topographic__elevation'].copy()
                    
                    DEMdf_r = DEMf-self._grid.at_node['topographic__initial_elevation']
            
                    self.DEMdfD[c+1] = {'DEMdf_r':DEMdf_r.sum()*self._grid.dx*self._grid.dy}     
                    self.df_evo_maps[mw_i][c+1] = self._grid.at_node['energy__elevation'].copy()
                    ### save maps for video
                    self.df_evo_maps[mw_i][c+1] = self._grid.at_node['energy__elevation'].copy()
                    self.topo_evo_maps[mw_i][c+1] = self._grid.at_node['topographic__elevation'].copy()
                    # data for tests
                    if self.VaryDp:
                        self.pd_r[mw_id].append(self._grid.at_node['particle__diameter'].copy())
                    self.st_r[mw_id].append(self._grid.at_node['soil__thickness'].copy())
                    self.tss_r[mw_id].append(self._grid.at_node['topographic__steepest_slope'].copy())
                    self.frn_r[mw_id].append(self._grid.at_node['flow__receiver_node'].copy())
                    self.frp_r[mw_id].append(self._grid.at_node['flow__receiver_proportions'].copy())
                    self.arv_r[mw_id].append(self.arvL)
                    self.arn_r[mw_id].append(self.arnL)
                    self.arpd_r[mw_id].append(self.arpdL)     
                    self.arndn_r[mw_id].append(self.arndn)

                
                
                
                # update iteration counter
                c+=1
                
                if c%self._check_frequency ==0:
                    print(c)  
                    
                    if self._anti_sloshing:
                        self._sloshing_count(mw_id,mw_i)

    def _sloshing_count(self,mw_id,mw_i):
  
          # if self.c>5 and [list(x) for x in self.arndn_r[mw_id][-1]] == [list(x) for x in self.arndn_r[mw_id][-2]]:

        
        if self.c>self._check_frequency:
            # check for small change in deposition
            sumdif = []
            irR = max(self.df_evo_maps[mw_i].keys()); irL = irR-self._check_frequency
            for w in np.linspace(irL,irR,self._check_frequency+1):                  
                dif = self.df_evo_maps[mw_i][w]-self._grid.at_node['topographic__initial_elevation']
                sumdif.append(dif[dif>0].sum())
            sumdif = np.array(sumdif)    
            difs = sumdif[:-1]-sumdif[1:]
        
            self.difsmn = np.abs((np.nanmean(difs)*self._grid.dx*self._grid.dy)/self._lsvol)
            print('difsmn:{}'.format(self.difsmn))
            
            # check for repeat delivery nodes
            a1 = self.arndn_r[mw_id][-1]; a2 = self.arndn_r[mw_id][-5]
            l_m = (len(a1)+len(a2))/2
            rnr = len(np.intersect1d(a1,a2))/l_m
            # print('repeat node ratio{},a1:{}, a2:{}'.format(rnr, a1, a2))
            
            if self.difsmn < 0.001 or rnr > 0.3:
                 self.sloshed = self.sloshed+1
                 print('##############################################   SLOSHEDDDDDDDDDDDDDD')
                 
             
    def _update_disturbance_map(self):
        """map of boolian values indicating if a node has been disturbed (changed 
        topographic__elevation) at any point in the model run"""
        self._grid.at_node['disturbance_map'][np.abs(self.dif>0)] = True 
 
    
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
            
            # soil thickness at node
            s_t = self._grid.at_node.dataset['soil__thickness'].values[ni]
            
            # remove soil (landslide) thickness at node
            self._grid.at_node.dataset['topographic__elevation'][ni] =  (                   
            self._grid.at_node.dataset['topographic__elevation'][ni] - s_t)
            
            # update soil thickness at node (now = 0)
            self._grid.at_node['soil__thickness'][ni] = (
                self._grid.at_node['soil__thickness'][ni]-s_t)
            
            if ci>0: # use slope based on surface for first node to start landslide 
                # (don't update slope to reflect change in elevation), but topography 
                # of failure surface after remove entire soil depth from node 
                # (regardless of release parameters) update slope to reflect node 
                # material removed from DEM
                self._update_topographic_slope()
             
            # get receiving nodes of node ni in mw index mw_i
            rn = self._grid.at_node.dataset['flow__receiver_node'].values[ni]
            rn = rn[np.where(rn != -1)]
            

            # receiving proportion of qso from cell n to each downslope cell
            rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[ni]
            rp = rp[np.where(rp > 0)] # only downslope cells considered
            

            
            # initial mass wasting thickness, (total thickness/number of pulses)  
            imw_t =s_t/self.nps[mw_i]
            # get volume (out) of node ni   
            vo = imw_t*self._grid.dx*self._grid.dy# initial volume 
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
            

        
        # set receiving nodes (self.rn) and proportions (self.rp) based on underlying topography                 
        self.rp = self._grid.at_node['flow__receiver_proportions'].copy()
        self.rn = self._grid.at_node['flow__receiver_node'].copy()

        # landslide release nodes, volumes and diameters - saved for incremental release
        self.rni = rni
        self.rvi = rvi
        self.rpdi = rpdi
        
        self.arndn = np.ones([len(rni)])*np.nan
        self.arn = rni
        self.arv = rvi
        self.arpd = rpdi

        
    def _scour_entrain_deposit_updatePD(self):
        """ mass conservation at a grid cell: determines the erosion, deposition
        change in topographic elevation and flow out of a cell as a function of
        debris flow friction angle, particle diameter and the underlying DEM
        slope
        
        
        slpn needs to be topographic__elevation determined
        rn can be either topographic__elevation or energy__elevation
        rp can be either topographic__elevation or energy__elevation
        """
        
        # list of deposition depths, used in settlement algorithm
        self.D_L = []
        def SEDU(vq_r):
            """function for iteratively determing scour, entrainment and
            deposition depths using node id to look up incoming flux and
            downslope nodes and slopes"""            

            n = vq_r[0]; vin = vq_r[1]; qsi = vq_r[2]; 
            
            if self.effecitve_qsi:
                qsi_ = min(qsi,self.qsi_max)
            else:
                qsi_ = qsi
            
            dn = self.arndn[self.arn == n]

            # get adjacent nodes
            adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
            self._grid.diagonal_adjacent_nodes_at_node[n]))
            
            # maximum slope at node n
            slpn = self._grid.at_node['topographic__steepest_slope'][n].max()
            
            # slpn = self._grid.at_node['topographic__steepest_slope'][adj_n].mean()

            
            # look up critical slope at node n
            if len(self.mw_dict['critical slope'])>1: # if option 1, critical slope is not constant but depends on location
                self.slpc = self.a*self._grid.at_node['drainage_area'][n]**self.b    
            # print('vin:{} ###################################'.format(vin))
            # incoming particle diameter (weighted average)
            pd_in = self._particle_diameter_in(n,vin) # move

            # can flow continue? need rn node to avoid direction flow towards self.
            # 
            
            # rn: receiver nodes based on either the pre-flow underlying topographic 
            # slope at node n or slope of the flow at node n once it is over node n
            # rn = self._grid.at_node.dataset['flow__receiver_node'].values[n] ## receiving node needs to be based on energy el
            rn = self.rn[n]
            rn = rn[np.where(rn != -1)] 
            rn = rn[~np.isin(rn,dn)]  # material can not be sent back to the delivery nodes
            
            # rn_g: receiver nodes based on the pre-flow underlying topographic slope at node n
            rn_g = self._grid.at_node.dataset['flow__receiver_node'].values[n]
            rn_g = rn_g[np.where(rn_g != -1)]
            # rn_g = rn_g[~np.isin(rn_g,dn)]
            # print('dn:{}, rn:{}'.format(dn, rn))
            # if qsi less than the vegetation SD in an undisturbed cell
            if ((qsi <=(self.SD_v*self.veg_factor)) and (self._grid.at_node['disturbance_map'][n] == False)):
                D = qsi 
                qso = 0 
                E = 0 
                deta = D 
                pd_up = 0
                Tbs = 0
                u = 0
                # print('less than SD veg, it:{}, node:{}, qsi:{}, D:{}, qso:{}'.format(self.c,n,qsi,D,qso))
            # if qsi is less than SD in a disturbed cell or the iteration limit has been reached or their are no receiving cells
            # if no receiving cells, then deposition is at node n
            # ADD TO MWR ##############
            elif (qsi <=self.SD_v) or self.c == self.itL-1 or (len(rn) < 1) or ((len(rn) == 1) and (len(dn) == 1) and ([dn] == [rn])):# 
                D = qsi
                qso = 0
                E = 0
                deta = D
                pd_up = 0
                Tbs = 0
                u = 0
                # print('qsi less than SD or rn <1 or sent to self, it:{}, node:{}, qsi:{}, D:{}, qso:{}'.format(self.c,n,qsi,D,qso))
                # print('SDv:{}, rn:{}'.format(self.SD_v, rn))
            else:
                D = self._deposit(qsi_, slpn, n) # function of qsi and topographic elevation before settle/scour by qsi                
                
                # scour a function of steepes topographic slope at node, determined before settle/scour by qsi
               
                if self.VaryDp:   
                    opt = 2
                else:
                    opt = 1                        
                E, pd_up, Tbs, u = self._scour(n, qsi_, slpn, opt = opt, pd_in = pd_in)   
                
                # contstrain that E
                # Sn = self._grid.at_node['topographic__slope']
                # if (Sn <= self.slpc) and ((Sn+E/self._grid.dx)>self.slpc):
                #     E = self.slpc*self.dx-Sn*self.dx
                if D > 0:#0.33*qsi: 
                    E = 0
                    pd_up = 0

                ## flow out
                qso = qsi-D+E                
                # small qso are considered zero
                qso  = np.round(qso,decimals = 8)
                # print('qso:'+str(qso))
                # chage elevation
                deta = D-E 
                
            # model behavior tracking
            self.enL.append(E)
            self.DpL.append(D)
            self.dfdL.append(qsi)
            # model behavior tracking
            self.TdfL.append(Tbs)
            self.slopeL.append(slpn) # slope
            self.velocityL.append(u) # velocity (if any)

            # updated node particle diameter (weighted average)
            n_pd = self._particle_diameter_node(n,pd_in,E,D)
            
            # list of deposition depths at cells in iteration 
            self.D_L.append(D)            

            return deta, qso, n_pd, pd_up, pd_in, qsi, E, D
    
        
        # apply SEDU function to all unique nodes in arn (arn_u)
        # create nudat, an np.array of data for updating fields at each node
        # that can be applied using vector operations
        ll=np.array([SEDU(r) for r in self.vqdat],dtype=object)     
        arn_ur = np.reshape(self.vqdat[:,0],(-1,1))
        nudat = np.concatenate((arn_ur,ll),axis=1)
        # print(nudat)
        return nudat # qso, rn not used


    def _determine_rnp_attributes(self):
        """ after mass conservation at the cell determined use pre-deposition/scour
        slope (before results of mass conservation applied to dem) to determine
        how outgoing volume is partioned to downslope cells and attributes of each
        paritioned volume"""
        
        """need to add: if self, deposit first, then send to receiving nodes
        
        call deposit fucntion?"""
        
        def rn_rp(nudat_r):
            n = nudat_r[0]; qso = nudat_r[2]; pd_up = nudat_r[4]; pd_in = nudat_r[5]
            qsi = nudat_r[6]; E = nudat_r[7]; D = nudat_r[8]
            
            # nodes that sent material to node n
            dn = self.arndn[self.arn == n]
            
            # nodes that receive material from node n
            
            # topographically determined receiving nodes 
            # rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
            rn = self.rn[n] #  defining rn from the energy elevation causes flow to slosh
            rn_ = rn.copy()
            # remove -1 node ids
            rn_ = rn_[np.where(rn_ != -1)] 
            
            # material can only move forward or stay at self
            if self.c > 0: # no previous step 
            
                # if np.isnan(and_rM): # if no receiving nodes, deposit some of the materials
                    # if self.effecitve_qsi:
                    #     qsi_ = min(qsi,self.qsi_max)
                    # else:
                    #     qsi_ = qsi
                        
                    # D = self._deposit(qsi_, slpn, n) # function of qsi and topographic elevation before settle/scour by qsi                
                    
                    # # scour a function of steepes topographic slope at node, determined before settle/scour by qsi
                   
                    # if self.VaryDp:   
                    #     opt = 2
                    # else:
                    #     opt = 1                        
                    # E, pd_up, Tbs, u = self._scour(n, qsi_, slpn, opt = opt, pd_in = pd_in)   
                    
                    # # contstrain that E
                    # # Sn = self._grid.at_node['topographic__slope']
                    # # if (Sn <= self.slpc) and ((Sn+E/self._grid.dx)>self.slpc):
                    # #     E = self.slpc*self.dx-Sn*self.dx
                    # if D > 0:#0.33*qsi: 
                    #     E = 0
                    #     pd_up = 0

                    # ## flow out
                    # qso = qsi-D+E                
                    # # small qso are considered zero
                    # qso  = np.round(qso,decimals = 8)
                    # # print('qso:'+str(qso))
                    # # chage elevation
                    # deta = D-E 
                
                rn_ = self._forward_nodes(n, rn_)
            
            # exclude boundary nodes
            rn_ = rn_[~np.isin(rn_, self._grid.boundary_nodes)] 
            
            # material can not be sent back to delivery node
            rn_ = rn_[~np.isin(rn_,dn)]
            

            if qso>0 and n not in self._grid.boundary_nodes: # redundant?
                # move this out of funciton, need to determine rn and rp after material is in cell, not before
                
                vo = qso*self._grid.dx*self._grid.dy # convert qso to volume
                # receiving proportion of qso from cell n to each downslope cell
                if len(rn_)<1: # if number of receiving nodes less than 1, no receiving cells, material deposits 
                    # print('length rn less than 1#########')
                    rp = np.array([1])
                    rn = np.array([n])
                    rv = rp*vo
                    # rn_new = rn
                    # rp_new = rp
                # if (len(rn[np.where(rn != -1)]) == 1) and ([n] == rn[np.where(rn != -1)]):# or [dn] == rn[np.where(rn != -1)]):

                else:
                    # rp_ = self.rp[n][np.where(rn != -1)]
                    # rp_ = self._grid.at_node.dataset['flow__receiver_proportions'].values[n][np.where(rn != -1)]
                    # rn = rn[np.where(rn != -1)] 
                    # rp = rp_[~np.isin(rn,dn)]
                    # rn = rn[~np.isin(rn,dn)]
                    # rp = rp[np.where(rp > 0)] # only downslope cells considered
                    # rp = rp/rp.sum() # if some proportion
                    
                    # rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[n]
                    rp = self.rp[n]
                    rp = rp[np.isin(rn,rn_)]
                    rp = rp/rp.sum()
                    rn = rn_

                    # receiving volume
                    rv = rp*vo
                    
                #     if (len(rn) ==1) and (n == rn): # if node is a pit
                #         rn_new = rn
                #         rp_new = rp   
                #     else:
                #         if self.c == 0: # no previous step to determine ang_rM, ang_rN = ang_rT                       
                #             ang_rN = self._topographic_azimuth(n, rn, rv)
                #         else:
                #             # get equivalent azimuth of resultant qT from dilivery nodes
                #             ang_rM, rn_M = self._incoming_azimuth(n)
                #             # get equivalent azimuth of resultant qT from rn and rp
                #             ang_rT = self._topographic_azimuth(n, rn, rv)
                #             # approximate magnitude of incoming momentum and topographically directed momentum
                #             a, b = self._det_a_and_b( n, qsi, rn_M)
                #             # compute New resultant qT from incoming material and topographically determined azimuths
                #             ang_rN = self._new_direction(ang_rM, a, ang_rT, b)
                #             # print('ang_rT:{}, ang_rN:{}'.format(ang_rT,ang_rN))
                #         # from New azimuth, get new set of rn
                #         rn_new, new_flow_directions = self._new_rn(n, ang_rN, qsi)
                #         # from new set of rn, and their respective azimuths, determine new rp
                #         rp_new = self._new_proportions(new_flow_directions, ang_rN)
                        
                # if self.momentum:
                #     # print('n{},rn{},rp{}'.format(n,rn,rp))
                #     # print('n{},rn_new{},rp_new{}'.format(n,rn_new,rp_new))
                #     rn = rn_new
                #     rv = rp_new*vo
 
                # if np.isnan(rv).any():

                #     msg = 'rv is nan!!!!!!!!!!!!!!!!!!!!!, rv:{}, rn:{}, n:{}, rn_:{}, rp:{}, proportions:{}'.format(rv, self.rn[n], n, rn_, rp,self._grid.at_node.dataset['flow__receiver_proportions'].values[n])
                #     # print("n_pd{}, pd_in{}, E{}, D{}, n{}".format(n_pd, pd_in, E, D, n))
                #     raise ValueError(msg)  
                rndn = (np.ones(len(rn))*n).astype(int) # receiving node delivery node (list of node n, length eqaul to number of proportions of qso sent to receiving cells)

                # particle diameter out (weighted average)
                pd_out = self._particle_diameter_out(pd_up,pd_in,qsi,E,D)    
                rpd = np.ones(len(rv))*pd_out

                # print('iteration:{}, n:{},  rn:{}, rp:{}, rv:{}, dn{}:, pd_out:{}, slp:{}'.format(self.c+1,n, rn, rp, rv, rndn, pd_out, self._grid.at_node['topographic__steepest_slope'][rn]))
                # store receiving nodes and volumes in temporary arrays
                self.arndn_ns = np.concatenate((self.arndn_ns, rndn), axis = 0) # next step delivery nodes
                self.arn_ns = np.concatenate((self.arn_ns, rn), axis = 0) # next step receiving node list
                self.arv_ns = np.concatenate((self.arv_ns, rv), axis = 0) # next step receiving node incoming volume list
                self.arpd_ns = np.concatenate((self.arpd_ns, rpd), axis = 0) # next step receiving node incoming particle diameter list
   
                self.arnL.append(rn)
                self.arvL.append(rv)
                self.arpdL.append(rpd)
                self.arndnL.append(rndn)

        # change to this
        # nudat_ = self.nudat[self.nudat[:,2]>0] # only run on nodes with qso>0
        # ll = np.array([rn_rp(r) for r in self.nudat],dtype=object)
        
        ll = np.array([rn_rp(r) for r in self.nudat],dtype=object)


    def _vin_qsi(self,arn_u):
        """determine volume and depth of incoming material
        returns vqdat: np array of receiving nodes [column 0], 
        vin [column 1] and qsi [column 2]"""

        def VQ(n):           
            # total incoming volume
            
            vin = np.sum(self.arv[self.arn == n])
            # convert to flux/cell width
            qsi = vin/(self._grid.dx*self._grid.dx)
            # print('arv:{}, arn:{}, n:{}, qsi:{}'.format(self.arv,self.arn,n,qsi))
            return vin, qsi

        ll=np.array([VQ(n) for n in arn_u], dtype=object)     
        arn_ur = np.reshape(arn_u,(-1,1))
        vqdat = np.concatenate((arn_ur,ll),axis=1)
    
        return vqdat

    def _incoming_azimuth(self, n):
        """determine weighted average azimuth of incoming material flux
        may be material from self, which assumed nearly stationary (velocity ~ 0)
        and is thus excluded from the azimuth"""
        # adjacent node ids to no de n
        adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
        self._grid.diagonal_adjacent_nodes_at_node[n]))
        # get all deliverying nodes
        ard_n = self.arndn[np.where(self.arn == n)]
        # only use nodes other than self to determine incoming azimuth (exclude self)
        exclude_n_mask = ~np.isin(ard_n, n)
        ard_n=ard_n[exclude_n_mask]
        # get incoming volume
        arv_n = self.arv[np.where(self.arn == n)]
        arv_n = arv_n[exclude_n_mask]
        #incoming azimuth from adjacent nodes, ordered: R,T,L,B,TR,TL,BL,BR
        adj_n_incoming_dir = np.array([270,180,90,360,225,135,45,315])
        dir_ = []
        # print('ard_n:{}'.format(ard_n))
        for an in ard_n:           
            dir_.append(adj_n_incoming_dir[adj_n == an][0])
        # magnitude of incoming material, excluding self
        qT = arv_n.sum()
        # print('azimuths:{}'.format(dir_))
        # resultant direction, weighted average using qsi
        ang_rM = (np.array(dir_)*arv_n).sum()/qT
        # node closest to resultant direciton
        adj_n_outgoing_dir = np.array([90,360,270,180,45,315,225,135])
        dif = adj_n_outgoing_dir - ang_rM;
        rn_M = adj_n[dif == dif.min()] # receiving node in line with direction
        # print('ang_rM:'.format(ang_rM))
        return ang_rM, rn_M

    def _forward_nodes(self, n, rn_):
        """at a pit, then only node in forward node list is self, gets stuck,
        """
        ang_rM, rn_M = self._incoming_azimuth(n)
        # print(ang_rM)
        if np.isnan(ang_rM):
            rn_ = np.array([])
        else:
            adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
            self._grid.diagonal_adjacent_nodes_at_node[n]))
            adj_n_outgoing_dir = np.array([90,360,270,180,45,315,225,135])
            mask = np.abs(self._aba(adj_n_outgoing_dir, ang_rM)) <90 # foward nodes less than a 90 degree angle for the direction of movement
            # print('adjacent nodes:{}, mask:{}'.format(adj_n, mask))
            forward_nodes = np.concatenate([adj_n[mask], np.array([n])]) # include self node
            rn_ = rn_[np.isin(rn_, forward_nodes)]    
        return rn_

    def _topographic_azimuth(self, n, rn, rv):
        """determine weighted average azimuth of topographically directed flux"""
        # adjacent node ids to no de n
        adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
        self._grid.diagonal_adjacent_nodes_at_node[n]))
        adj_n_outgoing_dir = np.array([90,360,270,180,45,315,225,135])
        dir_ = []
        for r in rn:
            # print('n{},adj_n{}'.format(n,adj_n))
            dir_.append(adj_n_outgoing_dir[adj_n == r][0])
        # magnitude
        qT = rv.sum()
        # resultant direction
        ang_rT = (np.array(dir_)*rv).sum()/qT
        return ang_rT
    
    def _det_a_and_b(self, n, qsi, rn_M):
        # get max topographic slope
        adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
        self._grid.diagonal_adjacent_nodes_at_node[n]))
        zo_min = self._grid.at_node['topographic__elevation'][adj_n].min()
        zi = self._grid.at_node['topographic__elevation'][n]
        slp_T = max((zi-zo_min)/self._grid.dx, 0.00001)
        if len(rn_M) > 0: # if there is a delivery node other than self, compute slope of qsi
            zo_rnM = self._grid.at_node['topographic__elevation'][rn_M]
            slp_M = max(((zi+qsi)-zo_rnM)/self._grid.dx, 0.00001)
            a = 0.1*slp_M/slp_T; b = 1
        else: # if the delivery node is self
            zo_rnM = None; slp_M=None; a = None; b = 1
        # print('values of rn_M:{}, zo_rnM:{}, slpM:{}, slpT:{}, a:{}, b:{}'.format(rn_M,zo_rnM,slp_M,slp_T,a,b))
        return a, b
    
    @staticmethod
    def _new_direction(ang_rM, a, ang_rT, b):
        """determine new direction"""
        if a:     
            ang_rN = (a*ang_rM+b*ang_rT)/(a+b)
        else:
            ang_rN = ang_rT
        return ang_rN
    
    @staticmethod
    def _angle_xy(angle):
        """convert an angle (degrees) to a unit vector equivalent"""
        angle_ = angle -90
        y = np.sin(np.radians(angle_))*-1
        x = np.cos(np.radians(angle_))
        return x,y
    
    def _aba(self, angle1, angle2):
        """get Angle Between two Azimuths (degrees)"""
        x1, y1 =self._angle_xy(angle1)
        x2, y2 =self._angle_xy(angle2)
        dot = x1*x2 + y1*y2      # dot product
        det = x1*y2 - y1*x2      # determinant
        angle = np.arctan2(det, dot)  
        return np.degrees(angle)
    
    def _new_rn(self, n, ang_rN, qsi):
        """from the new flux azimuth, determine the receiving nodes"""
        adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
        self._grid.diagonal_adjacent_nodes_at_node[n]))
        adj_n_outgoing_dir = np.array([90,360,270,180,45,315,225,135])
        mask = np.abs(self._aba(adj_n_outgoing_dir, ang_rN)) <90
        new_rn = adj_n[mask]
        # can only go to nodes less than height of qsi + z
        mask2 = self._grid.at_node['topographic__elevation'][new_rn]<(self._grid.at_node['topographic__elevation'][new_rn]+qsi)
        new_rn = new_rn[mask2]
        # cannot include boundary nodes
        mask3 = ~np.isin(new_rn, self._grid.boundary_nodes)
        new_rn = new_rn[mask3]
        new_directions = adj_n_outgoing_dir[mask]
        new_directions = new_directions[mask2]
        new_directions = new_directions[mask3]
        # print('new_rn:{}, new_directions:{}'.format(new_rn, new_directions))
        return new_rn, new_directions

    def _new_proportions(self, new_directions, ang_rN):
        """from the the new flux azimuth and the new receiving nodes,
        determine the proportion of flux sent to each node, using a weighted 
        average of the component of each receiving node azimuth in the direction
        of the new flux azimuth"""
        defl = np.abs(self._aba(new_directions, ang_rN))
        uv = np.cos(np.radians(defl))
        return(uv/uv.sum())
    
    
    def _update_E_dem(self):
        """update energy__elevation"""
    
        n = self.vqdat[:,0].astype(int); qsi = self.vqdat[:,2];         
        # energy slope is equal to the topographic elevation potential energy
        self._grid.at_node['energy__elevation'] = self._grid.at_node['topographic__elevation'].copy()
        # plus the pressure potential energy
        self._grid.at_node['energy__elevation'][n] = self._grid.at_node['energy__elevation'].copy()[n]+qsi


    def _update_energy_slope(self):
        """updates the topographic__slope and flow directions fields using the 
        energy__elevation field"""
        fd = FlowDirectorMFD(self._grid, surface="energy__elevation", diagonals=True,
                partition_method = self.routing_partition_method)
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
                partition_method = self.routing_partition_method)
        fd.run_one_step()

    
    def _update_channel_particle_diameter(self):
        """ for each unique node in receiving node list, update the grain size
        using the grain size determined in _scour_entrain_deposit
        """

        n = self.nudat[:,0].astype(int); new_node_pd = self.nudat[:,3]
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
            if self.D_L[ii] >0: # only settle if node has had deposition...use dif?
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
    
                # only consider areas of depsoition (steep areas in original dem are excluded)
                # rndif = self.dif[rn]
                # slpn = slpn[abs(rndif)>0]  
                # rn = rn[abs(rndif)>0]
    
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
                    
                    if qso_s < 0: # no negative 
                        # print("negative outlflow settling, n, {}, rn, {}, qso_s, {}".format(n,rn,qso_s))
                        qso_s = 0
                    
                    if qso_s > self.D_L[ii]: # settlement out can not exceed deposit
                        qso_s= self.D_L[ii]
                    
                    qso_s_i = qso_s*pp # proportion sent to each receiving cell                
                    
                    # print("qso_s, {}".format(qso_s))
                    # print("qso_s_i, {}".format(qso_s_i))
                    
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
        
        if opt ==1:
            # following Frank et al., 2015, approximate erosion depth as a linear
            # function of total stress under uniform flow conditions
            
            # erosion depth,:
            Tbs = self.rodf*self.g*depth*(np.sin(theta))
            Ec = self.cs*(Tbs)**self.eta
            u = 5
        if opt == 2:
            if depth < Dp: # grain size dependent erosion breaks if depth<Dp
                Dp = depth*.99
            # shear stress apprixmated as a power functino of inertial shear stress
            phi = np.arctan(self.slpc) # approximate phi [radians] from criticle slope [L/L]
            
            # inertial stresses
            us = (self.g*depth*slope)**0.5
            u = us*5.75*np.log10(depth/Dp)
            
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
        # self.TdfL.append(Tbs)
        # self.slopeL.append(slope) # slope
        # self.velocityL.append(u) # velocity (if any)
        
        return(E, pd_up, Tbs, u)

                
    def _deposit(self,qsi,slpn,n):
        """determine deposition depth as the minimum of L*qsi and
        an angle of repose determined depth. Where L is computed following
        Campforts, et al., 2020 but is flux per unit contour width (rather than flux), 
        and L is (1-(slpn/slpc)**2) rather than dx/(1-(slpn/slpc)**2)"""
        if self.sloshed < self.slosh_limit:
            if self.deposition_rule == "L_metric":
                D = self._deposit_L_metric(qsi, slpn)
            elif self.deposition_rule == "critical_slope":
                D = self._deposit_friction_angle(qsi, n)
            elif self.deposition_rule == "both":
                DL = self._deposit_L_metric(qsi, slpn)
                Dc = self._deposit_friction_angle(qsi, n) 
                D = min(DL,Dc)
        else: 
            D = qsi
        return(D)


    def _determine_zo(self, n, zi, qsi):
        """determine the mean energy elevation of the nodes lower than the
        incoming energy elevation of node n"""
        
        # get adjacent nodes
        adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
        self._grid.diagonal_adjacent_nodes_at_node[n]))
                        
        # incoming energy at node i
        ei = qsi+zi
        
        # nodes below incoming energy surface
        rn_e = adj_n[self._grid.at_node['topographic__elevation'][adj_n]<ei]
 
        if len(rn_e) > 0: 
                       
            zo = self._grid.at_node['topographic__elevation'][rn_e].min()
            
        else:  # a pit in the energy elevation surface
            zo = None
            
        return zo

    def _deposit_L_metric(self, qsi, slpn):
        """
        ----------
        qsi : float
            in coming flux per unit contour width
        slpn : float
            slope of node, measured in downslope direction (downslope is postive)
        """
        
        Lnum = np.max([(1-(slpn/self.slpc)**2),0])
        
        DL = qsi*Lnum
        
        return(DL)
    
    def _deposit_friction_angle(self, qsi, n):
        # ADD TO MWR ##############
        slp_h = self.slpc*self._grid.dx
        
            # elevation at node i
        zi = self._grid.at_node['topographic__elevation'][n]
        
        zo = self._determine_zo(n, zi, qsi )
        # print('n:{}, zo:{}, qsi:{}'.format(n, zo, qsi))
        if zo is None:# a pit in the energy elevation surface
            Dc = qsi 
        else:    
            if self._deposit_style == 'downslope_deposit':
                rule = ((zi-zo)<=((qsi+slp_h)/2))#rule = ((zi-zo)<=(slp_h))# deposition occurs if slope*dx is less than downslope deposit thickness
                def eq(qsi, zo, zi, slp_h):
                    # D = min(0.5*qsi+(zo-zi+slp_h)/2,qsi)
                    D = min(qsi-(((qsi+(zi-zo)-slp_h)/2)+(qsi+(zi-zo)-slp_h)/8-slp_h/4),qsi)
                    return np.round(D,decimals = 5) # for less than zero, greater than zero contraints
                
            elif self._deposit_style == 'downslope_deposit_sc':
                rule = ((zi-zo)<=((qsi*self.slpc+slp_h)))#rule = ((zi-zo)<=(slp_h))# deposition occurs if slope*dx is less than downslope deposit thickness
                def eq(qsi, zo, zi, slp_h):
                    D = min(qsi-(((qsi+(zi-zo)-slp_h)/2)+(qsi+(zi-zo)-slp_h)/8-slp_h/4),qsi)
                    return np.round(D,decimals = 5) # for less than zero, greater than zero contraints
            
            elif self._deposit_style == 'downslope_deposit_sc2':
                rule = ((zi-zo)<=(slp_h))
                def eq(qsi, zo, zi, slp_h):
                    D = min(0.5*qsi+(zo-zi+slp_h)/2,qsi)#D = min(zo-zi+slp_h,qsi)
                    return np.round(D,decimals = 5)
            
            elif self._deposit_style == 'downslope_deposit_sc3':
                rule = ((zi-zo)<=(slp_h))#rule = ((zi-zo)<=(slp_h))# deposition occurs if slope*dx is less than downslope deposit thickness
                def eq(qsi, zo, zi, slp_h):
                    D = min(qsi-(((qsi+(zi-zo)-slp_h)/2)+(qsi+(zi-zo)-slp_h)/8-slp_h/4),qsi)
                    return np.round(D,decimals = 5) # for less than zero, greater than zero contraints
            
            elif self._deposit_style == 'no_downslope_deposit_sc':
                rule = ((zi-zo)<=(slp_h))
                def eq(qsi, zo, zi, slp_h):
                    D = min(zo-zi+slp_h,qsi)
                    return np.round(D,decimals = 5)
            # elif self._deposit_style == "nolans_rule":
                
            
        # print('it:{}, node:{}, zi:{}, zo:{}, rule value:{}'.format(self.c,n, zi, zo,rule))
        

        
            if zo>zi:            

                Dc = eq(qsi,zo,zi, slp_h)
                # print('zo>zi, qsi:{}, D:{}'.format(qsi, Dc))
            
            elif (zo<=zi) and rule:#((zi-zo)<=(slp_h)):#######: #
                
                Dc = eq(qsi,zo,zi, slp_h)
                # print('zo<=zi, qsi:{}, D:{}'.format(qsi, Dc))
            else:
                Dc = 0
                # print('slope exceeds slp_h or slp_h+qsi, qsi:{}, D:{}'.format(qsi, Dc))
            if Dc <0:
                Dc = 0
                # print('D less than zero, qsi:{}, D:{}'.format(qsi, Dc))
                # print("negative deposition!! n {}, qsi{}, ei {}, DL {}, Dc {}".format(n,qsi,ei,DL,Dc))
                # raise(ValueError)
            
            
        # print('slp_h = {}, zi = {}, qsi ={}, zo ={}, D ={} '.format(slp_h, zi, qsi, zo, Dc))
        return(Dc)

    
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
            
            if self.VaryDp:
                inpd = self._grid.at_node['particle__diameter'][n]
            else:
                inpd = self.Dp
            
            n_pd = (inpd* (self._grid.at_node['soil__thickness'][n]-E)+ 
            pd_in*D)/(D+self._grid.at_node['soil__thickness'][n]-E)
        
        else:
        
            n_pd = 0
            
        if (n_pd <0) or (np.isnan(n_pd)) or (np.isinf(n_pd)):
            msg = "node particle diameter is negative, nan or inf"
            # print("n_pd{}, pd_in{}, E{}, D{}, n{}".format(n_pd, pd_in, E, D, n))
            raise ValueError(msg)        
        return n_pd


    @staticmethod
    def _particle_diameter_out(pd_up,pd_in,qsi,E,D):
        """determine the weighted average particle diameter of the outgoing
        flow"""
        
        pd_out = np.sum((pd_up*E+pd_in*(qsi-D))/(qsi-D+E))
                
        if (pd_out <=0) or (np.isnan(pd_out)) or (np.isinf(pd_out)):
            msg = "out-flowing particle diameter is zero, negative, nan or inf"
            print("pd_up{}, pd_in{}, qsi{}, E{}, D{}".format(pd_up, pd_in, qsi, E, D))
            raise ValueError(msg)
        
        return pd_out