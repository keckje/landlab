# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from landlab import RasterModelGrid
from landlab import Component
from landlab.components import FlowDirectorMFD

class MassWastingRunout(Component):
    
    '''a cellular automata mass wasting runout model that routes an initial mass 
    wasting body (e.g., a landslide) through a watershed, determines erosion and 
    aggradation depths, evolves the terrain and regolith and tracks attributes of 
    the regolith. This model is intended for modeling the runout extent, topographic
    change and sediment transport caused by a mapped landslide(s) or landslides 
    inferred from a landslide hazard map. 
    
    
    Examples
    ----------

    
    References
    ----------
    Keck et al., (2023), submitted to Earth Surface Dynamics.
    
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
    mw_dict,
    tracked_attributes = None,
    deposition_rule = "critical_slope",
    dist_to_full_qsc_constraint = 0,
    itL = 1000,
    grain_shear = True,
    effective_qsi = True,
    settle_deposit = False,
    E_constraint = True,
    save = False,
    run_id = 0
    ):
        
        """
        
        Parameters
        ----------
        grid : landlab raster model grid
        
        mw_dict : dictionary
            a dictionary of parameters that control the behavoir of the model.
            The dicitonary must include the following keys: 'critical slope', 
            'threshold flux' and 'scour coefficient'.
            
               
                where: 
                    critical slope: list of floats
                        critical slope (angle of repose if no cohesion) of mass 
                        wasting material , L/L list of length 1 for a basin uniform Sc value
                        list of length 2 for hydraulic geometry defined Sc, where the first 
                        and second values in the list are the coefficient and exponent of a 
                        user defined function for crictical slope that varies with contributing
                        area [m2] to a node (e.g., [0.146,0.051] sets Sc<=0.1 at 
                        contributing area > ~1100 m2).                                            
                    threshold flux: float
                        minimum volumetric flux per unit contour width, (i.e., 
                        volume/(grid.dx*grid.dx)/iteration or L/iteration).
                        flux below this threshold stops at the cell as a deposit
                    erosion coefficient: float
                        coefficient that converts total shear stress [kPa] 
                        at the base of the runout material to a scour depth [m]
                        
                 The mw_dict dictionary can also include the following keys. Values listed next
                 to the key are default values.
                        mw_dict = {'critical slope': user_input,
                                   'threshold flux': user_input,
                                   'scour coefficient': user_input,
                                   'scour exponent, m': 0.5,
                                   'vol solids concentration': 0.6,
                                   'density solids': 2650,
                                   'typical flow thickness, scour': 3,
                                   'typical slope, scour': 0.4,
                                   'max observed flow depth': 4}
        
                         
        tracked_attributes : list or None
            A list of the attributes that will be tracked by the runout model. 
            Attributes in tracked_attributes must also be a field on the model grid
            and names in list must match the grid field names. Default is None.
         
        deposition_rule : str
            Can be either "critical_slope", "L_metric" or "both". 
            "critical_slope" is deposition rule describe in Keck et al. 2023.
            "L_metric" is varation of rule described by Cartier et al., 2016 and 
            Campforts et al. 2020.
            "both" uses the minimum value of both rules.
            
            All results in Keck et al. 2023 use "critical_slope". Default value is 
            "critical_slope".
            
        dist_to_full_qsc_constraint : float
            distance in meters at which qsc is applied to runout. If the landslide
            initiates on relatively flat terrain, it may be difficult to identify
            a qsc value that allows the model start and deposit in a way that matches
            the observed. In Keck et al. 2023, dist_to_full_qsc_constraint = 0, but
            other landslides may need dist_to_full_qsc_constraint = 20 to 50 meters.
         
        itL : int
            maximum number of iterations the model runs before it
            is forced to stop. The default is 1000. Ideally, if properly parameterized,
            the model should stop on its own. All modeled runout in Keck et al. 2023
            stopped on its own.
         
        grain_shear : boolean
            indicate whether to define shear stress at the base of the runout material
            as a function of grain size (Equation 13, True) or the depth-slope 
            approximation (Equation 12, False). Default is True.
            
        effective_qsi : boolean
            indicate wheter to limit the flux appled to erosion and aggradation rules
            to the maximum observed flow depth (i.e., the effective qsi). All results in
            Keck et al. 2023 use this constraint. Default is True.
            
        settle_deposit : boolean
            indicate whether to allow deposits to settle before the next model iteration
            is implemented. Settlement is determined the critical slope as evaluated from 
            the lowest adjacent node to the deposit. This is not used in Keck et al. 2023
            but tends to allow model to better reproduce smooth, evenly sloped deposits.
            Default is false.
         
        E_constraint : boolean
             indicate if erosion can occur simultaneously with aggradation. If True, if 
             aggradation > 0, then erosion = 0. This is True in Keck et al., 2023. 
             Default is True.
         
        save : boolean
            Save topographic elevation of watershed after each model iteration? 
            The default is False.
         
        run_id : float, int or str 
            label for landslide run, can be the time or some other identifier. This
            can be updated each time model is implemnted with "run_one_step"
        
        Returns
        -------
        None.
        
        """

        super().__init__(grid)

        self.mw_dict = mw_dict
        self.run_id = run_id
        self.itL = itL
        self.settle_deposit = settle_deposit
        self.grain_shear = grain_shear 
        self.deposition_rule = deposition_rule 
        self.dist_to_full_qsc_constraint = dist_to_full_qsc_constraint 
        self.effecitve_qsi = effective_qsi 
        self.E_constraint = E_constraint 
        self._tracked_attributes = tracked_attributes 
        self.save = save
        self.routing_partition_method = 'slope' # 'square_root_of_slope', see flow director       
        self.print_model_iteration_frequency = 20 # how often to pring to screen
        
        if tracked_attributes:
            self.track_attributes = True
            
            # check attributes are included in grid
            for key in self._tracked_attributes:
                if self._grid.has_field('node', key) == False:
                    raise ValueError("{} not included as field in grid".format(key))
                
            # if using grain size dependent erosion, check 
            # particle_diameter is included as an attribute
            if self.grain_shear == True:
                if 'particle__diameter' in self._tracked_attributes:
                    print(' running with spatially variable Dp ')
                else:
                    raise ValueError("{} not included as field in grid and/or key in tracked_attributes".format(key))
        else:
            self.track_attributes = False
      
        if len(self.mw_dict['critical slope'])>1: # change this to check for list input if defined as func of CA
            self.a = self.mw_dict['critical slope'][0]
            self.b = self.mw_dict['critical slope'][1]
        else:
            self.slpc = self.mw_dict['critical slope'][0]
        
        self.qsc = self.mw_dict['minimum flux']
        
        self.cs = self.mw_dict['scour coefficient']
        
        if 'scour exponent' in self.mw_dict:
            self.eta = self.mw_dict['scour exponent']
        else:
            self.eta = 0.2               
           
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

        # density of runout mixture
        self.ro_mw = self.vs*self.ros+(1-self.vs)*self.rof
        # distance equivalent iteration
        self.d_it = int(self.dist_to_full_qsc_constraint/self._grid.dx)
        
        # define initial topographic + mass wasting thickness topography
        self._grid.at_node['energy__elevation'] = self._grid.at_node['topographic__elevation'].copy()
        self._grid.at_node['topographic__initial_elevation'] = self._grid.at_node['topographic__elevation'].copy()
        
       
    def run_one_step(self, run_id):
        """route a list of debritons through a DEM and update
        the DEM based on the scour, entrainment and depostion depths at each 
        cell    
        
        Parameters
        ----------
        run_id : label for landslide run, can be a time stamp or some other identifier

        Returns
        -------
        None.

        """
        # set instance time stamp
        self.run_id = run_id
                   
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
               
        
        # data containers for saving model images and behavior statistics    
        self.arndn_r = {}
        if self.save:
            cL = {} 
            # lists and dictionaries that save values computed for the variables 
            # listed below, useful for checking model behavior
            self.EL = [] # entrainment depth / regolith depth
            self.AL = [] # aggradation (deposition) depth
            self.qsiL = [] # incoming flux (qsi)
            self.TauL = [] # basal shear stress 
            self.slopeL = [] # slope
            self.velocityL = [] # velocity (if any)
            self.arqso_r = {} # flux out
            self.arn_r = {} # receiver nodes
            self.aratt_r = {} #  arriving attributes
            
            # dictionaries, that save the entire model grid field, for all fields listed below
            # usefull for creating movies of how the flow and terrain evolve or checking
            # model behavior
            self.runout_evo_maps = {} # runout material + topographic__elevation
            self.topo_evo_maps = {}# topographic__elevation
            self.att_r = {} # attribute value
            self.st_r = {} # soil__thickness
            self.tss_r ={} # topographic__steepest_slope
            self.frn_r = {} # flow__receiver_node
            self.frp_r = {} # 'flow__receiver_proportions'
            self.arqso_r = {} # flux out
            self.arn_r = {} # receiver nodes
            self.aratt_r = {} #  arriving attributes
            
        # For each mass wasting event in list:
        for mw_i,inn in enumerate(innL):
                         
            mw_id = self.mw_ids[mw_i]
            self._lsvol = self._grid.at_node['soil__thickness'][inn].sum()*self._grid.dx*self._grid.dy
            
            if self.qsi_max == None:
                self.qsi_max = self._grid.at_node['soil__thickness'][inn].max()
            
            # prep data containers for landslide mw_id
            self.arndn_r[mw_id] = []
            if self.save:
                cL[mw_i] = []
                self.runout_evo_maps[mw_i] = {}
                self.topo_evo_maps[mw_i] = {}
                self.DEMdfD = {}
                self.st_r[mw_id] = []
                self.tss_r[mw_id] = []
                self.frn_r[mw_id] = []
                self.frp_r[mw_id] = []            
                self.arqso_r[mw_id] = []
                self.arn_r[mw_id] = []
                if self.track_attributes:
                    self.att_r[mw_id] = dict.fromkeys(self._tracked_attributes, []) # this becomes the data container for each attribute
                    self.aratt_r[mw_id] = dict.fromkeys(self._tracked_attributes, [])  
                    
            
            # prepare initial mass wasting material (precipitons) for release 
            self._prep_initial_mass_wasting_material(inn, mw_i)

            self.arndn_r[mw_id].append(self.arndn)
            if self.save:
                # save first set of data
                self.runout_evo_maps[mw_i][0] = self._grid.at_node['energy__elevation'].copy()
                self.topo_evo_maps[mw_i][0] = self._grid.at_node['topographic__elevation'].copy()
                self.DEMdfD[0] = {'DEMdf_r':0}            
                if self.track_attributes:
                    for key in self._tracked_attributes:
                        self.att_r[mw_id][key].append(self._grid.at_node[key].copy()) # for each attribute, a copy of entire grid
                        self.aratt_r[mw_id][key].append(self.aratt) 
                self.st_r[mw_id].append(self._grid.at_node['soil__thickness'].copy())
                self.tss_r[mw_id].append(self._grid.at_node['topographic__steepest_slope'].copy())
                self.frn_r[mw_id].append(self._grid.at_node['flow__receiver_node'].copy())
                self.frp_r[mw_id].append(self._grid.at_node['flow__receiver_proportions'].copy())
                self.arqso_r[mw_id].append(self.arqso)
                self.arn_r[mw_id].append(self.arn)
                   

            # now loop through each receiving node in list rni, 
            # determine next set of recieving nodes
            # repeat until no more receiving nodes (material deposits)             
            c = 0 # model iteration counter
            
            while len(self.arn)>0 and c < self.itL:

                self.c = c
                if self.d_it == 0:
                    self.qsc_v = self.qsc
                else:
                    self.qsc_v = self.qsc*(min(c/self.d_it,1))

                # temporary data containers for storing receiving node, flux and attributes
                # these become the input for the next iteration
                self.arndn_ns = np.array([]) # next iteration donor nodes
                self.arn_ns = np.array([]) # next iteration receiver nodes
                self.arqso_ns = np.array([]) # next iteration flux to receiver nodes
                self.arnL = [] # list of receiver nodes
                self.arqsoL = [] # list of flux out
                self.arndnL = [] # list of donor nodes
                if self.track_attributes:
                    self.aratt_ns = dict.fromkeys(self._tracked_attributes, np.array([])) #
                    self.arattL = dict.fromkeys(self._tracked_attributes, [])
                
                # for each unique node in receiving node list self.arn
                self.arn_u = np.unique(self.arn).astype(int)  # unique arn list
                
                # determine the incoming flux to each node in self.arn_u
                self.qsi_dat = self._determine_qsi() ##
                
                # update energy elevation as node elevation plus incoming flow thickness
                # this happens even if using topographic__elevation to route so that 
                # the thickness of the debris flow is tracked for plotting
                self._update_E_dem()
                
                # determine scour, entrain, deposition and outflow depths and
                # the outflowing particle diameter, arranged in array nudat
                self.nudat = self._E_A_qso_determine_attributes()                

                # determine the receiving nodes, fluxes and grain size
                self._determine_rn_proportions_attributes()                               
                
                # update grid field: topographic__elevation with the values in 
                # nudat. Do this after directing flow, because assume deposition 
                # does not impact flow direction
                self._update_dem()
               
                ### update topographic slope field for deposition to detemine where
                # settling will occur - move this innto settle_deposit
                self._update_topographic_slope()  

                ### update attribute grid fields: particle__diameter with the values in 
                # nudat
                if self._tracked_attributes:
                    for key in self._tracked_attributes:
                        self._update_attribute_at_node(key)
                              
                self.dif  = self._grid.at_node['topographic__elevation']-self._grid.at_node['topographic__initial_elevation']
                
                if self.settle_deposit:             
                    self._settle()
                    self._update_topographic_slope()

                # once all cells in iteration have been evaluated, temporary receiving
                # temporary node, node flux and node attributes stored for
                # for next iteration
                self.arndn = self.arndn_ns.astype(int)
                self.arn = self.arn_ns.astype(int)
                self.arqso = self.arqso_ns #
                if self.track_attributes:
                    self.aratt = self.aratt_ns

                if self.save:

                    cL[mw_i].append(c)
                    
                    DEMf = self._grid.at_node['topographic__elevation'].copy()
                    
                    DEMdf_r = DEMf-self._grid.at_node['topographic__initial_elevation']
            
                    self.DEMdfD[c+1] = {'DEMdf_r':DEMdf_r.sum()*self._grid.dx*self._grid.dy}     
                    self.runout_evo_maps[mw_i][c+1] = self._grid.at_node['energy__elevation'].copy()
                    ### save maps for video
                    self.runout_evo_maps[mw_i][c+1] = self._grid.at_node['energy__elevation'].copy()
                    self.topo_evo_maps[mw_i][c+1] = self._grid.at_node['topographic__elevation'].copy()
                    # data for tests
                    if self.track_attributes:
                        for key in self._tracked_attributes:
                            self.att_r[mw_id][key].append(self._grid.at_node[key].copy())
                            self.aratt_r[mw_id][key].append(self.arattL) 
                    self.st_r[mw_id].append(self._grid.at_node['soil__thickness'].copy())
                    self.tss_r[mw_id].append(self._grid.at_node['topographic__steepest_slope'].copy())
                    self.frn_r[mw_id].append(self._grid.at_node['flow__receiver_node'].copy())
                    self.frp_r[mw_id].append(self._grid.at_node['flow__receiver_proportions'].copy())
                    self.arqso_r[mw_id].append(self.arqsoL)
                    self.arn_r[mw_id].append(self.arnL)                          
                    self.arndn_r[mw_id].append(self.arndn)

                # update iteration counter
                c+=1
                
                if c%self.print_model_iteration_frequency == 0:
                    print(c)  
                                            
    def _prep_initial_mass_wasting_material(self, inn, mw_i):
        """ Algorithm 1 - from an initial source area (landslide), prepare the 
        initial lists of receiving nodes and incoming fluxes and attributes 
        and remove the source material from the topographic DEM
        
        Parameters
        ----------
        inn: np.array 
             node id's that make up the area of the initial mass wasting area
        mw_i: int
            index of the initial mass wasting area (e.g., if there are two landslides
                                                    the first landslide will be mw_i = 0,
                                                    the second will be mw_i = 0)
        """
        # data containers for initial recieving node, outgoing flux and attributes
        rni = np.array([])
        rqsoi = np.array([])
        rpdi = np.array([])
        if self._tracked_attributes:
           att = dict.fromkeys(self._tracked_attributes, np.array([]))
        
        # order source area nodes from lowest to highest elevation
        node_z = self._grid.at_node.dataset['topographic__elevation'][inn]
        zdf = pd.DataFrame({'nodes':inn,'z':node_z})
        zdf = zdf.sort_values('z')            
        
        for ci, ni in enumerate(zdf['nodes'].values):
            
            # regolith (soil) thickness at node. soil thickness in source area
            # represents landslide thickness
            s_t = self._grid.at_node.dataset['soil__thickness'].values[ni]
            
            # remove soil (landslide) thickness at node
            self._grid.at_node.dataset['topographic__elevation'][ni] =  (                   
            self._grid.at_node.dataset['topographic__elevation'][ni] - s_t)
            
            # update soil thickness at node (now = 0)
            self._grid.at_node['soil__thickness'][ni] = (
                self._grid.at_node['soil__thickness'][ni]-s_t)
            
            if ci>0: # use surface slope for first node to start movement of landslide 
                # for all other nodes, update slope to reflect material removed from DEM
                self._update_topographic_slope()
             
            # get receiving nodes of node ni in mw index mw_i
            rn = self._grid.at_node.dataset['flow__receiver_node'].values[ni]
            rn = rn[np.where(rn != -1)]
            
            # receiving proportion of qso from cell n to each downslope cell
            rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[ni]
            rp = rp[np.where(rp > 0)] # only downslope cells considered

            # initial mass wasting thickness
            imw_t =s_t
            # get flux out of node ni   
            qso = imw_t
            # divide into proportions going to each receiving node
            rqso = rp*qso
            
            if self._tracked_attributes:
                # get initial mass wasting attributes moving (out) of node ni
                att_out = {}
                self.att_ar_out = {}
                for key in self._tracked_attributes:
                    att_val = self._grid.at_node.dataset[key].values[ni]

                    # particle diameter to each recieving node
                    self.att_ar_out[key] = np.ones(len(rqso))*att_val

                    att[key] = np.concatenate((att[key],self.att_ar_out[key]), axis = 0) 
       
            # append receiving node ids, fluxes and attributes to initial lists
            rni = np.concatenate((rni,rn), axis = 0) 
            rqsoi = np.concatenate((rqsoi,rqso), axis = 0) 
        
        self.arndn = np.ones([len(rni)])*np.nan # TODO: set this to node id
        self.arn = rni
        self.arqso = rqsoi
        if self._tracked_attributes:
            self.aratt = att

        
    def _E_A_qso_determine_attributes(self):
        """ Algorithm 2 - mass conservation at a grid cell, implemented using 
        the EAqA function below.
        """
        
        # list of deposition depths, used in settlement algorithm
        self.D_L = []
        def EAqA(qsi_r):
            """function for iteratively determining Erosion and Aggradiation depths, 
            the outgoing flux (qso) and Attribute values at each affected node and 
            attribute values of the outgoing flux.
            
            thoughts for speeding this up: use cython or try to restructure 
            EAqA function so that it can be applied with vector operations, 
            then filter all changes to correct cells where computations 
            should not have occurred
            
            Parameters
            ----------
            qsi_r: np array
                a row of the self.qsi_dat array

            """            

            n = qsi_r[0]; qsi = qsi_r[1]; 
            
            # maximum slope at node n
            slpn = self._grid.at_node['topographic__steepest_slope'][n].max()
            
            # proportion of flow in steepest direction
            pn = 1#self._grid.at_node['flow__receiver_proportions'][n].max() 
            
            if self.effecitve_qsi:
                qsi_ = min(qsi*pn,self.qsi_max)
            else:
                qsi_ = qsi*pn

            dn = self.arndn[self.arn == n]

            # get adjacent nodes
            adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
            self._grid.diagonal_adjacent_nodes_at_node[n]))
            
            # look up critical slope at node n
            if len(self.mw_dict['critical slope'])>1: # if option 1, critical slope is not constant but depends on location
                self.slpc = self.a*self._grid.at_node['drainage_area'][n]**self.b                      
            
            # incoming attributes (weighted average)
            if self._tracked_attributes:
                att_in = self._attributes_in(n,qsi)
            else:
                att_in = None 
           
            rn_g = self._grid.at_node.dataset['flow__receiver_node'].values[n]
            rn_g = rn_g[np.where(rn_g != -1)]

            # if qsi less than the vegetation qsc in an undisturbed cell
            if qsi <=(self.qsc_v):
                A = qsi 
                qso = 0
                E = 0 
                deta = A 
                if self._tracked_attributes:
                    att_up = dict.fromkeys(self._tracked_attributes, 0)
                else:
                    att_up = None
                Tau = 0
                u = 0
 
            else:
                A = min(qsi, self._aggradation(qsi_, slpn, n)) # function of qsi and topographic elevation before settle/scour by qsi                

                # erosion a function of steepes topographic slope at node, determined before settle/scour by qsi
                
                if A > 0 and self.E_constraint:
                    E = 0
                    if self._tracked_attributes:
                        att_up = dict.fromkeys(self._tracked_attributes, 0)
                    else:
                        att_up = None
                else:
                    if self.grain_shear:   
                        opt = 2
                    else:
                        opt = 1  
                    E, att_up, Tau, u = self._erosion(n, qsi_, slpn, att_in = att_in)   
                    # model behavior tracking
                    if self.save:
                        self.TauL.append(Tau)
                        self.velocityL.append(u) # velocity (if any)

                ## flow out
                qso = qsi-A+E                
                # small qso are considered zero
                qso  = np.round(qso,decimals = 8)
                # chage elevation
                deta = A-E 
            
            # model behavior tracking
            if self.save:
                self.EL.append(E)
                self.AL.append(A)
                self.qsiL.append(qsi)
                self.slopeL.append(slpn) # slope

            # updated node particle diameter (weighted average)
            if self._tracked_attributes:
                n_att = self._attributes_node(n,att_in,E,A)
            else:
                n_att = None
            
            # list of deposition depths at cells in iteration 
            self.D_L.append(A)
            
            # n_att, att_up, att_in are dictionaries of values of each attribute (keys of dictionary)            
            return deta, qso, qsi, E, A, n_att, att_up, att_in
        
        # apply EAqA function to all unique nodes in arn (arn_u)
        # create nudat, an np.array of data for updating fields at each node
        # that can be applied using vector operations
        ll=np.array([EAqA(r) for r in self.qsi_dat],dtype=object)     
        arn_ur = np.reshape(self.qsi_dat[:,0],(-1,1))
        nudat = np.concatenate((arn_ur,ll),axis=1)

        return nudat # qso, rn not used


    def _determine_rn_proportions_attributes(self):
        """ determine how outgoing flux is partioned to downslope cells and 
        attributes of each parition"""
        
        def rn_proportions_attributes(nudat_r):
            n = nudat_r[0]; qso = nudat_r[2];
            qsi = nudat_r[3]; E = nudat_r[4]; A = nudat_r[5]
            
            att_up = nudat_r[7]; att_in = nudat_r[8]
            
            # get donor node ids
            dn = self.arndn[self.arn == n]

            rn = self._grid.at_node.dataset['flow__receiver_node'].values[n]
            rn_ = rn.copy()
            rn_ = rn_[np.where(rn_ != -1)]
            rn_ = rn_[~np.isin(rn_,dn)]
            
            
            if qso>0 and n not in self._grid.boundary_nodes: 
                # move this out of funciton, need to determine rn and rp after material is in cell, not before

                # receiving proportion of qso from cell n to each downslope cell
                if len(rn_)<1:
                    rp = np.array([1])
                    rn = np.array([n])
                    rqso = rp*qso
                else:
                    rp = self._grid.at_node.dataset['flow__receiver_proportions'].values[n]
                    rp = rp[np.isin(rn,rn_)]
                    rp = rp/rp.sum()
                    rn = rn_
                    rqso = rp*qso

                # delivery node to the receiver nodes (length equal to number of receiver nodes)
                rndn = (np.ones(len(rn))*n).astype(int) 

                # particle diameter out (weighted average)
                if self._tracked_attributes:
                    att_out = self._attribute_out(att_up,att_in,qsi,E,A) 
                    
                    rpd_ns = {}
                    for key_n, key in enumerate(self._tracked_attributes):
                        ratt = np.ones(len(rqso))*att_out[key]
                        self.aratt_ns[key] = np.concatenate((self.aratt_ns[key], ratt), axis = 0) # next step receiving node incoming particle diameter list
                        self.arattL[key].append(ratt)
                        
                # store receiving nodes and fluxes in temporary arrays
                self.arndn_ns = np.concatenate((self.arndn_ns, rndn), axis = 0) # next iteration delivery nodes
                self.arn_ns = np.concatenate((self.arn_ns, rn), axis = 0) # next iteration receiving nodes
                self.arqso_ns = np.concatenate((self.arqso_ns, rqso), axis = 0) # next iteration qsi
                self.arnL.append(rn)
                self.arqsoL.append(rqso)           
                self.arndnL.append(rndn)

        nudat_ = self.nudat[self.nudat[:,2]>0] # only run on nodes with qso>0
        ll = np.array([rn_proportions_attributes(r) for r in self.nudat],dtype=object)


    def _determine_qsi(self):
        """determine flux of incoming material (qsi) to a node.
        returns self.qsi_dat: np array of receiving nodes [column 0], 
        and qsi to those nodes [column 1]
        """
        def _qsi(n):           
            """ sum the incoming flux to node n"""
            qsi = np.sum(self.arqso[self.arn == n])
            return qsi

        ll=np.array([_qsi(n) for n in self.arn_u], dtype=object)
        ll = np.reshape(ll,(-1,1))
        arn_ur = np.reshape(self.arn_u,(-1,1))
        qsi_dat = np.concatenate((arn_ur,ll),axis=1)
        return qsi_dat
    
    
    def _update_E_dem(self):
        """update energy__elevation"""
        n = self.qsi_dat[:,0].astype(int); qsi = self.qsi_dat[:,1];         
        # energy elevation is equal to the topographic elevation plus qsi
        self._grid.at_node['energy__elevation'] = self._grid.at_node['topographic__elevation'].copy()
        self._grid.at_node['energy__elevation'][n] = self._grid.at_node['energy__elevation'].copy()[n]+qsi


    def _update_energy_slope(self):
        """updates the topographic__slope and flow directions grid fields using the 
        energy__elevation field. This function is presently not used but may be useful
        for future implementations of MWR"""
        fd = FlowDirectorMFD(self._grid, surface="energy__elevation", diagonals=True,
                partition_method = self.routing_partition_method)
        fd.run_one_step()

        
    def _update_dem(self):
        """updates the topographic elevation of the landscape dem and soil 
        thickness fields"""
        n = self.nudat[:,0].astype(int); deta = self.nudat[:,1]
        self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]+deta   
        self._grid.at_node['topographic__elevation'][n] = self._grid.at_node['topographic__elevation'][n]+deta


    def _update_topographic_slope(self):
        """updates the topographic__slope and flow directions fields using the 
        topographic__elevation field"""
        fd = FlowDirectorMFD(self._grid, surface="topographic__elevation", diagonals=True,
                partition_method = self.routing_partition_method)
        fd.run_one_step()

    
    def _update_attribute_at_node(self, key):
        """ for each unique node in receiving node list, update the attribute
        using attribute value determined in the _E_A_qso_determine_attributes method

        Parameters
        ----------
        key: string
            one of the tracked attributes        
        """
        n = self.nudat[:,0].astype(int); 
        new_node_pd = np.array([d[key] for d in  self.nudat[:,6]])
        if np.isnan(np.sum(new_node_pd)):
            raise ValueError("{} is {}".format(key, new_node_pd))
        self._grid.at_node[key][n] = new_node_pd 
                
         
    def _settle(self):
        """ for each unique node in receiving node list, after erosion, aggradation 
        and change in node elevation have been determined, check that the height of the node 
        is not greater than permitted by angle of repose/critical slope as evaluated from 
        the lowest cell. Note, slope is not updated in this function. It is updated
        simultaneously at a later stage during the iteration. 
        """
        for ii, n in enumerate(self.arn_u): # for each node in the list, use the slope field, computed from the previous iteration, to compute settlment and settlment direction to adjacent cells
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

    
                # if slope to downlsope nodes > Sc, adjust elevation of node n
                if len(rn)>=1:
                    
                    # destribute material to downslope nodesbased on weighted
                    # average slope (same as multiflow direciton proportions, 
                    # but here only determined for downslope nodes in which
                    # S  > Sc )
                             
                    sslp = sum(slpn)
                    pp = slpn/sslp
                    
                    # determine the total flux / unit width sent to S > Sc downslope cells
                    # mean/min downslope cell elevation
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

                    # update the topographic elevation
                    self._grid.at_node['topographic__elevation'][n]=self._grid.at_node['topographic__elevation'][n]-qso_s
                    self._grid.at_node['topographic__elevation'][rn]=self._grid.at_node['topographic__elevation'][rn]+qso_s_i
                    
                    # update the soil thickness
                    self._grid.at_node['soil__thickness'][n] = self._grid.at_node['soil__thickness'][n]-qso_s
                    self._grid.at_node['soil__thickness'][rn] = self._grid.at_node['soil__thickness'][rn]+qso_s_i
                    
                    # update tracked attributes for sediment movement during settlement
                    for key in self._tracked_attributes:
                        pd_ = self._grid.at_node[key][n]                    
                        for v, n_ in enumerate(rn):
                            A = qso_s_i[v]
                            self._grid.at_node[key][n_] = self._attributes_node(n_,pd_,0,A)
    

    def _erosion(self, n, depth, slpn, att_in = None):
        """determines the erosion depth based on user selected method.
        
        Parameters
        ----------
        n : int
            node id
        depth : float
            erosion depth
        slpn : float
            slope in [l/L] 
        att_in: dict
            dictionary of the value of each attribute, this function
            only uses particle__diameter
            
        Returns
        -------
        E : float
            erosion depth [L]
        att_up : dict
            dictionary of the value of each attribute at node n
        Tau : float
            basal shear stress [Pa]
        u : float
            flow velocity [m/s]
        """
        theta = np.arctan(slpn) # convert tan(theta) to theta
        # particle size of scoured material
        if self._tracked_attributes:
            att_up = {}
            for key in self._tracked_attributes:
                att_up[key] = self._grid.at_node[key][n]
        else:
            att_up = None

        
        if self.grain_shear:
            Dp = att_in['particle__diameter']
            if depth < Dp: # grain size dependent erosion breaks if depth<Dp
                Dp = depth*.99
            # shear stress apprixmated as a power functino of inertial shear stress
            # phi = np.arctan(self.slpc) # approximate phi [radians] from criticle slope [L/L]
            phi = np.arctan(0.32)
            
            # inertial stresses
            us = (self.g*depth*slpn)**0.5
            u = us*5.75*np.log10(depth/Dp)
            
            dudz = u/depth
            Tcn = np.cos(theta)*self.vs*self.ros*(Dp**2)*(dudz**2)
            Tau = Tcn*np.tan(phi)

            Ec = (self.cs*Tau**self.eta)      

        else:
            # following Frank et al., 2015, approximate erosion depth as a linear
            # function of total stress under uniform flow conditions
            
            # erosion depth,:
            Tau = self.ro_mw*self.g*depth*(np.sin(theta))
            Ec = self.cs*(Tau)**self.eta
            u = np.nan
        
        dmx = self._grid.at_node['soil__thickness'][n]
                
        E = min(dmx, Ec) # convert Tb to kPa
        
        return(E, att_up, Tau, u)

                
    def _aggradation(self,qsi,slpn,n):
        """determine deposition depth as the minimum of L*qsi and/or
        a function of a critical-slope. Where the L metric is computed following
        Campforts, et al., 2020 but is expressed as 1-(slpn/slpc)**2 rather 
        than dx/(1-(slpn/slpc)**2)
        
        Parameters
        ----------
        qsi : float
            incoming flux [l3/iteration/l2]
        slpn : float
            slope at node in [L/L]
        n : int
            node id
            
        Returns
        -------
        A : float
            aggradation depth [L]
        """

        if self.deposition_rule == "L_metric":
            A = self._deposit_L_metric(qsi, slpn)
        elif self.deposition_rule == "critical_slope":
            A = self._deposit_friction_angle(qsi, n)
        elif self.deposition_rule == "both":
            A_L = self._deposit_L_metric(qsi, slpn)
            A_f = self._deposit_friction_angle(qsi, n) 
            A = min(A_L,A_f)

        return(A)


    def _determine_zo(self, n, zi, qsi):
        """determine the minimum elevation of the adjacent nodes. If all adjacent
        nodes are higher than the elevation of the node + qsi, zo is set to zi
        
        Parameters
        ----------
        n : int
            node id
        zi : float
            topographic elevation at node n (eta_n)
        qsi : float
            incoming flux [l3/iteration/l2]
            
        Returns
        -------
        zo : float
            topographic elevation of the lowest elevation node [l], 
            adjacent to node n      
        """
        
        # get adjacent nodes
        adj_n = np.hstack((self._grid.adjacent_nodes_at_node[n],
        self._grid.diagonal_adjacent_nodes_at_node[n]))
        
        # exclude closed boundary nodes               
        adj_n = adj_n[~np.isin(adj_n,self._grid.closed_boundary_nodes)]
                        
        # elevation of flow surface at node... may not need this
        ei = qsi+zi
        
        # nodes below elevation of node n
        rn_e = adj_n[self._grid.at_node['topographic__elevation'][adj_n]<ei]
 
        if len(rn_e) > 0: 
            zo = self._grid.at_node['topographic__elevation'][rn_e].min()
            
        else:  # an obstruction in the DEM
            zo = zi 
        return zo


    def _deposit_L_metric(self, qsi, slpn):
        """
        determine the L metric similar to Campforts et al. (2020)
        
        Parameters
        ----------
        qsi : float
            in coming flux per unit contour width
        slpn : float
            slope of node, measured in downslope direction (downslope is postive)
            
        Returns
        -------
        A_L : float
            aggradation depth [L]        
        """
        Lnum = np.max([(1-(slpn/self.slpc)**2),0])
        A_L = qsi*Lnum
        
        return(A_L)
    

    def _deposit_friction_angle(self, qsi, n):
        """ determine deposition depth following equations 4 though 9
        
        Parameters
        ----------
        qsi : float
            incoming flux [l3/iteration/l2]
        n : int
            node id
            
        Returns
        -------
        A_f : float
            aggradation depth [L]           
        """         
        slp_h = self.slpc*self._grid.dx
        zi = self._grid.at_node['topographic__elevation'][n]       
        zo = self._determine_zo(n, zi, qsi )
        rule = ((zi-zo)<=(slp_h))
        def eq(qsi, zo, zi, slp_h):
            dx = self._grid.dx
            sc = self.slpc
            s = (zi-zo)/dx
            sd = sc-s              
            D1 = sc*dx/2
            a = 0.5*dx*sd
            b = D1-0.5*dx*sd
            c = -qsi
            N1 = -b+(((b**2)-4*a*c)**0.5)/(2*a)
            N2 = -b-(((b**2)-4*a*c)**0.5)/(2*a)
            ndn = np.round(max([N1,N2,1]))
            A = min((1/ndn)*qsi+((ndn-1)/2)*dx*sd, qsi)
            return A 

        if rule:
            A_f = eq(qsi,zo,zi,slp_h)
        else:
            A_f = 0
        
        if A_f <0:
            print("negative deposition!! n {}, qsi{}, D {}, zo{}, zi{}".format(n,qsi,Dc,zo,zi))
            # raise(ValueError)            

        return(A_f)

        
    def _attributes_in(self,n,qsi):
        """determine the weighted average attribute value of the incoming
        flow
        
        Parameters
        ----------
        n : int
            node id
        qsi : float
            incoming flux [l3/iteration/l2]
        
        Returns
        -------
        att_in: dict
            dictionary of each attribute value flowing into the node      
        """       
        if (qsi == 0):
            att_in = dict.fromkeys(self._tracked_attributes, 0)
        elif (np.isnan(qsi)) or (np.isinf(qsi)):
            msg = "in-flowing flux is nan or inf"
            raise ValueError(msg)
        else:
            att_in = {}
            for key in self._tracked_attributes:
                att_in[key] = np.sum((self.aratt[key][self.arn == n])*(self.arqso[self.arn == n])/qsi)        
        return att_in


    def _attributes_node(self,n,att_in,E,A):
        """determine the weighted average attributes of the newly aggraded material
        + the inplace regolith
        
        Parameters
        ----------
        n : int
            node id       
        att_in: dict
            dictionary of the value of each attribute flowing into the node 
        E : float
            erosion depth [L]
        A : float
            aggradation depth [L] 

        Returns
        -------
        n_att_d: dict
            dictionary of each attribute value at the node after erosion
            and aggradation
        """
        def weighted_avg_at_node(key):
            if (A+self._grid.at_node['soil__thickness'][n]-E > 0):                
                inatt = self._grid.at_node[key][n] # attribute value at node                
                n_att = (inatt* (self._grid.at_node['soil__thickness'][n]-E)+ 
                att_in[key]*A)/(A+self._grid.at_node['soil__thickness'][n]-E)            
            else:            
                n_att = 0                
            if (n_att <0) or (np.isnan(n_att)) or (np.isinf(n_att)):
                msg = "node particle diameter is negative, nan or inf"
                raise ValueError(msg)                      
            return n_att
 
        n_att_d = {}
        for key in self._tracked_attributes:
            n_att_d[key] = weighted_avg_at_node(key)
        
        return n_att_d


    def _attribute_out(self,att_up,att_in,qsi,E,A):
        """determine the weighted average attributes of the outgoing
        flux
        
        Parameters
        ----------
        att_up: dict
            dictionary of each attribute value at the node before erosion
            or aggradation
        att_in: dict
            dictionary of each attribute value flowing into the node
        qsi : float
            incoming flux [l3/iteration/l2]        
        E : float
            erosion depth [L]
        A : float
            aggradation depth [L]    
            
        Returns
        -------
        att_out: dict
            dictionary of each attribute value flowing out of the node
        """
        att_out = {}
        for key in self._tracked_attributes:
            att_out[key] = np.sum((att_up[key]*E+att_in[key]*(qsi-A))/(qsi-A+E))
            check_val = att_out[key]    
            if (check_val <=0) or (np.isnan(check_val)) or (np.isinf(check_val)):
                msg = "out-flowing particle {} is zero, negative, nan or inf".format(key)
                raise ValueError(msg)
        return att_out
