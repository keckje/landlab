import warnings

import numpy as np
import pandas as pd

from landlab import Component
from landlab.components import FlowDirectorMFD
from landlab.components.mass_wasting_runout.mass_wasting_saver import MassWastingSaver

##TODO
# fix aggradation function so that it donest spit out a warning, check that it is applied corectly, are there nan values from the function?
# speed up _determine_rn_proportions_attributes_v
# reduce number of np concat calls
# clean up


class MassWastingRunout(Component):
    """a cellular-automata mass wasting runout model that routes an initial mass
    wasting body (e.g., a landslide) through a watershed, determines erosion and
    aggradation depths, evolves the terrain and regolith and tracks attributes of
    the regolith. This model is intended for modeling the runout extent, topographic
    change and sediment transport caused by a mapped landslide(s) or landslides
    inferred from a landslide hazard map.


    Examples
    ----------
    Import necessary packages and components

    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import FlowDirectorMFD
    >>> from landlab.components.mass_wasting_runout import MassWastingRunout

    Define the topographic__elevation field of a 7 columns by 7 rows, 10-meter
    raster model grid

    >>> dem = np.array(
    ...     [
    ...         [10, 8, 4, 3, 4, 7.5, 10],
    ...         [10, 9, 3.5, 4, 5, 8, 10],
    ...         [10, 9, 6.5, 5, 6, 8, 10],
    ...         [10, 9.5, 7, 6, 7, 9, 10],
    ...         [10, 10, 9.5, 8, 9, 9.5, 10],
    ...         [10, 10, 10, 10, 10, 10, 10],
    ...         [10, 10, 10, 10, 10, 10, 10],
    ...     ]
    ... )

    >>> dem = np.hstack(dem).astype(float)
    >>> mg = RasterModelGrid((7, 7), 10)
    >>> _ = mg.add_field("topographic__elevation", dem, at="node")

    Define boundary conditions

    >>> mg.set_closed_boundaries_at_grid_edges(True, True, True, True)

    Add multiflow direction fields, soil thickness (here set to 1 meter)

    >>> fd = FlowDirectorMFD(mg, diagonals=True, partition_method="slope")
    >>> fd.run_one_step()
    >>> nn = mg.number_of_nodes
    >>> depth = np.ones(nn) * 1
    >>> _ = mg.add_field("soil__thickness", depth, at="node")

    Define the initial landslide. Any mass_wasting_id value >1 is considered a
    landslide. The landslide extent is defined by assigining all nodes withing
    the landslide the same mass_wasting_id value.
    Here, the landslide is represented by a single node (node 38), which assigned
    a mass_wasting_id value of 1:

    >>> mg.at_node["mass__wasting_id"] = np.zeros(nn).astype(int)
    >>> mg.at_node["mass__wasting_id"][np.array([38])] = 1

    Add attributes of the regolith as fields of the raster model grid that will
    be tracked by the model. These could be any attribute in which the tracking
    method used by MassWastingRunout reasonably represents movement of the
    attribute. Here we track the particle diameter and organic content
    of the regolith. Note, a particle__diameter field is required if shear stress
    is determined as a function of grain size.

    >>> np.random.seed(seed=7)
    >>> mg.at_node["particle__diameter"] = np.random.uniform(0.05, 0.25, nn)
    >>> mg.at_node["organic__content"] = np.random.uniform(0.01, 0.10, nn)

    Next define parameter values for MassWastingRunout and instantiate the model:

    >>> Sc = [0.03]  # Sc, note: defined as a list (see below)
    >>> qsc = 0.01  # qsc
    >>> k = 0.02  # k
    >>> h_max = 1
    >>> tracked_attributes = ["particle__diameter", "organic__content"]
    >>> example_square_MWR = MassWastingRunout(
    ...     mg,
    ...     critical_slope=Sc,
    ...     threshold_flux=qsc,
    ...     erosion_coefficient=k,
    ...     tracked_attributes=tracked_attributes,
    ...     effective_qsi=True,
    ...     max_flow_depth_observed_in_field=h_max,
    ...     save=True,
    ... )

    Run MassWastingRunout

    >>> example_square_MWR.run_one_step()

    By subtracting the initial DEM from the final DEM, which has evolvod as
    a consequence of the runout, we can see areas of aggradation (positive values)
    and erosion (negative values). Nodes with non-zero topographic change
    represent the runout extent.

    >>> DEM_initial = mg.at_node["topographic__initial_elevation"]
    >>> DEM_final = mg.at_node["topographic__elevation"]
    >>> DEM_final - DEM_initial
    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.96579201,
            0.52734339, -0.00778869,  0.        ,  0.        ,  0.        ,
            0.        , -0.00594927, -0.12261762, -0.0027898 ,  0.        ,
            0.        ,  0.        ,  0.        , -0.04562554, -0.10973222,
           -0.05776526,  0.        ,  0.        ,  0.        ,  0.        ,
           -0.01225359, -0.07973101, -0.04888238,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        , -1.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ])

    See how the landslide removes all of the regolith at node 38 (the negative -1)

    Look at the final spatial distribution of regolith particle diameter.

    >>> mg.at_node["particle__diameter"]
    array([0.06526166, 0.20598376, 0.13768185, 0.19469304, 0.2455979 ,
           0.15769917, 0.15022409, 0.06441023, 0.1036878 , 0.15619144,
           0.17680799, 0.21074781, 0.12618823, 0.06318727, 0.10762912,
           0.23191871, 0.09295928, 0.14042479, 0.23545374, 0.05497985,
           0.17010978, 0.2400259 , 0.09606058, 0.15969798, 0.23182567,
           0.07663389, 0.15468252, 0.20008197, 0.18380265, 0.14355057,
           0.09096982, 0.14815318, 0.12447694, 0.14548023, 0.12317808,
           0.2175836 , 0.2037295 , 0.11279894, 0.        , 0.10520981,
           0.14056859, 0.12059567, 0.18147989, 0.12407022, 0.1418186 ,
           0.19386482, 0.13259837, 0.23128465, 0.08609032])

    Also note that the attribute value is set to zero at any node in which the regolith
    depth is 0.


    References
    ----------
    Keck, J., Istanbulluoglu, E., Campforts, B., Tucker G., Horner-Devine A.,
    A landslide runout model for sediment transport, landscape evolution and hazard
    assessment applications, submitted to Earth Surface Dynamics (2023)

    """

    _name = "MassWastingRunout"

    _unit_agnostic = False

    _info = {
        "mass__wasting_id": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "interger or float id of each mass wasting area is assigned \
                to all nodes representing the mass wasting area.",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "soil__thickness": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "soil depth to restrictive layer",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__receiver_proportions": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of proportion of flow sent to each receiver.",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "particle__diameter": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "representative particle diameter at each node, this might \
            vary with underlying geology, contributing area or field observations",
        },
    }

    def __init__(
        self,
        grid,
        critical_slope=0.05,
        threshold_flux=0.25,
        erosion_coefficient=0.005,
        tracked_attributes=None,
        deposition_rule="critical_slope",
        grain_shear=True,
        effective_qsi=False,
        settle_deposit=False,
        E_constraint=True,
        save=False,
        typical_flow_thickness_of_erosion_zone=2,
        typical_slope_of_erosion_zone=0.15,
        erosion_exponent=0.2,
        max_flow_depth_observed_in_field=None,
        vol_solids_concentration=0.6,
        density_solids=2650,
        density_fluid=1000,
        gravity=9.81,
        dist_to_full_qsc_constraint=0,
        itL=1000,
        run_id=0,
    ):
        """
        Parameters
        ----------
        grid: landlab raster model grid

        critical_slope: list of floats
            critical slope (angle of repose if no cohesion) of mass
            wasting material , L/L list of length 1 for a basin uniform Sc value
            list of length 2 for hydraulic geometry defined Sc, where the first
            and second values in the list are the coefficient and exponent of a
            user defined function for crictical slope that varies with contributing
            area [m2] to a node (e.g., [0.146,0.051] sets Sc<=0.1 at
            contributing area > ~1100 m2).

        threshold_flux: float
            minimum volumetric flux per unit contour width, [L3/L2/iterataion] or
            [L/iteration]. Flux below this threshold stops at the cell as a deposit

        erosion_coefficient: float
            coefficient used to convert total basal shear stress [kPa] of the runout
            material to a scour depth [m]

        tracked_attributes : list of str or None
            A list of the attribute names (strings) that will be tracked by the
            runout model. Attributes in tracked_attributes must also be a field
            on the model grid and names in list must match the grid field names.
            Default is None.

        deposition_rule : str
            Can be either "critical_slope", "L_metric" or "both".
            "critical_slope" is deposition rule used in Keck et al. 2023.
            "L_metric" is a variation of rule described by Campforts et al. 2020.
            "both" uses the minimum value of both rules.
            Default value is "critical_slope".

        grain_shear : bool
            Indicate whether to define shear stress at the base of the runout material
            as a function of grain size using Equation 13 (True) or the depth-slope
            approximation using Equation 12 (False). Default is True.

        effective_qsi : bool
            Indicate whether to limit erosion and aggradation rates to <= the
            erosion and aggradation rates coorisponding to the maximum observed flow
            depth. All results in Keck et al. 2023 use this constraint. Default is True.

        E_constraint : bool
             Indicate if erosion can not simultaneously occur with aggradation. If True,
             aggradation > 0, then erosion = 0. This is True in Keck et al., 2023.
             Default is True.

        settle_deposit : bool
            Indicate whether to allow deposits to settle before the next model iteration
            is implemented. Settlement is determined the critical slope as evaluated from
            the lowest adjacent node to the deposit. This is not used in Keck et al. 2023
            but tends to allow model to better reproduce smooth, evenly sloped deposits.
            Default is False.

        save : bool
            Save topographic elevation of watershed after each model iteration?
            This uses a lot of memory but is helpful for illustrating runout.
            The default is False.


        Other Parameters
        ----------
        These parameters have a lesser impact on model behavior or may not be
        applicable, depending on model run options.

        typical_flow_thickness_of_erosion_zone: float
            field estimated flow thickness in the erosion-dominatedd reaches of
            the runout path [m], used to estimate erosion_coefficient k using
            erosion_coef_k function. Default value: 3

        typical slope_of_erosion_zone: float
            field or remote sensing estimated slope in the scour dominated reach
            of the runout path [L/L], used to estimate erosion_coefficient k using
            erosion_coef_k function. Default value: 0.4

        erosion_exponent: float
            The exponent of equation 11, that scales erosion depth as a function of
            shear stress. Default value: 0.5

        max_flow_depth_observed: float
            Maximum observed flow depth, over the entire
            runout path [m], h_max in equation 24. Only used effective_qsi is True.
            Default value: 4

        vol_solids_concentration: float
            The ratio of the volume of the solids to the total volume of the flow
            mixture. Default value: 0.6

        density solids: float
            The density of the solids [kg/m3]. Default value: 2650

        density fluid: float
            The density of the fluid [kg/m3]. Default value: 1000

        gravity: float
            Acceleration due to gravity [m2/s]. Default value: 9.81

        dist_to_full_qsc_constraint : float
            distance in meters at which qsc is applied to runout. If the landslide
            initiates on relatively flat terrain, it may be difficult to determine
            a qsc value that allows the model start and deposit in a way that matches
            the observed. In Keck et al. 2023, dist_to_full_qsc_constraint = 0, but
            other landslides may need dist_to_full_qsc_constraint = 20 to 50 meters.

        itL : int
            maximum number of iterations the model runs before it
            is forced to stop. The default is 1000. Ideally, if properly parameterized,
            the model should stop on its own. All modeled runout in Keck et al. 2023
            stopped on its own.

        run_id : float, int or str
            label for landslide run, can be the time or some other identifier. This
            can be updated each time model is implemnted with "run_one_step"

        Returns
        -------
        None
        """
        if isinstance(critical_slope, (float, int)):
            critical_slope = (critical_slope,)

        super().__init__(grid)

        if len(critical_slope) > 1:
            self.variable_slpc = True
            self.a = critical_slope[0]
            self.b = critical_slope[1]
        else:
            self.variable_slpc = False
            self.slpc = critical_slope[0]
        self.qsc = threshold_flux
        self.k = erosion_coefficient
        self._tracked_attributes = tracked_attributes
        self.deposition_rule = deposition_rule
        self.grain_shear = grain_shear
        self.effective_qsi = effective_qsi
        self.settle_deposit = settle_deposit
        self.E_constraint = E_constraint
        self.save = save
        self.h = typical_flow_thickness_of_erosion_zone
        self.s = typical_slope_of_erosion_zone
        self.f = erosion_exponent
        self.qsi_max = max_flow_depth_observed_in_field
        if self.effective_qsi and self.qsi_max is None:
            raise ValueError(
                "Need to define the 'max_flow_depth_observed_in_field'"
                " or set effective_qsi to False"
            )
        self.vs = vol_solids_concentration
        self.ros = density_solids
        self.rof = density_fluid
        self.g = gravity
        self.dist_to_full_qsc_constraint = dist_to_full_qsc_constraint
        self.itL = itL
        self.run_id = run_id

        if tracked_attributes:
            self.track_attributes = True

            # check attributes are included in grid
            for key in self._tracked_attributes:
                if not self._grid.has_field(key, at="node"):
                    raise ValueError(f"{key} not included as field in grid")

            # if using grain size dependent erosion, check
            # particle_diameter is included as an attribute
            if (                                    ##### NEED TO MOVE THIS CHECK OUT OF ABOVE IF STATEMENT
                self.grain_shear
                and "particle__diameter" not in self._tracked_attributes
            ):
                raise ValueError(
                    "'particle__diameter' not included as field in grid and/or"
                    " key in tracked_attributes"
                )
        else:
            self.track_attributes = False

        # flow routing option
        # 'square_root_of_slope', see flow director
        self.routing_partition_method = "slope"

        # density of runout mixture
        self.ro_mw = self.vs * self.ros + (1 - self.vs) * self.rof
        # number of model iterations needed to reach dist_to_full_qsc_constraint
        self.d_it = int(self.dist_to_full_qsc_constraint / self._grid.dx)

        # define initial topographic + mass wasting thickness topography
        self._grid.at_node["energy__elevation"] = self._grid.at_node[
            "topographic__elevation"
        ].copy()
        self._grid.at_node["topographic__initial_elevation"] = self._grid.at_node[
            "topographic__elevation"
        ].copy()
        # prepare data containers for saving model images and behavior statistics
        if self.save:
            self.saver = MassWastingSaver(self)
            self.saver.prep_data_containers()

    def run_one_step(self):
        """run MWR"""

        # get all nodes that define the mass wasting events
        mask = self._grid.at_node["mass__wasting_id"] > 0

        # separate the mass wasting event nodes into individual events
        # change this so that the timing of each event can be specified for each id
        # if timing same, landslides of the same id are released simultaneously
        self.mw_ids = np.unique(self._grid.at_node["mass__wasting_id"][mask])
        innL = []  # innL is list of lists of nodes in each mass wasting event
        for mw_id in self.mw_ids:
            ls_mask = self._grid.at_node["mass__wasting_id"] == mw_id
            innL.append(np.hstack(self._grid.nodes)[ls_mask])

        # For each mass wasting event in list:
        for mw_i, inn in enumerate(innL):
            mw_id = self.mw_ids[mw_i]
            self._lsvol = (
                self._grid.at_node["soil__thickness"][inn].sum()
                * self._grid.dx
                * self._grid.dy
            )

            # prepare temporary data containers for each mass wasting event mw_i
            if self.save:
                self.saver.prep_mw_data_containers(mw_i, mw_id)

            # Algorithm 1, prepare initial mass wasting material (debritons) for release
            self._prep_initial_mass_wasting_material(inn, mw_i)

            # self.arndn_r[mw_id].append(self.arndn)
            if self.save:
                # save first set of data to reflect scar created by landslide
                self.saver.save_conditions_before_runout(mw_i, mw_id)

            # Algorith 2, now loop through each receiving nodes,
            # determine next set of recieving nodes,
            # repeat until no more receiving nodes (material deposits)
            self.c = 0  # model iteration counter
            while len(self.arn) > 0 and self.c < self.itL:
                # set qsc: the qsc constraint does not fully apply until runout
                # has traveled dist_to_full_qsc_constraint
                if self.d_it == 0:
                    self.qsc_v = self.qsc
                else:
                    self.qsc_v = self.qsc * (min(self.c / self.d_it, 1))

                # temporary data containers for each iteration of the while loop,
                # that store receiving node, flux and attributes to become the
                # input for the next iteration
                self.arndn_ns = np.array([])  # next iteration donor nodes
                self.arn_ns = np.array([])  # next iteration receiver nodes
                self.arqso_ns = np.array([])  # next iteration flux to receiver nodes
                self.arnL = []  # list of receiver nodes
                self.arqsoL = []  # list of flux out
                self.arndnL = []  # list of donor nodes
                if self.track_attributes:
                    self.aratt_ns = dict.fromkeys(
                        self._tracked_attributes, np.array([])
                    )  #
                    self.arattL = dict.fromkeys(self._tracked_attributes, [])

                # for each unique node in receiving node list self.arn
                self.arn_u = np.unique(self.arn).astype(int)  # unique arn list

                # determine the incoming flux to each node in self.arn_u
                self._determine_qsi_v()

                # update node elevation plus incoming flow thickness
                # this happens even if using topographic__elevation to route so that
                # the thickness of the debris flow is tracked for plotting
                self._update_E_dem_v()

                # determine erosion, aggradation, qso and attributes,
                # arranged in array nudat
                self._E_A_qso_determine_attributes_v()

                # from qso and flow direction at node, determine flux and attributes
                # sent to each receiver node
                self._determine_rn_proportions_attributes_v()

                # update grid field: topographic__elevation with the values in
                # nudat. Do this after directing flow, because assume deposition
                # does not impact flow direction
                self._update_dem_v()

                # update topographic slope field
                self._update_topographic_slope()

                # update tracked attribute grid fields
                if self._tracked_attributes:
                    for key in self._tracked_attributes:
                        self._update_attribute_at_node_v(key)

                # optional settlment of deposits and redistribution of attributes
                if self.settle_deposit:
                    self._settle()
                    self._update_topographic_slope()
                # if self.c < self.itL-1: # for last iteration in a run truncated by the itL constraint, don't update class variables for next step
                    # once all nodes in this iteration have been processed, the lists of receiving
                    # nodes (arn), donor nodes (arndn, which are the recieving nodes of this step),
                    # outgoing node flux (arqso) and node attributes (artt) are updated
                    # for the next iteration
                self.arndn = self.arndn_ns.astype(int)
                self.arn = self.arn_ns.astype(int)
                self.arqso = self.arqso_ns  #
                if self.track_attributes:
                    self.aratt = self.aratt_ns

                if self.save:
                    self.saver.save_conditions_after_one_iteration(mw_i, mw_id)

            # update iteration counter
                self.c += 1

    def _prep_initial_mass_wasting_material(self, inn, mw_i):
        """THIS FUNCTION NEEDS TO LOOP THROUGH EACH NODE BECAUSE MOTION OF UPSLOPE 
        NODES IS DEPDENDENT ON HOW DOWNSLOPE NODES MODIFY THE TERRAIN, NOT A SIMULTANEOUS
        MOVEMENT LIKE LATER ITERATIONS OF THE MODEL => REWRITE SO THAT FIRST NODE
        SLOPE IS SURFACE SLOPE AND THEN SUBTRACT ALL DEBRISTONS FROM SURFACE IN ONE
        COMPUTATION TO GET INTIAL LANDSLIDE LIST
        Algorithm 1 - from an initial source area (landslide), prepare the
        initial lists of receiving nodes and incoming fluxes and attributes
        and remove the source material from the DEM

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
        if self._tracked_attributes:
            att = dict.fromkeys(self._tracked_attributes, np.array([]))

        # order source area nodes from lowest to highest elevation
        node_z = self._grid.at_node.dataset["topographic__elevation"][inn]
        sorted_nodes = inn[np.argsort(node_z)]

        for ci, ni in enumerate(sorted_nodes):
            # regolith (soil) thickness at node. soil thickness in source area
            # represents landslide thickness
            s_t = self._grid.at_node.dataset["soil__thickness"].values[ni]

            # remove soil (landslide) thickness at node
            self._grid.at_node.dataset["topographic__elevation"][ni] = (
                self._grid.at_node.dataset["topographic__elevation"][ni] - s_t
            )

            # update soil thickness at node (now = 0)
            self._grid.at_node["soil__thickness"][ni] = (
                self._grid.at_node["soil__thickness"][ni] - s_t
            )

            if (
                ci > 0
            ):  # use surface slope for first node to start movement of landslide
                # for all other nodes, update slope to reflect material removed from DEM
                self._update_topographic_slope()

            # get receiving nodes of node ni in mw index mw_i
            rn = self._grid.at_node.dataset["flow__receiver_node"].values[ni]
            rn = rn[np.where(rn != -1)]

            # receiving proportion of qso from cell n to each downslope cell
            rp = self._grid.at_node.dataset["flow__receiver_proportions"].values[ni]
            rp = rp[np.where(rp > 0)]  # only downslope cells considered

            # initial mass wasting thickness
            imw_t = s_t
            # get flux out of node ni
            qso = imw_t
            # divide into proportions going to each receiving node
            rqso = rp * qso

            if self._tracked_attributes:
                # get initial mass wasting attributes moving (out) of node ni
                self.att_ar_out = {}
                for key in self._tracked_attributes:
                    att_val = self._grid.at_node.dataset[key].values[ni]
                    # particle diameter to each recieving node
                    self.att_ar_out[key] = np.ones(len(rqso)) * att_val

                    # attribute value is zero at node after reglith leaves
                    self._grid.at_node[key][ni] = 0

                    att[key] = np.concatenate((att[key], self.att_ar_out[key]), axis=0)

            # append receiving node ids, fluxes and attributes to initial lists
            rni = np.concatenate((rni, rn), axis=0)
            rqsoi = np.concatenate((rqsoi, rqso), axis=0)

        self.arndn = np.ones([len(rni)]) * np.nan
        self.arn = rni.astype(int) # ensure node id is stored as an int
        self.arqso = rqsoi
        if self._tracked_attributes:
            self.aratt = att
            

    def _prep_initial_mass_wasting_material_v(self, inn, mw_i):
        """THIS FUNCTION NEEDS TO LOOP THROUGH EACH NODE BECAUSE MOTION OF UPSLOPE 
        NODES IS DEPDENDENT ON HOW DOWNSLOPE NODES MODIFY THE TERRAIN, NOT A SIMULTANEOUS
        MOVEMENT LIKE LATER ITERATIONS OF THE MODEL, SMALL MODIFICATIONS TO MATCH
        FORMAT OF NEW FUNCTIONS AND GET RID OF NP CONCATENATE
        Algorithm 1 - from an initial source area (landslide), prepare the
        initial lists of receiving nodes and incoming fluxes and attributes
        and remove the source material from the DEM

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
        if self._tracked_attributes:
            att = dict.fromkeys(self._tracked_attributes, np.array([]))

        # order source area nodes from lowest to highest elevation
        node_z = self._grid.at_node.dataset["topographic__elevation"][inn]
        sorted_nodes = inn[np.argsort(node_z)]
        
        

        for ci, ni in enumerate(sorted_nodes):
            # regolith (soil) thickness at node. soil thickness in source area
            # represents landslide thickness
            s_t = self._grid.at_node.dataset["soil__thickness"].values[ni]

            # remove soil (landslide) thickness at node
            self._grid.at_node.dataset["topographic__elevation"][ni] = (
                self._grid.at_node.dataset["topographic__elevation"][ni] - s_t
            )

            # update soil thickness at node (now = 0)
            self._grid.at_node["soil__thickness"][ni] = (
                self._grid.at_node["soil__thickness"][ni] - s_t
            )

            if (
                ci > 0
            ):  # use surface slope for first node to start movement of landslide
                # for all other nodes, update slope to reflect material removed from DEM
                self._update_topographic_slope()

            # get receiving nodes of node ni in mw index mw_i
            rn = self._grid.at_node.dataset["flow__receiver_node"].values[ni]
            rn = rn[np.where(rn != -1)]

            # receiving proportion of qso from cell n to each downslope cell
            rp = self._grid.at_node.dataset["flow__receiver_proportions"].values[ni]
            rp = rp[np.where(rp > 0)]  # only downslope cells considered

            # initial mass wasting thickness
            imw_t = s_t
            # get flux out of node ni
            qso = imw_t
            # divide into proportions going to each receiving node
            rqso = rp * qso

            if self._tracked_attributes:
                # get initial mass wasting attributes moving (out) of node ni
                self.att_ar_out = {}
                for key in self._tracked_attributes:
                    att_val = self._grid.at_node.dataset[key].values[ni]
                    # particle diameter to each recieving node
                    self.att_ar_out[key] = np.ones(len(rqso)) * att_val

                    # attribute value is zero at node after reglith leaves
                    self._grid.at_node[key][ni] = 0

                    att[key] = np.concatenate((att[key], self.att_ar_out[key]), axis=0)

            # append receiving node ids, fluxes and attributes to initial lists
            rni = np.concatenate((rni, rn), axis=0)
            rqsoi = np.concatenate((rqsoi, rqso), axis=0)

        self.arndn = np.ones([len(rni)]) * np.nan
        self.arn = rni.astype(int) # ensure node id is stored as an int
        self.arqso = rqsoi
        if self._tracked_attributes:
            self.aratt = att



    def _E_A_qso_determine_attributes_v(self):
    # developing vectorized replacement to EA func
    # run Cascades, 2009 3 steps to get a qsi_dat
        
        self.D_L = []  # list of deposition depths, if the _settle function is used, THIS CAN BE DELETED, TODO

    
        # syntax here will change once qsi dat is changed to list of np.arrays
        n = self.qsi_dat[0] # qsi_dat needs to be float not object
        qsi = self.qsi_dat[1]
        
        
        slpn = self.grid.at_node["topographic__steepest_slope"][n].max(axis=1)
        
        # qsi_ = min(qsi, self.qsi_max)
        qsi_ = np.where(qsi<self.qsi_max,qsi,self.qsi_max)
        
        # critical slope
        slpc = self.a * self._grid.at_node["drainage_area"][n] ** self.b if self.variable_slpc else np.full(len(n), self.slpc)
        
        # incoming attributes    
        att_in  = self._attributes_in_v(n, qsi) 
           
        # first compute an aggradation value at all nodes               
        A_ = self._aggradation_v(qsi, n)
        # where slp less than qsc, sett aggradation equal to qsi
        A = np.where(qsi<self.qsc_v,qsi,A_)
        # now compute E, att_up, Tau and u at each node
        E, att_up, Tau, u = self._erosion_v(n, qsi_, slpn, att_in=att_in)
        # print(f"E = {E}")
        # print(f"att_up = {att_up}")
        # update E for cosntraints
        # constraint 1, qsi less than qsi_v
        E = np.where(qsi<self.qsc_v, 0, E)
        # constraint 2, E_constraint, no E when A
        if self.E_constraint:
            E = np.where(A>0,0,E)
            Tau = np.where(A>0,0,Tau)
            u = np.where(A>0,0,u)
            for key in self._tracked_attributes:
                att_up[key] = np.where(A>0,0,att_up[key])
        # flux out
        qso = qsi - A + E
        # small qso are considered zero
        qso = np.round(qso, decimals=8)
        # chnage elevation
        deta = A - E
        
        # model behavior tracking
        if self.save:
            self.saver.save_flow_stats(E, A, qsi, slpn, Tau, u)
        
        # updated attribute values at node
        n_att = self._attributes_node_v(n, att_in, E, A)
        
        # list of deposition depths at cells in iteration
        self.D_L.append(A)
        
        self.nudat = [n, deta, qso, qsi, E, A] 
        self.n_att = n_att
        self.att_up = att_up
        self.att_in = att_in

    
    def _determine_rn_proportions_attributes_v(self):
        """modifying vectorized_rn_proportions_1 to do the rest of the _determine_rn_proportions_attribute
        function, this is still a bottle neck in the code because it iterates through all nodes to find the 
        donar nodes, try to improve this function"""
        # 1. Extract inputs from nudat

        # new format
        n_ids = self.nudat[0]
        qsi = self.nudat[3]
        qso = self.nudat[2]
        E = self.nudat[4]
        A = self.nudat[5]
    
        att_up = self.att_up
        att_in = self.att_in   
        
        # 2. Get receiver nodes and proportions for the nodes in nudat
        all_rn = self._grid.at_node.dataset["flow__receiver_node"].values[n_ids]
        all_rp = self._grid.at_node.dataset["flow__receiver_proportions"].values[n_ids]
        
        # 3. Vectorized Donor (back-flow) exclusion
        # We check which receivers (all_rn) should be excluded for each node n
        # Since 'dn' depends on 'n', we find rows in self.arn that match our n_ids
        # and use that to mask all_rn.
        
        # For each row (node n), we identify receivers that are also donors of n
        # A highly efficient way to do this at this scale is a row-wise 'isin' check
        # But since all_rn is small (width 8), we can broadcast:
        mask_exclude = np.zeros(all_rn.shape, dtype=bool)
        for i, n in enumerate(n_ids):
            dn = self.arndn[self.arn == n]
            mask_exclude[i] = np.isin(all_rn[i], dn)
        
        # Final valid mask: not in exclusion list AND not -1
        valid_mask = (~mask_exclude) & (all_rn != -1)
        
        # 4. Handle constraints (qso > 0 and not boundary)
        is_boundary = np.isin(n_ids, self._grid.boundary_nodes)
        can_flow = (qso > 0) & (~is_boundary)
        
    
            

        # 5. Filter and Renormalize proportions
        # Zero out proportions for invalid links or nodes that cannot flow
        filtered_rp = np.where(valid_mask & can_flow[:, None], all_rp, 0.0)
        
        # Row-wise sums for renormalization
        rp_sums = filtered_rp.sum(axis=1)
        
        # Logic for "No valid receivers": if rp_sums is 0 but can_flow is True, flow stays at n
        no_rec_mask = (rp_sums == 0) & can_flow
        
        # Normalize filtered proportions
        norm_rp = np.divide(filtered_rp, rp_sums[:, None], 
                            out=np.zeros_like(filtered_rp), 
                            where=rp_sums[:, None] != 0)
        
        # 6. Build the final 3 arrays (rndn, rn, rqso)
        # Extract indices where flow is actually occurring
        row_idx, col_idx = np.where(norm_rp > 0)
        
        rndn_final = n_ids[row_idx] # delivery nodes n to receiver nodes
        rn_final = all_rn[row_idx, col_idx] # receiver nodes from nodes n
        rqso_final = norm_rp[row_idx, col_idx] * qso[row_idx] # qso sent to each receiver node from node n
        
        # Add cases where flow stays at node n
        if np.any(no_rec_mask):
            stay_n = n_ids[no_rec_mask]
            rndn_final = np.concatenate([rndn_final, stay_n])
            rn_final = np.concatenate([rn_final, stay_n])
            rqso_final = np.concatenate([rqso_final, qso[no_rec_mask]])
            
            
        # update attributes and flux arrays:
        qsi = qsi#[can_flow]
        qso = qso#[can_flow]
        E = E#[can_flow]
        A = A#[can_flow]
        for key in self._tracked_attributes:
            att_up[key] = att_up[key]#can_flow]
            att_in[key] = att_in[key]#[can_flow]
        # outgoing attributes
        if self._tracked_attributes:
            # print(f"att_up 22222 = {att_up}")
            att_out = self._attribute_out_v(att_up,
                                       att_in,
                                       qsi,
                                       E,
                                       A)
            att_out_final = {}
            for key in self._tracked_attributes:
                att_out_final[key] = att_out[key][row_idx]
                # Again, add cases where flow stays at nodes n
                if np.any(no_rec_mask):
                    att_out_final[key] = np.concatenate([att_out_final[key], att_out[key][no_rec_mask]])
                # print(f"att_out[{key}] = {att_out[key]}")
                # print(f"row_idx = {row_idx}")
                # ratt = att_out[key][row_idx]
                self.aratt_ns[key] = att_out_final[key]
                # self.aratt_ns[key] = np.concatenate(
                #     (self.aratt_ns[key], ratt), axis=0
                # )  # next step receiving node incoming particle diameter list
                self.arattL[key].append(att_out_final[key])#ratt)   # this is not used TODO
                
        # this part will already work as written
        # store receiving nodes and fluxes in temporary arrays
        self.arndn_ns = np.concatenate((self.arndn_ns, rndn_final), axis=0)
        # next iteration donor nodes
        self.arn_ns = np.concatenate((self.arn_ns, rn_final), axis=0)
        # next iteration receiving nodes
        self.arqso_ns = np.concatenate((self.arqso_ns, rqso_final), axis=0)



    def _determine_qsi_v(self):
        # 1. Sum all flux (arqso) for every node ID present in arn.
        # weights=self.arqso tells bincount to sum the fluxes rather than counting occurrences.
        # minlength ensures the resulting array is large enough to cover all relevant node IDs.
        max_id = max(self.arn.max(), self.arn_u.max())
        total_flux_per_node = np.bincount(self.arn, weights=self.arqso, minlength=max_id + 1)
        
        # 2. Extract only the sums for the specific nodes requested in arn_u.
        ll = total_flux_per_node[self.arn_u]
        
        # 3. Reshape and concatenate to match the expected (Nodes, 2) structure
        self.qsi_dat = [self.arn_u, ll]
        
        

    
        
    
    def _update_E_dem_v(self):
        """update energy__elevation"""
        n = self.qsi_dat[0]
        qsi = self.qsi_dat[1]
        ### already vectorized, updated for new nudat format
        # energy elevation is equal to the topographic elevation plus qsi
        self._grid.at_node["energy__elevation"] = self._grid.at_node[
            "topographic__elevation"
        ].copy()
        self._grid.at_node["energy__elevation"][n] = (
            self._grid.at_node["energy__elevation"].copy()[n] + qsi
        )

    def _update_energy_slope(self):
        """updates the topographic__slope and flow directions grid fields using the
        energy__elevation field. This function is presently not used but may be useful
        for future implementations of MWR"""
        ### this is dependent on landlab function, doesnt change
        fd = FlowDirectorMFD(
            self._grid,
            surface="energy__elevation",
            diagonals=True,
            partition_method=self.routing_partition_method,
        )
        fd.run_one_step()

        
    def _update_dem_v(self):
        """updates the topographic elevation of the landscape dem and soil
        thickness fields"""
        ### already vectorized, updated for new nudat format
        n = self.nudat[0]
        deta = self.nudat[1]
        self._grid.at_node["soil__thickness"][n] = (
            self._grid.at_node["soil__thickness"][n] + deta
        )
        self._grid.at_node["topographic__elevation"][n] = (
            self._grid.at_node["topographic__elevation"][n] + deta
        )

    def _update_topographic_slope(self):
        """updates the topographic__slope and flow directions fields using the
        topographic__elevation field"""
        ### this is dependent on landlab function, doesnt change
        fd = FlowDirectorMFD(
            self._grid,
            surface="topographic__elevation",
            diagonals=True,
            partition_method=self.routing_partition_method,
        )
        fd.run_one_step()


    def _update_attribute_at_node_v(self, key):
        """TRY TO COMBINE THIS WITH ATTRIBUTE_NODE
        for each unique node in receiving node list, update the attribute
        using attribute value determined in the _E_A_qso_determine_attributes method

        Parameters
        ----------
        key: string
            one of the tracked attributes
        """
        n = self.nudat[0]
        new_node_pd = self.n_att[key]
        self._grid.at_node[key][n] = new_node_pd        
        
        
    def _erosion_v(self, n, depth, slpn, att_in=None):
        """if self.grain_shear is True, determines the erosion depth using
        equation (13), otherwise uses equation (12).
    
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
        ### already vectorized, done
        
        theta = np.arctan(slpn)  # convert tan(theta) to theta
        # attributes of eroded material
        if self._tracked_attributes:
            att_up = {}
            for key in self._tracked_attributes:
                att_up[key] = self._grid.at_node[key][n]
        else:
            att_up = None
    
        if self.grain_shear:
            # shear stress approximated as a power function of inertial shear stress
            Dp = att_in["particle__diameter"]
            # if depth < Dp:  # grain size dependent erosion breaks if depth<Dp
            #     Dp = depth * 0.99
            Dp = np.where(depth<Dp,depth*0.99,Dp)
            u = flow_velocity(Dp, depth, slpn, self.g)
    
            Tau = shear_stress_grains(self.vs, self.ros, Dp, depth, slpn, self.g)
    
            Ec = self._grid.dx * erosion_rate(self.k, Tau, self.f, self._grid.dx)
    
        else:
            # quasi-static approximation
            Tau = self.ro_mw * self.g * depth * (np.sin(theta))
            Ec = self.k * (Tau) ** self.f
            u = np.nan
    
        dmx = self._grid.at_node["soil__thickness"][n]
    
        E = np.minimum(dmx, Ec)
    
        return (E, att_up, Tau, u)
    
    
    def _aggradation_v(self, qsi, n):
        """vectorized aggradation function
        determine deposition depth following equations 4 though 9
        use only the friction angle depostion approach
    
        Parameters
        ----------
        qsi : np.array of float
            incoming flux [l3/iteration/l2]
        n : np.array of int
            node id
    
        Returns
        -------
        A_f : float
            aggradation depth [L]
        """
        slp_h = self.slpc * self._grid.dx
        zi = self._grid.at_node["topographic__elevation"][n]
        zo = self._determine_zo_v(n, zi, qsi)

        dx = self._grid.dx
        sc = self.slpc
        s = (zi - zo) / dx
        sd = sc - s
        D1 = sc * dx / 2
        a = 0.5 * dx * sd
        b = D1 - 0.5 * dx * sd
        c = -qsi
        radicand = (b**2) - 4 * a * c # doing this to avoid warning when taking sqrt of negative
        sqrt = np.sqrt(radicand, where=(radicand >= 0), out=np.zeros_like(a, dtype=float))
        N1 = -b + (sqrt) / (2 * a)
        N2 = -b - (sqrt) / (2 * a)
        ndn = np.round(np.max([N1,N2,np.ones(len(N1))],axis=0))
        A = np.min([(1 / ndn) * qsi + ((ndn - 1) / 2) * dx * sd, qsi],axis=0)

        # where slope is less than critical, apply the aggradation rule, otherwise
        # aggradation is zero
        A_f = np.where((zi - zo) <= (slp_h),A, 0)
        return A_f


    def _determine_zo_v(self, n, zi, qsi):
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
    
        # get adjacent nodes, -1 values are no nodes (edge)
        adj_n = np.hstack(
            (
                self._grid.adjacent_nodes_at_node[n],
                self._grid.diagonal_adjacent_nodes_at_node[n],
            )
        )
    
        # exclude closed boundary nodes
        #adj_n = adj_n[~np.isin(adj_n, self._grid.closed_boundary_nodes)]
        adj_n = np.where(np.isin(adj_n, self._grid.closed_boundary_nodes), -1, adj_n)
    
        # adj_n = np.where(adj_n == -1, np.nan,adj_n)
        zo_ = self._grid.at_node["topographic__elevation"][adj_n]
        zo = np.where(adj_n == -1, np.inf,zo_).min(axis=1)
        
        # check that lowest elevation node is less than flow depth + node elevation
        # if not, then outlet node elevation is set to the elvation of the current node
        # (flow obstructed)
        ei = qsi + zi
        zo  = np.where(zo<ei,zo,zi) # this should probably just compare zo to zi, TODO
        return zo


    def _attributes_in_v(self, n, qsi):
        """vectorized attributes in, for each attribute returns an array
        of the flux of the attribute to each node in n"""
        # 1. Pre-calculate the weighted sums for every attribute
        # bincount calculates: result[i] = sum(weights[j]) for all j where arn[j] == i
        # attribute_weighted_sums = {}
        attributes = {}
        # We only need to check the max node ID to size our result arrays
        max_id = self.arn.max()

        for key in self._tracked_attributes:
            # Calculate (Attribute * Flow) for every row
            # print(f"self.aratt[{key}] = {self.aratt[key]}")
            # print(f"self.arqso = {self.arqso}")
            weights = self.aratt[key] * self.arqso
            # Sum them up grouped by the node ID in 'DebrisFlows.arn'
            attribute_weighted_sums = np.bincount(self.arn, weights=weights, minlength=max_id+1)
            
            attributes[key] = attribute_weighted_sums[n] / qsi
            
            # set attributes to zero where no qsi, this step may not be needed    
            attributes[key] = np.where(qsi == 0, 0, attributes[key])
            
        
        return attributes


    def _attributes_node_v(self, n, att_in, E, A):
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
            """determine weighted average attribute at the node. If all soil
            is eroded, attribute value is zero"""
            # if A + self._grid.at_node["soil__thickness"][n] - E > 0:
            inatt = self._grid.at_node[key][n]
            
            atts = np.zeros_like(A)
            mask = (A + self._grid.at_node["soil__thickness"][n] - E)>0
            atts_non_zero = (inatt[mask] * (self._grid.at_node["soil__thickness"][n][mask] - E[mask])+ att_in[key][mask] * A[mask]) / (A[mask] + self._grid.at_node["soil__thickness"][n][mask] - E[mask])
            atts[mask] = atts_non_zero
            n_att = atts

            return n_att
    
        n_att_d = {}
        for key in self._tracked_attributes:
            n_att_d[key] = weighted_avg_at_node(key)
    
        return n_att_d


    def _attribute_out_v(self, att_up, att_in, qsi, E, A):
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
        ### already vecotrized, done
        att_out = {}
        for key in self._tracked_attributes:
            atts = np.zeros_like(qsi)
            mask = (qsi - A + E)>0
            atts_non_zero = (att_up[key][mask] * E[mask] + att_in[key][mask] * (qsi[mask] - A[mask])) / (qsi[mask] - A[mask] + E[mask])
            atts[mask] = atts_non_zero
            att_out[key] = atts
        return att_out


def flow_velocity(Dp, h, s, g):
    ### already vecotrized, done
    us = (g * h * s) ** 0.5
    u = us * 5.75 * np.log10(h / Dp)
    return u


def shear_stress_grains(vs, ros, Dp, h, s, g):
    ### already vecotrized, done
    theta = np.arctan(s)
    phi = np.arctan(0.32)
    u = flow_velocity(Dp, h, s, g)
    dudz = u / h
    Tcn = np.cos(theta) * vs * ros * (Dp**2) * (dudz**2)
    tau = Tcn * np.tan(phi)
    return tau


def shear_stress_static(vs, ros, rof, h, s, g):
    ### already vecotrized, done
    theta = np.arctan(s)
    rodf = vs * ros + (1 - vs) * rof
    tau = rodf * g * h * (np.sin(theta))
    return tau


def erosion_coef_k(E_l, tau, f, dx):
    ### already vecotrized, done
    k = E_l * dx / (tau**f)
    return k


def erosion_rate(k, tau, f, dx):
    ### already vecotrized, done
    E_l = (k * tau**f) / dx
    return E_l
