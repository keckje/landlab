# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:24:19 2021

@author: keckj
"""


import pandas as pd
import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
from landlab.components import(FlowDirectorD8, 
                                FlowDirectorDINF, 
                                FlowDirectorMFD, 
                                FlowDirectorSteepest)



class MassWastingScourAndDeposition:
    
    def __init__(
    self,
    grid,
    df_para_dict,
    save_df_dem = False, 
    run_id = 0, 
    itL = 1000):
        
    
        
        self.grid = grid
        self.df_para_dict = df_para_dict
        self.save = save_df_dem
        self.run_id = run_id
        self.itL = itL
    
    
    '''a cellular automata debris flow model that routes an initial landslide 
    volume as a debris flow through the dem described by the 'topographic__elevation' 
    field of a landlab raster model grid
    

    Parameters
    ----------
    mg : landlab raster model grid
        raster model grid
    ivL : list of floats
        list of initial landslide volumes
    innL : list of int
        list of nodes where landslide volumes are released on the dem. 
        len(innL) = len(ivL)
    df_paraL : list of dictionaries
        each entry in the df_paraL is a dictionary of parameters that control 
        the behavoir of the cellular-automata debris flow model
        if len(df_paraL = 1, then the same model parameters are used for all runs.
    save_df_dem : boolean
        Save topographic elevation after each model iteration?. This could creat up to 
        itterL number of maps for each debris flow. The default is False.
    itL : int
        maximum number of iterations the cellular-automata model runs before it
        is forced to stop. The default is 1000.

    Returns
    -------
    None.

    '''
    
    def __call__(self, ivL, innL):
        
        
        self._scour_and_deposit(ivL, innL)
        

    def _scour_and_deposit(self, innL, ivL):
            
        # release parameters for landslide
        nps = self.df_para_dict['number of pulses']
        nid = self.df_para_dict['iteration delay']    
        
        # critical slope at which debris flow stops
        # higher reduces spread and runout distance
        slpc = self.df_para_dict['critical slope']
        
        # forced stop volume threshold
        # material stops at cell when volume is below this,
        # higher increases spread and runout distance
        SV = self.df_para_dict['stop-volume']
    
        # entrainment coefficient
        # very sensitive to entrainment coefficient
        # higher causes higher entrainment rate, longer runout, larger spread
        cs = self.df_para_dict['entrainment-coefficient']
    
    
            
        cL = {}
        
        # ivL and innL can be a list of values. For each value in list:
        for i,inn in enumerate(innL):
            # print(i)
            cL[i] = []
            
    
    
            # set up initial landslide cell
            iv =ivL[i]/nps# initial volume (total volume/number of pulses) 
        
            
            # initial receiving nodes (cells) from landslide
            rn = self.grid.at_node.dataset['flow__receiver_node'].values[inn]
            rni = rn[np.where(rn != -1)]
            
            # initial receiving proportions from landslide
            rp = self.grid.at_node.dataset['flow__receiver_proportions'].values[inn]
            rp = rp[np.where(rp > 0)]
            rvi = rp*iv # volume sent to each receving cell
        
            # now loop through each receiving node, 
            # determine next set of recieving nodes
            # repeat until no more receiving nodes (material deposits)
            
            slpr = []
            c = 0
            c2=0
            arn = rni
            arv = rvi
            enL = []
            while len(arn)>0 or c < self.itL:
        
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
                for n in np.unique(arn):
                    n = int(n)
                    
                    # slope at cell (use highest slope)
                    slpn = self.grid.at_node['topographic__steepest_slope'][n].max()
                    
                    # incoming volume: sum of all upslope volume inputs
                    vin = np.sum(arv[arn == n])
                    
                    # determine deposition volume following Campforts, et al., 2020            
                    # since using volume (rather than volume/dx), L is (1-(slpn/slpc)**2) 
                    # rather than mg.dx/(1-(slpn/slpc)**2)           
                    Lnum = np.max([(1-(slpn/slpc)**2),0])
                    
                    dpd = vin*Lnum # deposition volume
                    
                    # determine erosion depth
                    
                    # debris flow depth over cell    
                    df_depth = vin/(self.grid.dx*self.grid.dx) #df depth
                    
                    #determine erosion volume (function of slope)
                    # dmx = dc*mg.at_node['soil__thickness'][n]
                    # er = (dmx-dmx*Lnum)*mg.dx*mg.dy
                    
                    T_df = 1700*9.81*df_depth*slpn # shear stress from df
         
                    # max erosion depth equals regolith (soil) thickness
                    dmx = self.grid.at_node['soil__thickness'][n]
                    
                    # erosion depth: 
                    er = min(dmx, cs*T_df)
                    
                    enL.append(er)
                    
                    # erosion volume
                    ev = er*self.grid.dx*self.grid.dy
                    
                    # volumetric balance at cell
                    
                    # determine volume sent to downslope cells
                    
                    # additional constraint to control debris flow behavoir
                    # if flux to a cell is below threshold, debris is forced to stop
                    if vin <=SV:
                        dpd = vin # all volume that enter cell is deposited 
                        vo = 0 # debris stops, so volume out is 0
        
                        # determine change in cell height
                        deta = (dpd)/(self.grid.dx*self.grid.dy) # (deposition)/cell area
        
                    else:
                        vo = vin-dpd+ev # vol out = vol in - vol deposited + vol eroded
                        
                        # determine change in cell height
                        deta = (dpd-ev)/(self.grid.dx*self.grid.dy) # (deposition-erosion)/cell area
        
                    
                    # update raster model grid regolith thickness and dem
                    
                    
                    # if deta larger than regolith thickness, deta equals regolith thickness (fresh bedrock is not eroded)
                    if self.grid.at_node['soil__thickness'][n]+deta <0:
                        deta = - self.grid.at_node['soil__thickness'][n]           
                    
                    # Regolith - difference between the fresh bedrock surface and the top surface of the dem
                    self.grid.at_node['soil__thickness'][n] = self.grid.at_node['soil__thickness'][n]+deta 
        
                    # Topographic elevation - top surface of the dem
                    self.grid.at_node['topographic__elevation'][n] = self.grid.at_node['topographic__elevation'][n]+deta
                    # print(mg.at_node['topographic__elevation'][n])
            
            
                    # build list of receiving nodes and receiving volumes for next iteration        
            
                    # material stops at node if transport volume is 0 OR node is a 
                    # boundary node
                    if vo>0 and n not in self.grid.boundary_nodes: 
                        
            
                        th = 0
                        # receiving proportion of volume from cell n to each downslope cell
                        rp = self.grid.at_node.dataset['flow__receiver_proportions'].values[n]
                        rp = rp[np.where(rp > th)]
                        # print(rp)        
                        
                        # receiving nodes (cells)
                        rn = self.grid.at_node.dataset['flow__receiver_node'].values[n]
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
                fd = FlowDirectorMFD(self.grid, diagonals=True,
                                partition_method = 'slope')
                fd.run_one_step()
                
                
                if self.save:
                    cL[i].append(c)
                    
                    # save DEM difference (DEM for iteration C - initial DEM)
                    _ = self.grid.add_field( 'topographic__elevation_run_id_'+str(run_id)+str(i)+'_'+str(c),
                                    self.grid.at_node['topographic__elevation'].copy(),
                                    at='node')
                    mpnm.append('topographic__elevation_run_id_'+str(run_id)+str(i)+'_'+str(c))
                       
                # update iteration counter
                c+=1
                
                # update pulse counter
                c2+=nid
                
                if c%20 ==0:
                    print(c)            
