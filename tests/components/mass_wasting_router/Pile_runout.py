# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:45:13 2023

@author: keckj


pile of material test
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from landlab import RasterModelGrid
from landlab.components import LandslideProbability
from landlab.components import SinkFillerBarnes, FlowAccumulator, FlowDirectorMFD
from landlab import imshow_grid, imshow_grid_at_node
from landlab.grid.network import NetworkModelGrid
from landlab.plot import graph

from landlab.components import ChannelProfiler, DepressionFinderAndRouter
from landlab import Component, FieldError


from landlab.components.mass_wasting_router import MassWastingRunout
os.chdir('C:/Users/keckj/Documents/GitHub/code/landlab/LandlabTools')
import LandlabTools as LLT


os.chdir('C:/Users/keckj/Documents/GitHub/landlab/landlab/components/mass_wasting_router/')
from mwru_calibrator import (MWRu_calibrator,
                                profile_distance,
                                profile_plot,
                                view_profile_nodes)


os.chdir('C:/Users/keckj/Documents/GitHub/code/preevents/paper2/')
import MassWastingRunoutEvaluationFunctions as MWF

#%%

def plot_values(mg,field,xmin,xmax,ymin,ymax, field_back = "topographic__elevation", cmap = 'terrain', background = True):
    """plot the field values on the node within the specified domain extent"""
    
    
    if background:
        plt.figure(field,figsize = (12,8))
        imshow_grid(mg, field_back, grid_units=('m','m'),var_name=field_back,plot_name = field,cmap = cmap)
    values = mg.at_node[field]
    if values.dtype != int:
        values = values.round(decimals = 4)
    values_test_domain = values[(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]
    ntd = mg.at_node['node_id'][(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]
    for i,val in enumerate(values_test_domain):
        plt.text(mg.node_x[ntd[i]],mg.node_y[ntd[i]],str(val),color = 'red', fontsize = 9)
    plt.xlim([xmin,xmax]); plt.ylim([ymin,ymax])    
        
        
   

def determine_alpha(ros,vs,h,s,eta,E_l,dx,slpc = 0.1, Dp = None):
    """
    determine the coeficient of equation 7 (alpha)
    
    Parameters
    ----------
    ros : density of grains in runout [kg/m3]
    vs  : volumetric ratio of solids to matrix [m3/m3]
    h   : depth [m] - typical runout depth
    s   : slope [m/m] - average slope of erosion part of runout path
    eta : exponent of scour model (equation 7) - 
    E_l : average erosion rate per unit length of runout [m/m]
    dx  : cell width [m]
    slpc: average slope at which positive net deposition occurs
    Dp  : representative grain size [m]

    Returns
    -------
    alpha

    """

    g= 9.81
    rof = 1000
    rodf = vs*ros+(1-vs)*rof
    theta = np.arctan(s)
    
       
    if Dp: 
        print('grain-inertia')
        phi = np.arctan(slpc)
        
        # inertial stresses
        us = (g*h*s)**0.5
        u = us*5.75*np.log10(h/Dp)
        
        dudz = u/h
        Tcn = np.cos(theta)*vs*ros*(Dp**2)*(dudz**2)
        tau = Tcn*np.tan(phi)
    else:
        print('quasi-static')
        tau = rodf*g*h*(np.sin(theta))

    
    alpha = E_l*dx/(tau**eta)
    
    return alpha, tau


#%% parameters

# odd number
r = 51
c = r
dxdy = 5
ls_h = 5
w = 5


PlanVisualize = True
ProfileVisualize = True


qsc = 0.01 # pick qsc
lam = 10 # coeficient multiplied by qsc to determine equivlanet alpha
slpc = 0.01# critical slope

ros = 2650 # density
vs = 0.6 # volumetric solids concentration
h = 2 # typical flow thickness
s = 0.6 # typical slope
eta = 0.2 # exponent
Dp = 0.2 # particle diameter
g_erosion = True
ls_h = 5 # landslide thickness
hs = 1 # soil thickness
deposition_rule = "critical_slope"#"L_metric"#"critical_slope"#
deposit_style = 'downslope_deposit_sc10'#'no_downslope_deposit_sc'#'downslope_deposit_sc9'
effective_qsi = True
settle_deposit = False

#%%create model grid

mg = RasterModelGrid((r,c),dxdy)

dem = mg.add_field('topographic__elevation',
                    np.ones(r*c)*1,
                    at='node')

mg.at_node['node_id'] = np.hstack(mg.nodes)


# domain for plots
xmin = mg.node_x.min(); xmax = mg.node_x.max(); ymin = mg.node_y.min(); ymax = mg.node_y.max()


# set boundary conditions, add flow direction
mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries



# flow directions
fa = FlowAccumulator(mg, 
                      'topographic__elevation',
                      flow_director='FlowDirectorD8')
fa.run_one_step()

# hillshade for plots
mg.at_node['hillshade'] = mg.calc_hillshade_at_node(elevs=dem, alt=37., az=210.)


# soil thickness
thickness = np.ones(mg.number_of_nodes)*hs
mg.add_field('node', 'soil__thickness',thickness)


# set particle diameter
if g_erosion:
    mg.at_node['particle__diameter'] = np.ones(len(mg.node_x))*Dp


# copy of initial topography
DEMi = mg.at_node['topographic__elevation'].copy()



# view node ids
field = 'node_id'
field_back= "topographic__elevation"
plot_values(mg,field,xmin,xmax,ymin,ymax)


#%% Add multiflow direction

# run flow director, add slope and receiving node fields
mg.delete_field(loc = 'node', name = 'flow__sink_flag')
mg.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
mg.delete_field(loc = 'node', name = 'flow__receiver_node')
mg.delete_field(loc = 'node', name = 'topographic__steepest_slope')

# run flow director, add slope and receiving node fields
fd = FlowDirectorMFD(mg, diagonals=True,
                      partition_method = 'square_root_of_slope')
# fd = FlowDirectorDINF(mg)
# fd = FlowDirectorD8(mg)
fd.run_one_step()


plt.figure('grid and flow directions',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])

#%%
# create pile
# find central point in domain
x = mg.node_x.max()/2
y = mg.node_y.max()/2

# find all nodes with radius of central point
dn = ((mg.node_x-x)**2+(mg.node_y-y)**2)**0.5
pile_nodes = np.hstack(mg.nodes)[dn<w*mg.dx]


# nd = int((np.array(mg.nodes.shape[0])-w)/2)
# pile_nodes = mg.nodes[nd:r-nd,nd:r-nd]


mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
mg.at_node['mass__wasting_id'][pile_nodes] = 1
# set thickness of landslide
mg.at_node['soil__thickness'][pile_nodes] = ls_h

mg.at_node['topographic__elevation'][pile_nodes] =   mg.at_node['topographic__elevation'][pile_nodes]+(ls_h-hs)


# plt.savefig("Flume.png", dpi = 300, bbox_inches='tight')
plt.figure('3d view', figsize=(8, 10))
LLT.surf_plot(mg,title = 'initial dem [m]', zr= 1, color_type = "grey", dv = 100,
                      constant_scale = True,s =-0.5, nv = 100 , zlim_min = 0,  elev = 35, azim = -130)


field = 'node_id'
field_back= "topographic__elevation"
plot_values(mg,field,xmin,xmax,ymin,ymax)


#%% set up MWR
# mass wasting ide


# run parameters
npu = [1] 
nid = [1] 



# select alpha as a functionn of qsc
E_l_alpha = (qsc/mg.dx)*lam
if g_erosion:
    alpha, Tau = determine_alpha(ros, vs, h, s, eta, E_l_alpha, mg.dx, slpc = slpc, Dp = Dp)
else:
    alpha, Tau = determine_alpha(ros, vs, h, s, eta, E_l_alpha, mg.dx, slpc = slpc)#, Dp = Dp)

# E needs to be less than qsc. alpha*(E_l*delta_x)**eta < qsc
# params_o = [0.01, 0.01, 0.5]
params_o = [slpc, qsc, alpha]
slpc = [params_o[0]]   
SD = params_o[1]
cs = params_o[2]




mw_dict = {'critical slope':slpc, 'minimum flux':SD,
            'scour coefficient':cs, 'scour exponent':eta,
            'effective particle diameter':Dp, 'vol solids concentration':vs,
            'density solids':ros, 'typical flow thickness, scour':h,
            'typical slope, scour':s}

release_dict = {'number of pulses':npu, 'iteration delay':nid }

MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True, itL = 1000,
                                  dist_to_full_flux_constraint = 0,
                                  routing_surface = "topographic__elevation",
                                  settle_deposit = settle_deposit,
                                  deposition_rule = deposition_rule,
                                  deposit_style = deposit_style,
                                  anti_sloshing = False,
                                  effective_qsi = effective_qsi)

#%% run
start_time = time.time()
MWRu.run_one_step(dt = 0)

print("--- %s seconds ---" % (time.time() - start_time))
#%% view obseved runout
mg.at_node['dem_dif_o'] = mg.at_node['topographic__elevation']-mg.at_node['topographic__initial_elevation']
LLT.plot_node_field_with_shaded_dem(mg,field = 'dem_dif_o', fontsize = 10,cmap = 'RdBu_r', plot_name = 'hillshade')
plt.clim(-0.5,0.5)

field = "dem_dif_o"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r')
plt.clim(-1,1)

#%% mass conservation check
DEMi = mg.at_node['topographic__initial_elevation']
DEMf = mg.at_node['topographic__elevation']
DEMdf = DEMf-DEMi
# assert DEMdf.sum()*mg.dx*mg.dy == 75, 'not equal'
print("difference in initial and final dem [m3] is:{}".format(np.round(DEMdf.sum()*mg.dx*mg.dy, decimals = 8)))



if PlanVisualize:
    # plot how DEM changes
    for i in np.arange(0,len(MWRu.mw_ids)):
    
        for c in MWRu.df_evo_maps[i].keys():                  
            plt.figure('topo+thick'+str(c)+str(i),figsize=(12, 12))
            mg.at_node['df_topo'] = MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
            mg.at_node['topo'] = MWRu.df_evo_maps[i][c]
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
            LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo', fontsize = 10,cmap = 'RdBu_r', plot_name = 'topo + thick{},{}'.format(i,c) )
            field = "node_id"
            # plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back, background = False)
            plt.xticks(fontsize= 8 )
            plt.yticks(fontsize= 8 )
            plt.clim(-1,1)
            plt.xlim([xmin*.8,xmax*1.2]); plt.ylim([ymin*.3,ymax])

            if c >50:
                break
            
        # for c in MWRu.df_evo_maps[i].keys():   
        #     plt.figure('topo'+str(c)+str(i),figsize=(12, 12))
        #     mg.at_node['df_topo'] = MWRu.topo_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
        #     mg.at_node['topo'] = MWRu.topo_evo_maps[i][c]
        #     # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
        #     LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo', fontsize = 10,cmap = 'RdBu_r', plot_name = 'topo{},{}'.format(i,c) )
        #     field = "topo"
        #     # plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back, background = False)
        #     plt.xticks(fontsize= 8 )
        #     plt.yticks(fontsize= 8 )
        #     plt.clim(-2.5,2.5)
        #     plt.xlim([xmin*.8,xmax*1.2]); plt.ylim([ymin*.3,ymax])        

#%% evolving profile

if ProfileVisualize:
    # plot how DEM changes
    pf = mg.nodes[int((r-1)/2),:]
    x_ = mg.node_x[pf]
    y = mg.at_node['topographic__initial_elevation'][pf]
     
       
    for i in np.arange(0,len(MWRu.mw_ids)):

        for c in MWRu.df_evo_maps[i].keys():                  
            etopo = MWRu.df_evo_maps[i][c]#-mg.at_node['topographic__initial_elevation']
            topo = MWRu.topo_evo_maps[i][c]
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')
            # if c>100:
            #     break                  
            y_ = topo[pf]
            _y_ = etopo[pf]
            plt.figure(figsize = (6,3))
            plt.plot(x_,y,'k--', alpha = 0.5, linewidth = 1,label = 'initial ground profile')
            plt.plot(x_,y_,'g-', alpha = 0.5, linewidth = 1, label = 'post ground profile')
            plt.plot(x_,_y_,'r-', alpha = 0.5, linewidth = 1, label = 'df profile')
            plt.ylim(0,ls_h*3); plt.ylabel(' elevation ')
            plt.xlim([0, max(mg.node_x)])
            plt.legend()
            plt.grid(alpha = 0.5)  
            plt.title('iteration '+str(c))
            if c >50:
                break

    # for i in np.arange(0,len(MWRu.mw_ids)):

    #     for c in MWRu.df_evo_maps[i].keys():                  
    #         topo = MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
    #         slope = MWRu.tss_r[i+1][c].max(axis = 1)
    #         mg.at_node['slope'] = slope
    #         e_el = MWRu.df_evo_maps[i][c]
    #         mg.at_node['e_el'] = e_el
    #         mg.at_node['topographic__elevation'] = topo
    #         plt.figure('grid and flow directions'+str(c),figsize=(12, 6))
    #         receivers = MWRu.frn_r[1][c]
    #         proportions = MWRu.frp_r[1][c]
    #         LLT.drainage_plot_jk(mg, receivers = receivers, proportions = proportions, title='Basic Ramp'+str(c),surf_cmap="RdBu_r",clim = [-2.5,10])
    #         field = "e_el"
    #         plot_values(mg,field,xmin,xmax,125,250,field_back= field_back, background = False)
    #         plt.ylim(125, 250)
    
