# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 07:23:30 2022

@author: keckj

seperate set of tests used to confirm routing and sloshing problem

"""

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


#%% functions used in tests
#######
# use this to get values for manual computations
###

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
        
        

def example_square_mg():
    """ sloped, convergent, irregular surface"""
    dem = np.array([[10,8,4,3,4,7.5,10],[10,9,3.5,4,5,8,10],
                    [10,9,6.5,5,6,8,10],[10,9.5,7,6,7,9,10],[10,10,9.5,8,9,9.5,10],
                    [10,10,10,10,10,10,10],[10,10,10,10,10,10,10]])

    dem = np.hstack(dem).astype(float)
    mg = RasterModelGrid((7,7),10)
    _ = mg.add_field('topographic__elevation',
                        dem,
                        at='node')
    
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries
    mg.set_watershed_boundary_condition_outlet_id(3,dem)   
    mg.at_node['node_id'] = np.hstack(mg.nodes)
    # fd = FlowDirectorMFD(mg, diagonals=True,
    #                       partition_method = 'slope')
    # fd.run_one_step()
    nn = mg.number_of_nodes
    mg.at_node['mass__wasting_id'] = np.zeros(nn).astype(int)
    mg.at_node['mass__wasting_id'][np.array([38])] = 1  
    depth = np.ones(nn)*0
    mg.add_field('node', 'soil__thickness',depth)
    np.random.seed(seed=7)
    mg.at_node['particle__diameter'] = np.random.uniform(0.05,0.25,nn)
    return(mg)



#%% set up the grid
pdir = "D:/UW_PhD/PreeventsProject/Paper_2_MWR/Landlab_Development/mass_wasting_runout/development_plots/tests_version2/"


mg = example_square_mg()

dem = mg.at_node['topographic__elevation']



# mass wasting locaiton
mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
mg.at_node['mass__wasting_id'][23:25] = 1  

mg.at_node['soil__thickness'] = np.zeros(mg.number_of_nodes).astype(int)
mg.at_node['soil__thickness'][23] = 2
mg.at_node['soil__thickness'][24] = 3.5

mg.at_node['topographic__elevation'][23] = mg.at_node['topographic__elevation'][23]+2
mg.at_node['topographic__elevation'][24] = mg.at_node['topographic__elevation'][24]+3.5



# domain for plots
xmin = mg.node_x.min(); xmax = mg.node_x.max(); ymin = mg.node_y.min(); ymax = mg.node_y.max()

# set boundary conditions, add flow direction
mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries


mg.set_watershed_boundary_condition_outlet_id(2,dem)
    
mg.at_node['node_id'] = np.hstack(mg.nodes)

# flow directions
fa = FlowAccumulator(mg, 
                      'topographic__elevation',
                      flow_director='FlowDirectorD8')
fa.run_one_step()

# hillshade for plots
mg.at_node['hillshade'] = mg.calc_hillshade_at_node(elevs=dem, alt=37., az=210.)



# copy of initial topography
DEMi = mg.at_node['topographic__elevation'].copy()

#%% view the dem

plt.figure('3d view_side', figsize=(8, 10))
LLT.surf_plot(mg,title = 'initial dem [m] side', zr= 1, color_type = "grey", dv = 100,
                      constant_scale = True,s =-0.5, nv = 100 , zlim_min = 0,  elev = 35, azim = -180)
# plt.savefig("Flume.png", dpi = 300, bbox_inches='tight')
plt.figure('3d view', figsize=(8, 10))
LLT.surf_plot(mg,title = 'initial dem [m]', zr= 1, color_type = "grey", dv = 100,
                      constant_scale = True,s =-0.5, nv = 100 , zlim_min = 0,  elev = 35, azim = -130)


field = 'node_id'
field_back= "topographic__elevation"
plot_values(mg,field,xmin,xmax,ymin,ymax)


field = 'topographic__steepest_slope'
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)


field= "topographic__elevation"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)

field = 'mass__wasting_id'
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)

field = 'soil__thickness'
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)




#%% Add multiflow direction

# run flow director, add slope and receiving node fields
mg.delete_field(loc = 'node', name = 'flow__sink_flag')
mg.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
mg.delete_field(loc = 'node', name = 'flow__receiver_node')
mg.delete_field(loc = 'node', name = 'topographic__steepest_slope')

# run flow director, add slope and receiving node fields
fd = FlowDirectorMFD(mg, diagonals=True,
                      partition_method = 'slope')
# fd = FlowDirectorDINF(mg)
# fd = FlowDirectorD8(mg)
fd.run_one_step()


plt.figure('grid and flow directions',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])


#%% instantiate


# run parameters
npu = [1] 
nid = [1] 
slpc = [0.03]   
SD = 0.2
cs = 0.2


mw_dict = {'critical slope':slpc, 'minimum flux':SD,
            'scour coefficient':cs}

release_dict = {'number of pulses':npu, 'iteration delay':nid }

example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                  routing_surface = "topographic__elevation",
                                  settle_deposit = False, average_velocity= 0,
                                  deposition_rule = "critical_slope", itL = 50)

#%% run
example_MWRu.run_one_step(dt = 0)
mg.at_node['dem_dif_o'] = mg.at_node['topographic__elevation']-mg.at_node['topographic__initial_elevation']

#%% view obseved runout
LLT.plot_node_field_with_shaded_dem(mg,field = 'dem_dif_o', fontsize = 10,cmap = 'RdBu_r', plot_name = 'hillshade')
plt.clim(-1,1)

field = "dem_dif_o"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r')
plt.clim(-1,1)



Visualize = True
if Visualize:
    # plot how DEM changes
    for i in np.arange(0,len(example_MWRu.mw_ids)):
    
        for c in example_MWRu.df_evo_maps[i].keys():                  
            plt.figure('topo+thick'+str(c)+str(i),figsize=(12, 12))
            mg.at_node['df_topo'] = example_MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
            mg.at_node['topo'] = example_MWRu.df_evo_maps[i][c]
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
            LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo', fontsize = 10,cmap = 'RdBu_r', plot_name = 'topo + thick{},{}'.format(i,c) )
            field = "topo"
            plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back, background = False)
            plt.xticks(fontsize= 8 )
            plt.yticks(fontsize= 8 )
            plt.clim(-2.5,2.5)
            plt.xlim([xmin*.8,xmax*1.2]); plt.ylim([ymin*.3,ymax])
            

            plt.figure('topo'+str(c)+str(i),figsize=(12, 12))
            mg.at_node['df_topo'] = example_MWRu.topo_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
            mg.at_node['topo'] = example_MWRu.topo_evo_maps[i][c]
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
            LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo', fontsize = 10,cmap = 'RdBu_r', plot_name = 'topo{},{}'.format(i,c) )
            field = "topo"
            plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back, background = False)
            plt.xticks(fontsize= 8 )
            plt.yticks(fontsize= 8 )
            plt.clim(-2.5,2.5)
            plt.xlim([xmin*.8,xmax*1.2]); plt.ylim([ymin*.3,ymax])        



    for i in np.arange(0,len(example_MWRu.mw_ids)):

        for c in example_MWRu.df_evo_maps[i].keys():                  
            topo = example_MWRu.df_evo_maps[i][c]#-mg.at_node['topographic__initial_elevation']
            mg.at_node['topographic__elevation'] = topo
            plt.figure('grid and flow directions'+str(c),figsize=(8, 10))
            receivers = example_MWRu.frn_r[1][c]
            proportions = example_MWRu.frp_r[1][c]
            LLT.drainage_plot_jk(mg, receivers = receivers, proportions = proportions, title='Basic Ramp'+str(c),surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])