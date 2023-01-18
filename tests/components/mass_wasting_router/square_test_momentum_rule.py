# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:23:58 2022

@author: keckj
"""

#%% flume
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


#######
# use this to get values for manual computations
###
def plot_values(mg,field,xmin,xmax,ymin,ymax):
    """plot the field values on the node within the specified domain extent"""
    values = mg.at_node[field]
    if values.dtype != int:
        values = values.round(decimals = 4)
    values_test_domain = values[(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]
    ntd = mg.at_node['node_id'][(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]
    for i,val in enumerate(values_test_domain):
        plt.text(mg.node_x[ntd[i]],mg.node_y[ntd[i]],str(val),color = 'red', fontsize = 7)
    

dem = np.array([[10,8,4,3,4,7.5,10],[10,9,3.5,4,5,8,10],
                [10,9,6.5,5,6,8,10],[10,9.5,7,6,7,9,10],[10,10,9.5,8,9,9.5,10],
                [10,10,10,10,10,10,10],[10,10,10,10,10,10,10]])


# cc+=1
dem = np.hstack(dem).astype(float)
mg = RasterModelGrid((7,7),10)
_ = mg.add_field('topographic__elevation',
                    dem,
                    at='node')

mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries
mg.set_watershed_boundary_condition_outlet_id(3,dem)   
mg.at_node['node_id'] = np.hstack(mg.nodes)
# run flow director, add slope and receiving node fields
fd = FlowDirectorMFD(mg, diagonals=True,
                      partition_method = 'slope')
fd.run_one_step()
# set up mass wasting router
nn = mg.number_of_nodes
# mass wasting ide
mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
# mg.at_node['mass__wasting_id'][np.array([38])] = 1  
mg.at_node['mass__wasting_id'][np.array([31,38])] = np.array([1,1]) 
# soil depth
depth = np.ones(nn)*1
mg.add_field('node', 'soil__thickness',depth)



# slope, aspect = mg.calculate_slope_aspect_at_nodes_burrough(vals='topographic__elevation',)
# sfb = SinkFillerBarnes(mg,'topographic__elevation', method='D8',fill_flat = False, ignore_overfill = False)
# sfb.run_one_step()
# write_esri_ascii(dem_dir_model+'CapitolForestModelNodeIds.asc', mg, 'node_id')

DEMi = mg.at_node['topographic__elevation'].copy()


# hillshade for plots
mg.at_node['hillshade'] = mg.calc_hillshade_at_node(elevs=dem, alt=37., az=210.)

# particle diameter
np.random.seed(seed=7)
nn = mg.number_of_nodes
mg.at_node['particle__diameter'] = np.random.uniform(0.05,0.25,nn)

# view test topography


# inn = 236 # initial cell (node) where debris flow material begins
# plot grid, flow directions and initial cell location


plt.figure('grid and flow directions',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])
# plt.savefig(pdir+"WideFlume_FlowDirections.png".format(c), dpi = 300, bbox_inches='tight')

# plt.plot(mg.x_of_node[inn],mg.y_of_node[inn],'r.',markersize =22)


xmin = mg.node_x.min(); xmax = mg.node_x.max(); ymin = mg.node_y.min(); ymax = mg.node_y.max()
field = 'node_id'

plt.figure(field,figsize = (12,8))
imshow_grid(mg,'topographic__elevation',grid_units=('m','m'),var_name='Elevation(m)',plot_name = field,cmap = 'terrain')
plot_values(mg,field,xmin,xmax,ymin,ymax)
plt.xlim([xmin,xmax]); plt.ylim([ymin,ymax])
z_d = dem[(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]



# get receiveiving ndoes at 24
n = 24
adj_n = np.hstack((mg.adjacent_nodes_at_node[n],
mg.diagonal_adjacent_nodes_at_node[n]))


receivers[n]

#%%

# qsiti = np.zeros(mg.number_of_nodes) #qsi at t = i
# qsiti[np.array([17,24,25,31])] = np.array([2,1.5,1.5,1])
# energy_elevation = qsiti+mg.at_node['topographic__elevation']
# mg.add_field('node', 'energy__elevation',energy_elevation)




# run parameters
npu = [1] 
nid = [1] 
slpc = [0.03]   
SD = 0.01
cs = 0.02

mw_dict = {'critical slope':slpc, 'minimum flux':SD,
            'scour coefficient':cs}

release_dict = {'number of pulses':npu, 'iteration delay':nid }


example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                  routing_surface = "energy__elevation", settle_deposit = True)

# example_MWRu._grid.at_node['soil_thickness'] = np.ones(nn)*0.01

example_MWRu.itL = 0

example_MWRu.run_one_step(dt = 0)


plt.figure(field+'_post',figsize = (12,8))
imshow_grid(mg,'topographic__elevation',grid_units=('m','m'),var_name='Elevation(m)',plot_name = field,cmap = 'terrain')
plot_values(mg,field,xmin,xmax,ymin,ymax)
plt.xlim([xmin,xmax]); plt.ylim([ymin,ymax])
z_d = dem[(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]




# plot how DEM changes
for i in np.arange(0,len(example_MWRu.mw_ids)):

    for c in example_MWRu.df_evo_maps[i].keys():   
        if c<50:             
            plt.figure('dif'+str(c)+str(i),figsize=(12, 12))
            mg.at_node['df_topo_d'] = example_MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
            LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo_d', fontsize = 10,cmap = 'RdBu_r', plot_name = 'dem dif{},{}'.format(i,c) )
    
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
            plt.xticks(fontsize= 8 )
            plt.yticks(fontsize= 8 )
            plt.clim(-1,1)
            # plt.savefig(pdir+'DebrisFlowPlanView_{}.png'.format(c), dpi = 300, bbox_inches='tight')

dem = mg.at_node['topographic__elevation']
plt.figure('grid and flow directions',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])