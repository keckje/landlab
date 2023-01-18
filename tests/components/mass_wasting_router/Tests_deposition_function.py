# -*- coding: utf-8 -*-
"""

visualize tests for deposition rule, using a pit
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


#%% functions for script


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

    dem = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0.5,1,1],[1,1,1,1,1],[1,1,1,1,1]])


    dem = np.hstack(dem).astype(float)
    mg = RasterModelGrid((5,5),10)
    _ = mg.add_field('topographic__elevation',
                        dem,
                        at='node')
    
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries
    mg.set_watershed_boundary_condition_outlet_id(3,dem)   
    mg.at_node['node_id'] = np.hstack(mg.nodes)
    fd = FlowDirectorMFD(mg, diagonals=True,
                          partition_method = 'slope')
    fd.run_one_step()
    nn = mg.number_of_nodes
    mg.at_node['mass__wasting_id'] = np.zeros(nn).astype(int)
    # mg.at_node['mass__wasting_id'][np.array([38])] = 1  
    depth = np.ones(nn)*1
    mg.add_field('node', 'soil__thickness',depth)
    np.random.seed(seed=7)
    mg.at_node['particle__diameter'] = np.random.uniform(0.05,0.25,nn)
    return (mg)

mg = example_square_mg()
# domain for plots
xmin = mg.node_x.min(); xmax = mg.node_x.max(); ymin = mg.node_y.min(); ymax = mg.node_y.max()
#%% plot node id and elevation

plt.figure()
plot_values(mg,'node_id',xmin,xmax,ymin,ymax)
plt.figure()
plot_values(mg,'topographic__elevation',xmin,xmax,ymin,ymax)


#%% set up mass wasting runout

# mass wasting ide
mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
mg.at_node['mass__wasting_id'][12] = 1  

# run parameters
npu = [1] 
nid = [1] 

params_o = [0.03, 0.1, 0.2]
slpc = [params_o[0]]   
SD = params_o[1]
cs = params_o[2]


mw_dict = {'critical slope':slpc, 'minimum flux':SD,
            'scour coefficient':cs}

release_dict = {'number of pulses':npu, 'iteration delay':nid }

MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True, itL = 100,
                                  dist_to_full_flux_constraint = 0,
                                  routing_surface = "energy__elevation",
                                  settle_deposit = False,
                                  deposition_rule = "critical_slope")


#%% test 1, zo = zi
# add 1 m to node 17
mg.at_node['topographic__elevation'][17] = mg.at_node['topographic__elevation'][17] +1
plt.figure(label = 'before')
plot_values(mg,'topographic__elevation',xmin,xmax,ymin,ymax)

print(MWRu._deposit_friction_angle(1,24))
print(MWRu._deposit_friction_angle_v2(1,24))
#%% test 2, make adjacent nodes = zi+qsi
mg.at_node['topographic__elevation'][np.array([16,17,18])] = mg.at_node['topographic__elevation'][np.array([16,17,18])] + np.array([0.5,1,1])

plt.figure(label = 'before')
plot_values(mg,'topographic__elevation',xmin,xmax,ymin,ymax)


print(MWRu._deposit_friction_angle(1,24))
print(MWRu._deposit_friction_angle_v2(1,24))

#%% test 3, zi = zo+qsi+hs
mg.at_node['topographic__elevation'][17] = mg.at_node['topographic__elevation'][17] -2.3
plt.figure()
plot_values(mg,'topographic__elevation',xmin,xmax,ymin,ymax)

print(MWRu._deposit_friction_angle(1,24))
print(MWRu._deposit_friction_angle_v2(1,24))

#%% test 4, zi > zo+qsi+hs

mg.at_node['topographic__elevation'][17] = mg.at_node['topographic__elevation'][17] -1
plt.figure()
plot_values(mg,'topographic__elevation',xmin,xmax,ymin,ymax)

print(MWRu._deposit_friction_angle(1,24))
print(MWRu._deposit_friction_angle_v2(1,24))

#%% test where material is routed too in each of the above scenarios
dem = mg.at_node['topographic__elevation']
plt.figure('grid and flow directions',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])