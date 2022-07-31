# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 11:43:46 2022

@author: keckj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 09:10:45 2022

@author: keckj

Test using Flow_Director_Accumulator_PriorityFlood to remove bumps in DEM and to route

following the_Flow_Director_Accumulator_PriorityFlood jupyter notebook
"""

# setup
import os
import time
import copy as cp
import richdem as rd
## import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib as mpl


import pandas as pd
import numpy as np

## import necessary landlab components
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
from landlab.components import(FlowDirectorD8, 
                                FlowDirectorDINF, 
                                FlowDirectorMFD, 
                                FlowDirectorSteepest)

from landlab.components.mass_wasting_router import MassWastingRunout

## import landlab plotting functionality
from landlab.plot.drainage_plot import drainage_plot
from landlab import imshow_grid_at_node

## import functions
from landlab.io.esri_ascii import write_esri_ascii
from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf

from landlab.components import ChannelProfiler, FlowAccumulator, DepressionFinderAndRouter

from landlab.components import PriorityFloodFlowRouter

os.chdir('C:/Users/keckj/Documents/GitHub/code/landlab/LandlabTools')
import LandlabTools as LLT


os.chdir('C:/Users/keckj/Documents/GitHub/code/preevents/paper2/')
import MassWastingRunoutEvaluationFunctions as MWF

from landlab import imshowhs_grid, imshow_grid

#%% load dem
bdfdem = 'd0510m_c4.asc' # before debris flow dem
bdfdem_hs = 'hs_05_2m_c4.asc' # before debris flow dem hs


# dem
dem_dir = 'D:/UW_PhD/PreeventsProject/Paper_2_MWR/RunoutValidation/CapitolForest/maps/layers/two_models/camp4/'
mg, z = read_esri_ascii(dem_dir+bdfdem, name='topographic__elevation')
mg.set_watershed_boundary_condition(z) # finds lowest point in dem and sets it as an open node

# high res hillshade for plot background
mg_hs, hs = read_esri_ascii(dem_dir+bdfdem_hs, name='hillshade')
mg_hs.at_node['hillshade_arc'] = hs


#%% DEM smoothing

# cross section xlsx file name
xlsxm = 'profile_nodes_Camp4_t2.xlsx'
# cross section file location
xs_dir = 'D:/UW_PhD/PreeventsProject/Paper_2_MWR/RunoutValidation/CapitolForest/maps/layers/two_models/camp4/'

cl = pd.read_excel(xs_dir+xlsxm,sheet_name = 'cl')
cln = cl['MAJORITY'].values
xsd = cln

#%%re-load dem

mg, z = read_esri_ascii(dem_dir+bdfdem, name='topographic__elevation')
# save an copy of the initial elevation for dem differencing
# _ = mg.add_field('topographic__initial_elevation',
#                     mg.at_node['topographic__elevation'],
#                     at='node',
#                     copy = True)
mg.at_node['topographic__initial_elevation'] = cp.deepcopy(mg.at_node['topographic__elevation'])
mg.at_node['dem_dif'] = mg.at_node['topographic__initial_elevation'] - mg.at_node['topographic__elevation']
mg.set_watershed_boundary_condition(z) # finds lowest point in dem and sets it as an open node

# no difference in DEMS
print("max difference in dems: {}".format(mg.at_node['dem_dif'].max()))


#%%

flow_metric = "D8"
hill_flow_metric = "Quinn"
start_time = time.time()
fa_PF = PriorityFloodFlowRouter(
    mg,
    surface="topographic__elevation",
    flow_metric=flow_metric,
    suppress_out=True,
    depression_handler="breach",
    accumulate_flow=True,
    separate_hill_flow=True,
    accumulate_flow_hill=True,
    update_hill_flow_instantaneous=True,
    hill_flow_metric=hill_flow_metric,
)

start_time = time.time()
fa_PF.run_one_step()
print("--- %s seconds ---" % (time.time() - start_time))

mg.at_node['dem_dif'] = mg.at_node['topographic__initial_elevation'] - mg.at_node['topographic__elevation']
print("max difference in dems: {}".format(mg.at_node['dem_dif'].max()))

# DEM does not seem to change
#%% try using rd 

dem = cp.deepcopy(rd.rdarray(
    mg.at_node["topographic__elevation"].reshape(mg.shape),
    no_data=-9999))
dem.geotransform = [0, 1, 0, 0, 0, -1]


rd.BreachDepressions(
  dem,
  in_place = True,
  topology = 'D8'
)

dem_1d = np.array(dem.reshape(mg.number_of_nodes))
mg.at_node['depression_free_dem'] = dem_1d

mg.at_node['dem_dif'] = mg.at_node['topographic__initial_elevation'] - mg.at_node['depression_free_dem']
print("max difference in dems: {}".format(mg.at_node['dem_dif'].max()))


plt.figure()
LLT.plot_node_field_with_shaded_dem_Arc(mg, mg_hs = mg_hs, field = 'dem_dif',fontsize = 10,cmap = 'RdBu_r',alpha = .5)
plt.clim(-.1,.1)


MWF.profile_plot(mg, xsd, 'topographic__elevation', ef = 2, figsize = (10,10))
MWF.profile_plot(mg, xsd, 'depression_free_dem', ef = 2, figsize = (10,10))

# dem changes, but changes do not remove bumps in channel flood plane.

