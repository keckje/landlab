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
# high res hillshade for plot background
mg_hs, hs = read_esri_ascii(dem_dir+bdfdem_hs, name='hillshade')
mg_hs.at_node['hillshade_arc'] = hs

# dem
dem_dir = 'D:/UW_PhD/PreeventsProject/Paper_2_MWR/RunoutValidation/CapitolForest/maps/layers/two_models/camp4/'
mg, z = read_esri_ascii(dem_dir+bdfdem, name='topographic__elevation')
mg.set_watershed_boundary_condition(z) # finds lowest point in dem and sets it as an open node


# extent for zoom-n plot of flow directions
mvx = 491300; mvx_ = 491500
mvy = 5203300; mvy_ = 5203500


#%% ploting functions from notebook
# create a plotting routine to make a 3d plot of our surface.
def surf_plot(mg, surface="topographic__elevation", title="Surface plot of topography"):

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    Z = mg.at_node[surface].reshape(mg.shape)
    color = cm.gray((Z - Z.min()) / (Z.max() - Z.min()))
    surf = ax.plot_surface(
        mg.x_of_node.reshape(mg.shape),
        mg.y_of_node.reshape(mg.shape),
        Z,
        rstride=1,
        cstride=1,
        facecolors=color,
        linewidth=0.0,
        antialiased=False,
    )
    ax.view_init(elev=35, azim=-120)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Elevation")
    plt.title(title)
    plt.show()
    
    
def plotting(
    grid, topo=True, DA=True, hill_DA=False, flow_metric="D8", hill_flow_metric="Quinn"
):
    if topo:
        azdeg = 200
        altdeg = 20
        ve = 1
        plt.figure()
        plot_type = "DEM"
        ax = imshowhs_grid(
            grid,
            "topographic__elevation",
            grid_units=("deg", "deg"),
            var_name="Topo, m",
            cmap="terrain",
            plot_type=plot_type,
            vertical_exa=ve,
            azdeg=azdeg,
            altdeg=altdeg,
            default_fontsize=12,
            cbar_tick_size=10,
            cbar_width="100%",
            cbar_or="vertical",
            bbox_to_anchor=[1.03, 0.3, 0.075, 14],
            colorbar_label_y=-15,
            colorbar_label_x=0.5,
            ticks_km=False,
        )
    if DA:
        # % Plot first instance of drainage_area
        grid.at_node["drainage_area"][grid.at_node["drainage_area"] == 0] = (
            grid.dx * grid.dx
        )
        plot_DA = np.log10(grid.at_node["drainage_area"] * 111e3 * 111e3)

        plt.figure()
        plot_type = "Drape1"
        drape1 = plot_DA
        thres_drape1 = None
        alpha = 0.5
        myfile1 = "temperature.cpt"
        cmap1 = "terrain"
        ax = imshowhs_grid(
            grid,
            "topographic__elevation",
            grid_units=("deg", "deg"),
            cmap=cmap1,
            plot_type=plot_type,
            drape1=drape1,
            vertical_exa=ve,
            azdeg=azdeg,
            altdeg=altdeg,
            thres_drape1=thres_drape1,
            alpha=alpha,
            default_fontsize=12,
            cbar_tick_size=10,
            var_name="$log^{10}DA, m^2$",
            cbar_width="100%",
            cbar_or="vertical",
            bbox_to_anchor=[1.03, 0.3, 0.075, 14],
            colorbar_label_y=-15,
            colorbar_label_x=0.5,
            ticks_km=False,
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0.6)
        textstr = flow_metric
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

    if hill_DA:
        # Plot second instance of drainage_area (hill_drainage_area)
        grid.at_node["hill_drainage_area"][grid.at_node["hill_drainage_area"] == 0] = (
            grid.dx * grid.dx
        )
        plotDA = np.log10(grid.at_node["hill_drainage_area"] * 111e3 * 111e3)
        # plt.figure()
        # imshow_grid(grid, plotDA,grid_units=("m", "m"), var_name="Elevation (m)", cmap='terrain')

        plt.figure()
        plot_type = "Drape1"
        # plot_type='Drape2'
        drape1 = np.log10(grid.at_node["hill_drainage_area"])
        thres_drape1 = None
        alpha = 0.5
        myfile1 = "temperature.cpt"
        cmap1 = "terrain"
        ax = imshowhs_grid(
            grid,
            "topographic__elevation",
            grid_units=("deg", "deg"),
            cmap=cmap1,
            plot_type=plot_type,
            drape1=drape1,
            vertical_exa=ve,
            azdeg=azdeg,
            altdeg=altdeg,
            thres_drape1=thres_drape1,
            alpha=alpha,
            default_fontsize=10,
            cbar_tick_size=10,
            var_name="$log^{10}DA, m^2$",
            cbar_width="100%",
            cbar_or="vertical",
            bbox_to_anchor=[1.03, 0.3, 0.075, 14],
            colorbar_label_y=-15,
            colorbar_label_x=0.5,
            ticks_km=False,
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0.6)
        textstr = hill_flow_metric
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )
        
#%% view using default FlowAccumulator

start_time = time.time()
fa_LL = FlowAccumulator(
    mg, flow_director="D8", depression_finder="DepressionFinderAndRouter"
)
fa_LL.run_one_step()

print("--- %s seconds ---" % (time.time() - start_time))
# Plot output products
plotting(mg)


#%% multiflow using default FlowAccumulator
# mg.delete_field(loc = 'node', name = 'flow__sink_flag')
# mg.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
# mg.delete_field(loc = 'node', name = 'flow__receiver_node')
# mg.delete_field(loc = 'node', name = 'topographic__steepest_slope')
# mg.delete_field(loc = 'node', name = "flow__receiver_proportions")

# run flow director, add slope and receiving node fields
start_time = time.time()
fd = FlowDirectorMFD(mg, diagonals=True,
                      partition_method = 'slope')
fd.run_one_step()
print("--- %s seconds ---" % (time.time() - start_time))

#%% try using PriorityFloodFlowRouter
# # Here, we only calculate flow directions using the first instance of the flow accumulator
mg.delete_field(loc = 'node', name = 'flow__sink_flag')
mg.delete_field(loc = 'node', name = 'flow__link_to_receiver_node')
mg.delete_field(loc = 'node', name = 'flow__receiver_node')
mg.delete_field(loc = 'node', name = 'topographic__steepest_slope')
# mg.delete_field(loc = 'node', name = "flow__receiver_proportions")



flow_metric = "D8"
start_time = time.time()
fa_PF = PriorityFloodFlowRouter(
    mg,
    surface="topographic__elevation",
    flow_metric=flow_metric,
    suppress_out=True,
    depression_handler="fill",
    accumulate_flow=True,
)

fa_PF.run_one_step()
print("--- %s seconds ---" % (time.time() - start_time))

# Plot output products
plotting(mg)

#%% PriorityFloodFlowRouter with multiflow director
flow_metric = "D8"
hill_flow_metric = "Quinn"
start_time = time.time()
fa_PF = PriorityFloodFlowRouter(
    mg,
    surface="topographic__elevation",
    flow_metric=flow_metric,
    suppress_out=True,
    depression_handler="fill",
    accumulate_flow=True,
    separate_hill_flow=True,
    accumulate_flow_hill=True,
    update_hill_flow_instantaneous=False,
    hill_flow_metric=hill_flow_metric,
)


fa_PF.run_one_step()
fa_PF.update_hill_fdfa()
print("--- %s seconds ---" % (time.time() - start_time))
# 4. Plot output products
plotting(mg, hill_DA=True, flow_metric="D8", hill_flow_metric="Quinn")


# conclusion: PriorityFloodFlowrouter may not be quicker for this model scale

#%%
surf_plot(mg, title="Grid 1")


#%% # use D8, PriorityFloodFlowRouter
# # this is the same as writing:
fa = PriorityFloodFlowRouter(
    mg,
    surface="topographic__elevation",
    flow_metric="D8",
    update_flow_depressions=True,
    runoff_rate=None,
    depression_handler="fill",
)

start_time = time.time()
fa.run_one_step()
print("--- %s seconds ---" % (time.time() - start_time))

plt.figure()
drainage_plot(mg)
plt.xlim([mvx,mvx_])
plt.ylim([mvy, mvy_])

plt.figure()
drainage_plot(mg, "drainage_area",surf_cmap='prism')
plt.xlim([mvx,mvx_])
plt.ylim([mvy, mvy_])

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
fa.run_one_step()
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
plt.clim(-1,1)


MWF.profile_plot(mg, xsd, 'topographic__elevation', ef = 2, figsize = (10,10))
MWF.profile_plot(mg, xsd, 'depression_free_dem', ef = 2, figsize = (10,10))

# dem changes, but changes do not remove bumps in channel flood plane.

