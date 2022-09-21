# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 05:07:38 2022

@author: keckj


Are the calibration functions working correctly? draft tests for test file
functions to test

simulation


# sloshing issue; deposit, slope updated, steep, 

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
def plot_values(mg,field,xmin,xmax,ymin,ymax, field_back = "topographic__elevation", cmap = 'terrain'):
    """plot the field values on the node within the specified domain extent"""
    
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
        
        
def flume_maker(rows = 5, columns = 3, slope_above_break =.5, slope_below_break =.05, slope_break = 0.7, ls_width = 1, ls_length = 1, dxdy= 10):
    """
    

    Parameters
    ----------
    rows : integer
        number of rows in model domain. First and last row are used as boundaries
    columns : integer
        number of columns in domain. First and last column are used as boundaries
    slope_above_break : float
        slope of flume above break [m/m]
    slope_below_break : float
        slope of flume below break [m/m]
    slope_break : float
        ratio of length the slope break is placed, measured from the outlet
        slope break will be placed at closest node to slope_break value
    ls_width : odd value, integer
        width of landslide in number of cells, , must be <= than rows-2
    ls_length : integer
        length of landslide in number of cells, must be <= than rows-2
    dxdy : float
        side length of sqaure cell, [m]. The default is 10.

    Returns
    -------
    mg : raster model grid
        includes the field topographic__elevation
    lsn : np array
        0-d array of node id's that are the landslide
    pf : np array
        0-d array of node id's along the center of the flume. Used for profile
        plots.
    cc : np array
        0-d array of landslide node column id's'

    """
    
    r = rows; #rows
    c = columns; # columns
    sbr_r = slope_break; # 
    if ls_width == 1:
        cc = int(c/2)+1
    elif (ls_width <= c) and (ls_width%2 == 1):
        cc = []
        for i in range(ls_width):
            dif = -((ls_width)%2)+i
            cc.append(int(c/2)+dif+1)
        cc = np.array(cc)#[int(c/2)-1,int(c/2),int(c/2)+1])

    mg = RasterModelGrid((r,c+2),dxdy)
    ycol = np.reshape(mg.node_y,(r,c+2))[:,0]
    yn = np.arange(r)
    yeL = []
    sb = int(mg.node_y.max()*sbr_r)
    sbr = yn[ycol>sb].min()
    sb = ycol[sbr]

    for y in ycol:
        if y<=sb:
            ye = 0 #flat
            ye = y*slope_below_break
        else:
            # hillslope
            ye = (y-sb)*slope_above_break+sb*slope_below_break
        yeL.append(ye)
    dem = np.array([yeL,]*c).transpose()

    # dem[sbr:,cc] = dem[sbr:,cc]
    wall = np.reshape(dem[:,0],(len(dem[:,0]),1))+1.5
    dem = np.concatenate((wall,dem,wall),axis =1)
    dem = np.hstack(dem).astype(float)
    
    # flume

    _ = mg.add_field('topographic__elevation',
                        dem,
                        at='node')
    # profile nodes for profile plot
    if ls_width == 1:
        pf = mg.nodes[:,cc]
    elif ls_width > 1:
        pf = mg.nodes[:,int(c/2)+1]

    # landslide nodes
    lsn = mg.nodes[-(ls_length+1):-1,cc]

    return mg, lsn, pf, cc, sbr



#%% create the flume
pi = 1 # plot index

pdir = "D:/UW_PhD/PreeventsProject/Paper_2_MWR/Landlab_Development/mass_wasting_runout/development_plots/tests_version2/"

# rows = 27, columns = 15, slope_break = 0.8

dxdy = 1
rows = 95
columns = 31
ls_width = 1
ls_length = 2
slope_above_break = 0.6
slope_below_break = 0.0
slope_break = 0.3
soil_thickness = 0.67


flume_width = 2
cid = np.linspace(0,columns,columns+1).astype(int)
flume_cx = cid[int((columns-1)/2):int((columns-1)/2+2)]


mg, lsn, pf, cc, sbr = flume_maker(rows = rows, columns = columns, slope_above_break = slope_above_break
                              , slope_below_break = slope_below_break, slope_break = slope_break, ls_width = ls_width, ls_length = ls_length, dxdy = dxdy)


for n in np.hstack(mg.nodes):
    if ((mg.node_x[n]==14) or (mg.node_x[n]==17)) and (mg.node_y[n]>sbr):
        mg.at_node['topographic__elevation'][n] = mg.at_node['topographic__elevation'][n]+1.5
    if ((mg.node_x[n]<14) or (mg.node_x[n]>17)) and (mg.node_y[n]>sbr):
        
        mg.at_node['topographic__elevation'][n] = -.009999

dem = mg.at_node['topographic__elevation']        
mg.set_nodata_nodes_to_closed(dem, -.009999)

#%% assign landslide nodes

lsn = np.hstack(mg.nodes)[(((mg.node_x == 15) | (mg.node_x == 16)) & ((mg.node_y>87) & (mg.node_y<94)))]

#%%

# domain for plots
xmin = mg.node_x.min(); xmax = mg.node_x.max(); ymin = mg.node_y.min(); ymax = mg.node_y.max()



# set boundary conditions, add flow direction
mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries


mg.set_watershed_boundary_condition_outlet_id(cc,dem)
    
mg.at_node['node_id'] = np.hstack(mg.nodes)

# flow directions
fa = FlowAccumulator(mg, 
                      'topographic__elevation',
                      flow_director='FlowDirectorD8')
fa.run_one_step()

# hillshade for plots
mg.at_node['hillshade'] = mg.calc_hillshade_at_node(elevs=dem, alt=37., az=210.)


# soil thickness
mg.at_node['topographic__elevation'][lsn] = mg.at_node['topographic__elevation'][lsn]+soil_thickness
thickness = np.zeros(mg.number_of_nodes)
thickness[lsn] = soil_thickness
mg.add_field('node', 'soil__thickness',thickness)


# copy of initial topography
DEMi = mg.at_node['topographic__elevation'].copy()

#%%
# view the flume

plt.figure('3d view_side', figsize=(8, 10))
LLT.surf_plot(mg,title = 'initial dem [m] side', zr= 1, color_type = "grey", dv = 100,
                      constant_scale = True,s =-0.5, nv = 100 , zlim_min = 0,  elev = 35, azim = -180)
# plt.savefig("Flume.png", dpi = 300, bbox_inches='tight')
plt.figure('3d view', figsize=(8, 10))
LLT.surf_plot(mg,title = 'initial dem [m]', zr= 1, color_type = "grey", dv = 100,
                      constant_scale = True,s =-0.5, nv = 100 , zlim_min = 0,  elev = 35, azim = -130)


field = 'node_id'
field_back= "topographic__elevation"
plot_values(mg,field,13,18,80,94)
# plt.xlim(0,30); plt.ylim(80,100)


field = 'topographic__steepest_slope'
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)


field= "topographic__elevation"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)

el= mg.at_node['topographic__elevation'][pf]
pf_x = mg.node_y[pf]
plt.figure(label= 'initial el')
plt.plot(pf_x, el)
plt.gca().set_aspect('equal')

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

# zoom in to top of flume
plt.figure('grid and flow directions_top',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])
plt.plot(mg.node_x[lsn], mg.node_y[lsn], 'r.', markersize = 20)
plt.xlim(13,18); plt.ylim(80,95)

# zoom in to bottom of flume
plt.figure('grid and flow directions_bottom',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])
plt.xlim(10,21); plt.ylim(20,40)

#%% set up mass wasting runout

# mass wasting ide
mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
mg.at_node['mass__wasting_id'][lsn] = 1  

# run parameters
npu = [1] 
nid = [1] 
slpc = [0.005]   
SD = 0.05
cs = 0.2


mw_dict = {'critical slope':slpc, 'minimum flux':SD,
            'scour coefficient':cs}

release_dict = {'number of pulses':npu, 'iteration delay':nid }

example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True,
                                  routing_surface = "topographic__elevation",
                                  settle_deposit = False,
                                  deposition_rule = "critical_slope")

#%% set up calibrator

params = {'SD':[0.05,0.3,0.1],
               'cs':[0.001, 0.02, 0.005]}
el_l = 0
el_h = 20
channel_nodes= pf
cL = mg.dx
channel_distance = profile_distance(mg, channel_nodes)
profile_calib_dict = {"el_l":el_l, "el_h": el_h, "channel_nodes": channel_nodes,
                      "channel_distance":channel_distance, "cL":cL}



calibrate = MWRu_calibrator(example_MWRu, params, profile_calib_dict = profile_calib_dict,
                            prior_distribution = "uniform", jump_size = 0.2)


#%% run MWRu with known parameter values to create a dem_diff that can be set as the observed


example_MWRu.run_one_step(dt = 0)
mg.at_node['dem_dif_o'] = mg.at_node['topographic__elevation']-mg.at_node['topographic__initial_elevation']

# for tests, can use the values below for dem_dif_o if the grid is set up as:
    # dxdy = 10
    # rows = 7
    # columns = 5
    # ls_width = 3
    # slope = 0.6
    # slope_break = 0.4
    # soil_thickness = 2

    # with mw_dict = {'critical slope': [0.01], 'minimum flux': 0.01, 'scour coefficient': 0.02}


# # use these values 
# mg.at_node['dem_dif_o'] = np.array([ 0.        ,  0.        ,  0.40076924,  0.36420471,  0.44467475,
#         0.        ,  0.        ,  0.        ,  0.26736791,  0.22753467,
#         0.26913587,  0.22046527,  0.18074323,  0.        ,  0.        ,
#         0.2531977 ,  0.32927602,  0.31883648,  0.2523311 ,  0.23189692,
#         0.        ,  0.        ,  0.41055648,  0.74767565,  0.94991489,
#         0.75247022,  0.44863293,  0.        ,  0.        , -0.19464647,
#        -0.27282608, -0.28101959, -0.21054368, -0.11064826,  0.        ,
#         0.        ,  0.        , -2.        , -2.        , -2.        ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ])

# mg.at_node['topographic__elevation'] = mg.at_node['topographic__elevation']+mg.at_node['dem_dif_o']
#%% view obseved runout
LLT.plot_node_field_with_shaded_dem(mg,field = 'dem_dif_o', fontsize = 10,cmap = 'RdBu_r', plot_name = 'hillshade')
plt.clim(-1,1)

field = "dem_dif_o"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r')
plt.clim(-1,1)


el_p = mg.at_node['topographic__elevation'][pf]
el  = mg.at_node['topographic__initial_elevation'][pf]
plt.figure(label= 'post ls el')

plt.plot(pf, el, 'k-', label = 'initial dem')
plt.plot(pf, el_p, 'r-', alpha= 0.5, label = 'post failure dem')
# plt.gca().set_aspect('equal')
plt.legend()

Visualize = True
if Visualize:
    # plot how DEM changes
    for i in np.arange(0,len(example_MWRu.mw_ids)):
    
        for c in example_MWRu.df_evo_maps[i].keys():
            if c>100:
                break                  
            plt.figure('dif'+str(c)+str(i),figsize=(12, 12))
            mg.at_node['df_topo'] = example_MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
            LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo', fontsize = 10,cmap = 'RdBu_r', plot_name = 'dem dif{},{}'.format(i,c) )

            plt.xticks(fontsize= 8 )
            plt.yticks(fontsize= 8 )
            plt.clim(-0.5,0.5)
            plt.xlim([xmin*.8,xmax*1.2]); plt.ylim([ymin*.3,ymax])
            
    x_ = mg.node_y[pf]
    y = mg.at_node['topographic__initial_elevation'][pf]
     
    ef = 1   
    for i in np.arange(0,len(example_MWRu.mw_ids)):

        for c in example_MWRu.df_evo_maps[i].keys():                  
            topo = example_MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')
            if c>100:
                break                  
            y_ = mg.at_node['topographic__initial_elevation'][pf]+topo[pf]*ef
            plt.figure(figsize = (6,3))
            plt.plot(x_,y,'k--', alpha = 0.5, linewidth = 1,label = 'initial profile')
            plt.plot(x_,y_,'r-', alpha = 0.5, linewidth = 1, label = 'post df profile')
            plt.ylim([-3,max(y)]); plt.ylabel(' elevation ')
            plt.xlim([0, max(x_)])
            plt.legend()
            plt.grid(alpha = 0.5)  
       

    # for i in np.arange(0,len(example_MWRu.mw_ids)):

    #     for c in example_MWRu.df_evo_maps[i].keys():                  
    #         topo = example_MWRu.df_evo_maps[i][c]#-mg.at_node['topographic__initial_elevation']
    #         mg.at_node['topographic__elevation'] = topo
    #         plt.figure('grid and flow directions'+str(c),figsize=(8, 10))
    #         receivers = example_MWRu.frn_r[1][c]
    #         proportions = example_MWRu.frp_r[1][c]
    #         LLT.drainage_plot_jk(mg, receivers = receivers, proportions = proportions, title='Basic Ramp'+str(c),surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])

#%% visually check profile values using calibrator functions
mbLdf_o = calibrate._channel_profile_deposition("observed")

# profile from calibrator function
plt.figure()
plt.plot(mbLdf_o['node'], el, 'k-', label = 'initial dem')
plt.plot(mbLdf_o['node'], mbLdf_o['elevation'], 'r-', alpha = .5, label = 'post failure dem, calibrator')
# plt.gca().set_aspect('equal')
plt.legend()



#%% manually computed Vd

field = "node_id"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r')
plt.clim(-1,1)

field = "dem_dif_o"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r')
plt.clim(-1,1)

field = "topographic__elevation"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "topographic__elevation", cmap = 'terrain')
plt.clim(-1,1)

plt.figure()
plt.plot(mbLdf_o['node'], mbLdf_o['Vd'], 'k-', label = 'initial dem')


def manual_Vd(cn,datatype = "observed"):
    """used to manually determine Vd and compare to calibrator function cv_mass_change
    in all comparisons with manually summed Vd values, this function matched, and can
    now be used to compare to the calibrator function if MWRu is updated and the test
    needs to be updated for new values"""
    
# cn = 38
# datatype = "observed"

    if datatype == "modeled":
        demd = mg.at_node['dem_dif_m']
    elif datatype == "observed":
        demd = mg.at_node['dem_dif_o']
    el = mg.at_node['topographic__elevation'][cn]
    dd = np.nansum(demd[(dem<=el) & (dem>=el_l)])
    return dd

#%%
ddc = []
# # manual check, if using example setup, look at plots of elevation and node id to get dd
# # note for each manual check, 
# # cn = 3
# ddc.append(0.3642+0.2674+0.2275+0.2691+0.2205+0.1807+0.2532+0.3293+0.3188+0.2523+0.2319)

# # cn= 10
# ddc.append(.2674+0.2275+0.2691+0.2205+0.1807+0.2532+0.2523+0.2319)

# # cn = 17
# ddc.append(0.2674+0.2275+0.2691+0.2205+0.1807+0.2532+0.3188+0.2523+0.2319)

# # cn = 24
# ddc.append(0.4008+0.3642+0.444+0.2674+0.2275+0.2691+0.2205+0.1807+0.2532+0.3293+0.3188+0.2523+0.2319+0.4106+0.7477+0.9499+0.7525+0.4486)

# # cn = 31
# ddc.append(0.4008+0.3642+0.444+0.2674+0.2275+0.2691+0.2205+0.1807+0.2532+0.3293+0.3188+0.2523+0.2319+0.4106+0.7477+0.9499+0.7525+0.4486-0.281)

# # cn = 38
# ddc.append(0.4008+0.3642+0.444+0.2674+0.2275+0.2691+0.2205+0.1807+0.2532+0.3293+0.3188+0.2523+0.2319+0.4106+0.7477+0.9499+0.7525+0.4486-0.1946-0.2728-0.281-0.2105-0.1106-2-2-2)

# # cn = 45
# ddc.append(0.4008+0.3642+0.444+0.2674+0.2275+0.2691+0.2205+0.1807+0.2532+0.3293+0.3188+0.2523+0.2319+0.4106+0.7477+0.9499+0.7525+0.4486-0.1946-0.2728-0.281-0.2105-0.1106-2-2-2)


for cn in pf:
  ddc.append(manual_Vd(cn,datatype = "observed"))

ddc = np.array(ddc)*cL**2
plt.figure()
plt.plot(mbLdf_o['distance'], ddc, 'k-', label = "manual check")
plt.plot(mbLdf_o['distance'], mbLdf_o['Vd'], 'r-', alpha = 0.5, label = "function")
plt.xlabel("distance [m]")
plt.legend()
plt.ylabel("downstream cumulative volumetric change")


#%% Check RMSE funciton

# using Vd
# run a simulartion
calibrate(max_number_of_runs = 1)

# get the modeled profile values
mbLdf_m = calibrate._channel_profile_deposition("modeled")

plt.figure()
plt.plot(mbLdf_o['distance'], mbLdf_o['Vd'], 'k-', label = "observed")
plt.plot(mbLdf_m['distance'], mbLdf_m['Vd'], 'r-', alpha = 0.5, label = "calib run")
plt.xlabel("distance [m]")
plt.ylabel("downstream cumulative volumetric change")


# plt.figure()
# plt.plot(calibrate.mbLdf_o['distance'], calibrate.mbLdf_o['Vd'], 'k-', label = "observed")
# plt.plot(calibrate.mbLdf_m['distance'], calibrate.mbLdf_m['Vd'], 'r-', alpha = 0.5, label = "calib run")
# plt.xlabel("distance [m]")
# plt.ylabel("downstream cumulative volumetric change")

# manually computed RMSE
RMSE =(sum((mbLdf_o['Vd']-mbLdf_m['Vd'])**2)/len(mbLdf_o['Vd']))**0.5
RMSE_man = 1/RMSE

# from _RMSE function
RMSE_fun = calibrate.LHvals['1/RMSE'].iloc[-1]

print("RMSE manually determined: {}, function determined: {}".format(RMSE_man, RMSE_fun))


# using the profile

demdifo = mg.at_node['dem_dif_o'][mbLdf_o['node']]
demdifm = mg.at_node['dem_dif_m'][mbLdf_m['node']]

plt.figure()
plt.plot(mbLdf_o['distance'], demdifo, 'k-', label = "observed")
plt.plot(mbLdf_m['distance'], demdifm, 'r-', alpha = 0.5, label = "calib run")
plt.xlabel("distance [m]")
plt.ylabel("change dem, [m]")

RMSE =(sum((demdifo-demdifm)**2)/len(demdifo))**0.5
RMSE_man = 1/RMSE

# from _RMSE function
RMSE_fun = calibrate.LHvals['1/RMSE p'].iloc[-1]

print("RMSE manually determined: {}, function determined: {}".format(RMSE_man, RMSE_fun))


# using all grid cells
RMSE_man = 1/(sum((mg.at_node['dem_dif_o']-mg.at_node['dem_dif_m'])**2)/len(mg.at_node['dem_dif_m']))**0.5

# from _RMSE function
RMSE_fun = calibrate.LHvals['1/RMSE m'].iloc[-1]

print("RMSE manually determined: {}, function determined: {}".format(RMSE_man, RMSE_fun))

#%% check the omega function





#%% check mass is conserved

DEMdfDf = pd.DataFrame.from_dict(example_MWRu.DEMdfD, orient = 'index')


DEMi = mg.at_node['topographic__initial_elevation']

DEMf = mg.at_node['topographic__elevation']

DEMdf = DEMf-DEMi


# assert DEMdf.sum()*mg.dx*mg.dy == 75, 'not equal'
print("difference in initial and final dem [m3]")
print(DEMdf.sum()*mg.dx*mg.dy)
#%% are profiles as expected?

#%%

# # plot how DEM changes
# pi= 1
# for i in np.arange(0,len(example_MWRu.mw_ids)):

#     for c in example_MWRu.df_evo_maps[i].keys():   
#         if c%pi == 0:             
#             plt.figure('dif'+str(c)+str(i),figsize=(12, 12))
#             mg.at_node['df_topo_d'] = example_MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
#             LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo_d', fontsize = 10,cmap = 'RdBu_r', plot_name = 'dem dif{},{}'.format(i,c) )
#             plt.title(c)
#             # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
#             plt.xticks(fontsize= 8 )
#             plt.yticks(fontsize= 8 )
#             plt.clim(-2,2)
#             plt.savefig(pdir+'DebrisFlowPlanView_{}.png'.format(c), dpi = 300, bbox_inches='tight')

        




# # plot of cumulative DEM difference
# plt.figure()
# plt.plot(DEMdfDf.index, DEMdfDf['DEMdf_r'], 'k--', linewidth = 2,label = 'before settlement')
# # plt.plot(DEMdfDf.index, DEMdfDf['DEMdf_rd'], 'k-', linewidth = 2, alpha = 0.5,label = 'after settlement')
# plt.grid(alpha = 0.5)
# plt.xlabel("iteration")
# plt.ylabel('initial - evolved DEM, [m3]')
# plt.legend()
# # plt.gca().set_aspect('equal')
# plt.show()

        
        
#%% profile change 
# minimum channel threshold




# x_ = mg.node_y[pf]
# y = mg.at_node['topographic__initial_elevation'][pf]

# for i, topo in enumerate(example_MWRu.te_r[1]):
#     if i%pi == 0:
#         # x_,y_ = profiler_xy(profiler, topo)    
#         y_ = topo[pf]
#         plt.figure(figsize = (6,3))
#         plt.plot(x_,y,'k--', alpha = 0.5, linewidth = 1,label = 'initial profile')
#         plt.plot(x_,y_,'r-', alpha = 0.5, linewidth = 1, label = 'post df profile')
#         plt.ylim([-3,max(y)]); plt.ylabel(' elevation ')
#         plt.xlim([0, max(x_)])
#         plt.legend()
#         plt.grid(alpha = 0.5)
#         # plt.savefig(pdir+'DebrisFlowProfile_{}.png'.format(i), dpi = 300, bbox_inches='tight')
#         # plt.gca().set_aspect('equal')


# xs1 = np.array([83,84,85,86,87])
# xs2 = np.array([56,57,58,59,60])

# xsx = mg.node_x[xs1]; x1y = mg.node_y[xs1]
# x2x = mg.node_x[xs2]; x2y = mg.node_y[xs2]
# b1z = DEMi[xs1]

# for i, topo in enumerate(example_MWRu.te_r[1]):
#     plt.figure(figsize = (6,2))
#     plt.plot(xsx,b1z,'k--', alpha = 0.5, linewidth = 1,label = 'initial profile')
#     plt.plot(xsx,topo[xs1],'r-', alpha = 0.5, linewidth = 1, label = 'xs1')
#     plt.plot(xsx,topo[xs2],'r-', alpha = 0.5, linewidth = 1, label = 'xs2')
