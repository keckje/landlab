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
def plot_values(mg,field,xmin,xmax,ymin,ymax, field_back = "topographic__elevation", cmap = 'terrain', name= '_'):
    """plot the field values on the node within the specified domain extent"""
    
    if field:
        plt.figure(name,figsize = (12,8))
        imshow_grid(mg, field_back, grid_units=('m','m'),var_name=field_back,plot_name = field,cmap = cmap)
        values = mg.at_node[field]
        if values.dtype != int:
            values = values.round(decimals = 4)
        values_test_domain = values[(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]
        ntd = mg.at_node['node_id'][(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]
    
        for i,val in enumerate(values_test_domain):
            plt.text(mg.node_x[ntd[i]],mg.node_y[ntd[i]],str(val),color = 'red', fontsize = 9)  
    else:
        plt.figure(name,figsize = (12,8))
        imshow_grid(mg, field_back, grid_units=('m','m'),var_name=field_back,plot_name = name,cmap = cmap)
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
    wall = np.reshape(dem[:,0],(len(dem[:,0]),1))+10
    dem = np.concatenate((wall,dem,wall),axis =1)
    dem = np.hstack(dem).astype(float)

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

    return mg, lsn, pf, cc




def _scour(ros,vs,h,s,eta,cs,slpc = 0.1, Dp = None):
    """
    determine potential scour depth (not limited by regolith depth)
    using MWRu scour function
    
    Parameters
    ----------
    ros : density of grains in runout [kg/m3]
    vs  : volumetric ratio of solids to matrix [m3/m3]
    h   : depth [m] - typical runout depth
    s   : slope [m/m] - average slope of erosion part of runout path
    eta : exponent of scour model (equation 7) - 
    slpc: average slope at which positive net deposition occurs
    Dp  : representative grain size [m]

    Returns
    -------
    Ec: potential erosion depth [m]
    Tbs: basal shear stress [Pa]
    u: velocity [m/s]

    """
    g= 9.81
    rof = 1000
    rodf = vs*ros+(1-vs)*rof
    theta = np.arctan(s)
    
    if Dp: # if Dp => use this for item by item analysis
        # if h< Dp:
        #     Dp = h
        print('grain-inertia')
        phi = np.arctan(slpc)
        
        # inertial stresses
        us = (g*h*s)**0.5
        u = us*5.75*np.log10(h/Dp)
        
        dudz = u/h
        Tcn = np.cos(theta)*vs*ros*(Dp**2)*(dudz**2)
        Tbs = Tcn*np.tan(phi)
               
    else:
        print('quasi-static')
        Tbs = rodf*g*h*(np.sin(theta))
        u = 5
        
    Ec = (cs*Tbs**eta) 

    return(Ec, Tbs, u)



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

def determine_E_l(ros,vs,h,s,eta,alpha,dx,slpc = 0.1, Dp = None):
    """
    determine average erosion depth for comparison with qsc
    
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

    E_l = (alpha*(tau**eta))/dx
    
    return E_l, tau




#%% create the flume
pi = 1 # plot index

mdir = "D:/UW_PhD/PreeventsProject/Paper_2_MWR/Landlab_Development/mass_wasting_runout/development_plots/tests_version2/"
svnm = 'calib_tests_'
# rows = 27, columns = 15, slope_break = 0.8

dxdy = 10
rows = 27
columns = 15
ls_width = 3
ls_length = 3
slope_above_break = 0.6
slope_below_break = 0.01
slope_break = 0.8
soil_thickness = 2.5

mg, lsn, pf, cc = flume_maker(rows = rows, columns = columns, slope_above_break = slope_above_break
                              , slope_below_break = slope_below_break, slope_break = slope_break, ls_width = ls_width, ls_length = ls_length)

dem = mg.at_node['topographic__elevation']
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
thickness = np.ones(mg.number_of_nodes)*soil_thickness
mg.add_field('node', 'soil__thickness',thickness)


Dp = 0.1
# particle diameter
Dp_ = np.ones(mg.number_of_nodes)*Dp
mg.add_field('node', 'particle__diameter', Dp_)

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
plot_values(mg,field,xmin,xmax,ymin,ymax)


field = 'topographic__steepest_slope'
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)


field= "topographic__elevation"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back= field_back)

el= mg.at_node['topographic__elevation'][pf]
plt.figure(label= 'initial el')
plt.plot(pf, el)
plt.gca().set_aspect('equal')

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

#%% set up mass wasting runout

# mass wasting ide
mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
mg.at_node['mass__wasting_id'][lsn] = 1  

# run parameters
npu = [1] 
nid = [1] 
# slpc, qsc, alpha
params_o = [0.015, 0.02, 0.005]
# params_o = [0.05, 0.82, 0.1]
# params_o = [0.05, 0.78160412935585255, 0.023565070615077743] 
# params_o = [0.05, 0.87959043570258, 0.023963226618357404]
slpc = [params_o[0]]   
SD = params_o[1]
cs = params_o[2]


mw_dict = {'critical slope':slpc, 'minimum flux':SD,
            'scour coefficient':cs}

release_dict = {'number of pulses':npu, 'iteration delay':nid }

example_MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True, itL = 1000,
                                  # dist_to_full_flux_constraint = 10,
                                  routing_surface = "topographic__elevation",
                                  settle_deposit = False,
                                  deposition_rule = "critical_slope",
                                  deposit_style = 'downslope_deposit')

#%% set up calibrator


example_MWRu.cs
ros = 2650; vs = 0.6; h = 1; s = 0.5; eta = 0.2; cs = example_MWRu.cs
# convert alpha to equivalent scour depth, threshold flux needs to be larger than this value
E_l, tau, = determine_E_l(ros,vs,h,s,eta,example_MWRu.cs,mg.dx,slpc = slpc, Dp = Dp)

SDmin = E_l*mg.dx/5

#
params_c = {'slpc':[0.001,.15,0.05], 'SD': [SDmin/5, SDmin*10, SDmin*1.3]}
el_l = 0
el_h = 20
channel_nodes= pf
cL = mg.dx
channel_distance = profile_distance(mg, channel_nodes)
profile_calib_dict = {"el_l":el_l, "el_h": el_h, "channel_nodes": channel_nodes,
                      "channel_distance":channel_distance, "cL":cL}

jump_size = 0.09
alpha_max = 0.8
alpha_min = 0.6#alpha_max-0.4
N_cycles = 100

calibrate = MWRu_calibrator(example_MWRu, params_c, profile_calib_dict = profile_calib_dict, N_cycles = N_cycles,
                            prior_distribution = "uniform", jump_size = jump_size, alpha_max = alpha_max, alpha_min = alpha_min, plot_tf = True, seed = 7)


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

Visualize = False
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
            plt.clim(-1,1)
            plt.xlim([xmin*.8,xmax*1.2]); plt.ylim([ymin*.3,ymax])
            plt.show()
            
    x_ = mg.node_y[pf]
    y = mg.at_node['topographic__initial_elevation'][pf]
     
       
    for i in np.arange(0,len(example_MWRu.mw_ids)):

        for c in example_MWRu.df_evo_maps[i].keys():                  
            topo = example_MWRu.df_evo_maps[i][c]#-mg.at_node['topographic__initial_elevation']
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')
            if c>100:
                break                  
            y_ = topo[pf]
            plt.figure(figsize = (6,3))
            plt.plot(x_,y,'k--', alpha = 0.5, linewidth = 1,label = 'initial profile')
            plt.plot(x_,y_,'r-', alpha = 0.5, linewidth = 1, label = 'post df profile')
            plt.ylim([-3,max(y)]); plt.ylabel(' elevation ')
            plt.xlim([0, max(x_)])
            plt.legend()
            plt.grid(alpha = 0.5)  
       

    for i in np.arange(0,len(example_MWRu.mw_ids)):

        for c in example_MWRu.df_evo_maps[i].keys():                  
            topo = example_MWRu.df_evo_maps[i][c]#-mg.at_node['topographic__initial_elevation']
            mg.at_node['topographic__elevation'] = topo
            plt.figure('grid and flow directions'+str(c),figsize=(8, 10))
            receivers = example_MWRu.frn_r[1][c]
            proportions = example_MWRu.frp_r[1][c]
            LLT.drainage_plot_jk(mg, receivers = receivers, proportions = proportions, title='Basic Ramp'+str(c),surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])
            plt.show()
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
# found error, corrected

# using Vd
# run a simulartion
calibrate(max_number_of_runs = 100)


#%%
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
RMSE =(sum((mbLdf_o['Vd']-mbLdf_m['Vd'])**2)/len(mbLdf_o['Vd']))
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

RMSE =(sum((demdifo-demdifm)**2)/len(demdifo))
RMSE_man = 1/RMSE

# from _RMSE function
RMSE_fun = calibrate.LHvals['1/RMSE p'].iloc[-1]

print("RMSE manually determined: {}, function determined: {}".format(RMSE_man, RMSE_fun))


# using all grid cells
RMSE_man = 1/(sum((mg.at_node['dem_dif_o']-mg.at_node['dem_dif_m'])**2)/len(mg.at_node['dem_dif_m']))

# from _RMSE function
RMSE_fun = calibrate.LHvals['1/RMSE m'].iloc[-1]

print("RMSE manually determined: {}, function determined: {}".format(RMSE_man, RMSE_fun))

#%% check the omega function
# found error in intersection node computation, corrected

# first view two runout extents

field = "node_id"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "topographic__elevation", cmap = 'Greys', name = " nodes")


field = "node_id"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r', name = " o")
plt.clim(-1e-20,1e-20)


field = "node_id"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_m", cmap = 'RdBu_r', name = " m")
plt.clim(-1e-20,1e-20)


na = 100
n_a = mg.nodes.reshape(mg.shape[0]*mg.shape[1])
n_o = n_a[np.abs(mg.at_node['dem_dif_o']) > 0]
n_m = n_a[np.abs(mg.at_node['dem_dif_m']) > 0]


n_x = n_o[np.isin(n_o,n_m)]#np.unique(np.concatenate([n_o,n_m])) # intersection nodes
n_u = n_o[~np.isin(n_o,n_m)] # underestimate
n_o = n_m[~np.isin(n_m,n_o)] # overestimate

X = len(n_x)*na
U = len(n_u)*na
O = len(n_o)*na
T = X+U+O
omegaT = X/T-U/T-O/T



#%% check mass is conserved

DEMdfDf = pd.DataFrame.from_dict(example_MWRu.DEMdfD, orient = 'index')


DEMi = mg.at_node['topographic__initial_elevation']

DEMf = mg.at_node['topographic__elevation']

DEMdf = DEMf-DEMi


# assert DEMdf.sum()*mg.dx*mg.dy == 75, 'not equal'
print("difference in initial and final dem [m3]")
print(DEMdf.sum()*mg.dx*mg.dy)


#%% are profiles as expected?
x_m = calibrate.mbLdf_m['distance']
y_m = calibrate.mbLdf_m['elevation']

x_m_ = channel_distance
y_m_ = mg.at_node['topographic__elevation'][channel_nodes]

plt.figure()
plt.plot(x_m_, y_m_, 'k-', label = "manual check")
plt.plot(x_m, y_m, 'r-', alpha = 0.5, label = "function")
plt.xlabel("distance [m]")
plt.ylabel("downstream cumulative volumetric change")
plt.legend()



x_o = calibrate.mbLdf_o['distance']
y_o = calibrate.mbLdf_o['elevation']

x_o_ = channel_distance
y_o_ = mg.at_node['topographic__initial_elevation'][channel_nodes]+mg.at_node['dem_dif_o'][channel_nodes]

plt.figure()
plt.plot(x_o_, y_o_, 'k-', label = "manual check")
plt.plot(x_o, y_o, 'r-', alpha = 0.5, label = "function")
plt.xlabel("distance [m]")
plt.ylabel("downstream cumulative volumetric change")
plt.legend()



#%% reformat calibrator to use RMSE of depositon, profile and 2d rmse?

# comparison of sensitivity of each metric reveals RMSE profile and DTE are least sensitive. 
# RMSE map doesnt always line up with best?

observed = mg.at_node['dem_dif_o'] 
modeled = mg.at_node['dem_dif_m']
RMSE_map = calibrate._RMSE(observed, modeled)

observed = mg.at_node['dem_dif_o'][mbLdf_o['node']] 
modeled = mg.at_node['dem_dif_m'][mbLdf_m['node']]
RMSE_pf = calibrate._RMSE(observed, modeled)

observed = mbLdf_o['Vd']; modeled = mbLdf_m['Vd']
RMSE_Vd = calibrate._RMSE(observed, modeled)



# field = "dem_dif_o"
# field = None
# plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r', name = " o")
# plt.clim(-1,1)

# mg.at_node['model_dif_compare'] = calibrate.dem_dif_m_dict[it_best]

# field = "model_dif_compare"
# field = None
# plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "model_dif_compare", cmap = 'RdBu_r', name = " m")
# plt.clim(-1,1)


#%%plot results of MCMC


# RMSE_map = []
# mask =  np.abs(mg.at_node['dem_dif_o'])>0           

# for key in calibrate.dem_dif_m_dict:
#         observed = mg.at_node['dem_dif_o'][mask]
#         modeled = calibrate.dem_dif_m_dict[key]       
#         mask_m =  np.abs(modeled)<=0 
#         modeled[mask_m] = np.abs(modeled).max()
#         modeled = modeled[mask]
#         RMSE_map.append(calibrate._RMSE(observed, modeled))
# calibrate.LHvals['1/RMSE m2'] = 1/np.array(RMSE_map)



# RMSE_oT = []
 

# for key in calibrate.dem_dif_m_dict:
        
#         calibrate.mg.at_node['dem_dif_m'] =calibrate.dem_dif_m_dict[key]       
#         RMSE_oT.append(_RMSEomegaT())
# calibrate.LHvals['1/RMSE m2'] = np.array(RMSE_oT)






it_best = calibrate.LHvals['iteration'][calibrate.LHvals['candidate_posterior'] == calibrate.LHvals['candidate_posterior'].max()].values[0]






# check RMSE values
itm = calibrate.LHvals['iteration'].max()
plt.figure(figsize = (15,5))
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE']/calibrate.LHvals['1/RMSE'].max(),'k-', alpha = 0.5, label = 'RMSE V')
# plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE p']/calibrate.LHvals['1/RMSE p'].max(),'r-', alpha = 0.5, label = 'RMSE p')
# plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE m']/calibrate.LHvals['1/RMSE m'].max(),'g-', alpha = 0.5, label = 'RMSE m')
# plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE m2']/calibrate.LHvals['1/RMSE m2'].max(),'o-', alpha = 0.5, label = 'RMSE m2')
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['RMSEomegaT']/calibrate.LHvals['RMSEomegaT'].max(),'o-', alpha = 0.5, label = 'RMSEOmegaT')
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['candidate_posterior']/calibrate.LHvals['candidate_posterior'].max(),'c-', alpha = 0.5, label = 'likelihood')
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['omegaT']/calibrate.LHvals['omegaT'].max(),'m-', alpha = 0.5, label = 'OmegaT')
plt.xlim(0,calibrate.LHvals['iteration'].max()*1.05)
plt.xticks(np.arange(0,itm+1, step=int(1+itm/20)))
plt.legend(fontsize = 8, loc = "right")
plt.xlim(it_best-50,it_best+50)
plt.savefig(mdir+svnm+"_likelihood_metrics.png", dpi = 300, bbox_inches='tight')
plt.show()



# check RMSE values
itm = calibrate.LHvals['iteration'].max()
plt.figure(figsize = (15,5))
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE']/calibrate.LHvals['1/RMSE'].max(),'k-', alpha = 0.5, label = 'RMSE V')
# plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE p']/calibrate.LHvals['1/RMSE p'].max(),'r-', alpha = 0.5, label = 'RMSE p')
# plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE m']/calibrate.LHvals['1/RMSE m'].max(),'g-', alpha = 0.5, label = 'RMSE m')
# plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['1/RMSE m2']/calibrate.LHvals['1/RMSE m2'].max(),'o-', alpha = 0.5, label = 'RMSE m2')
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['RMSEomegaT']/calibrate.LHvals['RMSEomegaT'].max(),'o-', alpha = 0.5, label = 'RMSEOmegaT')
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['candidate_posterior']/calibrate.LHvals['candidate_posterior'].max(),'c-', alpha = 0.5, label = 'likelihood')
plt.plot(calibrate.LHvals['iteration'], calibrate.LHvals['omegaT']/calibrate.LHvals['omegaT'].max(),'m-', alpha = 0.5, label = 'OmegaT')
plt.xlim(0,calibrate.LHvals['iteration'].max()*1.05)
plt.xticks(np.arange(0,itm+1, step=int(1+itm/20)))
plt.legend(fontsize = 8, loc = "right")
plt.savefig(mdir+svnm+"_likelihood_metrics_best.png", dpi = 300, bbox_inches='tight')
# plt.xlim(it_best-50,it_best+50)
plt.show()

plt.figure(figsize = (15,5))
plt.plot(calibrate.jstracking,'k-', alpha = 0.5, linewidth = 2)
plt.ylabel('jump-size distribution $\sigma$', fontsize = 16)
plt.xlabel('n*10 iteration', fontsize = 16)




# calibration plots
# params_c = params
results = calibrate.LHvals

x_mn = params_c['SD'][0]
x_mx = params_c['SD'][1]
y_mn = params_c['slpc'][0]
y_mx = params_c['slpc'][1]

# plot jumps
plt.figure(figsize = (6,3))
plt.plot(results['candidate_value_SD'], results['candidate_value_slpc'])
plt.xlim([x_mn,x_mx])
plt.ylim([y_mn,y_mx])
plt.xlabel('crtical volumetric flux, below which everything stops $qs_c$, [m]')
plt.ylabel('critical slope $slp_c$')
plt.savefig(mdir+svnm+"_jumps.png", dpi = 300, bbox_inches='tight')

# plot liklihood value of each jump
# see for format: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
import scipy as sc
grid_x, grid_y = np.mgrid[x_mn:x_mx:20j,y_mn:y_mx:20j]

points = results[['candidate_value_SD','candidate_value_slpc']].values

values = results['candidate_posterior'].values

grid_z1 = sc.interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

plt.figure(figsize = (6,3))
plt.imshow(grid_z1.T, extent=(x_mn,x_mx,y_mn,y_mx), origin='lower')
plt.xlabel('crtical flow depth, below which everything stops $qs_c$, [m]')
plt.ylabel(r'scour coef., $\alpha$')
plt.savefig(mdir+svnm+"_likelihood.png", dpi = 300, bbox_inches='tight')


# X = grid_x[:,0]
# Y = grid_y[:,0]
# plt.figure(figsize = (3,3))
# plt.imshow(grid_z1.T, extent=(x_mn,x_mx,y_mn,y_mx), origin='lower', cmap = 'Greys_r', alpha = 0.5)
# plt.contour(grid_x,grid_y,grid_z1,np.linspace(np.nanmin(grid_z1),np.nanmax(grid_z1),7), colors='k', linewidth = 0.5)
# plt.xlabel('$qs_c$, [m]')
# plt.ylabel(r'$\alpha$')



# plt.figure(figsize = (6,3))
# n = results.shape[0]
# counts, xedge,yedge,image =  plt.hist2d(results['selected_value_SD'], results['selected_value_slpc'], bins = 15)#int(n**0.5))
# plt.xlabel('crtical flow depth, below which everything stops $qs_c$, [m]')
# plt.ylabel(r'scour coef., $\alpha$')
# plt.title('histogram')
# plt.colorbar()




def parameter_uncertainty(results, parameter, parameter2):
    # parameter diagnostic plots
    # check that tested parameter values display proper variability (see MCMC_notes.pdf Figure 3)
    col = 'selected_value_'+parameter
    plt.figure(figsize = (3,2))
    plt.plot(results[col], color = 'k', alpha = 0.8, linewidth = 1)
    plt.ylabel(parameter)
    plt.xlabel('iteration')
    plt.savefig(mdir+svnm+col+"_iterations.png", dpi = 300, bbox_inches='tight')

    # get count in each parameter value bin
    # "The distributions of counts in each bin gives the probabilility distribution of the parameter as a function of the given the rules" 
    # (Jessica class note, Markov_models_lecture2017, pg 63). 95% confidence intervals are referred to as credibility intervals, rather than 
    # confidence interval.
    col = 'selected_value_'+parameter
    plt.figure(figsize = (3,2))
    n = results[col].shape[0]
    counts, edges, plot = plt.hist(results[col],bins = int(n**0.5),color = 'k',alpha = 0.8)
    plt.grid(alpha = 0.5)
    plt.xlabel(parameter)
    plt.ylabel('count')
    plt.savefig(mdir+svnm+col+"_histogram.png", dpi = 300, bbox_inches='tight')
    
    # sum the histogram bins to get the cdf, using the bin center
    bin_cntr = ((edges[1:]-edges[0:-1])).cumsum() 
    bin_cnt = counts.cumsum() 
    
    plt.figure(figsize = (3,2))
    cp = bin_cnt/counts.sum()
    plt.plot(bin_cntr, cp, color = 'k', alpha = 0.8, linewidth = 1)
    plt.grid(alpha = 0.5)
    plt.xlabel(parameter)
    plt.ylabel(r'P( X$\leq$x )')
    plt.savefig(mdir+svnm+col+"_cdf.png", dpi = 300, bbox_inches='tight')
    
    def intp(x,y,x1,message = None): 
        f = sc.interpolate.interp1d(x,y)   
        y1 = f(x1)
        return y1
    
    lb = intp([0]+list(cp),[0]+list(bin_cntr),0.0501010)
    up = intp([0]+list(cp),[0]+list(bin_cntr),0.9510101)
    
    
    # col = 'candidate_value_'+parameter
    # plt.figure(figsize = (3,2))
    # plt.plot(results[col], results['candidate_posterior'], 'k.', markersize = 5, alpha = 0.6, linewidth = 1)
    # plt.grid(alpha = 0.5)
    # plt.ylabel("likelihood")
    # plt.xlabel(parameter)

    col = 'candidate_value_'+parameter
    plt.figure(figsize = (3,2))
    plt.scatter(results[col], results['candidate_posterior'], s=2, c=results['candidate_value_'+ parameter2], cmap='viridis')
    plt.ylabel('likelihood', fontsize = 12)
    plt.colorbar(label =  parameter2)
    plt.xlabel(col)
    plt.savefig(mdir+svnm+col+'_'+parameter2+"_likelihood.png", dpi = 300, bbox_inches='tight')

    
    return lb, up
try:
    parameter_uncertainty(results, 'slpc', 'SD')
    parameter_uncertainty(results, 'slpc', 'cs')   
    parameter_uncertainty(results, 'SD', 'slpc')   
    parameter_uncertainty(results, 'SD', 'cs')
    parameter_uncertainty(results, 'cs', 'SD')
    parameter_uncertainty(results, 'cs', 'slpc')
except:
    parameter_uncertainty(results, 'slpc', 'SD')
    parameter_uncertainty(results, 'SD', 'slpc')
