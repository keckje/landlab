# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 07:23:30 2022

@author: keckj

seperate set of tests used to confirm routing and sloshing problem

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




#%% create the flume
pi = 1 # plot index

pdir = "D:/UW_PhD/PreeventsProject/Paper_2_MWR/Landlab_Development/mass_wasting_runout/development_plots/tests_version2/"

# rows = 27, columns = 15, slope_break = 0.8

dxdy = 10
rows = 27
columns = 1
ls_width = 1
ls_length = 1
slope_above_break = 0.0
slope_below_break = 0.0
slope_break = 0.8
soil_thickness = 5

mg, lsn, pf, cc = flume_maker(rows = rows, columns = columns, slope_above_break = slope_above_break
                              , slope_below_break = slope_below_break, slope_break = slope_break, ls_width = ls_width, ls_length = ls_length)
lsn = 76
dem = mg.at_node['topographic__elevation']

mg.at_node['topographic__elevation'][76] = mg.at_node['topographic__elevation'][76]+5
mg.at_node['topographic__elevation'][79] = mg.at_node['topographic__elevation'][79]+7
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
mg.at_node['soil__thickness'][76] = 5#mg.at_node['topographic__elevation'][70]+5

# # no soil thickness
# thickness = np.zeros(mg.number_of_nodes)*soil_thickness
# mg.add_field('node', 'soil__thickness',thickness)
# mg.at_node['soil__thickness'][lsn] = soil_thickness*np.ones(len(lsn))




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

params_o = [0.01, 0.01, 0.05]
slpc = [params_o[0]]   
SD = params_o[1]
cs = params_o[2]



# mg.at_node['particle__diameter'] = np.ones(len(mg.node_x))*0.15

mw_dict = {'critical slope':slpc, 'minimum flux':SD,
            'scour coefficient':cs}

release_dict = {'number of pulses':npu, 'iteration delay':nid }

MWRu = MassWastingRunout(mg,release_dict,mw_dict, save = True, itL = 150,
                                  dist_to_full_flux_constraint = 0,
                                  routing_surface = "energy__elevation",
                                  settle_deposit = False,
                                  deposition_rule = "critical_slope",
                                  deposit_style = 'no_downslope_deposit')


#%% run
start_time = time.time()
MWRu.run_one_step(dt = 0)

print("--- %s seconds ---" % (time.time() - start_time))
#%%
mg.at_node['dem_dif_o'] = mg.at_node['topographic__elevation']-mg.at_node['topographic__initial_elevation']

#%% view obseved runout
LLT.plot_node_field_with_shaded_dem(mg,field = 'dem_dif_o', fontsize = 10,cmap = 'RdBu_r', plot_name = 'hillshade')
plt.clim(-1,1)

field = "dem_dif_o"
plot_values(mg,field,xmin,xmax,ymin,ymax,field_back = "dem_dif_o", cmap = 'RdBu_r')
plt.clim(-1,1)

#%% mass conservation check
DEMi = mg.at_node['topographic__initial_elevation']
DEMf = mg.at_node['topographic__elevation']
DEMdf = DEMf-DEMi
# assert DEMdf.sum()*mg.dx*mg.dy == 75, 'not equal'
print("difference in initial and final dem [m3] is:{}".format(np.round(DEMdf.sum()*mg.dx*mg.dy, decimals = 8)))



#%% evoloving surface

Visualize = False
if Visualize:
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
Visualize = True
if Visualize:
    # plot how DEM changes

    x_ = mg.node_y[pf]
    y = mg.at_node['topographic__initial_elevation'][pf]
     
       
    for i in np.arange(0,len(MWRu.mw_ids)):

        for c in MWRu.df_evo_maps[i].keys():                  
            etopo = MWRu.df_evo_maps[i][c]#-mg.at_node['topographic__initial_elevation']
            topo = MWRu.topo_evo_maps[i][c]
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')
            if c>100:
                break                  
            y_ = topo[pf]
            _y_ = etopo[pf]
            plt.figure(figsize = (6,3))
            plt.plot(x_,y,'k--', alpha = 0.5, linewidth = 1,label = 'initial ground profile')
            plt.plot(x_,y_,'g-', alpha = 0.5, linewidth = 1, label = 'post ground profile')
            plt.plot(x_,_y_,'r-', alpha = 0.5, linewidth = 1, label = 'df profile')
            plt.ylim([-3,max(y)]); plt.ylabel(' elevation ')
            plt.xlim([0, max(x_)])
            plt.legend()
            plt.grid(alpha = 0.5)  
            plt.title('iteration:'+str(c-1))


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
    
    
#%% look at delivery and receiving node squence if a small flume

dr_dq = pd.DataFrame(zip(MWRu.arndn_r[1],MWRu.arn_r[1]), columns = ['deliverying nodes','receiving nodes'])
print(dr_dq)

#%% flow characteristics 

#         self.df_evo_maps = {} # copy of dem after each routing iteration
#         self.topo_evo_maps = {}
#         self.enL = [] # entrainment depth / regolith depth
#         self.DpL = [] # deposition depth
#         self.dfdL = [] # incoming debris flow thickness (qsi)
#         self.TdfL = [] # basal shear stress 
#         self.slopeL = [] # slope
#         self.velocityL = [] # velocity (if any)

# slope vs shear stress
plt.figure()
plt.semilogy(MWRu.slopeL, MWRu.TdfL,'k.', alpha = 0.5)
plt.xlabel('slope'); plt.ylabel('shear stress [Pa]')

plt.figure()
plt.plot(MWRu.slopeL, MWRu.dfdL,'k.', alpha = 0.5)
plt.xlabel('slope'); plt.ylabel('flow thickness [m]]')


plt.figure()
plt.plot(MWRu.slopeL, MWRu.enL,'k.', alpha = 0.5)
plt.xlabel('slope'); plt.ylabel('scour depth [m]]')


plt.figure()
plt.plot(MWRu.slopeL, MWRu.DpL,'k.', alpha = 0.5)
plt.xlabel('slope'); plt.ylabel('deposition depth [m]]')
