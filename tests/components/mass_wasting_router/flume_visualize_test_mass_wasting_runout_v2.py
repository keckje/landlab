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
    

flume = "wide"

pi = 1 # plot index

pdir = "D:/UW_PhD/PreeventsProject/Paper_2_MWR/Landlab_Development/mass_wasting_runout/development_plots/202201_status_images/"

# figure out a way to create a shoot to a wide basin

# wide: r30,c11, sbr_r = 0.5
if flume == "wide":

    # wide
    r = 30; #rows
    c =11; # columns
    sbr_r = 0.5; # ratio of length the slope break is placed, measured from the outlet
    cc = int(c/2)
    ypr = 0.5
    xpr = 0.67


if flume == "narrow":
    # narrow
    r = 20; #rows
    c =1; # columns
    sbr_r = 0.5; # ratio of length the slope break is placed, measured from the outlet
    cc = int(c/2)
    ypr = 1
    xpr = 1


dxdy = 10 # cell width and height of raster model grid
mg = RasterModelGrid((r,c+2),dxdy)
ycol = np.reshape(mg.node_y,(r,c+2))[:,0]
yn = np.arange(r)
yeL = []
sb = int(mg.node_y.max()*sbr_r)
sbr = yn[ycol>sb].min()

for y in ycol:
    
    if y<=sb:
        ye = 1 #flat
        
    else:
        # hillslope
        ye = 6+(y-sb)*.5
        
    yeL.append(ye)
    
dem = np.array([yeL,]*c).transpose()



dem[sbr:,cc] = dem[sbr:,cc]-5

wall = np.reshape(dem[:,0],(len(dem[:,0]),1))+10

dem = np.concatenate((wall,dem,wall),axis =1)

cc+=1
dem = np.hstack(dem).astype(float)

_ = mg.add_field('topographic__elevation',
                    dem,
                    at='node')

# profile nodes
pf = mg.nodes[:,cc]

# ls nodes, last 6 excluding the very last
lsn = pf[-6:-1]

dem_copy = dem.copy()


plt.figure('3d view', figsize=(8, 10))
LLT.surf_plot(mg,title = 'initial dem [m]', zr= 2, color_type = "grey", dv = 100,
                      constant_scale = True,s =-0.5, nv = 100 , zlim_min = 0,  elev = 35, azim = -130)
plt.savefig("WideFlume.png".format(c), dpi = 300, bbox_inches='tight')



mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries


mg.set_watershed_boundary_condition_outlet_id(cc,dem)
    
mg.at_node['node_id'] = np.hstack(mg.nodes)

# write_esri_ascii(dem_dir_model+'CapitolForestModelNodeIds.asc', mg, 'node_id')

DEMi = mg.at_node['topographic__elevation'].copy()


# hillshade for plots
mg.at_node['hillshade'] = mg.calc_hillshade_at_node(elevs=dem, alt=37., az=210.)


# flow directions
fa = FlowAccumulator(mg, 
                      'topographic__elevation',
                      flow_director='FlowDirectorD8')
fa.run_one_step()


# LLT.plot_node_field_with_shaded_dem(mg,field = 'surface_water__discharge', fontsize = 10,cmap = 'RdBu_r',alpha = .5)



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

# view test topography


# inn = 236 # initial cell (node) where debris flow material begins
# plot grid, flow directions and initial cell location


plt.figure('grid and flow directions',figsize=(8, 10))
receivers = mg.at_node['flow__receiver_node']
proportions = mg.at_node['flow__receiver_proportions']
LLT.drainage_plot_jk(mg, proportions = proportions, title='Basic Ramp',surf_cmap="Greys",clim = [dem.min(),dem.max()*1.2])
plt.savefig(pdir+"WideFlume_FlowDirections.png".format(c), dpi = 300, bbox_inches='tight')

# plt.plot(mg.x_of_node[inn],mg.y_of_node[inn],'r.',markersize =22)


xmin = mg.node_x.min(); xmax = mg.node_x.max(); ymin = mg.node_y.min(); ymax = mg.node_y.max()
field = 'node_id'

plt.figure(field,figsize = (12,8))
imshow_grid(mg,'topographic__elevation',grid_units=('m','m'),var_name='Elevation(m)',plot_name = field,cmap = 'terrain')
plot_values(mg,field,xmin,xmax,ymin,ymax)
plt.xlim([xmin,xmax]); plt.ylim([ymin,ymax])
z_d = dem[(mg.node_x>=xmin) & (mg.node_x<=xmax) & (mg.node_y>=ymin) & (mg.node_y<=ymax)]


#%%
# set up mass wasting router
nn = mg.number_of_nodes

# mass wasting ide
mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)

mg.at_node['mass__wasting_id'][lsn] = 1  
# soil depth
depth = np.ones(nn)*2
mg.add_field('node', 'soil__thickness',depth)


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

example_MWRu.run_one_step(dt = 0)


# plot how DEM changes
for i in np.arange(0,len(example_MWRu.mw_ids)):

    for c in example_MWRu.df_evo_maps[i].keys():   
        if c%pi == 0:             
            plt.figure('dif'+str(c)+str(i),figsize=(12, 12))
            mg.at_node['df_topo_d'] = example_MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
            LLT.plot_node_field_with_shaded_dem(mg,field = 'df_topo_d', fontsize = 10,cmap = 'RdBu_r', plot_name = 'dem dif{},{}'.format(i,c) )
    
            # imshow_grid_at_node(mg,'df_topo',cmap ='RdBu_r')  
            plt.xticks(fontsize= 8 )
            plt.yticks(fontsize= 8 )
            plt.clim(-1,1)
            plt.savefig(pdir+'DebrisFlowPlanView_{}.png'.format(c), dpi = 300, bbox_inches='tight')

for i in np.arange(0,len(example_MWRu.mw_ids)):

    for c in example_MWRu.df_evo_maps[i].keys():
        df_topo = example_MWRu.df_evo_maps[i][c]-mg.at_node['topographic__initial_elevation']
        _ = mg.add_field('df_topo',
                    df_topo,
                    at='node',
                    clobber = True)
        
        
        LLT.surf_plot(mg,zr= 2,surface='df_topo',title = 'evolving dem + df thickness minus initial dem [m]',color_type = "grey", min_s = 1, max_s = 1, constant_scale = True, dv = 1.5,
                      s =-.5, nv = 1.5 , zlim_min = 0, zlim_max =10, elev = 35, azim = -130)
        plt.savefig(pdir+'DebrisFlowDifference_{}.png'.format(c), dpi = 300, bbox_inches='tight')
        
#%% check mass is conserved

DEMdfDf = pd.DataFrame.from_dict(example_MWRu.DEMdfD, orient = 'index')


DEMi = mg.at_node['topographic__initial_elevation']

DEMf = mg.at_node['topographic__elevation']

DEMdf = DEMf-DEMi


# assert DEMdf.sum()*mg.dx*mg.dy == 75, 'not equal'
print("difference in initial and final dem [m3]")
print(DEMdf.sum()*mg.dx*mg.dy)



# plot of cumulative DEM difference
plt.figure()
plt.plot(DEMdfDf.index, DEMdfDf['DEMdf_r'], 'k--', linewidth = 2,label = 'before settlement')
# plt.plot(DEMdfDf.index, DEMdfDf['DEMdf_rd'], 'k-', linewidth = 2, alpha = 0.5,label = 'after settlement')
plt.grid(alpha = 0.5)
plt.xlabel("iteration")
plt.ylabel('initial - evolved DEM, [m3]')
plt.legend()
# plt.gca().set_aspect('equal')
plt.show()

        
        
#%% profile change 
# minimum channel threshold




x_ = mg.node_y[pf]
y = mg.at_node['topographic__initial_elevation'][pf]

for i, topo in enumerate(example_MWRu.te_r[1]):
    if i%pi == 0:
        # x_,y_ = profiler_xy(profiler, topo)    
        y_ = topo[pf]
        plt.figure(figsize = (6,3))
        plt.plot(x_,y,'k--', alpha = 0.5, linewidth = 1,label = 'initial profile')
        plt.plot(x_,y_,'r-', alpha = 0.5, linewidth = 1, label = 'post df profile')
        plt.ylim([-3,max(y)*ypr]); plt.ylabel(' elevation ')
        plt.xlim([50, max(x_)*xpr])
        plt.legend()
        plt.grid(alpha = 0.5)
        plt.savefig(pdir+'DebrisFlowProfile_{}.png'.format(i), dpi = 300, bbox_inches='tight')
        # plt.gca().set_aspect('equal')


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
    
    