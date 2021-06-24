# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:24:19 2021

@author: keckj
"""

# # setup
import os

# import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib as mpl
import pandas as pd
# import numpy
import numpy as np

# import necessary landlab components
from landlab import RasterModelGrid, HexModelGrid
from landlab.components import FlowAccumulator
from landlab.components import(FlowDirectorD8, 
                                FlowDirectorDINF, 
                                FlowDirectorMFD, 
                                FlowDirectorSteepest)

# import landlab plotting functionality
from landlab.plot.drainage_plot import drainage_plot
from landlab import imshow_grid_at_node

# import functions
from landlab.io.esri_ascii import write_esri_ascii
from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf


from landlab.components import ChannelProfiler, FlowAccumulator, DepressionFinderAndRouter

os.chdir('C:/Users/keckj/Documents/GitHub/code/landlab/LandlabPlots')
import LandLabPlots as LLP



saveimages = False



class DebrisFlowScourAndDeposition:
    
    def __init__(
    self,
    grid,
    df_para_dict,
    save_df_dem = False, 
    run_id = 0, 
    itL = 1000):
        
    
        
        self.grid = grid
        self.df_para_dict = df_para_dict
        self.save = save_df_dem
        self.run_id = run_id
        self.itL = itL
    
    '''a cellular automata debris flow model that routes an initial landslide 
    volume as a debris flow through the dem described by the 'topographic__elevation' 
    field of a landlab raster model grid
    

    Parameters
    ----------
    mg : landlab raster model grid
        raster model grid
    ivL : list of floats
        list of initial landslide volumes
    innL : list of int
        list of nodes where landslide volumes are released on the dem. 
        len(innL) = len(ivL)
    df_paraL : list of dictionaries
        each entry in the df_paraL is a dictionary of parameters that control 
        the behavoir of the cellular-automata debris flow model
        if len(df_paraL = 1, then the same model parameters are used for all runs.
    save_df_dem : boolean
        Save topographic elevation after each model iteration?. This could creat up to 
        itterL number of maps for each debris flow. The default is False.
    itL : int
        maximum number of iterations the cellular-automata model runs before it
        is forced to stop. The default is 1000.

    Returns
    -------
    None.

    '''
    
    def __call__(self, ivL, innL):
        
        
        self._scour_and_deposit(ivL, innL)
        

    def _scour_and_deposit(self, innL, ivL):
            
        # release parameters for landslide
        nps = self.df_para_dict['number of pulses']
        nid = self.df_para_dict['iteration delay']    
        
        # critical slope at which debris flow stops
        # higher reduces spread and runout distance
        slpc = self.df_para_dict['critical slope']
        
        # forced stop volume threshold
        # material stops at cell when volume is below this,
        # higher increases spread and runout distance
        SV = self.df_para_dict['stop-volume']
    
        # entrainment coefficient
        # very sensitive to entrainment coefficient
        # higher causes higher entrainment rate, longer runout, larger spread
        cs = self.df_para_dict['entrainment-coefficient']
    
    
            
        cL = {}
        
        # ivL and innL can be a list of values. For each value in list:
        for i,inn in enumerate(innL):
            # print(i)
            cL[i] = []
            
    
    
            # set up initial landslide cell
            iv =ivL[i]/nps# initial volume (total volume/number of pulses) 
        
            
            # initial receiving nodes (cells) from landslide
            rn = self.grid.at_node.dataset['flow__receiver_node'].values[inn]
            rni = rn[np.where(rn != -1)]
            
            # initial receiving proportions from landslide
            rp = self.grid.at_node.dataset['flow__receiver_proportions'].values[inn]
            rp = rp[np.where(rp > 0)]
            rvi = rp*iv # volume sent to each receving cell
        
            # now loop through each receiving node, 
            # determine next set of recieving nodes
            # repeat until no more receiving nodes (material deposits)
            
            slpr = []
            c = 0
            c2=0
            arn = rni
            arv = rvi
            enL = []
            while len(arn)>0 or len(arn) < self.itL:
        
                # add pulse (fraction) of total landslide volume
                # at interval "nid" until total number of pulses is less than total
                # for landslide volume (nps*nid)
        
                if ((c+1)%nid ==0) & (c2<=nps*nid):
                    arn = np.concatenate((arn,rni))
                    arv = np.concatenate((arv,rvi))
                    
                arn_ns = np.array([])
                arv_ns = np.array([])
                # print('arn:'+str(arn))
                
                # for each unique cell in receiving node list arn
                for n in np.unique(arn):
                    n = int(n)
                    
                    # slope at cell (use highest slope)
                    slpn = self.grid.at_node['topographic__steepest_slope'][n].max()
                    
                    # incoming volume: sum of all upslope volume inputs
                    vin = np.sum(arv[arn == n])
                    
                    # determine deposition volume following Campforts, et al., 2020            
                    # since using volume (rather than volume/dx), L is (1-(slpn/slpc)**2) 
                    # rather than mg.dx/(1-(slpn/slpc)**2)           
                    Lnum = np.max([(1-(slpn/slpc)**2),0])
                    
                    dpd = vin*Lnum # deposition volume
                    
                    # determine erosion depth
                    
                    # debris flow depth over cell    
                    df_depth = vin/(self.grid.dx*self.grid.dx) #df depth
                    
                    #determine erosion volume (function of slope)
                    # dmx = dc*mg.at_node['soil__thickness'][n]
                    # er = (dmx-dmx*Lnum)*mg.dx*mg.dy
                    
                    T_df = 1700*9.81*df_depth*slpn # shear stress from df
         
                    # max erosion depth equals regolith (soil) thickness
                    dmx = self.grid.at_node['soil__thickness'][n]
                    
                    # erosion depth: 
                    er = min(dmx, cs*T_df)
                    
                    enL.append(er)
                    
                    # erosion volume
                    ev = er*self.grid.dx*self.grid.dy
                    
                    # volumetric balance at cell
                    
                    # determine volume sent to downslope cells
                    
                    # additional constraint to control debris flow behavoir
                    # if flux to a cell is below threshold, debris is forced to stop
                    if vin <=SV:
                        dpd = vin # all volume that enter cell is deposited 
                        vo = 0 # debris stops, so volume out is 0
        
                        # determine change in cell height
                        deta = (dpd)/(self.grid.dx*self.grid.dy) # (deposition)/cell area
        
                    else:
                        vo = vin-dpd+ev # vol out = vol in - vol deposited + vol eroded
                        
                        # determine change in cell height
                        deta = (dpd-ev)/(self.grid.dx*self.grid.dy) # (deposition-erosion)/cell area
        
                    
                    # update raster model grid regolith thickness and dem
                    
                    
                    # if deta larger than regolith thickness, deta equals regolith thickness (fresh bedrock is not eroded)
                    if self.grid.at_node['soil__thickness'][n]+deta <0:
                        deta = - self.grid.at_node['soil__thickness'][n]           
                    
                    # Regolith - difference between the fresh bedrock surface and the top surface of the dem
                    self.grid.at_node['soil__thickness'][n] = self.grid.at_node['soil__thickness'][n]+deta 
        
                    # Topographic elevation - top surface of the dem
                    self.grid.at_node['topographic__elevation'][n] = self.grid.at_node['topographic__elevation'][n]+deta
                    # print(mg.at_node['topographic__elevation'][n])
            
            
                    # build list of receiving nodes and receiving volumes for next iteration        
            
                    # material stops at node if transport volume is 0 OR node is a 
                    # boundary node
                    if vo>0 and n not in self.grid.boundary_nodes: 
                        
            
                        th = 0
                        # receiving proportion of volume from cell n to each downslope cell
                        rp = self.grid.at_node.dataset['flow__receiver_proportions'].values[n]
                        rp = rp[np.where(rp > th)]
                        # print(rp)        
                        
                        # receiving nodes (cells)
                        rn = self.grid.at_node.dataset['flow__receiver_node'].values[n]
                        rn = rn[np.where(rn != -1)]            
                        # rn = rn[np.where(rp > th)]
                        
                        # receiving volume
                        rv = rp*vo
                   
                        # store receiving nodes and volumes in temporary arrays
                        arn_ns =np.concatenate((arn_ns,rn), axis = 0) # next step receiving node list
                        arv_ns = np.concatenate((arv_ns,rv), axis = 0) # next steip receiving node incoming volume list
                        
        
                    
                # once all cells in iteration have been evaluated, temporary receiving
                # node and node volume arrays become arrays for next iteration
                arn = arn_ns
                arv = arv_ns
                
                # update DEM slope 
                # fd = FlowDirectorDINF(mg) # update slope
                fd = FlowDirectorMFD(self.grid, diagonals=True,
                                partition_method = 'slope')
                fd.run_one_step()
                
                
                if self.save:
                    cL[i].append(c)
                    
                    # save DEM difference (DEM for iteration C - initial DEM)
                    _ = self.grid.add_field( 'topographic__elevation_run_id_'+str(run_id)+str(i)+'_'+str(c),
                                    self.grid.at_node['topographic__elevation'].copy(),
                                    at='node')
                    mpnm.append('topographic__elevation_run_id_'+str(run_id)+str(i)+'_'+str(c))
                       
                # update iteration counter
                c+=1
                
                # update pulse counter
                c2+=nid
                
                if c%20 ==0:
                    print(c)            

    

#%% test run


# imported DEM
# load inputs    
wdir = 'D:/UW_PhD/PreeventsProject/Paper_2_MWR/Landlab_Development/output/'
pdir = wdir+'plots/'
os.chdir(wdir)


dem_dir_model = 'D:/UW_PhD/PreeventsProject/Paper_2_MWR/Validation/'


mg, z = read_esri_ascii(dem_dir_model+'d05_m_m_10mk.asc', name='topographic__elevation')

# mg.status_at_node[np.isclose(z, -9999.)] = mg.BC_NODE_IS_CLOSED
mg.set_watershed_boundary_condition(z) #finds lowest point in dem and sets it as an open node


mg.at_node['node_id'] = np.hstack(mg.nodes)

# write_esri_ascii(dem_dir_model+'CapitolForestModelNodeIds.asc', mg, 'node_id')

# using uniform depth
depth = np.ones(mg.at_node['topographic__elevation'].shape[0])*1.2
mg.add_field('node', 'soil__thickness',depth)

dxdy = mg.dx

# hillshade for plots
mg.at_node['hillshade'] = mg.calc_hillshade_at_node(elevs=z, alt=37., az=210.)
        

fa = FlowAccumulator(mg, 
                      'topographic__elevation',
                      flow_director='FlowDirectorD8')
fa.run_one_step()

# fill depressions to correct surface area determination
df_4 = DepressionFinderAndRouter(mg)
df_4.map_depressions()

LLP.plot_node_field_with_shaded_dem(mg,field = 'surface_water__discharge', fontsize = 10,cmap = 'RdBu_r',alpha = .5)


# load cross sections
tmp, tz = read_esri_ascii(dem_dir_model+'xs_scour1.asc', name='temp')
vals = tmp.at_node['temp']; xs_sc1 = vals[vals!=-9999]

tmp, tz = read_esri_ascii(dem_dir_model+'xs_scour2.asc', name='temp')
vals = tmp.at_node['temp']; xs_sc2 = vals[vals!=-9999]

# tmp, tz = read_esri_ascii(dem_dir_model+'xs1.asc', name='temp')
# vals = tmp.at_node['temp']; xs1 = vals[vals!=-9999]

# tmp, tz = read_esri_ascii(dem_dir_model+'xs2.asc', name='temp')
# vals = tmp.at_node['temp']; xs2 = vals[vals!=-9999]

def XSras_to_df(dem_dir,Xnm,sortv = 'x'):
    tmp, tz = read_esri_ascii(dem_dir+Xnm, name='temp')
    vals = tmp.at_node['temp']; vals = vals[vals!=-9999].astype('int')
    vals_x  =  mg.node_x[vals]
    vals_y =  mg.node_y[vals]
    tmpdf = pd.DataFrame(zip(vals,vals_x,vals_y),columns = ['node','x','y'])
    tmpdf = tmpdf.sort_values(sortv)
    dist = ((tmpdf['x'][1:].values-tmpdf['x'][0:-1].values)**2+(tmpdf['y'][1:].values-tmpdf['y'][0:-1].values)**2)**0.5
    dist = np.concatenate([np.array([0]),dist])
    tmpdf['dist'] = dist
    tmpdf['c_dist'] = tmpdf['dist'].cumsum()
    xs_df = tmpdf
    xs_df['el_0'] = mg.at_node['topographic__elevation'][xs_df['node']]   
    return xs_df





#%%
# if ana_ch2 == 1:
#     # channel 1
#     # outlet long profile
#     dem_dir = dem_dir_model
#     Xnm = 'xs1_out_lp.asc'
#     out_lp_df = XSras_to_df(dem_dir_model,Xnm,sortv = 'x')
    
#     # outlet x-sec 1
#     Xnm = 'xs1.asc'
#     xs1_df = XSras_to_df(dem_dir_model,Xnm,sortv = 'y')
    
#     # outlet x-sec 2
#     Xnm = 'xs2.asc'
#     xs2_df = XSras_to_df(dem_dir_model,Xnm,sortv = 'y')

# if ana_ch2 == 2:
#     # channel 2
#     # outlet long profile
#     dem_dir = 'D:/UW_PhD/PreeventsProject/Paper_2_MWR/Validation/ch2/'
#     Xnm = 'ch2_outlet_lp.txt'
#     out_lp_df = XSras_to_df(dem_dir,Xnm,sortv = 'x')
    
#     # outlet x-sec 1
#     Xnm = 'xs1.txt'
#     xs1_df = XSras_to_df(dem_dir,Xnm,sortv = 'x')
    
#     # outlet x-sec 2
#     Xnm = 'xs3.txt'
#     xs3_df = XSras_to_df(dem_dir,Xnm,sortv = 'x')
    
#     # outlet x-sec 2
#     Xnm = 'xs5.txt'
#     xs5_df = XSras_to_df(dem_dir,Xnm,sortv = 'x')
    
#     # outlet x-sec 2
#     Xnm = 'xs7.txt'
#     xs7_df = XSras_to_df(dem_dir,Xnm,sortv = 'y')

#%% make a copy of the initial DEM for elevation differencing

# flume
# _ = mg.add_field('topographic__initial_elevation',
#                     (mg.y_of_node**1.5)/5+np.abs((mg.x_of_node-5*dxdy)**2)/3,
#                     at='node')


# imported DEM
_ = mg.add_field('topographic__initial_elevation',
                    mg.at_node['topographic__elevation'],
                    at='node',
                    copy = True)


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

#%% test routing and DEM response
ana_ch2 = 1

# find the node location of a landslide by zooming in to specific range

# view test topography and landslide locations
plt.figure(figsize=(10, 10))
imshow_grid_at_node(mg,'hillshade',cmap='gray')
# adjust plot range to narrow down which node needed
plt.xlim([490500,491500])
plt.ylim([5204000, 5205000])
plt.grid()
plt.show()

# search in narrowed down range for node  id
ns = np.hstack(mg.nodes)
ns[(mg.node_x>490660) & (mg.node_x<490700) & (mg.node_y>5204550) & (mg.node_y<5204560)]

if ana_ch2 ==1:
    ivL = [10000]# total volume of landslide, estimated from lidar
    innL = [167995] # ch1 initial node where landslide material is produced

if ana_ch2 ==2:
    innL = [215592] # ch2
    ivL = [15000]# total volume of landslide, estimated from lidar


# view test topography and landslide locations
plt.figure(figsize=(10, 10))
# LLP.drainage_plot(mg, title='Basic Ramp')
imshow_grid_at_node(mg,'hillshade',cmap='gray')
for inp in innL:
    plt.plot(mg.x_of_node[inp],mg.y_of_node[inp],'r.',markersize =15)
plt.title('Landslide location on DEM')
plt.grid()
plt.show()



# debris flow control parameters
# how is the volume released
npu = 8 # sL = [8,10,25] # number of pulses, list for each landslide
nid = 5 #[5,5,5] # delay between pulses (iterations), list for each landslide

# depostion location parameters

# visual best: 0.02, 1.5, 3.5e-5

# critical slope at which debris flow stops
# higher reduces spread and runout distance
slpc = 0.02

# material stops at cell when volume is below this, 
# higher increases spread and runout distance
SV = 1.5#3.3

# very sensitive to entrainment coefficient
# higher causes higher entrainment rate, longer runout, larger spread
# cs = 3.9e-5
cs = 2.5e-5



df_para_dict = {'critical slope':slpc, 'stop-volume':SV,
           'entrainment-coefficient':cs, 
           'number of pulses':npu, 'iteration delay':nid }


#%%


DebrisFlows = DebrisFlowScourAndDeposition(mg, df_para_dict)


DebrisFlows(innL,ivL)



#%% cumulative deposition and erosion on channel nodes


mg.at_node['topographic__initial_elevation']
mg.at_node['topographic__elevation']

diff = mg.at_node['topographic__elevation'] - mg.at_node['topographic__initial_elevation']
mg.at_node['deta_dstorm_1'] = diff

if ana_ch2 == 1:
    LLP.plot_node_field_with_shaded_dem(mg,field = 'deta_dstorm_1', fontsize = 10,cmap = 'RdBu_r')
    plt.clim(-1,1)
    # plt.plot(mg.node_x[out_lp_df['node']],mg.node_y[out_lp_df['node']],'k.',markersize = 1)
    # plt.plot(mg.node_x[xs1_df['node']],mg.node_y[xs1_df['node']],'k.',markersize = 1)
    # plt.plot(mg.node_x[xs3_df['node']],mg.node_y[xs3_df['node']],'k.',markersize = 1)
    # plt.plot(mg.node_x[xs5_df['node']],mg.node_y[xs5_df['node']],'k.',markersize = 1)
    # plt.plot(mg.node_x[xs7_df['node']],mg.node_y[xs7_df['node']],'k.',markersize = 1)
    plt.title('slope: '+str(slpc)+', SV: '+str(SV)+', cs: '+str(cs))
    plt.xlim([490000,494000])
    plt.ylim([5202400, 5205050])
    plt.savefig(pdir+'slope_'+str(slpc)+'_SV_'+str(SV)+'_cs_'+str(cs)+'_runout_.png', dpi = 300, bbox_inches='tight')
                

DF_vol_tot = (diff[diff>0]*mg.dx*mg.dy).sum()
print(DF_vol_tot)