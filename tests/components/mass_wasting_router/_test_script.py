# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 08:58:05 2023

@author: keckje
"""
import numpy as np
from landlab import RasterModelGrid
from landlab.components import SinkFillerBarnes, FlowAccumulator, FlowDirectorMFD
from landlab.components.mass_wasting_router import MassWastingRunout


import os
import pytest
os.chdir('C:/Users/keckje/Documents/GitHub/landlab/tests/components/mass_wasting_router/')


# !pytest test_mass_wasting_runout.py



# code to get output values for smoke tests
#%%
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
    # mg.set_watershed_boundary_condition_outlet_id(3,dem)   
    mg.at_node['node_id'] = np.hstack(mg.nodes)
    fd = FlowDirectorMFD(mg, diagonals=True,
                          partition_method = 'slope')
    fd.run_one_step()
    nn = mg.number_of_nodes
    mg.at_node['mass__wasting_id'] = np.zeros(nn).astype(int)
    mg.at_node['mass__wasting_id'][np.array([38])] = 1  
    depth = np.ones(nn)*1
    mg.add_field('node', 'soil__thickness',depth)
    np.random.seed(seed=7)
    mg.at_node['particle__diameter'] = np.random.uniform(0.05,0.25,nn)
    mg.at_node['organic__content'] = np.random.uniform(0.01,0.10,nn) 
    return(mg)

example_square_mg = example_square_mg()

#%%

def example_square_MWRu(example_square_mg):
    slpc = [0.03]   
    qsc = 0.01
    k = 0.02
    mofd = 1


    tracked_attributes = ['particle__diameter','organic__content']
        
    example_square_MWRu = MassWastingRunout(example_square_mg,
                                            critical_slope=slpc,
                                            threshold_flux=qsc,
                                            erosion_coefficient=k,
                                            max_flow_depth_observed_in_field=mofd,
                                            save = True,                                    
                                            tracked_attributes = tracked_attributes)
    return(example_square_MWRu)


example_square_MWRu = example_square_MWRu(example_square_mg)

#%%

def example_flat_mg():
    "small, flat surface"
    dem = np.ones(25)
    mg = RasterModelGrid((5,5),10)
    _ = mg.add_field('topographic__elevation',
                        dem,
                        at='node')
    
    
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries
    mg.set_watershed_boundary_condition_outlet_id(3,dem)   
    mg.at_node['node_id'] = np.hstack(mg.nodes)
    nn = mg.number_of_nodes
    mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
    depth = np.ones(nn)*1
    mg.add_field('node', 'soil__thickness',depth)    
    return(mg)

example_flat_mg = example_flat_mg()

#%%

def example_bumpy_mg():
    """sloped, irregular surface"""
    dem = np.ones(25)
    mg = RasterModelGrid((5,5),10)
    _ = mg.add_field('topographic__elevation',
                        dem,
                        at='node')        
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True) #close all boundaries
    mg.set_watershed_boundary_condition_outlet_id(3,dem)   
    mg.at_node['node_id'] = np.hstack(mg.nodes)
    mg.at_node['topographic__elevation'][np.array([6,7,8,11,13,16,17,18])] = np.array([3,2,5,5,7,9,8,11])   
    nn = mg.number_of_nodes
    mg.at_node['mass__wasting_id'] = np.zeros(mg.number_of_nodes).astype(int)
    depth = np.ones(nn)*1
    mg.add_field('node', 'soil__thickness',depth)    
    return(mg)

example_bumpy_mg = example_bumpy_mg()
