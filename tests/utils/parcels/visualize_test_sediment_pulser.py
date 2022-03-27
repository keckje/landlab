# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:37:22 2022

@author: keckj

used this script for initial debugging and drafts of tests for pulser
"""

import pandas as pd
import numpy as np
from landlab import NetworkModelGrid
from landlab.utils.parcels.sediment_pulser_at_links_v2 import SedimentPulserAtLinks
from landlab.utils.parcels.sediment_pulser_each_parcel_v2 import SedimentPulserEachParcel

# example_nmg
y_of_node = (0, 100, 200, 200, 300, 400, 400, 125)
x_of_node = (0, 0, 100, -50, -100, 50, -150, -100)
nodes_at_link = ((1, 0), (2, 1), (1, 7), (3, 1), (3, 4), (4, 5), (4, 6))

grid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)

# add variables to grid
grid.add_field(
    "topographic__elevation", [0.0, 0.1, 0.3, 0.2, 0.35, 0.45, 0.5, 0.6], at="node"
)
grid.add_field(
    "bedrock__elevation", [0.0, 0.1, 0.3, 0.2, 0.35, 0.45, 0.5, 0.6], at="node"
)
grid.add_field(
    "reach_length",
    [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
    at="link",
)  # m

grid.add_field("channel_width", 15 * np.ones(grid.size("link")), at="link")

grid.add_field(
    "flow_depth", 0.01121871 * np.ones(grid.size("link")), at="link"
)  # Why such an odd, low flow depth? Well, it doesn't matter... and oops.-AP

grid.add_field(
    "channel_slope", 
    [0.05,0.1,0.07,0.1,0.08,0.12,0.1],
    at = "link",
    )    

grid.add_field(
    "drainage_area", 
    [1.7,0.2,1,0.3,0.8,0.2,0.4],
    at = "link",
    ) # km2



####
# test parcelDF
####

# use defaul grain size
make_pulse = SedimentPulserEachParcel(grid)


parcel_df = pd.DataFrame({'pulse_volume': [0.2, 2.5, 1.5, 1],
                                  'link_#': [1, 3, 5, 2],
                                  'normalized_downstream_distance': [0.8,0.7,0.5,0.2]})



time = 12
pulse0 = make_pulse(time,PulseDF=parcel_df)


# specify grain size

make_pulse = SedimentPulserEachParcel(grid)



parcel_df = pd.DataFrame({'pulse_volume': [0.2, 2.5, 1.5, 1],
                                  'link_#': [1, 3, 5, 2],
                                  'normalized_downstream_distance': [0.8,0.7,0.5,0.2],
                         'D50': [0.15, 0.2, 0.22, 0.1],
                         'D_sd': [0, 0, 0, 0]})
                         # 'D stdev [m]': [0.05, 0.07, 0.1, 0.15, 0.27])

time = 13
pulse0 = make_pulse(time,PulseDF=parcel_df)

print(pulse0.variable_names)
print(pulse0.dataset['D'])
print(pulse0.dataset['element_id'])
print(pulse0.dataset['density'])
print(pulse0.dataset['abrasion_rate'])
print(pulse0.dataset['time_arrival_in_link'])
print(pulse0.dataset['volume'])





def time_to_pulse(time):
    return True

# Instantiate 'SedimentPulserAtLinks' utility for the network model
# grid and pulse criteria

make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)

## normal_0, only time specified, should use defaults
time = 10
pulse0 = make_pulse(time = time)        


# D = pulse1.dataset['D']
# np.testing.assert_allclose(rn, rn_e, rtol = 1e-4)
# np.testing.assert_allclose(rv, rv_e, rtol = 1e-4)

print(pulse0.variable_names)
print(pulse0.dataset['D'])
print(pulse0.dataset['density'])
print(pulse0.dataset['abrasion_rate'])
print(pulse0.dataset['time_arrival_in_link'])



# Instantiate 'SedimentPulserAtLinks' utility for the network model
# grid and pulse criteria

make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)

# Run the instance with inputs for the time, link location, number of
# parcels using the instance grain characterstics for all links


## normal_1, density and abrasion specified, all others specified
time = 10
links = [2, 6]
n_parcels_at_link = [3, 5]
d50 = [0.3, 0.12]
std_dev =  [0.2, 0.1]   
parcel_volume = [1, 0.5]
pulse1 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
      d50=d50, std_dev=std_dev,parcel_volume = parcel_volume)        


# De = 
# D = pulse1.dataset['D']
# np.testing.assert_allclose(rn, rn_e, rtol = 1e-4)
# np.testing.assert_allclose(rv, rv_e, rtol = 1e-4)

print(pulse1.variable_names)
print(pulse1.dataset['D'])
print(pulse1.dataset['density'])
print(pulse1.dataset['abrasion_rate'])
print(pulse1.dataset['time_arrival_in_link'])

# # check contents of DataRecord



# Run the instance with inputs for the time, link location, number of
# parcels and grain charactersitics specific to each link

## normal_2


make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)

time = 11
links = [2, 6]
n_parcels_at_link = [3, 5]
d50 = [0.3, 0.12]
std_dev =  [0.2, 0.1]   
parcel_volume = [1, 0.5]
rho_sediment = [2650, 2500]
abrasion_rate = [.1, .3]
pulse2 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
      d50=d50, std_dev=std_dev,parcel_volume = parcel_volume, 
      rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)

print(pulse2.variable_names)
print(pulse2.dataset['D'])
print(pulse2.dataset['density'])
print(pulse2.dataset['abrasion_rate'])
print(pulse2.dataset['time_arrival_in_link'])


## special_1
# specify time that a pulse can occur


def time_to_pulse_L(time):
    Ptime = [19,20,22,23,24,75,76]
    return time in Ptime

make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse_L)

time = 21
links = [2, 6]
n_parcels_at_link = [3, 5]
d50 = [0.3, 0.12]
std_dev =  [0.2, 0.1]   
parcel_volume = [1, 0.5]
rho_sediment = [2650, 2500]
abrasion_rate = [.1, .3]
pulse3 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
      d50=d50, std_dev=std_dev,parcel_volume = parcel_volume, 
      rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)

time = 22
pulse3 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
      d50=d50, std_dev=std_dev,parcel_volume = parcel_volume, 
      rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)

# should only be 8 parcels in network

print(pulse3.variable_names)
print(pulse3.dataset['D'])
print(pulse3.dataset['time_arrival_in_link'])


## special_2
# standard deviation is 0
make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)

time = 11
links = [2, 6]
n_parcels_at_link = [3, 5]
d50 = [0.3, 0.12]
std_dev =  [0, 0.1]   
parcel_volume = [1, 0.5]
rho_sediment = [2650, 2500]
abrasion_rate = [.1, .3]
pulse4 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
      d50=d50, std_dev=std_dev,parcel_volume = parcel_volume, 
      rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)

print(pulse4.variable_names)
print(pulse4.dataset['D'])
print(pulse4.dataset['time_arrival_in_link'])

## special_3
make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)
# pulses sent to the same link
time = 11
links = [2, 2]
n_parcels_at_link = [3, 5]
d50 = [0.3, 0.12]
std_dev =  [0, 0.1]   
parcel_volume = [1, 0.5]
rho_sediment = [2650, 2500]
abrasion_rate = [.1, .3]
pulse4 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
      d50=d50, std_dev=std_dev,parcel_volume = parcel_volume, 
      rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)

print(pulse4.variable_names)
print(pulse4.dataset['D'])
print(pulse4.dataset['density'])
print(pulse4.dataset['abrasion_rate'])
print(pulse4.dataset['time_arrival_in_link'])



## bad_values_1
make_pulse = SedimentPulserAtLinks(grid, time_to_pulse=time_to_pulse)

time = 11
links = [2, 6]
n_parcels_at_link = [3, 5]
d50 = [0.3, 0.12]
std_dev =  [0, 0.1]   
parcel_volume = [1, 0.5]
rho_sediment = [2650, 2500]
abrasion_rate = [.1, .3]
pulse4 = make_pulse(time=time, links=links, n_parcels_at_link=n_parcels_at_link,
      d50=d50, std_dev=std_dev,parcel_volume = parcel_volume, 
      rho_sediment = rho_sediment, abrasion_rate = abrasion_rate)

print(pulse4.variable_names)
print(pulse4.dataset['D'])
print(pulse4.dataset['time_arrival_in_link'])



