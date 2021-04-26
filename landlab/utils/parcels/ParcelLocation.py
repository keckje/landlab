# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 22:44:26 2021

@author: keckj


function for determining the distance on link of each parcel given the pulse volume

part of MWR, used to create the final parcel table

"""
## set up workspace
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.optimize

# for debris flow
# compute deposition location
# compute begin deposition location
# collect parcels up to begin deposition location
# deposit parcels based on shape of deposit


# this section determines depostion location:
# assume all on one link

# hypothetical emperical formulas for deposit Length, upstream width, downstream width and downstream depth
# as a function of pulse volume
# L = 0.0433*v+13.92
# W1 = 0.0012*v+4.4309
# W2 = 0.0084*v+5.1007
# D = 0.003*v+1.4939

# volume of a deposit given deposit downstream width, depth and length
# Vold = ((Wd*Dd)/2)*Ld


# dimensions of deposit estimated from hypothetical emperical observations
# see Hypothetical_DebrisFlowDepositGeometry.xlsx
# dimension = func(pulse volume)

####
# data available in SedimentPulseDF from MWR
####
    
LinkList = np.array([20,8,2,1,0]) # all links in the debris flow path => THIS is not yet tracked in MWR

LinkDist = np.array([30,50,15,75,60]) # distance of each link

Link = 0 # deposition link

DepLinkLoc = 0.2 # locaiton on link

v = 150

parcel_volume = 0.5
    
dtoBr = 0.15 # deposit to bedload ratio

def ParcelDeposit(v, LinkList, LinkDist, Link, DepLinkLoc, parcel_volume, dtoBr):

    Dd = 0.003*v+1.4939
    Ld = 0.0433*v+13.92
    W1d = 0.0012*v+4.4309
    W2d = 0.0084*v+5.1007
    
    

    n = int((v*dtoBr)/parcel_volume)
    
    parcel_dx = []
    for i in range(0,n+1):
        Voldx = parcel_volume*i
        print(Voldx)
        def j(x):
            return np.abs(Voldx - (((Dd/Ld)*x)*(((W2d-W1d)/Ld)*x+W1d)/2)*x)
        
        r = scipy.optimize.minimize_scalar(j,method='bounded',bounds = [0,Ld])
        parcel_dx.append(r.x)
    
    # distance upstream from deposition point
    
    parcel_dx_up = Ld - np.array(parcel_dx)
    
    # number of parcels in pulse
    nmp = parcel_dx_up.shape[0]
        
    # create fields
    
    # index in linkList of link
    Li = list(LinkList).index(Link)
    
    # create the deposition link array for the pulse
    LinkArray = np.ones(nmp).astype(int)*Link # link of each parcel
    
    # create the deposition location array for the pulse
    pulse_head_loc = DepLinkLoc*LinkDist[Li] # determine distance along link from inlet to head of deposit
    
    parcel_loc_ = pulse_head_loc - parcel_dx_up # distance along link from inlet of each parcel
    parcel_loc = parcel_loc_/LinkDist[Li] # convert to distance along link from inlet ratio (DataRecord format)
       
    # if parcels are located upstream of link inlet, change link number and position
    c = 1
    while (parcel_loc_ < 0).any():
        nMask = parcel_loc_ < 0 #
        LinkArray[nMask] = LinkList[Li-c] # update link number
        parcel_loc_[nMask] = LinkDist[Li-c]+parcel_loc_[nMask] # update distance on link from inlet
        parcel_loc[nMask] = parcel_loc_[nMask]/LinkDist[Li-c] # update distance on link from inlet ratio
        c+=1

    return (LinkArray, parcel_loc)


ParcelDeposit(v, LinkList, LinkDist, Link, DepLinkLoc)



Linkarray = np.expand_dims(Linkarray, axis=1) # not sure if needs to be in this format
parcel_loc = np.expand_dims(parcel_loc, axis=1)

#(3)create 1xnum_pulse_parcels array that lists the link each parcel is entered into.
newpar_element_id = np.array([]) 
for i, row in parcelDF.iterrows():       
    newpar_element_id = np.concatenate(LinkArray)  

    #(4)create 1xn array of zeros to append to array of distance parcels traveled before tiemestep t (zero because parcels did not exist before timestep)
      newpar_dist = np.zeros(num_pulse_parcels,dtype=int)          
               
      #(5) create time stamp of zero for each parcel before parcel existed        
      new_time_arrival_in_link = time* np.ones(
          np.shape(newpar_element_id)) #arrives at current time in nst model
      
      #(6) compute total volume of all parcels entered into network during timestep
      new_volume = parcel_vol*np.ones(np.shape(newpar_element_id)) #parcelDF['vol [m^3]'].values /100  # volume of each parcel (m3) divide by 100 because large parcels break model
      #new_volume = np.expand_dims(new_volume, axis=1)
      
      #(7) assign grain properties -lithology ,activity, density, abrasion rate, diameter,- this can come from dataframe parcelDF
      new_lithology = ["pulse_material"] * np.size(
          newpar_element_id)  
      
      new_active_layer = np.ones(
          np.shape(newpar_element_id))  # 1 = active/surface layer; 0 = subsurface layer
      
      new_density = 2650 * np.ones(np.size(newpar_element_id))  # (kg/m3)
          
      new_abrasion_rate = 0 * np.ones(np.size(newpar_element_id))
      
      try:
          p_parcel_D  = parcelDF['d50 [m]'] # grain size in parcel : Change to read parcelDF
      except:
          p_parcel_D = self.d50
          
      new_D = p_parcel_D * np.ones(np.shape(newpar_element_id))
  
  
      #(8) assign part of grid that parcel is deposited (node vs link)    
      newpar_grid_elements = np.array(
          np.empty(
              (np.shape(newpar_element_id)), dtype=object)) 
      
      newpar_grid_elements.fill("link")
      
      item_id = {"grid_element": newpar_grid_elements,
               "element_id": newpar_element_id}
  
      #(9) construct dictionary of all parcel variables to be entered into data recorder
      variables = {
          "starting_link": (["item_id"], new_starting_link),
          "abrasion_rate": (["item_id"], new_abrasion_rate),
          "density": (["item_id"], new_density),
          #"lithology": (["item_id"], new_lithology),
          "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link),
          "active_layer": (["item_id", "time"], new_active_layer),
          "location_in_link": (["item_id", "time"], new_location_in_link),
          "D": (["item_id", "time"], new_D),
          "volume": (["item_id", "time"], new_volume),
      }
      
      return variables,item_id
