# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 00:43:45 2021

@author: keckj
"""


class ParcelCollector():
    
        '''
        parcelcollector data frame - volume removed, location removed
        
        if volume removed exceeds, parcel volume, parcel location chages
        
        changes volume or location of parcels in DataRecord based on parcel location
        
        setting the grid_element and element_id variables, reset volume, to nan when an item exits the system. 