# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:59:10 2023

@author: keckje
"""

import numpy as np
import pandas as pd
import xarray as xr
from landlab.io import read_esri_ascii
from landlab import RasterModelGrid

class downscale_to_landlab_grid():
    """given the raw appended gridded hydrology outputs from a distributed hydrology model, 
    esri ascii files of the DHSVM dem and dem wetness index and the landlab DEM and wetness 
    index, interpolates depth to water table at the landlab grid scale
    
    TODO: add soil thickness as input parameter, create topmodel lambda in downscaler (get ca, slope using landlab)
    
    Parameters
    ----------
    DSHVM_grid : string
        file path to an esri ascii file of the DHSVM digitial elevation model (DEM)
    landlab_grid : string
        file path to an esri ascii file of the landlab DEM
    DHSVM_lambda : string
        file path to an esri ascii file of the DHSVM DEM wetness index
    landlab_lambda : string
        file path to an esri ascii file of the landlab DEM wetness index
    appended_maps : string
        file path to the appended depth to water table maps ascii file. The ascii
        file is a text version of the binary file output by DHSVM.To convert to 
        ascii, use myconvert.c. See DHSVM documentation for use of myconvert.c
    map_dates : list
        list the map dates text from DHSVM
    
    Returns the following xarray data array class variables.
    ---------- 

        DSHVM_grid
        landlab_grid
        landlab_lambda
        DHSVM_lambda_lin_interp
    
    Returns the following xarray data set class variables. The data sets have a
        time coordinate defined by the map_dates input.
    ----------
        
        DHSVM_dtw_lin_interp: linearly interpolated DHSVM dtw at the landlab 
            grid domain
        landlab_dtw: DHSVM dtw interpolated using the topmodel wetness index and
            approach of Doten et al., 2006.
            
    To visually check input and output, use built in xarray plotting functions:
    >>> # visual check input (stored as a dataarray)
    >>> plt.figure(figsize=(8, 5))
    >>> instance_name.DHSVM_grid.plot(vmin=0, vmax=3000,)
    >>> plt.gca().set_aspect('equal')        

    >>> # visual check output (stored as a dataset)
    >>> plt.figure(figsize=(8, 5))
    >>> instance_name.landlab_dtw.isel(time=[0]).to_array().plot(vmin=0, vmax=1) 
    >>> plt.gca().set_aspect('equal')    
    
    author: Jeff Keck
    """
    
    
    def __init__(self, H_grid,
                  landlab_grid,
                  appended_maps,
                  map_dates,
                  H_lambda = None,
                  landlab_lambda = None,
                  landlab_soild = None,
                  resample = False):

        print('loading hydrology model and landlab grids')
        # dhsvm grid
        grid_d, z_d = read_esri_ascii(H_grid, name='topographic__elevation')
        self.DHSVM_grid = self._grid_to_da(grid_d,'topographic__elevation')
    
        # landlab grid
        grid, z = read_esri_ascii(landlab_grid, name='topographic__elevation')
        self.landlab_grid = self._grid_to_da(grid,'topographic__elevation')
        self.y1 = self.landlab_grid.y
        self.x1 = self.landlab_grid.x
        
        self.resample = resample
    
        # dhsvm wetness index
        g2, wi_d = read_esri_ascii(DHSVM_lambda, name='lambda')
        self.DHSVM_lambda = self._grid_to_da(g2,'lambda')
            
        # landlab wetness index
        g2, wi_l = read_esri_ascii(landlab_lambda, name='lambda')
        self.landlab_lambda = self._grid_to_da(g2,'lambda')
        
        # landlab soil depth
        g2, sd = read_esri_ascii(landlab_soild, name='soild')
        self.landlab_soild = self._grid_to_da(g2,'soild')        
        
        print('loading appeneded maps')
        self.dtw_maps = np.loadtxt(appended_maps)
        
        self.map_dates = map_dates

        self._appended_maps_to_dataset()
        
        if self.resample:
            self._resample(by = self.resample['by'], metric = self.resample['metric'])
        
        if DHSVM_lambda and landlab_lambda and landlab_soild: 
            print('interpolating maps to landlab grid using the topographic wetness index')
            self._interpolate_DHSVM_to_landlab()
        else:
            print('linearly interpolating maps to landlab grid')
            self._lin_interpolate_H_to_landlab()
            
            
    def _grid_to_da(self, grid, field):
        """get a 2-d xarray data array, including coordinates, from a landlab 
        raster model grid"""
        r = grid.number_of_cell_rows+2; c =  grid.number_of_cell_columns+2 #2 more nodes than cells
        ycoords = grid.node_axis_coordinates(1).reshape(r,c)[0,:]
        xcoords = grid.node_axis_coordinates(0).reshape(r,c)[:,0]
        data = grid.at_node[field].reshape(r,c)
        da = xr.DataArray(data,coords=[("x", xcoords), ("y", ycoords)],dims = ["x","y"])    
        return da

    def _appended_maps_to_dataset(self):
        """split the large np array into individual arrays for each day"""
        # get number of days
        num = len(self.map_dates)
        # split into list of arrays
        list_arrays = np.vsplit(self.dtw_maps, num)
        list_object = map(np.flipud, list_arrays)
        new_list_arrays = list(list_object)
        # change format to array of arrays
        separate_maps =  np.asarray(new_list_arrays)
        # create and xarray dataset (set of dataarrays)
        DHSVM_dwt = xr.Dataset(data_vars = {'wt': (('time', 'x', 'y'), separate_maps)})
        # assign coordinates and plot again 
        DHSVM_dwt['time'] = pd.to_datetime(self.map_dates)
        DHSVM_dwt['y'] = self.DHSVM_grid.y
        DHSVM_dwt['x'] = self.DHSVM_grid.x
        self.DHSVM_dwt = DHSVM_dwt
        
    def _resample(self, by = 'water_year', metric = 'max'):
        """aggregate values into water year or other duration. If duration is 
        less than frequency of data, then values are filled using fillna and 
        forward fill (i.e., if annual mean values are provided and the maps
        are resampled to hourly data, then the annual mean value is listed 
        for each hour"""
        water_year = None
        ds = self.DHSVM_dwt
        if by == 'water_year':
            # change time format to water year
            print('resampling based on WATER YEAR')
            water_year = (ds.time.dt.month >= 10) + ds.time.dt.year
            ds.coords['time'] = pd.to_datetime(water_year, format = '%Y')
            by = '1Y'
        if metric == 'max':
            ds = ds.resample(time=by).max()
        elif metric == 'mean':
            ds = ds.resample(time=by).mean() 
        elif metric == 'min':
            ds = ds.resample(time=by).min()    
        else:
            msg = "'{}' not a resampling option".format(metric)
            raise ValueError(msg)
        
        if water_year is not None:
            # for water year, year by beginning of year
            ds.coords['time'] = pd.to_datetime(pd.to_datetime(ds.time.values).year, format = '%Y')
                
        self.DHSVM_dwt = ds
                
    def _lin_interpolate_H_to_landlab(self):
            self.dtw_d_l = self.DHSVM_dwt.interp(y = self.y1, x = self.x1) 

    def _topmodel_interpolate_H_to_landlab(self):
        """interpolate DHSVM grid values to finer landlab gridding 
        scheme using the Topmodel wetness index following Doten et al., 2006"""
    
        # first linearly interpolate DHSVM grid values to landlab grid
        # lambda
        lambda_d_l = self.DHSVM_lambda.interp(y = self.y1, x = self.x1)   
        # depth to water table maps
        dtw_d_l = self.DHSVM_dwt.interp(y = self.y1, x = self.x1)    
    
        # now apply Topmodel approximation to account for topographic controls on 
        # local depth to water table given the average depth to water table
        
        # create the dataset of interpolated depth to water
        landlab_dtw_ = [] # list of arrays
        for i,m in enumerate(self.dtw_d_l.time):
            dtw_d_l_single = np.squeeze(self.dtw_d_l.isel(time=[i])['wt'])
            landlab_dtw_single = dtw_d_l_single + (lambda_d_l-self.landlab_lambda)/2
            
            # apply constraints to interpolated dtw values:
            
            # dtw can not be less than 0
            landlab_dtw_single.data[landlab_dtw_single.data<0] = 0
        
            # dtw can not exceed soil thickness
            landlab_dtw_single.data[landlab_dtw_single.data>self.landlab_soild.data] = \
                self.landlab_soild.data[landlab_dtw_single.data>self.landlab_soild.data]
                  
            landlab_dtw_.append(landlab_dtw_single.values)
        
        landlab_dtw = xr.Dataset(data_vars = {'wt': (('time', 'x', 'y'), landlab_dtw_)})
        #assign coordinates and plot again 
        if not self.resample:
            landlab_dtw['time'] = self.map_dates
        else:
            landlab_dtw['time'] = self.DHSVM_dwt.time
            
        landlab_dtw['y'] = self.y1
        landlab_dtw['x'] = self.x1
        
        self.landlab_dtw = landlab_dtw
        self.DHSVM_lambda_lin_interp = lambda_d_l