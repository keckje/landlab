# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:42:20 2022

@author: keckj
"""


# setup
import os

## import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate


#%% define plotting functions
def both_para_results(results, params):
    
    x_mn = params['SD'][0]
    x_mx = params['SD'][1]
    y_mn = params['cs'][0]
    y_mx = params['cs'][1]
        
    plt.figure(figsize = (6,3))
    plt.plot(results['selected_value_SD'], results['selected_value_cs'])
    plt.xlim([x_mn,x_mx])
    plt.ylim([y_mn,y_mx])
    plt.xlabel('crtical flow depth, below which everything stops $qs_c$, [m]')
    plt.ylabel(r'scour coef., $\alpha$')
    
    grid_x, grid_y = np.mgrid[x_mn:x_mx:20j,y_mn:y_mx:20j]
    
    points = results[['selected_value_SD','selected_value_cs']].values
    
    values = results['selected_posterior'].values
    
    grid_z1 = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
    
    plt.figure(figsize = (6,3))
    plt.imshow(grid_z1.T, extent=(x_mn,x_mx,y_mn,y_mx), origin='lower')
    plt.xlabel('crtical flow depth, below which everything stops $qs_c$, [m]')
    plt.ylabel(r'scour coef., $\alpha$')
    
    
    X = grid_x[:,0]
    Y = grid_y[:,0]
    plt.figure(figsize = (3,3))
    plt.imshow(grid_z1.T, extent=(x_mn,x_mx,y_mn,y_mx), origin='lower', cmap = 'Greys_r', alpha = 0.5)
    plt.contour(grid_x,grid_y,grid_z1,np.linspace(np.nanmin(grid_z1),np.nanmax(grid_z1),7), colors='k', linewidth = 0.5)
    plt.xlabel('$qs_c$, [m]')
    plt.ylabel(r'$\alpha$')
    # plt.savefig(wdir+"calibration"+svnm+".png", dpi = 300, bbox_inches='tight')
    
    
    
    plt.figure(figsize = (6,3))
    n = results.shape[0]
    counts, xedge,yedge,image =  plt.hist2d(results['selected_value_SD'], results['selected_value_cs'], bins = 15)#int(n**0.5))
    plt.xlabel('crtical flow depth, below which everything stops $qs_c$, [m]')
    plt.ylabel(r'scour coef., $\alpha$')
    plt.title('histogram')
    plt.colorbar()
    # plt.savefig(wdir+"histogram_"+svnm+".png", dpi = 300, bbox_inches='tight')
    



def parameter_uncertainty(results, parameter):
    """parameter can be either 'cs' or 'SD' """
    # parameter diagnostic plots
    # check that tested parameter values display proper variability (see MCMC_notes.pdf Figure 3)
    col = 'candidate_value_'+parameter
    plt.figure(figsize = (3,2))
    plt.plot(results[col], color = 'k', alpha = 0.8, linewidth = 1)
    plt.ylabel(parameter)
    plt.xlabel('iteration')
    
    # get count in each parameter value bin
    # "The distributions of counts in each bin gives the probabilility distribution of the parameter as a function of the given the rules" 
    # (Jessica class note, Markov_models_lecture2017, pg 63). 95% confidence intervals are referred to as credibility intervals, rather than 
    # confidence interval.
    plt.figure(figsize = (3,2))
    n = results[col].shape[0]
    counts, edges, plot = plt.hist(results[col],bins = int(n**0.5),color = 'k',alpha = 0.8)
    plt.grid(alpha = 0.5)
    plt.xlabel(parameter)
    plt.ylabel('count')
    
    
    # sum the histogram bins to get the cdf, using the bin center
    bin_cntr = ((edges[1:]-edges[0:-1])/2).cumsum() 
    bin_cnt = counts.cumsum() 
    
    plt.figure(figsize = (3,2))
    cp = bin_cnt/counts.sum()
    plt.plot(bin_cntr, cp, color = 'k', alpha = 0.8, linewidth = 1)
    plt.grid(alpha = 0.5)
    plt.xlabel(parameter)
    plt.ylabel('P(x)')
    
    def intp(x,y,x1,message = None): 
        f = interpolate.interp1d(x,y)   
        y1 = f(x1)
        return y1
    
    lb = intp([0]+list(cp),[0]+list(bin_cntr),0.0501010)
    up = intp([0]+list(cp),[0]+list(bin_cntr),0.9510101)
    
    
    
    # by likihood? 
    window = int(results.shape[0]/20)
    sorted_results = results.sort_values(by = col)
    
    average = sorted_results['candidate_posterior'].rolling(window=window).mean()
    plt.figure(figsize = (3,2))
    plt.plot(results[col], results['candidate_posterior'], 'k.', markersize = 5, alpha = 0.2)
    plt.plot(sorted_results[col], average, 'r-', alpha = 0.75, linewidth = 2)
    plt.grid(alpha = 0.5)
    plt.ylabel("likelihood")
    plt.xlabel(parameter)

    average = sorted_results['1/RMSE'].rolling(window=window).mean()
    plt.figure(figsize = (3,2))
    plt.plot(results[col], results['1/RMSE'], 'k.', markersize = 5, alpha = 0.2)
    plt.plot(sorted_results[col], average, 'r-', alpha = 0.75, linewidth = 2)
    plt.grid(alpha = 0.5)
    plt.ylabel('1/RMSE')
    plt.xlabel(parameter) 
    
    average = sorted_results['1/RMSE p'].rolling(window=window).mean()
    plt.figure(figsize = (3,2))
    plt.plot(results[col], results['1/RMSE p'], 'k.', markersize = 5, alpha = 0.2)
    plt.plot(sorted_results[col], average, 'r-', alpha = 0.75, linewidth = 2)
    plt.grid(alpha = 0.5)
    plt.ylabel('1/RMSE p')
    plt.xlabel(parameter) 
    
    average = sorted_results['1/RMSE m'].rolling(window=window).mean()
    plt.figure(figsize = (3,2))
    plt.plot(results[col], results['1/RMSE m'], 'k.', markersize = 5, alpha = 0.2)
    plt.plot(sorted_results[col], average, 'r-', alpha = 0.75, linewidth = 2)
    plt.grid(alpha = 0.5)
    plt.ylabel('1/RMSE m')
    plt.xlabel(parameter)   
    
    average = sorted_results['DTE'].rolling(window=window).mean()
    plt.figure(figsize = (3,2))
    plt.plot(results[col], results['DTE'], 'k.', markersize = 5, alpha = 0.2)
    plt.plot(sorted_results[col], average, 'r-', alpha = 0.75, linewidth = 2)
    plt.grid(alpha = 0.5)
    plt.ylabel('DTE')
    plt.xlabel(parameter)    
    
    average = sorted_results['omegaT'].rolling(window=window).mean()
    plt.figure(figsize = (3,2))
    plt.plot(results[col], results['omegaT'], 'k.', markersize = 5, alpha = 0.2)
    plt.plot(sorted_results[col], average, 'r-', alpha = 0.75, linewidth = 2)
    plt.grid(alpha = 0.5)
    plt.ylabel('omegaT')
    plt.xlabel(parameter)      

#%%

params = {'SD': [0.001, 1.5, 0.6], 'cs': [0.001, 1.5, 0.15]}
mdir = 'D:/UW_PhD/PreeventsProject/Paper_2_MWR/RunoutValidation/S1000/output/MCMC_v1/'


x_mn = params['SD'][0]
x_mx = params['SD'][1]
y_mn = params['cs'][0]
y_mx = params['cs'][1]








# all
# csvnm = 's1000_22_1000_all_metricsmcmc.csv'
csvnm = 's1000_22_1000_all_300itlmcmc.csv'
# no DTE
# csvnm = 's1000_22_1000_noDTEmcmc.csv'
# no DTE, no Omega
# csvnm = 's1000_22_1000_noDTEnoOmegamcmc.csv' 
# Omega, RMSE only
# csvnm = 's1000_22_1000_RMSEVdOmegaTmcmc.csv'

# 300



results = pd.read_csv(mdir+csvnm)


# check RMSE values
itm = results['iteration'].max()
plt.figure()
plt.plot(results['iteration'], results['1/RMSE']/results['1/RMSE'].max(),'k-', alpha = 0.5, label = 'RMSE V')
plt.plot(results['iteration'], results['1/RMSE p']/results['1/RMSE p'].max(),'r-', alpha = 0.5, label = 'RMSE p')
plt.plot(results['iteration'], results['1/RMSE m']/results['1/RMSE m'].max(),'g-', alpha = 0.5, label = 'RMSE m')
plt.plot(results['iteration'], results['DTE']/results['DTE'].max(),'c-', alpha = 0.5, label = 'DTE')
plt.plot(results['iteration'], results['omegaT']/results['omegaT'].max(),'m-', alpha = 0.5, label = 'OmegaT')
plt.xlim(0,results['iteration'].max()*1.20)
# plt.xticks(np.arange(0,itm+1, step=int(1+itm/20)))
plt.legend(fontsize = 8, loc = "right")
plt.show()


# look at calibration results
both_para_results(results, params)

parameter_uncertainty(results, "SD")


parameter_uncertainty(results, "cs")


# 
it_best_all =results['iteration'][results['candidate_posterior'] == results['candidate_posterior'].max()].values[0]
it_best_r =results['iteration'][results['1/RMSE'] == results['1/RMSE'].max()].values[0]
it_best_r_p =results['iteration'][results['1/RMSE p'] == results['1/RMSE p'].max()].values[0]
it_best_r_m =results['iteration'][results['1/RMSE m'] == results['1/RMSE m'].max()].values[0]

results.iloc[it_best]


# calibration is best at SD 0.7+/-0.1
# find all SD 0.7 +/-0.1 values with a high likihood (0.8 quantile)
best = results[((results['candidate_value_SD']>0.55) & (results['candidate_value_SD']<0.75) 
 & results['candidate_posterior']>results['candidate_posterior'].quantile(0.95))]

plt.figure()
plt.boxplot(best['candidate_value_cs'])
best['candidate_value_cs'].quantile(0.5)

plt.figure()
plt.boxplot(best['candidate_value_SD'])
best['candidate_value_SD'].quantile(0.5)
