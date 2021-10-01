import pandas as pd
import numpy as np
import sys
import os
from netCDF4 import Dataset, date2num
import glob
import scipy.io
import tqdm
import xarray as xr
import os

import numpy as np
import pandas as pd
import os

from scipy.optimize import dual_annealing

from scipy.stats import genextreme as gev
from scipy.stats import gumbel_l as gumbel
from scipy.stats import gamma as gamma
from scipy.stats import expon as exp
from scipy.stats import lognorm as logn
from scipy.stats import gumbel_r as gumbel_r
from scipy.stats import powerlaw as powerlaw

import scipy.stats as st

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import statsmodels.api as sma
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
from pyproj import Proj, transform
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set(); sns.set_context('notebook')

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def Ajuste_GEV_KMA_Frechet(data, plot=False):
    #Ajuste Gumbel
    func=getattr(st,'genextreme')
    paramt = func.fit(data,fc=0.0000000001)
    nlogL = np.log(func.pdf(data,*paramt)).sum()
    parmhatgev = func.fit(data)
    nlogLGev = np.log(func.pdf(data,*parmhatgev)).sum()
    
    if -parmhatgev[0]>0:
        isgev = np.sum((np.abs(nlogLGev)-np.abs(nlogL))>=1.92)
        if  isgev==1:
                paramGev= parmhatgev
        else:
             paramGev = paramt
    else:
        paramGev= parmhatgev

    if plot == True:
        fig, ax=plt.subplots(2, 2, figsize=(12, 8))
        data.plot(kind='hist', bins=50, density =True, alpha=0.5, color='red',ax=ax[0,0])
        dataYLim = ax[0,0].get_ylim()
        distribution=getattr(st,'genextreme')
        best_distribution = distribution
        best_dist = getattr(st,'genextreme')
        best_fit_params = paramGev
        best_fit_name='genextreme'

        #----------------------------------------------------------------------------------- 
        pdf = make_pdf(best_dist, best_fit_params)

        pdf.plot(lw=2, label='PDF', legend=True, ax=ax[0,0])
        data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax[0,0])

        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)

        ax[0,0].set_title(u'Best probability density function \n' + dist_str)
        ax[0,0].set_xlabel(u'Data')
        ax[0,0].set_ylabel('Frequency')
        ax[0,0].set_ylim(dataYLim)
        #-----------------------------------------------------------------------------------     
        data=data.values.flatten()
        PQmax_1 = best_dist.cdf(data.astype(float),*best_fit_params);
        Qemp = np.sort(data);
        kk = np.arange(1,len(data)+1);
        prob = kk/(len(data));
        Qlogn  = best_dist.ppf(prob,*best_fit_params);
        ax[0,1].plot(Qemp,Qlogn,'.k')
        ax[0,1].plot(np.arange(min(data),max(data)),np.arange(min(data),max(data)),'-b')
        ax[0,1].set_title('Q-Q Plot \n' + dist_str)
        ax[0,1].set_xlabel(u'Empirical')
        ax[0,1].set_ylabel('Model')
        #-----------------------------------------------------------------------------------     
        ecdf=ECDF(data)
        x = np.linspace(min(data),max(data),200)
        y = best_dist.cdf(x, *best_fit_params)
        ax[1,0].plot(x, y)
        ax[1,0].scatter(data.astype(float),ecdf(data))
        ax[1,0].set_title('Best Cumulative density function \n' + dist_str)
        ax[1,0].set_xlabel(u'Data')
        ax[1,0].set_ylabel('Prob')
        #-----------------------------------------------------------------------------------    
        x_r=np.arange(0.01,1000)
        prob_t=(1-1/x_r)
        y_r=best_dist.ppf(prob_t,*best_fit_params)
        ax[1,1].plot(x_r,y_r,'-b')

        ecdf=ECDF(data)
        t_r=1/(1-ecdf(data))
        ax[1,1].plot(t_r,data,'.k')
        ax[1,1].set_xscale("log")
        ax[1,1].set_xlabel(u'Return Periods')
        ax[1,1].set_ylabel('Return Values')
        ax[1,1].set_title('Return Values Plot \n' + dist_str)

        print(best_fit_params)
        fig.tight_layout()  
    func=getattr(st,'genextreme')
    
    PeriodoRetorno=[2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,200,500,1000]
    TWL = pd.DataFrame(index=PeriodoRetorno,columns=['TWL'])
    for i in PeriodoRetorno:
        TWL.loc[i] = func.ppf((1-1/i),*paramGev)
        
    return TWL

def gumbel_loglik(params, datos):
    func=getattr(st,'gumbel')
    loc, scale = params
    return -func(loc=loc, scale=scale).logpdf(datos).sum()

def fit_EVAGumbel(data,T):
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf=ECDF(data.flatten())
    func=getattr(st,'gumbel')
   
    params = func.fit(data.astype(float))
    loc = params[-2]
    scale = params[-1]
    y_ = ecdf(data.flatten())
    cdf = func.cdf(data.flatten(), loc=loc, scale=scale)
    sse = np.sum(np.power(y_ - cdf, 2.0))
    if sse>0.5:
        limites = [(-200., 200.), (1e-3, 200)]
        res = dual_annealing(gumbel_loglik, limites, args=[data,func])
        params = res.x
    return params
    
def RP_EVAGumbel(parms, T):
    func=getattr(st,'gumbel')
    TWL=func.ppf((1-1/T),*params)
    return TWL


def EVAGumbel_plot(data,params):
    func=getattr(st,'gumbel')
    fig, ax=plt.subplots(2, 2, figsize=(12, 8))
    data.plot(kind='hist', bins=50, density =True, alpha=0.5, color='red',ax=ax[0,0])
    dataYLim = ax[0,0].get_ylim()
    distribution=getattr(st,'gumbel')
    best_distribution = distribution
    best_dist = getattr(st,'gumbel')
    best_fit_params = paramGev
    best_fit_name='gumbel'

    #----------------------------------------------------------------------------------- 
    pdf = make_pdf(best_dist, best_fit_params)

    pdf.plot(lw=2, label='PDF', legend=True, ax=ax[0,0])
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax[0,0])

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax[0,0].set_title(u'Best probability density function \n' + dist_str)
    ax[0,0].set_xlabel(u'Data')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_ylim(dataYLim)
    #-----------------------------------------------------------------------------------     
    data=data.values.flatten()
    PQmax_1 = best_dist.cdf(data.astype(float),*best_fit_params);
    Qemp = np.sort(data);
    kk = np.arange(1,len(data)+1);
    prob = kk/(len(data));
    Qlogn  = best_dist.ppf(prob,*best_fit_params);
    ax[0,1].plot(Qemp,Qlogn,'.k')
    ax[0,1].plot(np.arange(min(data),max(data)),np.arange(min(data),max(data)),'-b')
    ax[0,1].set_title('Q-Q Plot \n' + dist_str)
    ax[0,1].set_xlabel(u'Empirical')
    ax[0,1].set_ylabel('Model')
    #-----------------------------------------------------------------------------------     
    ecdf=ECDF(data)
    x = np.linspace(min(data),max(data),200)
    y = best_dist.cdf(x, *best_fit_params)
    ax[1,0].plot(x, y)
    ax[1,0].scatter(data.astype(float),ecdf(data))
    ax[1,0].set_title('Best Cumulative density function \n' + dist_str)
    ax[1,0].set_xlabel(u'Data')
    ax[1,0].set_ylabel('Prob')
    #-----------------------------------------------------------------------------------    
    x_r=np.arange(0.01,1000)
    prob_t=(1-1/x_r)
    y_r=best_dist.ppf(prob_t,*best_fit_params)
    ax[1,1].plot(x_r,y_r,'-b')

    ecdf=ECDF(data)
    t_r=1/(1-ecdf(data))
    ax[1,1].plot(t_r,data,'.k')
    ax[1,1].set_xscale("log")
    ax[1,1].set_xlabel(u'Return Periods')
    ax[1,1].set_ylabel('Return Values')
    ax[1,1].set_title('Return Values Plot \n' + dist_str)

    print(best_fit_params)
    fig.tight_layout()  
    
def EVAGumbel_inversa(params,TWLpeak):
    func=getattr(st,'gumbel')
    quantil=func.cdf(TWLpeak,*params)
    
    Resturn_period = 1/(1-quantil)
    
    return Resturn_period

