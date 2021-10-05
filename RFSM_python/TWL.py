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

import scipy.stats as st

import statsmodels as sm
import statsmodels.api as sma
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from pyproj import Proj, transform
from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns
sns.set(); sns.set_context('notebook')

from RFSM_python.utils import *

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


######### GEV_KMA_Frechet ######
def Gev_loglik(params, datos):
    func=getattr(st,'genextreme')
    c, loc, scale = params
    return -func(c, loc=loc, scale=scale).logpdf(datos).sum()


def fit_GEV_KMA_Frechet(data):
    from statsmodels.distributions.empirical_distribution import ECDF
    
    ecdf=ECDF(data.flatten())
    func=getattr(st,'genextreme')
    
    paramt      = func.fit(data.astype(float),fc=0.0000000001)
    nlogL       = np.log(func.pdf(data,*paramt)).sum()
    
    arg = params[:-2]
    loc   = paramt[-2]
    scale = paramt[-1]
    y_ = ecdf(data.flatten())
    
    cdf = func.cdf(data.flatten(), loc=loc, scale=scale, c =0.0000000001)
    sse = np.sum(np.power(y_ - cdf, 2.0))
    if sse>0.5:
        limites = [(0,0.0000000001),(-200., 200.), (1e-3, 200)]
        res = dual_annealing(Gev_loglik, limites, args=[data,func])
        paramt = res.x
    
    
    parmhatgev  = func.fit(data)
    nlogLGev    = np.log(func.pdf(data,*parmhatgev)).sum()
    
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    y_ = ecdf(data.flatten())
    cdf = func.cdf(data.flatten(), loc=loc, scale=scale, *arg)
    sse = np.sum(np.power(y_ - cdf, 2.0))
    if sse>0.5:
        limites = [(-2, 2),(-200., 200.), (1e-3, 200)]
        res = dual_annealing(Gev_loglik, limites, args=[data,func])
        parmhatgev = res.x
        
    
    if -parmhatgev[0]>0:
        isgev = np.sum((np.abs(nlogLGev)-np.abs(nlogL))>=1.92)
        if  isgev==1:
                paramGev= parmhatgev
        else:
             paramGev = paramt
    else:
        paramGev= parmhatgev
        
    return parmhatgev

def RP_GEV_KMA_Frechet(params, T):
    func=getattr(st,'genextreme')
    TWL=func.ppf((1-1/T),*params)
    return TWL



# def Ajuste_GEV_KMA_Frechet(data, plot=False):
#     #Ajuste Gumbel
#     func=getattr(st,'genextreme')
#     paramt = func.fit(data,fc=0.0000000001)
#     nlogL = np.log(func.pdf(data,*paramt)).sum()
#     parmhatgev = func.fit(data)
#     nlogLGev = np.log(func.pdf(data,*parmhatgev)).sum()
    
#     if -parmhatgev[0]>0:
#         isgev = np.sum((np.abs(nlogLGev)-np.abs(nlogL))>=1.92)
#         if  isgev==1:
#                 paramGev= parmhatgev
#         else:
#              paramGev = paramt
#     else:
#         paramGev= parmhatgev

def GEV_KMA_Frechet_plot(data,params):
    func=getattr(st,'genextreme')
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

    fig.tight_layout()  
    
    
def GEV_KMA_Frechet_inversa(params,TWLpeak):
    func=getattr(st,'genextreme')
    quantil=func.cdf(TWLpeak,*params)
    
    Resturn_period = 1/(1-quantil)
    
    return Resturn_period


############ Gumbel ##################

def gumbel_loglik(params, datos):
    func=getattr(st,'gumbel_r')
    loc, scale = params
    return -func(loc=loc, scale=scale).logpdf(datos).sum()

def fit_EVAGumbel(data):
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf=ECDF(data.flatten())
    func=getattr(st,'gumbel_r')
   
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
    
def RP_EVAGumbel(params, T):
    func=getattr(st,'gumbel_r')
    loc   = params[0]
    scale = params[1]
    TWL=loc- scale*np.log(-np.log(np.exp(-1/T)) ) 
    return TWL


def EVAGumbel_plot(data,params):
    func=getattr(st,'gumbel_r')
    fig, ax=plt.subplots(2, 2, figsize=(12, 8))
    data.plot(kind='hist', bins=50, density =True, alpha=0.5, color='red',ax=ax[0,0])
    dataYLim = ax[0,0].get_ylim()
    distribution=getattr(st,'gumbel_r')
    best_distribution = distribution
    best_dist = getattr(st,'gumbel_r')
    best_fit_params = params
    best_fit_name='gumbel_r'

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

    fig.tight_layout()
    
def EVAGumbel_inversa(params,TWLpeak):
    func=getattr(st,'gumbel_r')
    quantil=func.cdf(TWLpeak,*params)
    
    Resturn_period = 1/(1-quantil)
    
    return Resturn_period


def setup_fun (Tp,Hs,ksetup,mfore,kswash):
    L0=9.81/(2.*np.pi)*(Tp**2)
    
    setup = ksetup*0.35*mfore*np.sqrt(Hs*L0)
    swash = kswash*0.03*np.sqrt(Hs*L0)
    Stockdon_setup = setup+swash

    return Stockdon_setup

def generate_files_extrems(path_inputs_dinamicas,path_outputs,pointsTWL,PeriodosRetorno, refNMMA, func_extrem, cod_epsg, plot=True):
    """ La siguiente función nos permite generar los ficheros de TWL para el escenario ROW.
        A continuación se explican cada uno de los parámetros de entrada necesarios y los outputs generados
        
        Parámetros:
        ---------------------
        path_inputs_dinamicas   : string. path donde se encuentran los netcdf de las dinámicas para cada uno de los puntos de estudio.
        path_outputs            : string. path donde se desean guardar los ficheros del análisis de extremos.
        pointsTWL               : pandas dataframe. Dataframe con las coordenadas y características de los puntos donde se van a asignar las dinámicas.
        PeriodosRetorno         : list. Lista con los períodos de retorno que se desea utilizar.
        refNMMA                 : float. Referencia nivel medio del mar.
        func_extrem             : string. Función de extremos que se desea utilizar.
        cod_epsg                : int. Código epsg del sistema de coordenadas utilizado en las coordendas de los puntos de estudio. 
        plot                    : True or flase. Si se quiere plotear el ajuste de extremos
        
        Salidas:
        ---------------------
        Ficheros csv para cada uno de modelos, escenarios y periodos.
    
    """
    
    # cargamos el shapefile con información del tipo de costa, difracción, marea y río
    pointsTWL = gpd.read_file(pointsTWL) # cargo el archivo puntos definidos para el cálculo del TWL con todos los atributos que contiene


    PtosObjtvo_ROW_his_nc = xr.open_dataset(path_inputs_dinamicas+'PtosObjtvo_DIN_DOW_his.nc')
    PtosObjtvo_MAT_nc = xr.open_dataset(path_inputs_dinamicas+'PtosObjtvo_MAT.nc')
    
    time1 = pd.to_datetime(PtosObjtvo_ROW_his_nc.time.data).round('H')
    time2 = pd.to_datetime(PtosObjtvo_MAT_nc.time.data).round('H')
    Time_final = time1.intersection(time2)

    #PeriodoRetorno=[2, 3, 4, 5, 6, 7, 8, 9, 10, 20,25, 30, 40, 50, 60, 70, 80, 90, 100,200,500,1000]
    string_PR = [str(int) for int in PeriodosRetorno]
    
    print('Calculando TWL')

    TWL_extrem_hist_tab = pd.DataFrame(index=pointsTWL.index,columns=string_PR+['MSLR','SLR_5','SLR_95'])
    TWL_extrem_hist_tab.loc[:,['MSLR','SLR_5','SLR_95']] = 0
    
    HS_extrem_hist_tab = pd.DataFrame(index=pointsTWL.index,columns=string_PR)
    
    TWL_serie = pd.DataFrame(index = Time_final, columns=pointsTWL.index)
    HS_serie  = pd.DataFrame(index = Time_final, columns=pointsTWL.index)
    
    PMVE_serie      = pd.DataFrame(index =['t0','t1','t2','t3','t4','t5','t6'], columns=pointsTWL.index)
    
    if func_extrem=='gumbel':
    
        Params_extrem_his_TWL    = pd.DataFrame(index=pointsTWL.index, columns=['loc','scale'])
        Params_extrem_his_HS    = pd.DataFrame(index=pointsTWL.index, columns=['loc','scale'])
    else:
        Params_extrem_his_TWL    = pd.DataFrame(index=pointsTWL.index, columns=['c','loc','scale'])
        Params_extrem_his_HS     = pd.DataFrame(index=pointsTWL.index, columns=['c','loc','scale'])
        
    inProj = Proj(init='epsg:'+str(cod_epsg))
    outProj = Proj(init='epsg:4326')

    for i in tqdm.tqdm(range(len(pointsTWL))):
        
        x_0 = pointsTWL.CX[i]
        y_0 = pointsTWL.CY[i]
        
        kswash = pointsTWL.kswash[i] # coeficiente de minoración o mayoración de la componente del swash en el TWL
        ksetup = pointsTWL.ksetup[i] # coeficiente de minoración o mayoración de la componente del setup en el TWL
        kr     = pointsTWL.kr[i] # coeficiciente de reducción del oleaje (difracción, interior estuario)
        mfore = pointsTWL.mforeshore[i] # pendiente intermareal (foreshore)
        ktide = pointsTWL.ktide[i] # coeficiente de minoración o mayoración de la marea astronómica
        
        
        x,y = transform(inProj,outProj,x_0,y_0)

        point_selec_his = PtosObjtvo_ROW_his_nc.point.data[np.sqrt((x-PtosObjtvo_ROW_his_nc.lon.data)**2+(y-PtosObjtvo_ROW_his_nc.lat.data)**2).argmin()]

        point_selec_MAT = PtosObjtvo_MAT_nc.point.data[np.sqrt((x-PtosObjtvo_MAT_nc.lon.data)**2+(y-PtosObjtvo_MAT_nc.lat.data)**2).argmin()]



        Hs_his = PtosObjtvo_ROW_his_nc.sel(point = point_selec_his,drop=True)['hs'].to_dataframe()
        Hs_his.index = Hs_his.index.round('H')
        try:
            Hs_his[Hs_his.hs<0]=0.001
        except:
            Hs_his = Hs_his.copy()
           
        Hs_his = Hs_his*kr
        
        HS_serie.iloc[:,i] = Hs_his.loc[HS_serie.index].values

        Tp_his = PtosObjtvo_ROW_his_nc.sel(point = point_selec_his,drop=True)['tps'].to_dataframe()
        Tp_his.index = Tp_his.index.round('H')
        try:
            Tp_his[Tp_his.tps<=0]=np.mean(Tp_his[Tp_his>0],axis=0)
        except:
            Tp_his = Tp_his

        SS_his = PtosObjtvo_ROW_his_nc.sel(point = point_selec_his,drop=True)['zeta'].to_dataframe()
        SS_his.index = SS_his.index.round('H')
        
        SS_his = SS_his-np.nanmean(SS_his)


        AT = PtosObjtvo_MAT_nc.sel(point = point_selec_MAT,drop=True)['tide'].to_dataframe()
        AT.index = AT.index.round('H')

        PMVE = AT.iloc[AT.values.argmax()-3:AT.values.argmax()+4]
        PMVE = PMVE*ktide+refNMMA
        PMVE_serie.iloc[:,i] = PMVE.values

        setup_hist  = pd.DataFrame(index=Tp_his.index, columns=['setup'])
        setup_hist.iloc[:,0]  = setup_fun(Tp_his.values,Hs_his.values, ksetup, mfore, kswash)

        ## Cálculo de la cota de inundación / TWL

        Result_hist  = pd.concat((setup_hist,SS_his,AT),axis = 1).dropna()
        

        TWL_hist  = Result_hist.iloc[:,0]+Result_hist.iloc[:,1]+Result_hist.iloc[:,2]+refNMMA
        
        TWL_serie.loc[Result_hist.index,pointsTWL.index[i]] = TWL_hist.values 
        
        ##### TWL ######
        return_period_TWL = []
        
        data = TWL_hist.resample('A').max().dropna().astype(float).values
        
        if func_extrem=='gumbel':
            params = fit_EVAGumbel(data)
            
            for T in PeriodosRetorno:
                return_period_TWL.append(RP_EVAGumbel(params, T))
            if plot==True:
                EVAGumbel_plot(TWL_hist.resample('A').max().dropna().astype(float), params) 
            
            
        elif func_extrem=='GEV_KMA_Frechet':
            params = fit_GEV_KMA_Frechet(data)
            for T in PeriodosRetorno:
                return_period_TWL.append(RP_GEV_KMA_Frechet(params, T))  
            if plot==True:
                GEV_KMA_Frechet_plot(TWL_hist.resample('A').max().dropna().astype(float),params)
            
              
        TWL_extrem_hist_tab.iloc[i,:len(PeriodosRetorno)]   = np.array(return_period_TWL).reshape(1,-1)
        Params_extrem_his_TWL.iloc[i,:]  = params
        
        
        ##### HS ######
        return_period_Hs = []
        
        data = Hs_his.resample('A').max().dropna().astype(float).values
        
        if func_extrem=='gumbel':
            params = fit_EVAGumbel(data)
            
            for T in PeriodosRetorno:
                return_period_Hs.append(RP_EVAGumbel(params, T))
            if plot==True:
                EVAGumbel_plot(TWL_hist.resample('A').max().dropna().astype(float), params) 
            
            
        elif func_extrem=='GEV_KMA_Frechet':
            params = fit_GEV_KMA_Frechet(data)
            for T in PeriodosRetorno:
                return_period_Hs.append(RP_GEV_KMA_Frechet(params, T))  
            if plot==True:
                GEV_KMA_Frechet_plot(TWL_hist.resample('A').max().dropna().astype(float),params)
            
              
        HS_extrem_hist_tab.iloc[i,:len(PeriodosRetorno)]   = np.array(return_period_Hs).reshape(1,-1)
        Params_extrem_his_HS.iloc[i,:]  = params
        
              
    TWL_extrem_hist_tab.to_csv(path_outputs+'TWL_extrem_ROW.csv')
    Params_extrem_his_TWL.to_csv(path_outputs+'Params_extrem_ROW.csv')
    
    HS_extrem_hist_tab.to_csv(path_outputs+'Hs_extrem_ROW.csv')
    Params_extrem_his_HS.to_csv(path_outputs+'Params_extrem_Hs.csv')
    
    PMVE_serie.to_csv(path_outputs+'Evento_PMVE.csv')
    
    
    
    return TWL_serie,HS_serie,PMVE_serie