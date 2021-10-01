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

from RFSM_python.utils import *

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
from pyproj import Proj, transform
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set(); sns.set_context('notebook')


def create_netcf_Dinamicas_DOW(path_files,path_output,model,scenario):
    directory = list() 
    for file in os.listdir(path_files):
        if file.startswith("Punto"):
            directory.append(file)
    names_point = list()
    lon = list()
    lat = list()
    for d, dd in enumerate(directory):
        names_point.append(dd[:-19])
        lon.append(float(dd[-17:-10]))
        lat.append(float(dd[-8:-1]))
        
    for i,ii in enumerate(tqdm.tqdm(directory)):
        if i ==0:
            mat    = scipy.io.loadmat(path_files+directory[i]+'/'+directory[i]+'_'+model+'_'+scenario+'.mat')
            time   = mat['time'].flatten()
            hs_o   = mat['hs'].reshape(1,-1)
            tps_o  = mat['tps'].reshape(1,-1)
            dir_o  = mat['dir'].reshape(1,-1)
            zeta_o = mat['zeta'].reshape(1,-1)
            tm02_o = mat['tm02'].reshape(1,-1)
        else:
            mat = scipy.io.loadmat(path_files+directory[i]+'/'+directory[i]+'_'+model+'_'+scenario+'.mat')
            hs_o   = np.concatenate((hs_o, mat['hs'].reshape(1,-1)), axis=0)
            tps_o  = np.concatenate((tps_o, mat['tps'].reshape(1,-1)), axis=0)
            dir_o  = np.concatenate((dir_o, mat['dir'].reshape(1,-1)), axis=0)
            zeta_o = np.concatenate((zeta_o, mat['zeta'].reshape(1,-1)), axis=0)
            tm02_o = np.concatenate((tm02_o, mat['tm02'].reshape(1,-1)), axis=0)
        
    time_py = matDatenum2PYDatetime(time,unitTime = 'D')[0]
    
    nc = Dataset(path_output+'PtosObjtvo_DIN_'+model+'_'+scenario+'.nc', 'w', format='NETCDF4')
    # Global Attributes 
    nc.description= 'Contiene las dinámicas del modelo '+model+' en el escenario '+scenario  
    # nc dimensions
    nc.createDimension('lon',  len(lon))
    nc.createDimension('lat',  len(lat))
    nc.createDimension('time',len(time))
    #nc.createDimension('point',len(names_point))
    # crear variables
    xx_nc=nc.createVariable('lon','float32', ('lon'))
    yy_nc=nc.createVariable('lat','float32', ('lat'))
    time_nc=nc.createVariable('time','float32',('time'))
    #point_nc = nc.createVariable('point','int',('point'))
    hs_nc=nc.createVariable('hs','float32', ('time','lon'))
    tps_nc=nc.createVariable('tps','float32', ('time','lon'))
    dir_nc=nc.createVariable('dir','float32', ('time','lon'))
    zeta_nc=nc.createVariable('zeta','float32', ('time','lon'))
    tm02_nc=nc.createVariable('tm02','float32', ('time','lon'))
    #units
    xx_nc.units = 'degrees_east'
    yy_nc.units = 'degrees_north'
    time_nc.units='days since '+str(time_py[0].year)+'-01-01'
    hs_nc.units='m'
    tps_nc.units='s'
    dir_nc.units='º'
    zeta_nc.units='º'
    tm02_nc.units='s'
    #long_name
    xx_nc.long_name = 'longitude coordinate'
    yy_nc.long_name = 'latitud coordinate'
    time_nc.long_name='dias del año'
    hs_nc.long_name='Altura de ola'
    tps_nc.long_name='Período de pico suavizado'
    dir_nc.long_name='Dirección'
    zeta_nc.long_name='Marea meteorológica'
    tm02_nc.long_name='Periodo medio'
    # calendar
    time_nc.calendar = 'standard'
    # rellenar variables
    #point_nc[:]=np.arange(1,len(names_point)+1)
    xx_nc[:]=lon
    yy_nc[:]=lat
    time_nc[:]=date2num(time_py.to_pydatetime(), units='days since '+str(time_py[0].year)+'-01-01', calendar='standard')
    hs_nc[:]=hs_o[:,:].T
    tps_nc[:]=tps_o[:,:].T
    dir_nc[:]=dir_o[:,:].T
    zeta_nc[:] = zeta_o[:,:].T
    tm02_nc[:]=tm02_o[:,:].T
    nc.close()   
    
    
def create_netcf_MareaAst(path_files,path_output):
    directory = list() 
    for file in os.listdir(path_files):
        if file.startswith("Punto"):
            directory.append(file)
    names_point = list()
    lon = list()
    lat = list()
    for d, dd in enumerate(directory):
        names_point.append(dd[:-23])
        lon.append(float(dd[-21:-14]))
        lat.append(float(dd[-12:-5]))
        
    for i,ii in enumerate(tqdm.tqdm(directory)):
        if i ==0:
            mat    = scipy.io.loadmat(path_files+directory[i])
            time   = mat['time'].flatten()
            lon_tide_o = mat['lon_tide']
            lat_tide_o = mat['lat_tide']
            tide_o  = mat['tide'].reshape(1,-1)
            u_o  = mat['u'].reshape(1,-1)
            v_o  = mat['v'].reshape(1,-1)
        else:
            mat    = scipy.io.loadmat(path_files+directory[i])
            tide_o  =  np.concatenate((tide_o,mat['tide'].reshape(1,-1)),axis=0)
            lon_tide_o = np.concatenate((lon_tide_o,mat['lon_tide']),axis=0)
            lat_tide_o = np.concatenate((lat_tide_o,mat['lat_tide']),axis=0)
            u_o  = np.concatenate((u_o,mat['u'].reshape(1,-1)),axis=0)
            v_o  = np.concatenate((v_o,mat['v'].reshape(1,-1)),axis=0)
        
    time_py = matDatenum2PYDatetime(time,unitTime = 'D')[0]
    nc = Dataset(path_output+'PtosObjtvo_MAT.nc', 'w', format='NETCDF4')
    # Global Attributes 
    nc.description= 'Contiene la variable Marea Astronómica' 
    # nc dimensions
    nc.createDimension('lon',  len(lon))
    nc.createDimension('lat',  len(lat))
    nc.createDimension('time',len(time))
    #nc.createDimension('point',len(names_point))
    # crear variables
    xx_nc=nc.createVariable('lon','float32', ('lon'))
    yy_nc=nc.createVariable('lat','float32', ('lat'))
    time_nc=nc.createVariable('time','float32',('time'))
    lon_tide_nc = nc.createVariable('lon_tide','float32', ('lon'))
    lat_tide_nc = nc.createVariable('lat_tide','float32', ('lat'))
    #point_nc = nc.createVariable('point','int',('point'))
    tide_nc=nc.createVariable('tide','float32', ('time','lon'))
    u_nc=nc.createVariable('u','float32', ('time','lon'))
    v_nc=nc.createVariable('v','float32', ('time','lon'))
    #units
    xx_nc.units = 'degrees_east'
    yy_nc.units = 'degrees_north'
    time_nc.units='days since '+str(time_py[0].year)+'-01-01'
    lon_tide_nc.units = 'degrees_east'
    lat_tide_nc.units = 'degrees_north'
    tide_nc.units='m'
    u_nc.units='m'
    v_nc.units='m'
    #long_name
    xx_nc.long_name = 'longitude coordinate'
    yy_nc.long_name = 'latitud coordinate'
    lon_tide_nc.long_name = 'longitude coordinate tide'
    lat_tide_nc.long_name = 'latitud coordinate tide'
    time_nc.long_name='dias del año'
    tide_nc.long_name='Altura'
    u_nc.long_name='Dirección horizontal'
    v_nc.long_name='Dirección vertical'
    
    # calendar
    time_nc.calendar = 'standard'
    # rellenar variables
    #point_nc[:]=np.arange(1,len(names_point)+1)
    xx_nc[:]=lon
    yy_nc[:]=lat
    lon_tide_nc[:] = lon_tide_o
    lat_tide_nc[:] = lat_tide_o
    time_nc[:]=date2num(time_py.to_pydatetime(), units='days since '+str(time_py[0].year)+'-01-01', calendar='standard')
    tide_nc[:]=tide_o[:,:].T
    u_nc[:]=u_o[:,:].T
    v_nc[:]=v_o[:,:].T
    nc.close()
    
    
def create_netcf_Dinamicas_SLR(path_files,path_output):
    directory = list() 
    for file in os.listdir(path_files):
        if file.startswith("Punto"):
            directory.append(file)
    names_point = list()
    lon = list()
    lat = list()
    for d, dd in enumerate(directory):
        names_point.append(dd[:-27])
        lon.append(float(dd[-25:-18]))
        lat.append(float(dd[-16:-9]))
        
    for i,ii in enumerate(tqdm.tqdm(directory)):
        if i ==0:
            mat    = scipy.io.loadmat(path_files+directory[i])
            time   = mat['time'].flatten()
            m_ensemble45_o  = mat['m_ensemble45'].reshape(1,-1)
            m_ensemble85_o  = mat['m_ensemble85'].reshape(1,-1)
            P5_45_o  = mat['P5_45'].reshape(1,-1)
            P5_85_o  = mat['P5_85'].reshape(1,-1)
            P95_45_o = mat['P95_45'].reshape(1,-1)
            P95_85_o = mat['P95_85'].reshape(1,-1)
        else:
            mat = scipy.io.loadmat(path_files+directory[i])
            m_ensemble45_o   = np.concatenate((m_ensemble45_o, mat['m_ensemble45'].reshape(1,-1)), axis=0)
            m_ensemble85_o  = np.concatenate((m_ensemble85_o, mat['m_ensemble85'].reshape(1,-1)), axis=0)
            P5_45_o  = np.concatenate((P5_45_o, mat['P5_45'].reshape(1,-1)), axis=0)
            P5_85_o  = np.concatenate((P5_85_o, mat['P5_85'].reshape(1,-1)), axis=0)
            P95_45_o = np.concatenate((P95_45_o, mat['P95_45'].reshape(1,-1)), axis=0)
            P95_85_o = np.concatenate((P95_85_o, mat['P95_85'].reshape(1,-1)), axis=0)
        
    time_py = matDatenum2PYDatetime(time,unitTime = 'D')[0]
    
    nc = Dataset(path_output+'PtosObjtvo_MSL.nc', 'w', format='NETCDF4')
    # Global Attributes 
    nc.description= 'Contiene la variable SLR' 
    # nc dimensions
    nc.createDimension('lon',  len(lon))
    nc.createDimension('lat',  len(lat))
    nc.createDimension('time',len(time))
    #nc.createDimension('point',len(names_point))
    # crear variables
    xx_nc=nc.createVariable('lon','float32', ('lon'))
    yy_nc=nc.createVariable('lat','float32', ('lat'))
    time_nc=nc.createVariable('time','float32',('time'))
    #point_nc = nc.createVariable('point','int',('point'))
    m_ensemble45_nc=nc.createVariable('m_ensemble45','float32', ('time','lon'))
    m_ensemble85_nc=nc.createVariable('m_ensemble85','float32', ('time','lon'))
    P5_45_nc=nc.createVariable('P5_45','float32', ('time','lon'))
    P5_85_nc=nc.createVariable('P5_85','float32', ('time','lon'))
    P95_45_nc=nc.createVariable('P95_45','float32', ('time','lon'))
    P95_85_nc=nc.createVariable('P95_85','float32', ('time','lon'))
    #units
    xx_nc.units = 'degrees_east'
    yy_nc.units = 'degrees_north'
    time_nc.units='days since '+str(time_py[0].year)+'-01-01'
    m_ensemble45_nc.units='m'
    m_ensemble85_nc.units='s'
    P5_45_nc.units='º'
    P5_85_nc.units='º'
    P95_45_nc.units='º'
    P95_85_nc.units='l'
    #long_name
    xx_nc.long_name = 'longitude coordinate'
    yy_nc.long_name = 'latitud coordinate'
    time_nc.long_name='dias del año'
    m_ensemble45_nc.long_name='Ensemble RCP 45'
    m_ensemble85_nc.long_name='Ensemble RCP 85'
    P5_45_nc.long_name='Percentil 5% para el RCP 45'
    P5_85_nc.long_name='Percentil 5% para el RCP 85'
    P95_45_nc.long_name='Percentil 95% para el RCP 45'
    P95_85_nc.long_name='Percentil 95% para el RCP 85'
    # calendar
    time_nc.calendar = 'standard'
    # rellenar variables
    #point_nc[:]=np.arange(1,len(names_point)+1)
    xx_nc[:]=lon
    yy_nc[:]=lat
    time_nc[:]=date2num(time_py.to_pydatetime(), units='days since '+str(time_py[0].year)+'-01-01', calendar='standard')
    m_ensemble45_nc[:]=m_ensemble45_o[:,:].T
    m_ensemble85_nc[:]=m_ensemble85_o[:,:].T
    P5_45_nc[:]=P5_45_o[:,:].T
    P5_85_nc[:]=P5_85_o[:,:].T
    P95_45_nc[:] = P95_45_o[:,:].T
    P95_85_nc[:]=P95_85_o[:,:].T
    nc.close()