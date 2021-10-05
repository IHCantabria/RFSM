# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr
import os
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
import tqdm
import shapely
from shapely.geometry import MultiPolygon, Point
from scipy.interpolate import griddata
from datetime import datetime
import glob
import os, shutil
from scipy import stats
import urllib.request
import fiona
import rasterio
import time
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely import wkt
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from rasterstats import zonal_stats
from pandas.io.json import json_normalize
from shapely.geometry import Point, LineString
gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")

from osgeo import ogr
import math

def make_dirs_RFSM(path_install_RFSM,path_project):
    if len(os.listdir(path_project))==0:
        copytree(path_install_RFSM, path_project, symlinks=False, ignore=None) ## Cambiar el path donde tenemos la instalación de RFSM

def make_dirs(path_project,TestDesc,Results):
    if os.path.exists(path_project+'tests/'+TestDesc) == False:
        os.mkdir(path_project+'tests/'+TestDesc)

    if os.path.exists(path_project+'tests/'+TestDesc+'/Input_xml/') == False:
        os.mkdir(path_project+'tests/'+TestDesc+'/Input_xml/')

    if os.path.exists(path_project+'tests/'+TestDesc+'/export/') == False:
        os.mkdir(path_project+'tests/'+TestDesc+'/export/')    

    if os.path.exists(path_project+'tests/'+TestDesc+'/Input_AccData/') == False:
        os.mkdir(path_project+'tests/'+TestDesc+'/Input_AccData/',mode=0o777) 

    if os.path.exists(path_project+'tests/'+TestDesc+'/Input_User/') == False:
        os.mkdir(path_project+'tests/'+TestDesc+'/Input_User/')
        
    tblkTestBCType=pd.DataFrame(index=np.arange(0,5),columns=['BCTypeID', 'BCType'])
    tblkTestBCType.iloc[:,0]=np.arange(1,6)
    tblkTestBCType.iloc[:,1]=['Discharge','Level','LevelOut','Levelln','LevelFlowRating']
    tblkTestBCType.to_csv(path_project+'tests/'+TestDesc+'/Input_User/tblkTestBCType.csv',index=False)

    if os.path.exists(path_project+'tests/'+TestDesc+'/log/') == False:
        os.mkdir(path_project+'tests/'+TestDesc+'/log/')

    if os.path.exists(path_project+'tests/'+TestDesc+'/Results_'+str(Results)+'/') == False:
        os.mkdir(path_project+'tests/'+TestDesc+'/Results_'+str(Results)+'/')        
        
def header_ascii(ascii):
    name = list()
    data = list()
    c = 0
    with open(ascii) as input_file:
        for line in input_file:
            c = c + 1
            if c < 7:
                a, b = (item.strip() for item in line.split(' ', 1))
                name.append(a)
                data.append(b)  
    ncols         =   int(data[0])     # number of rows of the grid in VIC
    nrows         =   int(data[1])     # number of cols of the grid in VIC
    xllcorner     =   np.float(data[2])
    yllcorner     =   np.float(data[3])
    cellsize      =   np.float(data[4])     # VIC cellsize desired
    NODATA_value  =   np.float(data[5])
    
    return ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value

def getWKT_PRJ (epsg_code):
    
    # access projection information
    wkt = urllib.request.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code))
    # remove spaces between charachters
    remove_spaces = wkt.read().decode("utf-8").replace(" ","")
    # place all the text on one line
    output = remove_spaces.replace("\n", "")
    return output

def rasterToASCII(filename, data, xmin, ymin, cellsize, nodata, header=True,type_data='float'):
    with open(filename, "wt") as fid:
        (nrows, ncols) = data.shape
        if header:
            fid.write("ncols {0:d}\nnrows {1:d}\n".format(ncols, nrows))
            fid.write("xllcorner {0:.8f}\nyllcorner {1:.7f}\n".format(xmin, ymin))
            fid.write("cellsize {0:.12f}\n".format(cellsize))
            fid.write("NODATA_value {:d}\n".format(int(nodata)))
        for i in range(nrows-1, -1, -1):
            for j in range(ncols):
                if data[i,j]!=nodata:
                    if type_data =='float':
                        fid.write("{0:.4f} ".format(data[i,j]))
                    else:
                        fid.write("{:d} ".format(int(data[i,j])))
                        
                else:
                    fid.write("{:d} ".format(int(nodata)))
            fid.write("\n")



def zonal_statistic(raster_mask,raster,statis='sum',only_values=[]):
    [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(raster_mask)
    xx = np.linspace(xllcorner+cellsize/2, xllcorner+cellsize/2 + (ncols-1)*cellsize, ncols)
    yy =np.linspace(yllcorner+cellsize/2, yllcorner+cellsize/2 + (nrows-1)*cellsize, nrows)
    [XX,YY] = np.meshgrid(xx,yy)

    coord = np.concatenate((XX.flatten().reshape(-1,1),YY.flatten().reshape(-1,1)),axis=1)
    raster_mask_values = np.flipud(np.loadtxt(raster_mask , skiprows = 6))
    raster_mask_values[raster_mask_values==NODATA_value] = np.nan

    Statistic = pd.DataFrame(index=['sum','mean','majority'])
    list_raster = raster_mask_values.flatten()
    src = rasterio.open(raster)
    
    
    if len(only_values) ==0:
        ids = np.unique(list_raster)
        ids = ids[~np.isnan(ids)]
    else:
        ids = only_values

    for i, ii in enumerate(ids):
        values = list()
        station_name = np.where(list_raster == ii)[0]
        for j in station_name:
            coord_s = coord[j]
            for v in src.sample([coord_s]):
                values.append(v[0])

        values = np.array(values)
        values = values[values!=NODATA_value]
        
        if len(values)==0:
            continue
        else:
            st = list()
            st.append(np.nansum(values))
            st.append(np.nanmean(values))
            st.append(stats.mode(np.array(values),nan_policy='omit')[0][0])
            Statistic[int(ii)] = st

    Statistic = Statistic.T
    return Statistic.loc[:,statis]

def close_holes(poly):
    from shapely.geometry import MultiPolygon, Polygon
    """
    Close polygon holes by limitation to the exterior ring.
    Args:
        poly: Input shapely Polygon
    Example:
        df.geometry.apply(lambda p: close_holes(p))
    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly
    
    
def convert_line_to_shp(shape_line, output):
    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(output)
    layer = ds.CreateLayer('', None, ogr.wkbLineString)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    ## If there are multiple geometries, put the "for" loop here

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(shape_line.wkb)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


def convert_polygon_to_shp(shape_polygon, output):
    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(output)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    ## If there are multiple geometries, put the "for" loop here

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(shape_polygon.wkb)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


def convert_index_datetime(dataframe):
    index_list=list()
    for i in range(len(dataframe)):
        A = str(dataframe.iloc[i,0])
        B = str(dataframe.iloc[i,1])
        C = str(dataframe.iloc[i,2])
        D = str(dataframe.iloc[i,3])
        index_list.append(datetime.strptime(A+'/'+B+'/'+C+'/'+D, '%Y/%M/%d/%H'))
    return index_list
    


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def delete_files(path):
    import os
    import glob
    files = glob.glob(path)
    for f in files:
        os.remove(f)


from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

def explode(indata):
    indf = gpd.GeoDataFrame.from_file(indata)
    outdf = gpd.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf


def zonal_statistic(raster_mask,raster,statis='sum',only_values=[]):
    [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(raster_mask)
    xx = np.linspace(xllcorner+cellsize/2, xllcorner+cellsize/2 + (ncols-1)*cellsize, ncols)
    yy =np.linspace(yllcorner+cellsize/2, yllcorner+cellsize/2 + (nrows-1)*cellsize, nrows)
    [XX,YY] = np.meshgrid(xx,yy)

    coord = np.concatenate((XX.flatten().reshape(-1,1),YY.flatten().reshape(-1,1)),axis=1)
    raster_mask_values = np.flipud(np.loadtxt(raster_mask , skiprows = 6))
    raster_mask_values[raster_mask_values==NODATA_value] = np.nan

    Statistic = pd.DataFrame(index=['sum','mean','majority'])
    list_raster = raster_mask_values.flatten()
    src = rasterio.open(raster)
    
    
    if len(only_values) ==0:
        ids = np.unique(list_raster)
        ids = ids[~np.isnan(ids)]
    else:
        ids = only_values

    for i, ii in enumerate(ids):
        values = list()
        station_name = np.where(list_raster == ii)[0]
        for j in station_name:
            coord_s = coord[j]
            for v in src.sample([coord_s]):
                values.append(v[0])

        values = np.array(values)
        values = values[values!=NODATA_value]
        
        if len(values)==0:
            continue
        else:
            st = list()
            st.append(np.nansum(values))
            st.append(np.nanmean(values))
            st.append(stats.mode(np.array(values),nan_policy='omit')[0][0])
            Statistic[int(ii)] = st

    Statistic = Statistic.T
    return Statistic.loc[:,statis]


def matDatenum2PYDatetime(datenumVec,unitTime = 'D'):
    datetimeVec = pd.to_datetime(datenumVec-719529, unit=unitTime,errors='coerce')
    datetimeNum = datenumVec-719529
    return datetimeVec,datetimeNum

def rasterize (shapefile,ID,raster_extent,output):
    from osgeo import gdal, ogr

    vector_layer = shapefile

    # open the raster layer and get its relevant properties
    tileHdl = gdal.Open(raster_extent, gdal.GA_ReadOnly)

    tileGeoTransformationParams = tileHdl.GetGeoTransform()
    projection = tileHdl.GetProjection()

    rasterDriver = gdal.GetDriverByName('GTiff')

    buildingPolys_ds = ogr.Open(vector_layer)
    buildingPolys = buildingPolys_ds.GetLayer()
    # Create the destination data source
    tempSource = rasterDriver.Create(output, 
                                     tileHdl.RasterXSize, 
                                     tileHdl.RasterYSize,
                                     1, #missed parameter (band)
                                     gdal.GDT_Float32)

    tempSource.SetGeoTransform(tileGeoTransformationParams)
    tempSource.SetProjection(projection)
    tempTile = tempSource.GetRasterBand(1)
    tempTile.Fill(-9999)
    tempTile.SetNoDataValue(-9999)

    gdal.RasterizeLayer(tempSource, [1], buildingPolys, options=["ATTRIBUTE="+ID])
    tempSource = None