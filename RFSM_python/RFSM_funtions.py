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
gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")

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
        
def extract_coastline(shp_coastline,path_DTM,path_DTM_new, cellsize):
    drv = gdal.Open(DTM)
    OutTile = gdal.Warp(path_DTM_new[:-3]+'tif',
                        [drv], 
                        format = 'GTiff',
                        xRes = cellsize, 
                        yRes = cellsize,
                        cutlineDSName=shp_coastline, 
                        cropToCutline=True, 
                        dstNodata = -9999)

    OutTile = None 
    ds = gdal.Open(path_DTM_new[:-3]+'tif')
    ds_out = gdal.Translate(path_DTM_new,ds,format='AAIGrid')
    ds_out = None

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

def line_cost_raster(raster,path_output,n_types=True):
    raster=np.flipud(np.loadtxt(raster,skiprows=6))
    [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(DTM)
    contour = raster.copy()*0
    if n_types==True:
        t=1
        for c in range(ncols):
            for r in range(nrows):
                try:
                    if raster[r,c]==-9999:
                        continue
                    else:
                        if ((raster[r,c-1])==-9999 or (raster[r,c+1])==-9999 or (raster[r+1,c]==-9999)):
                            contour[r,c]=t
                            t=t+1
                        elif raster[r,c-1]==-9999:
                            contour[r,c]=0
                except:
                    continue
        rasterToASCII(path_output+'coast.asc', contour, xllcorner, yllcorner+0.5, cellsize, NODATA_value, header=True,type_data='integer') 
    else:
        for c in range(ncols):
            for r in range(nrows):
                try:
                    if raster[r,c]==-9999:
                        continue
                    else:
                        if ((raster[r,c-1])==-9999 or (raster[r,c+1])==-9999 or (raster[r+1,c]==-9999)):
                            contour[r,c]=1
                        elif raster[r,c-1]==-9999:
                            contour[r,c]=0
                except:
                    continue
        rasterToASCII(path_output+'coast_Line.asc', contour, xllcorner, yllcorner+0.5, cellsize, NODATA_value, header=True,type_data='integer')                

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

def tusrBCFlowLevel (poits_x,point_y,path_project,TestDesc,Table_impact_zone,izcoast,n_periods,BCSetID,BCTypeID_COAST,
                    raw_q=True, point_X_river = None ,point_Y_river = None ,hidrograma=None):
    import geopandas as gpd
    Table_impact_zone_edit =Table_impact_zone.copy()
    Table_impact_zone_edit.loc[:,'BCTypeID'] = 0
    
    
    # % 1 overtopping; 2 level;
    
    points_dinamics = pd.read_csv(path_inputs_dinamicas+'Points_Marine_Dynamics_UTM.dat',delim_whitespace=True,
                                  header=None,
                                  names=['x coordinate or longitud','y coordinate or latitud','depth of closure','absolute id','relative id'])
    
    timeDOW=pd.read_csv(path_inputs_dinamicas+'WAVES/TimeHs.dat',delim_whitespace=True,header=None,dtype=int)
    timemm=pd.read_csv(path_inputs_dinamicas+'SS/TimeMM.dat',delim_whitespace=True, header=None,dtype=int)
    timeat=pd.read_csv(path_inputs_dinamicas+'AT/TimeAT.dat',delim_whitespace=True,header=None,dtype=int)
    
    timeDOW_index = convert_index_datetime(timeDOW)
    timemm_index  = convert_index_datetime(timemm)
    timeat_index  = convert_index_datetime(timeat)
    
    Table_impact_zone_edit.loc[izcoast.index,'BCTypeID'] = BCTypeID_COAST
    
    if raw_q==True:
        
        dist = np.sqrt((point_X_river-Table_impact_zone_edit.iloc[:,2])**2+(point_Y_river-Table_impact_zone_edit.iloc[:,3])**2)
        point_select=dist.idxmin()
        
        Table_impact_zone_edit.loc[point_select,'BCTypeID'] = 10
        
    Results_TWL = pd.DataFrame(index=np.arange(0,sum(Table_impact_zone_edit.BCTypeID.values>0)*n_periods),columns=['BCSetID', 'BCTypeID', 'IZID', 'Time', 'BCValue'])
    
    Table_impact_zone_edit_2=Table_impact_zone_edit[Table_impact_zone_edit.BCTypeID!=0].copy()
    it=0
    for iDZ in tqdm.tqdm(range(len(Table_impact_zone_edit_2))):
        if (Table_impact_zone_edit_2['BCTypeID'].iloc[iDZ]==2) or (Table_impact_zone_edit_2['BCTypeID'].iloc[iDZ]==1):

            x_pointIZID = Table_impact_zone_edit_2[' MidX'].values[iDZ]
            y_pointIZID = Table_impact_zone_edit_2[' MidY'].values[iDZ]
            IZID = Table_impact_zone_edit_2.index[iDZ]

            dist = np.sqrt((x_pointIZID-points_dinamics.iloc[:,0])**2+(y_pointIZID-points_dinamics.iloc[:,1])**2)
            point_select=dist.idxmin()

            TWL_def = list()
            W=pd.read_csv(path_inputs_dinamicas  +'WAVES/H_'+str(points_dinamics['relative id'][point_select])+'.dat',delim_whitespace=True,header=None)
            W.index = timeDOW_index
            MM=pd.read_csv(path_inputs_dinamicas + 'SS/MM_'+str(points_dinamics['relative id'][point_select])+'.dat',delim_whitespace=True,header=None)
            MM.index = timemm_index
            MA=pd.read_csv(path_inputs_dinamicas + 'AT/MA_'+str(points_dinamics['relative id'][point_select])+'.dat',delim_whitespace=True,header=None)
            MA.index = timeat_index

            Result = pd.concat([W,MM,MA],axis=1).dropna()
            Hs=Result.iloc[:,0]; T=Result.iloc[:,1]
            setup=0.05*T*(Hs)**(1/2)
            MM=Result.iloc[:,3]
            MA=Result.iloc[:,4]
            TWL=MM+MA+setup
            if BCTypeID_COAST ==1:
                TWL = TWL*izcoast.nCells[IZID]*cellsize
            else:
                TWL = TWL-izcoast.minH[IZID]
                TWL[TWL<0]=0
            MaximumTWL=np.where(TWL==np.max(TWL))[0]
            TWL_def = list()
            for n in range(int(-(n_periods-1)/2),int((n_periods-1)/2+1)):
                TWL_def.append(TWL.iloc[MaximumTWL+n].values[0])

            Results_TWL.iloc[it:it+n_periods,0] = BCSetID
            Results_TWL.iloc[it:it+n_periods,1] = BCTypeID_COAST
            Results_TWL.iloc[it:it+n_periods,2] = IZID
            Results_TWL.iloc[it:it+n_periods,3] = np.linspace(0,3600*n_periods,n_periods)
            Results_TWL.iloc[it:it+n_periods,4] = TWL_def

            it=it+n_periods

            del W, MM, MA, Hs, setup, TWL

        elif Table_impact_zone_edit_2['BCTypeID'].iloc[iDZ]==10:
            print('Río')
            Results_TWL.iloc[it:it+n_periods,0] = BCSetID
            Results_TWL.iloc[it:it+n_periods,1] = 1
            Results_TWL.iloc[it:it+n_periods,2] = IZID
            Results_TWL.iloc[it:it+n_periods,3] = np.linspace(0,3600*n_periods,n_periods)
            Results_TWL.iloc[it:it+n_periods,4] = hidrograma

            it=it+n_periods
            
    Results_TWL.to_csv(path_project+'tests/'+TestDesc+'/Input_User/tusrBCFlowLevel.csv',index=False)

def tusrBCRainfall (path_project,TestDesc,Table_impact_zone,n_periods,array_rain_intensity):
    Table_impact_zone_edit=Table_impact_zone.copy()
    RainfallZones = np.unique(Table_impact_zone_edit['RainfallZoneID'].values)
    tusrBCRainfallZones  = pd.DataFrame(index=np.arange(0,RainfallZones*n_periods),columns=['RainfallZoneID', 'Time', 'RainfallIntensity'])
    tusrBCRainfallRunoff = pd.DataFrame(index=np.arange(0,len(Table_impact_zone_edit)),columns=['IZID', 'RainfallZoneID', 'RunoffCoef'])
    it = 0
    for i,ii in enumerate(RainfallZones):
        tusrBCRainfallZones.iloc[it:it+n_periods,0] = ii
        tusrBCRainfallZones.iloc[it:it+n_periods,1] = np.linspace(0,3600*n_periods,n_periods)
        tusrBCRainfallZones.iloc[it:it+n_periods,2] = array_rain_intensity[i]
        it=it+n_periods
    for  j,jj in enumerate(Table_impact_zone.index):
        tusrBCRainfallRunoff.iloc[j,0] = jj
        tusrBCRainfallRunoff.iloc[j,1] = Table_impact_zone_edit.loc[jj,'RainfallZoneID']
        tusrBCRainfallRunoff.iloc[j,2] = Table_impact_zone_edit.loc[jj,'IZRunOffCoef']
        
    tusrBCRainfallZones.to_csv(path_project+'tests/'+TestDesc+'/Input_User/tusrBCRainfallZones.csv',index=False)
    tusrBCRainfallRunoff.to_csv(path_project+'tests/'+TestDesc+'/Input_User/tusrBCRainfallRunoff.csv',index=False)


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


def make_dirs(path_install_RFSM,path_project,TestDesc,Results):
    if len(os.listdir(path_project))==0:
        copytree(path_install_RFSM, path_project, symlinks=False, ignore=None) ## Cambiar el path donde tenemos la instalación de RFSM
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
    


def preprocess_RFSM(DTM,CoastLine,path_project,epsg_n):
    path_output = path_project+'ascii/'
    DTM_CLIP_LC = path_project+'ascii/DTM_LC.asc'
    [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(DTM)
    xx = np.linspace(xllcorner, xllcorner + (ncols-1)*cellsize, ncols)
    yy = np.linspace(yllcorner, yllcorner + (nrows-1)*cellsize, nrows)
    [XX,YY] = np.meshgrid(xx,yy)
    # Cortamos con la línea de Costa
    extract_coastline(CoastLine,DTM,DTM_CLIP_LC,cellsize)
    ### Cargamos el DTM
    DTM=np.loadtxt(DTM_CLIP_LC,skiprows=6)
    ### Filtramos valores negativos
    DTM[(DTM<=0)&(DTM!=NODATA_value)]=0
    ### Nos quedamos con el area para inundar por encima de una cota
    DTM_10=DTM.copy()
    DTM_10[DTM_10>COTA_ESTUDIO]=-9999
    ### Raster de Áreas de inundación
    FLOOD = DTM_10.copy()
    FLOOD[(FLOOD<=COTA_ESTUDIO)&(FLOOD>=0)]=1
    rasterToASCII(path_output+'topography.asc', np.flipud(DTM_10), xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='float')
    # create the .prj file
    prj = open(path_output+'topography.prj', "w")
    # call the function and supply the epsg code
    epsg = getWKT_PRJ(epsg_n)
    prj.write(epsg)
    prj.close()
    rasterToASCII(path_output+'floodareas.asc', np.flipud(FLOOD), xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='integer')
    # create the .prj file
    prj = open(path_output+'floodareas.prj', "w")
    # call the function and supply the epsg code
    epsg = getWKT_PRJ(epsg_n)
    prj.write(epsg)
    prj.close()
    
    prj = open(path_output+'topography.prj', "w")
    # call the function and supply the epsg code
    epsg = getWKT_PRJ(epsg_n)
    prj.write(epsg)
    prj.close()

def impact_zones_process(path_project,DTM_LC,cellsize,new_coast_Line=True):
    start = time.time()
    izd1_r = path_project+'ascii/check/izid1.asc'
    izd2_r = path_project+'ascii/check/izid2.asc'
    #[ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(DTM_LC)
    if new_coast_Line == True:
        dtm_lc_asc =  gdal.Open(DTM_LC)
        band = dtm_lc_asc.GetRasterBand(1)
        dtm = BandReadAsArray(band)
        dtm[dtm<10] = -9999
        dtm[dtm>=10] = 1

        driver = gdal.GetDriverByName("GTiff")
        dsOut = driver.Create(path_project+'ascii/DTM_OC.tif'
                              ,dtm_lc_asc.RasterXSize, dtm_lc_asc.RasterYSize, 1, band.DataType)
        CopyDatasetInfo(dtm_lc_asc,dsOut)
        bandOut=dsOut.GetRasterBand(1)
        bandOut.Fill(-9999)
        bandOut.SetNoDataValue(-9999)
        BandWriteArray(bandOut, dtm)

        #Close the datasets
        band1 = None
        band2 = None
        mdt_tif = None
        level_tif = None
        bandOut = None
        dsOut = None
        
        #dtm = np.loadtxt(DTM_LC,skiprows=6)
        
        #rasterToASCII(path_project+'ascii/DTM_OC.asc', np.flipud(dtm), xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='float')

        os.system('gdal_polygonize.py '+path_project+'ascii/DTM_OC.tif'+' '+path_project+'shp/DTM_OC.shp -b 1 -f "ESRI Shapefile" DTM_OC IZID')
    
    
    os.system('gdal_polygonize.py '+izd1_r+' '+path_project+'shp/izid1.shp -b 1 -f "ESRI Shapefile" izid1 IZID')
    os.system('gdal_polygonize.py '+izd2_r+' '+path_project+'shp/izid2.shp -b 1 -f "ESRI Shapefile" izid2 IZID')
    
    izd1 = path_project+'shp/izid1.shp'
    izd2 = path_project+'shp/izid2.shp'
    
    shape_1 = gpd.read_file(path_project+'shp/DTM_OC.shp')
    shape_1['CID'] = 1
    shape_1 = shape_1.dissolve(by='CID')

    shape_2 = gpd.read_file(path_project+'shp/izid2.shp')
    shape_2['CID'] = 1
    shape_2 = shape_2.dissolve(by='CID')

    result_2 = gpd.overlay(shape_1, shape_2, how='union')
    result_2['CID']=1
    result_2 = result_2.dissolve(by='CID')
    result_2.to_file(path_project+'shp/Final_cut_complet.shp')

    FC = gpd.read_file(path_project+'shp/Final_cut_complet.shp')
    FC = explode(path_project+'shp/Final_cut_complet.shp')
    FC = FC.geometry.apply(lambda p: close_holes(p))
    FC.to_file(path_project+'shp/Final_cut_complet_F.shp')
    
    
    extract_coastline(path_project+'shp/Final_cut_complet_F.shp',DTM_LC,path_project+'ascii/DTM_LC_2.tif', cellsize)
        
    line_cost_raster(path_project+'ascii/DTM_LC_2.tif',path_project+'ascii/')
    #line_cost_raster(path_project+'ascii/DTM_LC_2.tif',path_project+'ascii/',n_types=False)
    os.system('gdal_polygonize.py '+path_project+'ascii/coast.tif'+' '+path_project+'shp/coast_IH.shp -b 1 -f "ESRI Shapefile" coast_IH CID')
    
#     shapefile = gpd.read_file(path_project+'shp/coast_IH.shp' )
#     shapefile=shapefile[shapefile.CID==1]
#     pol = shapefile['geometry']
    
    # from shapely.ops import cascaded_union
    # u = cascaded_union(pol)
    # convert_polygon_to_shp(u,path_project+'shp/coast_IH.shp')
    
    
    ds = ogr.Open(path_project+'shp/coast_IH.shp')
    ly = ds.ExecuteSQL('SELECT ST_Centroid(geometry), * FROM coast_IH', dialect='sqlite')
    drv = ogr.GetDriverByName('Esri shapefile')
    ds2 = drv.CreateDataSource(path_project+'shp/coast_IH_s.shp')
    ds2.CopyLayer(ly, '')
    ly = ds = ds2 = None  # save, close
    
    from shapely.geometry import shape
    fc_A = gpd.read_file(path_project+'shp/coast_IH_s.shp')
    fc_B = gpd.read_file(path_project+'shp/izid2.shp')
    # List to collect pairs of intersecting features
    fc_intersect = np.unique(gpd.sjoin(fc_B,fc_A , op='intersects')['IZID'])
    shape_1=gpd.sjoin(fc_B,fc_A , op='intersects')
    shape_1=shape_1.drop_duplicates(subset=['geometry'])
    shape_1.to_file(path_project+'shp/izcoast.shp')
    
    Table_impact_zone=pd.read_csv(path_project+'ascii/tblImpactZone.csv',index_col=0)
    
    izcoast=Table_impact_zone.loc[fc_intersect,:].loc[:,[' NbCells',' MinLevel',' MidX',' MidY']].copy()
    izcoast.columns=['nCells','minH','MidX','MidY']
    #ZE = zonal_statistic(izd2_r,path_project+'ascii/coast_Line.tif',statis='sum',only_values=izcoast.index)
    ZE = zonal_stats(path_project+'shp/izcoast.shp',
                     path_project+'ascii/coast_Line.tif', 
                     stats=['sum'],
                     geojson_out=True,included_attributes=['IZID'])
    ZE = json_normalize(ZE)
    ZE.index = ZE.loc[:,'properties.IZID']
    ZE = ZE.drop_duplicates(subset='properties.IZID')
    for i,ii in enumerate(izcoast.index):
        izcoast.loc[ii,'nCells'] =  ZE.loc[ii,'properties.sum']
    end = time.time()
    print(end - start)
    izcoast.to_csv(path_project+'ascii/izcoast.csv')
    return Table_impact_zone, izcoast

def izcoast_modify(izcoast_shp,Table_impact_zone,output):
    iz_shp=gpd.read_file(izcoast_shp)
    izcoast=Table_impact_zone.loc[iz_shp.loc[:,'IZID'].values,:].loc[:,[' NbCells',' MinLevel',' MidX',' MidY']].copy()
    izcoast.columns=['nCells','minH','MidX','MidY']
    #ZE = zonal_statistic(izd2_r,path_project+'ascii/coast_Line.tif',statis='sum',only_values=izcoast.index)
    ZE = zonal_stats(izcoast_shp,
                     path_project+'ascii/coast_Line.tif', 
                     stats=['sum'],
                     geojson_out=True,included_attributes=['IZID'])
    ZE = json_normalize(ZE)
    ZE.index = ZE.loc[:,'properties.IZID']
    ZE = ZE.drop_duplicates(subset='properties.IZID')
    for i,ii in enumerate(izcoast.index):
        izcoast.loc[ii,'nCells'] =  ZE.loc[ii,'properties.sum']
    izcoast.to_csv(output)
    
def create_file_manning(path_project,raster_rugos,TestDesc,Table_impact_zone):
    tusrlZManning = pd.DataFrame(index=np.arange(0,len(Table_impact_zone)),columns=['BCSetID','IZID','CManning'])
    ZE = zonal_stats(path_project+'shp/izid2.shp',raster_rugos, stats=['mean'],geojson_out=True,included_attributes=['IZID'])
    ZE = json_normalize(ZE)
    ZE.index = ZE.loc[:,'properties.IZID']
    for i,ii in enumerate(tqdm.tqdm(Table_impact_zone.index)):
        tusrlZManning.iloc[i,0]=BCSetID
        tusrlZManning.iloc[i,1]=ii
        try:
            tusrlZManning.iloc[i,2]=ZE.loc[ii,'properties.mean']
        except:
            print('Fallo en el IZID:' + str(ii))
            break
    tusrlZManning.to_csv(path_project+'tests/'+TestDesc+'/Input_User/tusrIZManning.csv',index=False)


def create_xml(TestID,TestDesc,BCSetID,StartTime,EndTime,TimeStep,SaveTimeStep,MaxTimeStep,MinTimeStep,AlphaParameter,ManningGlobalValue,path_project,Results):
    lines = ['<settings>\n', 
    '<TestID>'+str(TestID)+'</TestID>\n',
    '<TestDesc>'+TestDesc+'</TestDesc>\n',
    '<BCSetID>'+str(BCSetID)+'</BCSetID>\n',
    '<StartTime>'+StartTime+'</StartTime>\n',
    '<EndTime>'+EndTime+'</EndTime>\n',
    '<TimeStep>'+str(TimeStep)+'</TimeStep>\n',
    '<SaveTimeStep>'+SaveTimeStep+'</SaveTimeStep>\n',
    '<MaxTimeStep>'+str(MaxTimeStep)+'</MaxTimeStep>\n',
    '<MinTimeStep>'+str(MinTimeStep)+'</MinTimeStep>\n',
    '<AlphaParameter>'+str(AlphaParameter)+'</AlphaParameter>\n',
    '<ManningGlobalValue>'+str(ManningGlobalValue)+'</ManningGlobalValue>\n',
    '<ModelType>csv</ModelType>\n',
    '<DbName>'+path_project+ 'tests/'+TestDesc+'</DbName>\n',
    '<FA_ID>1</FA_ID>\n',
    '<TimeStepMethod>a</TimeStepMethod>\n',
    '<LogVerbose>0</LogVerbose>\n',
    '<Results>'+str(Results)+'</Results>\n',
    '</settings>\n']

    if len(os.listdir(path_project+'tests/'+TestDesc+'/Input_AccData/'))>0:
        directory= glob.glob(path_project+'ascii/*csv')
        shutil.rmtree(path_project+'tests/'+TestDesc+'/Input_AccData/')
        os.mkdir(path_project+'tests/'+TestDesc+'/Input_AccData/')
        for file in directory:
            if os.path.isfile(file):
                shutil.copy2(file, path_project+'tests/'+TestDesc+'/Input_AccData/')   
    else:
        directory= glob.glob(path_project+'ascii/*csv')
        for file in directory:
            if os.path.isfile(file):
                shutil.copy2(file, path_project+'tests/'+TestDesc+'/Input_AccData/')

    with open(path_project+'tests/'+TestDesc+'/Input_xml/'+'input_'+str(TestID)+'.xml', "w") as fh:
        for line in (lines):
                fh.write(line)

def execute_RFSM(path_project,TestDesc,TestID):
    os.chdir(path_project+'tests/'+TestDesc+'/Input_xml/')
    try:
        test = os.uname()
        if test[0] == "Linux":
            os.system(path_project+'bin/RFSM/Windows7_x86/RFSM_Hydrodynamic.exe '+'input_'+str(TestID)+'.xml')
    except AttributeError:
        print("Assuming windows!")
        os.system(path_project+'bin/RFSM/Windows7_x86/RFSM_Hydrodynamic.exe '+'input_'+str(TestID)+'.xml')


def export_result_RFSM(path_project,TestDesc,Results):
    path_results = path_project + 'tests/'+TestDesc+'/Results_'+str(Results)+'/'
    tusrResultsIZMax = pd.read_csv(path_results+'tusrResultsIZMax.csv',index_col=1)
    shape_IZID2 = gpd.read_file(path_project+'shp/izid2.shp')
    shape_IZID2['Level'] = np.nan
    
    for IDZ in tqdm.tqdm(tusrResultsIZMax.index):
        posi_IDZ = np.where(shape_IZID2.IZID==IDZ)[0][0]
        shape_IZID2.iloc[posi_IDZ,-1] = tusrResultsIZMax.loc[IDZ,'MaxLevel']
    shape_IZID2.to_file(path_project+'shp/izid2_Level.shp')
    
    rasterize (path_project+'shp/izid2_Level.shp',
               'Level',
               path_project+'ascii/topography.asc',
               path_project + 'tests/'+TestDesc+'/export/Result_Level_IZID'+str(Results)+'.tif' )
    mdt_tif = gdal.Open(path_project+'ascii/topography.asc', GA_ReadOnly )
    band1 = mdt_tif.GetRasterBand(1)
    mdt = BandReadAsArray(band1)
    
    level_tif =  gdal.Open(path_project + 'tests/'+TestDesc+'/export/Result_Level_IZID'+str(Results)+'.tif')
    band2 = level_tif.GetRasterBand(1)
    level = BandReadAsArray(band2)
    
    
    MaxLevel_new = level - mdt
    pos_level_neg = np.where(MaxLevel_new<0.001)
    MaxLevel_new[pos_level_neg[0],pos_level_neg[1]] =-9999
    MaxLevel_new[np.isnan(MaxLevel_new)] = -9999
    
    
    #Write the out file
    driver = gdal.GetDriverByName("GTiff")
    dsOut = driver.Create(path_project + 'tests/'+TestDesc+'/export/MaxLevel'+str(Results)+'.tif'
                          , mdt_tif.RasterXSize, mdt_tif.RasterYSize, 1, band1.DataType)
    CopyDatasetInfo(mdt_tif,dsOut)
    bandOut=dsOut.GetRasterBand(1)
    bandOut.Fill(-9999)
    bandOut.SetNoDataValue(-9999)
    BandWriteArray(bandOut, MaxLevel_new)

    #Close the datasets
    mdt_tif = None
    band1 = None
    band2 = None
    mdt_tif = None
    level_tif = None
    bandOut = None
    dsOut = None