# _Autor:_    __Salavador Navas__
# _Revisión:_ __05/10/2020__

import numpy as np
import pandas as pd
from osgeo import gdal, ogr
import os
import gdalconst
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
import tqdm
import shapely
from shapely.geometry import MultiPolygon, Point
import geopandas as gpd
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
    gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
    drv = gdal.Open(path_DTM)
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


# function to generate .prj file information using spatialreference.org
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

def line_cost_raster(raster_file,path_output,n_types=True):
    raster=np.flipud(np.loadtxt(raster_file,skiprows=6))
    [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(raster_file)
    contour = raster.copy()*0+-9999
    if n_types==True:
        t=1
        for c in range(ncols):
            for r in range(nrows):
                try:
                    if raster[r,c]==-9999:
                        continue
                    else:
                        if ((raster[r,c-1])==-9999 or (raster[r,c+1])==-9999 or (raster[r+1,c]==-9999) or (raster[r-1,c]==-9999)):
                            contour[r,c]=t
                            t=t+1
#                         elif raster[r,c-1]==-9999:
#                             contour[r,c]=0
#                             t=t+1
                except:
                    continue
        rasterToASCII(path_output+'coast.asc', contour, xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='integer') 
    else:
        for c in range(ncols):
            for r in range(nrows):
                try:
                    if raster[r,c]==-9999:
                        continue
                    else:
                         if ((raster[r,c-1])==-9999 or (raster[r,c+1])==-9999 or (raster[r+1,c]==-9999) or (raster[r-1,c]==-9999)):
                            contour[r,c]=1
#                         elif raster[r,c-1]==-9999:
#                             contour[r,c]=0
                except:
                    continue
        rasterToASCII(path_output+'coast_Line.asc', contour, xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='integer')                

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
    
    
def zonal_statistic(raster_mask,raster,statis='sum'):
    raster_mask = path_project+'ascii/check/izid2.asc'
    raster = path_project+'ascii/coast_Line.asc'
    statis='sum'
    [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(raster_mask)
    xx = np.linspace(xllcorner+cellsize/2, xllcorner+cellsize/2 + (ncols-1)*cellsize, ncols)
    yy =np.linspace(yllcorner+cellsize/2, yllcorner+cellsize/2 + (nrows-1)*cellsize, nrows)
    [XX,YY] = np.meshgrid(xx,yy)

    coord = np.concatenate((XX.flatten().reshape(-1,1),YY.flatten().reshape(-1,1)),axis=1)
    raster_mask_values = np.flipud(np.loadtxt(raster_mask , skiprows = 6))
    raster_mask_values[raster_mask_values==NODATA_value]=np.nan

    Statistic = pd.DataFrame(index=['sum','mean','majority'])
    list_raster = raster_mask_values.flatten()
    src = rasterio.open(raster)
    ids = np.unique(list_raster)
    ids = ids[~np.isnan(ids)]
    for i, ii in enumerate(ids):
        values = list()
        station_name = np.where(list_raster == ii)[0]
        for j in station_name:
            coord_s = coord[j]
            for v in src.sample([coord_s]):
                values.append(v[0])

        st = list()
        st.append(np.nansum(values))
        st.append(np.nanmean(values))
        st.append(stats.mode(np.array(values),nan_policy='omit')[0][0])
        Statistic[int(ii)] = st

    Statistic = Statistic.T
    return Statistic.loc[:,statis]

class RFSM(object):
    def __init__(self,path_project,DTM,CoastLine,epsg_n):
        self.DTM=DTM
        self.path_project=path_project
        self.epsg_n = epsg_n
        self.CoastLine = CoastLine
        return

    def make_dirs(self,TestDesc,Results):

        dir_file=__file__
        path_install_RFSM=dir_file[:-28]+'IHRFSM_v1.0/'
        print(dir_file)
        
        if len(os.listdir(self.path_project))==0:
            copytree(path_install_RFSM, self.path_project, symlinks=False, ignore=None) ## Cambiar el path donde tenemos la instalación de RFSM
        if os.path.exists(self.path_project+'tests/'+TestDesc) == False:
            os.mkdir(self.path_project+'tests/'+TestDesc)

        if os.path.exists(self.path_project+'tests/'+TestDesc+'/Input_xml/') == False:
            os.mkdir(self.path_project+'tests/'+TestDesc+'/Input_xml/')

        if os.path.exists(self.path_project+'tests/'+TestDesc+'/export/') == False:
            os.mkdir(self.path_project+'tests/'+TestDesc+'/export/')    

        if os.path.exists(self.path_project+'tests/'+TestDesc+'/Input_AccData/') == False:
            os.mkdir(self.path_project+'tests/'+TestDesc+'/Input_AccData/',mode=0o777) 

        if os.path.exists(self.path_project+'tests/'+TestDesc+'/Input_User/') == False:
            os.mkdir(self.path_project+'tests/'+TestDesc+'/Input_User/')
            
        tblkTestBCType=pd.DataFrame(index=np.arange(0,5),columns=['BCTypeID', 'BCType'])
        tblkTestBCType.iloc[:,0]=np.arange(1,6)
        tblkTestBCType.iloc[:,1]=['Discharge','Level','LevelOut','Levelln','LevelFlowRating']
        tblkTestBCType.to_csv(self.path_project+'tests/'+TestDesc+'/Input_User/tblkTestBCType.csv',index=False)

        if os.path.exists(self.path_project+'tests/'+TestDesc+'/log/') == False:
            os.mkdir(self.path_project+'tests/'+TestDesc+'/log/')

        if os.path.exists(self.path_project+'tests/'+TestDesc+'/Results_'+str(Results)+'/') == False:
            os.mkdir(self.path_project+'tests/'+TestDesc+'/Results_'+str(Results)+'/')
        
    def preprocess_RFSM(self,COTA_ESTUDIO):
        path_output = self.path_project+'ascii/'
        DTM_CLIP_LC = self.path_project+'ascii/DTM_LC.asc'
        # create the .prj file
        prj = open(path_output+'DTM.prj', "w")
        # call the function and supply the epsg code
        epsg = getWKT_PRJ(self.epsg_n)
        prj.write(epsg)
        prj.close()
        [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(self.DTM)
        xx = np.linspace(xllcorner, xllcorner + (ncols-1)*cellsize, ncols)
        yy = np.linspace(yllcorner, yllcorner + (nrows-1)*cellsize, nrows)
        [XX,YY] = np.meshgrid(xx,yy)
        # Cortamos con la línea de Costa
        extract_coastline(self.CoastLine,self.DTM, DTM_CLIP_LC, cellsize)
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
        epsg = getWKT_PRJ(self.epsg_n)
        prj.write(epsg)
        prj.close()
        rasterToASCII(path_output+'floodareas.asc', np.flipud(FLOOD), xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='integer')
        # create the .prj file
        prj = open(path_output+'floodareas.prj', "w")
        # call the function and supply the epsg code
        epsg = getWKT_PRJ(self.epsg_n)
        prj.write(epsg)
        prj.close()
    def impact_zones_process(self):
        start = time.time()
        izd1_r = self.path_project+'ascii/check/izid1.asc'
        izd2_r = self.path_project+'ascii/check/izid2.asc'
        DTM_LC = self.path_project+'ascii/DTM_LC.asc'
        
        [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(DTM_LC)
        dtm = np.loadtxt(DTM_LC,skiprows=6)
        dtm[dtm<10] = -9999
        dtm[dtm>=10] = 1
        rasterToASCII(self.path_project+'ascii/DTM_OC.asc', np.flipud(dtm), xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='float')
        
        os.system('gdal_polygonize.py '+self.path_project+'ascii/DTM_OC.asc'+' '+self.path_project+'shp/DTM_OC.shp -b 1 -f "ESRI Shapefile" DTM_OC IZID')
        
        os.system('gdal_polygonize.py '+izd1_r+' '+self.path_project+'shp/izid1.shp -b 1 -f "ESRI Shapefile" izid1 IZID')
        os.system('gdal_polygonize.py '+izd2_r+' '+self.path_project+'shp/izid2.shp -b 1 -f "ESRI Shapefile" izid2 IZID')
        
        izd1 = self.path_project+'shp/izid1.shp'
        izd2 = self.path_project+'shp/izid2.shp'
        
        shape_1 = gpd.read_file(self.path_project+'shp/DTM_OC.shp')
        shape_1['CID'] = 1
        shape_1 = shape_1.dissolve(by='CID')

        shape_2 = gpd.read_file(self.path_project+'shp/izid2.shp')
        shape_2['CID'] = 1
        shape_2 = shape_2.dissolve(by='CID')

        result_2 = gpd.overlay(shape_1, shape_2, how='union')
        result_2['CID']=1
        result_2 = result_2.dissolve(by='CID')
        result_2.to_file(self.path_project+'shp/Final_cut_complet.shp')

        FC = gpd.read_file(self.path_project+'shp/Final_cut_complet.shp')
        FC = explode(self.path_project+'shp/Final_cut_complet.shp')
        FC = FC.geometry.apply(lambda p: close_holes(p))
        FC.to_file(self.path_project+'shp/Final_cut_complet_F.shp')
        
        
        extract_coastline(self.path_project+'shp/Final_cut_complet_F.shp',DTM_LC,self.path_project+'ascii/DTM_LC_2.asc', cellsize)
            
        line_cost_raster(self.path_project+'ascii/DTM_LC_2.asc',self.path_project+'ascii/',n_types=True)
        line_cost_raster(self.path_project+'ascii/DTM_LC_2.asc',self.path_project+'ascii/',n_types=False)
        os.system('gdal_polygonize.py '+self.path_project+'ascii/coast.asc'+' '+self.path_project+'shp/coast_IH.shp -b 1 -f "ESRI Shapefile" coast_IH CID')
        
        shapefile = gpd.read_file(self.path_project+'shp/coast_IH.shp' )
        shapefile=shapefile[shapefile.CID==1]
        pol = shapefile['geometry']
        
        # from shapely.ops import cascaded_union
        # u = cascaded_union(pol)
        # convert_polygon_to_shp(u,path_project+'shp/coast_IH.shp')
        
        
        ds = ogr.Open(self.path_project+'shp/coast_IH.shp')
        ly = ds.ExecuteSQL('SELECT ST_Centroid(geometry), * FROM coast_IH', dialect='sqlite')
        drv = ogr.GetDriverByName('Esri shapefile')
        ds2 = drv.CreateDataSource(self.path_project+'shp/coast_IH_s.shp')
        ds2.CopyLayer(ly, '')
        ly = ds = ds2 = None  # save, close
        
        from shapely.geometry import shape
        fc_A = gpd.read_file(self.path_project+'shp/coast_IH_s.shp')
        fc_B = gpd.read_file(self.path_project+'shp/izid2.shp')
        # List to collect pairs of intersecting features
        fc_intersect = np.unique(gpd.sjoin(fc_B,fc_A , op='intersects')['IZID'])
        gpd.sjoin(fc_B,fc_A , op='intersects').to_file(self.path_project+'shp/izcoast.shp')
        
        Table_impact_zone=pd.read_csv(self.path_project+'ascii/tblImpactZone.csv',index_col=0)
        
        izcoast=Table_impact_zone.loc[fc_intersect,:].loc[:,[' NbCells',' MinLevel',' MidX',' MidY']].copy()
        izcoast.columns=['nCells','minH','MidX','MidY']
        
        ZE = zonal_statistic(izd2_r,self.path_project+'ascii/coast_Line.asc',statis='sum')
        
        for i in tqdm.tqdm(izcoast.index):
            izcoast.loc[i,'nCells'] =  ZE.loc[i]
        end = time.time()
        print(end - start)
        return Table_impact_zone, izcoast
    
    def tusrBCFlowLevel (self,path_inputs_dinamicas,TestDesc,Table_impact_zone,izcoast,n_periods,BCSetID,BCTypeID_COAST,
                    raw_q=True, point_X_river = None ,point_Y_river = None ,hidrograma=None):
        
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

        Results_TWL.to_csv(self.path_project+'tests/'+TestDesc+'/Input_User/tusrBCFlowLevel.csv',index=False)
        
    def tusrBCRainfall (self,TestDesc,Table_impact_zone,n_periods,array_rain_intensity):
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

        tusrBCRainfallZones.to_csv(self.path_project+'tests/'+TestDesc+'/Input_User/tusrBCRainfallZones.csv',index=False)
        tusrBCRainfallRunoff.to_csv(self.path_project+'tests/'+TestDesc+'/Input_User/tusrBCRainfallRunoff.csv',index=False)
        
    def create_file_manning(self,TestDesc,Table_impact_zone,BCSetID):
        clc_code=pd.read_csv(self.path_project+'ascii/clc_legend.csv')
        usos_suelo = np.loadtxt(self.path_project+'ascii/land_uses.asc',skiprows=6)
        tusrlZManning = pd.DataFrame(index=np.arange(0,len(Table_impact_zone)),columns=['BCSetID','IZID','CManning'])

        [ncols_u,nrows_u,xllcorner_u,yllcorner_u,cellsize_u,NODATA_value_u] = header_ascii(self.path_project+'ascii/land_uses.asc')
        [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(self.path_project+'ascii/DTM_LC.asc')

        xx_LU = np.linspace(xllcorner_u, xllcorner_u + (ncols_u-1)*cellsize_u, ncols_u)
        yy_LU = np.linspace(yllcorner_u, yllcorner_u + (nrows_u-1)*cellsize_u, nrows_u)

        xx = np.linspace(xllcorner, xllcorner + (ncols-1)*cellsize, ncols)
        yy = np.linspace(yllcorner, yllcorner + (nrows-1)*cellsize, nrows)


        [XX_LU,YY_LU] = np.meshgrid(xx_LU,yy_LU)
        [XX,YY] = np.meshgrid(xx,yy)

        fnew=griddata((XX_LU.flatten(), YY_LU.flatten()), usos_suelo.flatten(), (XX, YY), method='nearest')
        rasterToASCII(self.path_project+'ascii/land_uses_r.asc', np.flipud(fnew), xllcorner, yllcorner, cellsize, NODATA_value, header=True,type_data='integer')

        izid2_r = self.path_project+'ascii/check/izid2.asc'

        ZE = zonal_statistic(izid2_r,self.path_project+'ascii/land_uses_r.asc','majority')
        for i,ii in enumerate(tqdm.tqdm(Table_impact_zone.index)):
            tusrlZManning.iloc[i,0]=BCSetID
            tusrlZManning.iloc[i,1]=ii
            try:
                tusrlZManning.iloc[i,2]=clc_code[clc_code.GRID_CODE==ZE[ii]]['n_manning'].values[0]
            except:
                tusrlZManning.iloc[i,2]=clc_code[clc_code.GRID_CODE==2]['n_manning'].values[0]
        tusrlZManning.to_csv(self.path_project+'tests/'+TestDesc+'/Input_User/tusrIZManning.csv',index=False)
        
    def create_xml(self,TestID,TestDesc,BCSetID,StartTime,EndTime,TimeStep,SaveTimeStep,MaxTimeStep,MinTimeStep,AlphaParameter,ManningGlobalValue,Results):
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
        '<DbName>'+self.path_project+ 'tests/'+TestDesc+'</DbName>\n',
        '<FA_ID>1</FA_ID>\n',
        '<TimeStepMethod>a</TimeStepMethod>\n',
        '<LogVerbose>0</LogVerbose>\n',
        '<Results>'+str(Results)+'</Results>\n',
        '</settings>\n']

        if len(os.listdir(self.path_project+'tests/'+TestDesc+'/Input_AccData/'))>0:
            directory= glob.glob(self.path_project+'ascii/*csv')
            shutil.rmtree(self.path_project+'tests/'+TestDesc+'/Input_AccData/')
            os.mkdir(self.path_project+'tests/'+TestDesc+'/Input_AccData/')
            for file in directory:
                if os.path.isfile(file):
                    shutil.copy2(file, self.path_project+'tests/'+TestDesc+'/Input_AccData/')   
        else:
            directory= glob.glob(self.path_project+'ascii/*csv')
            for file in directory:
                if os.path.isfile(file):
                    shutil.copy2(file, self.path_project+'tests/'+TestDesc+'/Input_AccData/')

        with open(self.path_project+'tests/'+TestDesc+'/Input_xml/'+'input_'+str(TestID)+'.xml', "w") as fh:
            for line in (lines):
                    fh.write(line)
    def execute_RFSM(self,TestDesc,TestID):
        os.chdir(self.path_project+'tests/'+TestDesc+'/Input_xml/')
        try:
            test = os.uname()
            if test[0] == "Linux":
                os.system(self.path_project+'bin/RFSM/Windows7_x86/RFSM_Hydrodynamic.exe '+'input_'+str(TestID)+'.xml')
        except AttributeError:
            print("Assuming windows!")
            os.system(self.path_project+'bin/RFSM/Windows7_x86/RFSM_Hydrodynamic.exe '+'input_'+str(TestID)+'.xml')
            
    def export_result_RFSM(self,TestDesc,Results):
        path_results = self.path_project + 'tests/'+TestDesc+'/Results_'+str(Results)+'/'
        tusrResultsIZMax = pd.read_csv(path_results+'tusrResultsIZMax.csv',index_col=1)
        izd2_r = self.path_project+'ascii/check/izid2.asc'
        izd2_r_array = np.loadtxt(izd2_r,skiprows=6)
        [ncols_r,nrows_r,xllcorner_r,yllcorner_r,cellsize_r,NODATA_value_r] = header_ascii(izd2_r)

        MaxLevel      = izd2_r_array.copy()
        MaxDepth      = izd2_r_array.copy()
        MaxDischarge  = izd2_r_array.copy()
        MaxVelocity   = izd2_r_array.copy()
        MaxHazard     = izd2_r_array.copy()
        for IDZ in tqdm.tqdm(tusrResultsIZMax.index):
            pos = np.where(izd2_r_array==float(IDZ))

            MaxLevel[pos[0],pos[1]]      = tusrResultsIZMax.loc[IDZ,'MaxLevel']
            MaxDepth[pos[0],pos[1]]      = tusrResultsIZMax.loc[IDZ,'MaxDepth']
            MaxDischarge[pos[0],pos[1]]  = tusrResultsIZMax.loc[IDZ,'MaxDischarge']
            MaxVelocity[pos[0],pos[1]]   = tusrResultsIZMax.loc[IDZ,'MaxVelocity']
            MaxHazard[pos[0],pos[1]]     = tusrResultsIZMax.loc[IDZ,'MaxHazard']

        rasterToASCII(self.path_project + 'tests/'+TestDesc+'/export/MaxLevel_'+str(Results)+'.asc', np.flipud(MaxLevel), 
                      xllcorner_r, yllcorner_r, cellsize_r, NODATA_value_r, header=True,type_data='float')

        rasterToASCII(self.path_project + 'tests/'+TestDesc+'/export/MaxDepth_'+str(Results)+'.asc', np.flipud(MaxDepth), 
                      xllcorner_r, yllcorner_r, cellsize_r, NODATA_value_r, header=True,type_data='float')

        rasterToASCII(self.path_project + 'tests/'+TestDesc+'/export/MaxDischarge_'+str(Results)+'.asc', np.flipud(MaxDischarge), 
                      xllcorner_r, yllcorner_r, cellsize_r, NODATA_value_r, header=True,type_data='float')

        rasterToASCII(self.path_project + 'tests/'+TestDesc+'/export/MaxVelocity_'+str(Results)+'.asc', np.flipud(MaxVelocity), 
                      xllcorner_r, yllcorner_r, cellsize_r, NODATA_value_r, header=True,type_data='float')

        rasterToASCII(self.path_project + 'tests/'+TestDesc+'/export/MaxHazard_'+str(Results)+'.asc', np.flipud(MaxHazard), 
                      xllcorner_r, yllcorner_r, cellsize_r, NODATA_value_r, header=True,type_data='float')
