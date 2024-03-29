# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

import numpy as np
import pandas as pd
import geopandas as gpd
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
import  zipfile
from shutil import copyfile
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
#from pyproj import _datadir, datadir
from fiona import _shim, schema
import os.path
import sys
import subprocess
import shlex
import json  # json is an easy way to send arbitrary ascii-safe lists of strings out of python

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
    xllcorner     =   float(data[2])
    yllcorner     =   float(data[3])
    cellsize      =   float(data[4])     # VIC cellsize desired
    NODATA_value  =   float(data[5])
    
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
    
    
def create_colormap(colors, position=None, bit=False, reverse=False, name='custom_colormap'):
    """
    returns a linear custom colormap
    Parameters
    ----------
    colors : array-like
        contain RGB values. The RGB values may either be in 8-bit [0 to 255]
        or arithmetic [0 to 1] (default).
        Arrange your tuples so that the first color is the lowest value for the
        colorbar and the last is the highest.
    position : array like
        contains values from 0 to 1 to dictate the location of each color.
    bit : Boolean
        8-bit [0 to 255] (in which bit must be set to
        True when called) or arithmetic [0 to 1] (default)
    reverse : Boolean
        If you want to flip the scheme
    name : string
        name of the scheme if you plan to save it
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        cmap with equally spaced colors
    """
    from matplotlib.colors import LinearSegmentedColormap
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f')
    if reverse:
        colors = colors[::-1]
    if position is not None and not isinstance(position, np.ndarray):
        position = np.array(position)
    elif position is None:
        position = np.linspace(0, 1, colors.shape[0])
    else:
        if position.size != colors.shape[0]:
            raise ValueError("position length must be the same as colors")
        elif not np.isclose(position[0], 0) and not np.isclose(position[-1], 1):
            raise ValueError("position must start with 0 and end with 1")
    if bit:
        colors[:] = [tuple(map(lambda x: x / 255., color)) for color in colors]
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    return LinearSegmentedColormap(name, cdict, 256)


def create_legend_kml(data, file_kml, path_output_image_leg,name_image_leg):
    from tifffile import imread, imsave
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    fig = plt.figure()
    # Create a list of RGB tuples
    colors = [(0, 240, 255), (0, 50, 170)] # This example uses the arithmetic RGB
    # If you are only going to use your colormap once you can
    # take out a step.
    #raster =  imread(data)
    #mpb = plt.imshow(raster, cmap=create_colormap(colors,bit=True),vmin=0, vmax=np.max(raster))
    #plt.colorbar()
    #plt.close()
    
    a = np.array([[0,2.5]])
    plt.figure(figsize=(8, 6))
    mpb = plt.imshow(a,cmap=create_colormap(colors,bit=True))
    plt.gca().set_visible(False)
    plt.close()

    
    fig,ax = plt.subplots(figsize=(8, 6))
    cbar = plt.colorbar(mpb,ax=ax)
    cbar.ax.set_ylabel('Calado (m)',fontsize = 20)
    cbar.ax.tick_params(labelsize=20) 
    fig.patch.set_facecolor('white')
    ax.remove()
    plt.savefig(path_output_image_leg+name_image_leg+'.png',bbox_inches='tight',transparent=False)
    plt.close()
    
    with open(file_kml, "r+") as f:
        data = f.readlines()
    doc = list()
    for i, ii in enumerate(data):
        if 'Document' in ii:
            doc.append(i)
    lines_1 = data[:doc[1]]
    lines_2 = ['    <ScreenOverlay>\n',
               '       <name>Legend</name>\n',
               '        <Icon>\n',
               '          <href>'+name_image_leg+'.png</href>\n',
               '        </Icon>\n',
               '        <overlayXY x="0" y="1" xunits="fraction" yunits="fraction"/>\n',
               '        <screenXY x="0" y="1" xunits="fraction" yunits="fraction"/>\n',
               '    </ScreenOverlay>\n',
               '  </Document>\n',
               '</kml>\n']
    f.close()

    with open(file_kml, "w") as fh:
        for line in (lines_1+lines_2):
            fh.write(line) 
            
            
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path)))
            
            
            

try:
    from osgeo import gdal
    gdal.TermProgress = gdal.TermProgress_nocb
except ImportError:
    import gdal

__version__ = '$id$'[5:-1]
verbose = 0
quiet = 0


# =============================================================================
def raster_copy( s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
                 t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
                 nodata=None ):

    if nodata is not None:
        return raster_copy_with_nodata(
            s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
            t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
            nodata )

    if verbose != 0:
        print('Copy %d,%d,%d,%d to %d,%d,%d,%d.' \
              % (s_xoff, s_yoff, s_xsize, s_ysize,
             t_xoff, t_yoff, t_xsize, t_ysize ))

    s_band = s_fh.GetRasterBand( s_band_n )
    t_band = t_fh.GetRasterBand( t_band_n )

    data = s_band.ReadRaster( s_xoff, s_yoff, s_xsize, s_ysize,
                              t_xsize, t_ysize, t_band.DataType )
    t_band.WriteRaster( t_xoff, t_yoff, t_xsize, t_ysize,
                        data, t_xsize, t_ysize, t_band.DataType )
        

    return 0
    
# =============================================================================
def raster_copy_with_nodata( s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
                             t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
                             nodata ):
    try:
        import numpy as Numeric
    except ImportError:
        import Numeric
    
    if verbose != 0:
        print('Copy %d,%d,%d,%d to %d,%d,%d,%d.' \
              % (s_xoff, s_yoff, s_xsize, s_ysize,
             t_xoff, t_yoff, t_xsize, t_ysize ))

    s_band = s_fh.GetRasterBand( s_band_n )
    t_band = t_fh.GetRasterBand( t_band_n )

    data_src = s_band.ReadAsArray( s_xoff, s_yoff, s_xsize, s_ysize,
                                   t_xsize, t_ysize )
    data_dst = t_band.ReadAsArray( t_xoff, t_yoff, t_xsize, t_ysize )

    nodata_test = Numeric.equal(data_src,nodata)
    to_write = Numeric.choose( nodata_test, (data_src, data_dst) )
                               
    t_band.WriteArray( to_write, t_xoff, t_yoff )

    return 0
    
# =============================================================================
def names_to_fileinfos( names ):
    """
    Translate a list of GDAL filenames, into file_info objects.
    names -- list of valid GDAL dataset names.
    Returns a list of file_info objects.  There may be less file_info objects
    than names if some of the names could not be opened as GDAL files.
    """
    
    file_infos = []
    for name in names:
        fi = file_info()
        if fi.init_from_name( name ) == 1:
            file_infos.append( fi )

    return file_infos

# *****************************************************************************
class file_info:
    """A class holding information about a GDAL file."""

    def init_from_name(self, filename):
        """
        Initialize file_info from filename
        filename -- Name of file to read.
        Returns 1 on success or 0 if the file can't be opened.
        """
        fh = gdal.Open( filename )
        if fh is None:
            return 0

        self.filename = filename
        self.bands = fh.RasterCount
        self.xsize = fh.RasterXSize
        self.ysize = fh.RasterYSize
        self.band_type = fh.GetRasterBand(1).DataType
        self.projection = fh.GetProjection()
        self.geotransform = fh.GetGeoTransform()
        self.ulx = self.geotransform[0]
        self.uly = self.geotransform[3]
        self.lrx = self.ulx + self.geotransform[1] * self.xsize
        self.lry = self.uly + self.geotransform[5] * self.ysize

        ct = fh.GetRasterBand(1).GetRasterColorTable()
        if ct is not None:
            self.ct = ct.Clone()
        else:
            self.ct = None

        return 1

    def report( self ):
        print('Filename: '+ self.filename)
        print('File Size: %dx%dx%d' \
              % (self.xsize, self.ysize, self.bands))
        print('Pixel Size: %f x %f' \
              % (self.geotransform[1],self.geotransform[5]))
        print('UL:(%f,%f)   LR:(%f,%f)' \
              % (self.ulx,self.uly,self.lrx,self.lry))

    def copy_into( self, t_fh, s_band = 1, t_band = 1, nodata_arg=None ):
        """
        Copy this files image into target file.
        This method will compute the overlap area of the file_info objects
        file, and the target gdal.Dataset object, and copy the image data
        for the common window area.  It is assumed that the files are in
        a compatible projection ... no checking or warping is done.  However,
        if the destination file is a different resolution, or different
        image pixel type, the appropriate resampling and conversions will
        be done (using normal GDAL promotion/demotion rules).
        t_fh -- gdal.Dataset object for the file into which some or all
        of this file may be copied.
        Returns 1 on success (or if nothing needs to be copied), and zero one
        failure.
        """
        t_geotransform = t_fh.GetGeoTransform()
        t_ulx = t_geotransform[0]
        t_uly = t_geotransform[3]
        t_lrx = t_geotransform[0] + t_fh.RasterXSize * t_geotransform[1]
        t_lry = t_geotransform[3] + t_fh.RasterYSize * t_geotransform[5]

        # figure out intersection region
        tgw_ulx = max(t_ulx,self.ulx)
        tgw_lrx = min(t_lrx,self.lrx)
        if t_geotransform[5] < 0:
            tgw_uly = min(t_uly,self.uly)
            tgw_lry = max(t_lry,self.lry)
        else:
            tgw_uly = max(t_uly,self.uly)
            tgw_lry = min(t_lry,self.lry)
        
        # do they even intersect?
        if tgw_ulx >= tgw_lrx:
            return 1
        if t_geotransform[5] < 0 and tgw_uly <= tgw_lry:
            return 1
        if t_geotransform[5] > 0 and tgw_uly >= tgw_lry:
            return 1
            
        # compute target window in pixel coordinates.
        tw_xoff = int((tgw_ulx - t_geotransform[0]) / t_geotransform[1] + 0.1)
        tw_yoff = int((tgw_uly - t_geotransform[3]) / t_geotransform[5] + 0.1)
        tw_xsize = int((tgw_lrx - t_geotransform[0])/t_geotransform[1] + 0.5) \
                   - tw_xoff
        tw_ysize = int((tgw_lry - t_geotransform[3])/t_geotransform[5] + 0.5) \
                   - tw_yoff

        if tw_xsize < 1 or tw_ysize < 1:
            return 1

        # Compute source window in pixel coordinates.
        sw_xoff = int((tgw_ulx - self.geotransform[0]) / self.geotransform[1])
        sw_yoff = int((tgw_uly - self.geotransform[3]) / self.geotransform[5])
        sw_xsize = int((tgw_lrx - self.geotransform[0]) \
                       / self.geotransform[1] + 0.5) - sw_xoff
        sw_ysize = int((tgw_lry - self.geotransform[3]) \
                       / self.geotransform[5] + 0.5) - sw_yoff

        if sw_xsize < 1 or sw_ysize < 1:
            return 1

        # Open the source file, and copy the selected region.
        s_fh = gdal.Open( self.filename )

        return \
            raster_copy( s_fh, sw_xoff, sw_yoff, sw_xsize, sw_ysize, s_band,
                         t_fh, tw_xoff, tw_yoff, tw_xsize, tw_ysize, t_band,
                         nodata_arg )


# =============================================================================
def Usage():
    print('Usage: gdal_merge.py [-o out_filename] [-of out_format] [-co NAME=VALUE]*')
    print('                     [-ps pixelsize_x pixelsize_y] [-separate] [-q] [-v] [-pct]')
    print('                     [-ul_lr ulx uly lrx lry] [-n nodata_value] [-init "value [value...]"]')
    print('                     [-ot datatype] [-createonly] input_files')
    print('                     [--help-general]')
    print('')

# =============================================================================
#
# Program mainline.
#

def gedal_merge( argv=None ):

    global verbose, quiet
    verbose = 0
    quiet = 0
    names = []
    format = 'GTiff'
    out_file = 'out.tif'

    ulx = None
    psize_x = None
    separate = 0
    copy_pct = 0
    nodata = None
    create_options = []
    pre_init = []
    band_type = None
    createonly = 0
    
    gdal.AllRegister()
    if argv is None:
        argv = sys.argv
    argv = gdal.GeneralCmdLineProcessor( argv )
    if argv is None:
        sys.exit( 0 )

    # Parse command line arguments.
    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg == '-o':
            i = i + 1
            out_file = argv[i]

        elif arg == '-v':
            verbose = 1

        elif arg == '-q' or arg == '-quiet':
            quiet = 1

        elif arg == '-createonly':
            createonly = 1

        elif arg == '-separate':
            separate = 1

        elif arg == '-seperate':
            separate = 1

        elif arg == '-pct':
            copy_pct = 1

        elif arg == '-ot':
            i = i + 1
            band_type = gdal.GetDataTypeByName( argv[i] )
            if band_type == gdal.GDT_Unknown:
                print('Unknown GDAL data type: ', argv[i])
                sys.exit( 1 )

        elif arg == '-init':
            i = i + 1
            str_pre_init = argv[i].split()
            for x in str_pre_init:
                pre_init.append(float(x))

        elif arg == '-n':
            i = i + 1
            nodata = float(argv[i])

        elif arg == '-f':
            # for backward compatibility.
            i = i + 1
            format = argv[i]

        elif arg == '-of':
            i = i + 1
            format = argv[i]

        elif arg == '-co':
            i = i + 1
            create_options.append( argv[i] )

        elif arg == '-ps':
            psize_x = float(argv[i+1])
            psize_y = -1 * abs(float(argv[i+2]))
            i = i + 2

        elif arg == '-ul_lr':
            ulx = float(argv[i+1])
            uly = float(argv[i+2])
            lrx = float(argv[i+3])
            lry = float(argv[i+4])
            i = i + 4

        elif arg[:1] == '-':
            print('Unrecognised command option: ', arg)
            Usage()
            sys.exit( 1 )

        else:
            # Expand any possible wildcards from command line arguments
            f = glob.glob( arg )
            if len(f) == 0:
                print('File not found: "%s"' % (str( arg )))
            names += f # append 1 or more files
        i = i + 1

    if len(names) == 0:
        print('No input files selected.')
        Usage()
        sys.exit( 1 )

    Driver = gdal.GetDriverByName(format)
    if Driver is None:
        print('Format driver %s not found, pick a supported driver.' % format)
        sys.exit( 1 )

    DriverMD = Driver.GetMetadata()
    if 'DCAP_CREATE' not in DriverMD:
        print('Format driver %s does not support creation and piecewise writing.\nPlease select a format that does, such as GTiff (the default) or HFA (Erdas Imagine).' % format)
        sys.exit( 1 )

    # Collect information on all the source files.
    file_infos = names_to_fileinfos( names )

    if ulx is None:
        ulx = file_infos[0].ulx
        uly = file_infos[0].uly
        lrx = file_infos[0].lrx
        lry = file_infos[0].lry
        
        for fi in file_infos:
            ulx = min(ulx, fi.ulx)
            uly = max(uly, fi.uly)
            lrx = max(lrx, fi.lrx)
            lry = min(lry, fi.lry)

    if psize_x is None:
        psize_x = file_infos[0].geotransform[1]
        psize_y = file_infos[0].geotransform[5]

    if band_type is None:
        band_type = file_infos[0].band_type

    # Try opening as an existing file.
    gdal.PushErrorHandler( 'CPLQuietErrorHandler' )
    t_fh = gdal.Open( out_file, gdal.GA_Update )
    gdal.PopErrorHandler()
    
    # Create output file if it does not already exist.
    if t_fh is None:
        geotransform = [ulx, psize_x, 0, uly, 0, psize_y]

        xsize = int((lrx - ulx) / geotransform[1] + 0.5)
        ysize = int((lry - uly) / geotransform[5] + 0.5)

        if separate != 0:
            bands = len(file_infos)
        else:
            bands = file_infos[0].bands

        t_fh = Driver.Create( out_file, xsize, ysize, bands,
                              band_type, create_options )
        if t_fh is None:
            print('Creation failed, terminating gdal_merge.')
            sys.exit( 1 )
            
        t_fh.SetGeoTransform( geotransform )
        t_fh.SetProjection( file_infos[0].projection )

        if copy_pct:
            t_fh.GetRasterBand(1).SetRasterColorTable(file_infos[0].ct)
    else:
        if separate != 0:
            bands = len(file_infos)
            if t_fh.RasterCount < bands :
                print('Existing output file has less bands than the number of input files. You should delete it before. Terminating gdal_merge.')
                sys.exit( 1 )
        else:
            bands = min(file_infos[0].bands,t_fh.RasterCount)

    # Do we need to pre-initialize the whole mosaic file to some value?
    if pre_init is not None:
        if t_fh.RasterCount <= len(pre_init):
            for i in range(t_fh.RasterCount):
                t_fh.GetRasterBand(i+1).Fill( pre_init[i] )
        elif len(pre_init) == 1:
            for i in range(t_fh.RasterCount):
                t_fh.GetRasterBand(i+1).Fill( pre_init[0] )

    # Copy data from source files into output file.
    t_band = 1

    if quiet == 0 and verbose == 0:
        gdal.TermProgress( 0.0 )
    fi_processed = 0
    
    for fi in file_infos:
        if createonly != 0:
            continue
        
        if verbose != 0:
            print("")
            print("Processing file %5d of %5d, %6.3f%% completed." \
                  % (fi_processed+1,len(file_infos),
                     fi_processed * 100.0 / len(file_infos)) )
            fi.report()

        if separate == 0 :
            for band in range(1, bands+1):
                fi.copy_into( t_fh, band, band, nodata )
        else:
            fi.copy_into( t_fh, 1, t_band, nodata )
            t_band = t_band+1
            
        fi_processed = fi_processed+1
        if quiet == 0 and verbose == 0:
            gdal.TermProgress( fi_processed / float(len(file_infos))  )
    
    # Force file to be closed.
    t_fh = None
###########################################################################################
def gdal_edit(argv):

    argv = gdal.GeneralCmdLineProcessor(argv)
    if argv is None:
        return -1

    datasetname = None
    srs = None
    ulx = None
    uly = None
    urx = None
    ury = None
    llx = None
    lly = None
    lrx = None
    lry = None
    nodata = None
    unsetnodata = False
    units = None
    xres = None
    yres = None
    unsetgt = False
    unsetstats = False
    stats = False
    setstats = False
    approx_stats = False
    unsetmd = False
    ro = False
    molist = []
    gcp_list = []
    open_options = []
    offset = []
    scale = []
    colorinterp = {}
    unsetrpc = False

    i = 1
    argc = len(argv)
    while i < argc:
        if argv[i] == '-ro':
            ro = True
        elif argv[i] == '-a_srs' and i < len(argv) - 1:
            srs = argv[i + 1]
            i = i + 1
        elif argv[i] == '-a_ullr' and i < len(argv) - 4:
            ulx = float(argv[i + 1])
            i = i + 1
            uly = float(argv[i + 1])
            i = i + 1
            lrx = float(argv[i + 1])
            i = i + 1
            lry = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-a_ulurll' and i < len(argv) - 6:
            ulx = float(argv[i + 1])
            i = i + 1
            uly = float(argv[i + 1])
            i = i + 1
            urx = float(argv[i + 1])
            i = i + 1
            ury = float(argv[i + 1])
            i = i + 1
            llx = float(argv[i + 1])
            i = i + 1
            lly = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-tr' and i < len(argv) - 2:
            xres = float(argv[i + 1])
            i = i + 1
            yres = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-a_nodata' and i < len(argv) - 1:
            nodata = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-scale' and i < len(argv) -1:
            scale.append(float(argv[i+1]))
            i = i + 1
            while i < len(argv) - 1 and ArgIsNumeric(argv[i+1]):
                scale.append(float(argv[i+1]))
                i = i + 1
        elif argv[i] == '-offset' and i < len(argv) - 1:
            offset.append(float(argv[i+1]))
            i = i + 1
            while i < len(argv) - 1 and ArgIsNumeric(argv[i+1]):
                offset.append(float(argv[i+1]))
                i = i + 1
        elif argv[i] == '-mo' and i < len(argv) - 1:
            molist.append(argv[i + 1])
            i = i + 1
        elif argv[i] == '-gcp' and i + 4 < len(argv):
            pixel = float(argv[i + 1])
            i = i + 1
            line = float(argv[i + 1])
            i = i + 1
            x = float(argv[i + 1])
            i = i + 1
            y = float(argv[i + 1])
            i = i + 1
            if i + 1 < len(argv) and ArgIsNumeric(argv[i + 1]):
                z = float(argv[i + 1])
                i = i + 1
            else:
                z = 0
            gcp = gdal.GCP(x, y, z, pixel, line)
            gcp_list.append(gcp)
        elif argv[i] == '-unsetgt':
            unsetgt = True
        elif argv[i] == '-unsetrpc':
            unsetrpc = True
        elif argv[i] == '-unsetstats':
            unsetstats = True
        elif argv[i] == '-approx_stats':
            stats = True
            approx_stats = True
        elif argv[i] == '-stats':
            stats = True
        elif argv[i] == '-setstats' and i < len(argv)-4:
            stats = True
            setstats = True
            if argv[i + 1] != 'None':
                statsmin = float(argv[i + 1])
            else:
                statsmin = None
            i = i + 1
            if argv[i + 1] != 'None':
                statsmax = float(argv[i + 1])
            else:
                statsmax = None
            i = i + 1
            if argv[i + 1] != 'None':
                statsmean = float(argv[i + 1])
            else:
                statsmean = None
            i = i + 1
            if argv[i + 1] != 'None':
                statsdev = float(argv[i + 1])
            else:
                statsdev = None
            i = i + 1
        elif argv[i] == '-units' and i < len(argv) - 1:
            units = argv[i + 1]
            i = i + 1
        elif argv[i] == '-unsetmd':
            unsetmd = True
        elif argv[i] == '-unsetnodata':
            unsetnodata = True
        elif argv[i] == '-oo' and i < len(argv) - 1:
            open_options.append(argv[i + 1])
            i = i + 1
        elif argv[i].startswith('-colorinterp_')and i < len(argv) - 1:
            band = int(argv[i][len('-colorinterp_'):])
            val = argv[i + 1]
            if val.lower() == 'red':
                val = gdal.GCI_RedBand
            elif val.lower() == 'green':
                val = gdal.GCI_GreenBand
            elif val.lower() == 'blue':
                val = gdal.GCI_BlueBand
            elif val.lower() == 'alpha':
                val = gdal.GCI_AlphaBand
            elif val.lower() == 'gray' or val.lower() == 'grey':
                val = gdal.GCI_GrayIndex
            elif val.lower() == 'undefined':
                val = gdal.GCI_Undefined
            else:
                sys.stderr.write('Unsupported color interpretation %s.\n' % val +
                                 'Only red, green, blue, alpha, gray, undefined are supported.\n')
                return Usage()
            colorinterp[band] = val
            i = i + 1
        elif argv[i][0] == '-':
            sys.stderr.write('Unrecognized option : %s\n' % argv[i])
            return Usage()
        elif datasetname is None:
            datasetname = argv[i]
        else:
            sys.stderr.write('Unexpected option : %s\n' % argv[i])
            return Usage()

        i = i + 1

    if datasetname is None:
        return Usage()

    if (srs is None and lry is None and yres is None and not unsetgt and
            not unsetstats and not stats and not setstats and nodata is None and
            not units and not molist and not unsetmd and not gcp_list and
            not unsetnodata and not colorinterp and
            scale is None and offset is None and not unsetrpc):
        print('No option specified')
        print('')
        return Usage()

    exclusive_option = 0
    if lry is not None:
        exclusive_option = exclusive_option + 1
    if lly is not None:  # -a_ulurll
        exclusive_option = exclusive_option + 1
    if yres is not None:
        exclusive_option = exclusive_option + 1
    if unsetgt:
        exclusive_option = exclusive_option + 1
    if exclusive_option > 1:
        print('-a_ullr, -a_ulurll, -tr and -unsetgt options are exclusive.')
        print('')
        return Usage()

    if unsetstats and stats:
        print('-unsetstats and either -stats or -approx_stats options are exclusive.')
        print('')
        return Usage()

    if unsetnodata and nodata:
        print('-unsetnodata and -nodata options are exclusive.')
        print('')
        return Usage()

    if open_options is not None:
        if ro:
            ds = gdal.OpenEx(datasetname, gdal.OF_RASTER, open_options=open_options)
        else:
            ds = gdal.OpenEx(datasetname, gdal.OF_RASTER | gdal.OF_UPDATE, open_options=open_options)
    # GDAL 1.X compat
    elif ro:
        ds = gdal.Open(datasetname)
    else:
        ds = gdal.Open(datasetname, gdal.GA_Update)
    if ds is None:
        return -1

    if scale:
        if len(scale) == 1:
            scale = scale * ds.RasterCount 
        elif len(scale) != ds.RasterCount:
            print('If more than one scale value is provided, their number must match the number of bands.')
            print('')
            return Usage()
    
    if offset:
        if len(offset) == 1:
            offset = offset * ds.RasterCount
        elif len(offset) != ds.RasterCount:
            print('If more than one offset value is provided, their number must match the number of bands.')
            print('')
            return Usage()
    
    wkt = None
    if srs == '' or srs == 'None':
        ds.SetProjection('')
    elif srs is not None:
        sr = osr.SpatialReference()
        if sr.SetFromUserInput(srs) != 0:
            print('Failed to process SRS definition: %s' % srs)
            return -1
        wkt = sr.ExportToWkt()
        if not gcp_list:
            ds.SetProjection(wkt)

    if lry is not None:
        gt = [ulx, (lrx - ulx) / ds.RasterXSize, 0,
              uly, 0, (lry - uly) / ds.RasterYSize]
        ds.SetGeoTransform(gt)

    elif lly is not None:  # -a_ulurll
        gt = [ulx, (urx - ulx) / ds.RasterXSize, (llx - ulx) / ds.RasterYSize,
              uly, (ury - uly) / ds.RasterXSize, (lly - uly) / ds.RasterYSize]
        ds.SetGeoTransform(gt)

    if yres is not None:
        gt = ds.GetGeoTransform()
        # Doh ! why is gt a tuple and not an array...
        gt = [gt[j] for j in range(6)]
        gt[1] = xres
        gt[5] = yres
        ds.SetGeoTransform(gt)

    if unsetgt:
        # For now only the GTiff drivers understands full-zero as a hint
        # to unset the geotransform
        if ds.GetDriver().ShortName == 'GTiff':
            ds.SetGeoTransform([0, 0, 0, 0, 0, 0])
        else:
            ds.SetGeoTransform([0, 1, 0, 0, 0, 1])

    if gcp_list:
        if wkt is None:
            wkt = ds.GetGCPProjection()
        if wkt is None:
            wkt = ''
        ds.SetGCPs(gcp_list, wkt)

    if nodata is not None:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
    elif unsetnodata:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).DeleteNoDataValue()

    if scale:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetScale(scale[i])

    if offset:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetOffset(offset[i])

    if units:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetUnitType(units)

    if unsetstats:
        for i in range(ds.RasterCount):
            band = ds.GetRasterBand(i + 1)
            for key in band.GetMetadata().keys():
                if key.startswith('STATISTICS_'):
                    band.SetMetadataItem(key, None)

    if stats:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).ComputeStatistics(approx_stats)

    if setstats:
        for i in range(ds.RasterCount):
            if statsmin is None or statsmax is None or statsmean is None or statsdev is None:
                ds.GetRasterBand(i+1).ComputeStatistics(approx_stats)
                min,max,mean,stdev = ds.GetRasterBand(i+1).GetStatistics(approx_stats,True)
                if statsmin is None:
                    statsmin = min
                if statsmax is None:
                    statsmax = max
                if statsmean is None:
                    statsmean = mean
                if statsdev is None:
                    statsdev = stdev
            ds.GetRasterBand(i+1).SetStatistics(statsmin, statsmax, statsmean, statsdev)

    if molist:
        if unsetmd:
            md = {}
        else:
            md = ds.GetMetadata()
        for moitem in molist:
            equal_pos = moitem.find('=')
            if equal_pos > 0:
                md[moitem[0:equal_pos]] = moitem[equal_pos + 1:]
        ds.SetMetadata(md)
    elif unsetmd:
        ds.SetMetadata({})

    for band in colorinterp:
        ds.GetRasterBand(band).SetColorInterpretation(colorinterp[band])

    if unsetrpc:
        ds.SetMetadata(None, 'RPC')

    ds = band = None

    return 0




def shell_split(cmd):
    """
    Like `shlex.split`, but uses the Windows splitting syntax when run on Windows.

    On windows, this is the inverse of subprocess.list2cmdline
    """
    if os.name == 'posix':
        return shlex.split(cmd)
    else:
        # TODO: write a version of this that doesn't invoke a subprocess
        if not cmd:
            return []
        full_cmd = '{} {}'.format(
            subprocess.list2cmdline([
                sys.executable, '-c',
                'import sys, json; print(json.dumps(sys.argv[1:]))'
            ]), cmd
        )
        ret = subprocess.check_output(full_cmd).decode()
        return json.loads(ret)

def gdal_polygonize (argv):

    def Usage():
        print("""
    gdal_polygonize [-8] [-nomask] [-mask filename] raster_file [-b band|mask]
                    [-q] [-f ogr_format] out_file [layer] [fieldname]
    """)
        sys.exit(1)


    def DoesDriverHandleExtension(drv, ext):
        exts = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
        return exts is not None and exts.lower().find(ext.lower()) >= 0


    def GetExtension(filename):
        ext = os.path.splitext(filename)[1]
        if ext.startswith('.'):
            ext = ext[1:]
        return ext


    def GetOutputDriversFor(filename):
        drv_list = []
        ext = GetExtension(filename)
        for i in range(gdal.GetDriverCount()):
            drv = gdal.GetDriver(i)
            if (drv.GetMetadataItem(gdal.DCAP_CREATE) is not None or
                drv.GetMetadataItem(gdal.DCAP_CREATECOPY) is not None) and \
               drv.GetMetadataItem(gdal.DCAP_VECTOR) is not None:
                if len(ext) > 0 and DoesDriverHandleExtension(drv, ext):
                    drv_list.append(drv.ShortName)
                else:
                    prefix = drv.GetMetadataItem(gdal.DMD_CONNECTION_PREFIX)
                    if prefix is not None and filename.lower().startswith(prefix.lower()):
                        drv_list.append(drv.ShortName)

        return drv_list


    def GetOutputDriverFor(filename):
        drv_list = GetOutputDriversFor(filename)
        if len(drv_list) == 0:
            ext = GetExtension(filename)
            if len(ext) == 0:
                return 'ESRI Shapefile'
            else:
                raise Exception("Cannot guess driver for %s" % filename)
        elif len(drv_list) > 1:
            print("Several drivers matching %s extension. Using %s" % (ext, drv_list[0]))
        return drv_list[0]

    # =============================================================================
    # 	Mainline
    # =============================================================================


    format = None
    options = []
    quiet_flag = 0
    src_filename = None
    src_band_n = 1

    dst_filename = None
    dst_layername = None
    dst_fieldname = None
    dst_field = -1

    mask = 'default'

    gdal.AllRegister()
    #argv = gdal.GeneralCmdLineProcessor(argv)
    argv = shell_split(argv)
    if argv is None:
        sys.exit(0)

    # Parse command line arguments.
    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == '-f' or arg == '-of':
            i = i + 1
            format = argv[i]

        elif arg == '-q' or arg == '-quiet':
            quiet_flag = 1

        elif arg == '-8':
            options.append('8CONNECTED=8')

        elif arg == '-nomask':
            mask = 'none'

        elif arg == '-mask':
            i = i + 1
            mask = argv[i]

        elif arg == '-b':
            i = i + 1
            if argv[i].startswith('mask'):
                src_band_n = argv[i]
            else:
                src_band_n = int(argv[i])

        elif src_filename is None:
            src_filename = argv[i]

        elif dst_filename is None:
            dst_filename = argv[i]

        elif dst_layername is None:
            dst_layername = argv[i]

        elif dst_fieldname is None:
            dst_fieldname = argv[i]

        else:
            Usage()

        i = i + 1

    if src_filename is None or dst_filename is None:
        Usage()

    if format is None:
        format = GetOutputDriverFor(dst_filename)

    if dst_layername is None:
        dst_layername = 'out'

    # =============================================================================
    # 	Verify we have next gen bindings with the polygonize method.
    # =============================================================================
    try:
        gdal.Polygonize
    except AttributeError:
        print('')
        print('gdal.Polygonize() not available.  You are likely using "old gen"')
        print('bindings or an older version of the next gen bindings.')
        print('')
        sys.exit(1)

    # =============================================================================
    # Open source file
    # =============================================================================

    src_ds = gdal.Open(src_filename)

    if src_ds is None:
        print('Unable to open %s' % src_filename)
        sys.exit(1)

    if src_band_n == 'mask':
        srcband = src_ds.GetRasterBand(1).GetMaskBand()
        # Workaround the fact that most source bands have no dataset attached
        options.append('DATASET_FOR_GEOREF=' + src_filename)
    elif isinstance(src_band_n, str) and src_band_n.startswith('mask,'):
        srcband = src_ds.GetRasterBand(int(src_band_n[len('mask,'):])).GetMaskBand()
        # Workaround the fact that most source bands have no dataset attached
        options.append('DATASET_FOR_GEOREF=' + src_filename)
    else:
        srcband = src_ds.GetRasterBand(src_band_n)

    if mask is 'default':
        maskband = srcband.GetMaskBand()
    elif mask is 'none':
        maskband = None
    else:
        mask_ds = gdal.Open(mask)
        maskband = mask_ds.GetRasterBand(1)

    # =============================================================================
    #       Try opening the destination file as an existing file.
    # =============================================================================

    try:
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        dst_ds = ogr.Open(dst_filename, update=1)
        gdal.PopErrorHandler()
    except:
        dst_ds = None

    # =============================================================================
    # 	Create output file.
    # =============================================================================
    if dst_ds is None:
        drv = ogr.GetDriverByName(format)
        if not quiet_flag:
            print('Creating output %s of format %s.' % (dst_filename, format))
        dst_ds = drv.CreateDataSource(dst_filename)

    # =============================================================================
    #       Find or create destination layer.
    # =============================================================================
    try:
        dst_layer = dst_ds.GetLayerByName(dst_layername)
    except:
        dst_layer = None

    if dst_layer is None:

        srs = None
        if src_ds.GetProjectionRef() != '':
            srs = osr.SpatialReference()
            srs.ImportFromWkt(src_ds.GetProjectionRef())

        dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)

        if dst_fieldname is None:
            dst_fieldname = 'DN'

        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = 0
    else:
        if dst_fieldname is not None:
            dst_field = dst_layer.GetLayerDefn().GetFieldIndex(dst_fieldname)
            if dst_field < 0:
                print("Warning: cannot find field '%s' in layer '%s'" % (dst_fieldname, dst_layername))

    # =============================================================================
    # Invoke algorithm.
    # =============================================================================

    if quiet_flag:
        prog_func = None
    else:
        prog_func = gdal.TermProgress_nocb

    result = gdal.Polygonize(srcband, maskband, dst_layer, dst_field, options,
                             callback=prog_func)

    srcband = None
    src_ds = None
    dst_ds = None
    mask_ds = None
    
    
def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        
        
        
import textwrap
from numbers import Number
from typing import Union, Tuple, Optional, Sequence, Dict
import argparse
import os
import os.path
import sys
import string
from collections import defaultdict

import numpy

from osgeo import gdal
from osgeo import gdal_array
from osgeo_utils.auxiliary.base import is_path_like, PathLikeOrStr, MaybeSequence
from osgeo_utils.auxiliary.util import GetOutputDriverFor, open_ds
from osgeo_utils.auxiliary.extent_util import Extent, GT
from osgeo_utils.auxiliary import extent_util
from osgeo_utils.auxiliary.rectangle import GeoRectangle
from osgeo_utils.auxiliary.color_table import get_color_table, ColorTableLike
from osgeo_utils.auxiliary.gdal_argparse import GDALArgumentParser, GDALScript

GDALDataType = int

# create alphabetic list (lowercase + uppercase) for storing input layers
AlphaList = list(string.ascii_letters)

# set up some default nodatavalues for each datatype
DefaultNDVLookup = {gdal.GDT_Byte: 255, gdal.GDT_UInt16: 65535, gdal.GDT_Int16: -32768,
                    gdal.GDT_UInt32: 4294967293, gdal.GDT_Int32: -2147483647,
                    gdal.GDT_Float32: 3.402823466E+38, gdal.GDT_Float64: 1.7976931348623158E+308}

# tuple of available output datatypes names
GDALDataTypeNames = tuple(gdal.GetDataTypeName(dt) for dt in DefaultNDVLookup.keys())

""" Perform raster calculations with numpy syntax.
Use any basic arithmetic supported by numpy arrays such as +-* along with logical
operators such as >. Note that all files must have the same dimensions, but no projection checking is performed.

Keyword arguments:
    [A-Z]: input files
    [A_band - Z_band]: band to use for respective input file

Examples:
add two files together:
    Calc("A+B", A="input1.tif", B="input2.tif", outfile="result.tif")

average of two layers:
    Calc(calc="(A+B)/2", A="input1.tif", B="input2.tif", outfile="result.tif")

set values of zero and below to null:
    Calc(calc="A*(A>0)", A="input.tif", A_band=2, outfile="result.tif", NoDataValue=0)

work with two bands:
    Calc(["(A+B)/2", "A*(A>0)"], A="input.tif", A_band=1, B="input.tif", B_band=2, outfile="result.tif", NoDataValue=0)

sum all files with hidden noDataValue
    Calc(calc="sum(a,axis=0)", a=['0.tif','1.tif','2.tif'], outfile="sum.tif", hideNoData=True)
"""

def Calc(calc: MaybeSequence[str], outfile: Optional[PathLikeOrStr] = None, NoDataValue: Optional[Number] = None,
         type: Optional[Union[GDALDataType, str]] = None, format: Optional[str] = None,
         creation_options: Optional[Sequence[str]] = None, allBands: str = '', overwrite: bool = False,
         hideNoData: bool = False, projectionCheck: bool = False,
         color_table: Optional[ColorTableLike] = None,
         extent: Optional[Extent] = None, projwin: Optional[Union[Tuple, GeoRectangle]] = None,
         user_namespace: Optional[Dict]=None,
         debug: bool = False, quiet: bool = False, **input_files):

    if debug:
        print(f"gdal_calc.py starting calculation {calc}")

    # Single calc value compatibility
    if isinstance(calc, (list, tuple)):
        calc = calc
    else:
        calc = [calc]
    calc = [c.strip('"') for c in calc]

    creation_options = creation_options or []

    # set up global namespace for eval with all functions of gdal_array, numpy
    global_namespace = {key: getattr(module, key)
                        for module in [gdal_array, numpy] for key in dir(module) if not key.startswith('__')}

    if user_namespace:
        global_namespace.update(user_namespace)

    if not calc:
        raise Exception("No calculation provided.")
    elif not outfile and format.upper() != 'MEM':
        raise Exception("No output file provided.")

    if format is None:
        format = GetOutputDriverFor(outfile)

    if isinstance(extent, GeoRectangle):
        pass
    elif projwin:
        if isinstance(projwin, GeoRectangle):
            extent = projwin
        else:
            extent = GeoRectangle.from_lurd(*projwin)
    elif not extent:
        extent = Extent.IGNORE
    else:
        extent = extent_util.parse_extent(extent)

    compatible_gt_eps = 0.000001
    gt_diff_support = {
        GT.INCOMPATIBLE_OFFSET: extent != Extent.FAIL,
        GT.INCOMPATIBLE_PIXEL_SIZE: False,
        GT.INCOMPATIBLE_ROTATION: False,
        GT.NON_ZERO_ROTATION: False,
    }
    gt_diff_error = {
        GT.INCOMPATIBLE_OFFSET: 'different offset',
        GT.INCOMPATIBLE_PIXEL_SIZE: 'different pixel size',
        GT.INCOMPATIBLE_ROTATION: 'different rotation',
        GT.NON_ZERO_ROTATION: 'non zero rotation',
    }

    ################################################################
    # fetch details of input layers
    ################################################################

    # set up some lists to store data for each band
    myFileNames = []  # input filenames
    myFiles = []  # input DataSets
    myBands = []  # input bands
    myAlphaList = []  # input alpha letter that represents each input file
    myDataType = []  # string representation of the datatype of each input file
    myDataTypeNum = []  # datatype of each input file
    myNDV = []  # nodatavalue for each input file
    DimensionsCheck = None  # dimensions of the output
    Dimensions = []  # Dimensions of input files
    ProjectionCheck = None  # projection of the output
    GeoTransformCheck = None  # GeoTransform of the output
    GeoTransforms = []  # GeoTransform of each input file
    GeoTransformDiffer = False  # True if we have inputs with different GeoTransforms
    myTempFileNames = []  # vrt filename from each input file
    myAlphaFileLists = []  # list of the Alphas which holds a list of inputs

    # loop through input files - checking dimensions
    for alphas, filenames in input_files.items():
        if isinstance(filenames, (list, tuple)):
            # alpha is a list of files
            myAlphaFileLists.append(alphas)
        elif is_path_like(filenames) or isinstance(filenames, gdal.Dataset):
            # alpha is a single filename or a Dataset
            filenames = [filenames]
            alphas = [alphas]
        else:
            # I guess this alphas should be in the global_namespace,
            # It would have been better to pass it as user_namepsace, but I'll accept it anyway
            global_namespace[alphas] = filenames
            continue
        for alpha, filename in zip(alphas * len(filenames), filenames):
            if not alpha.endswith("_band"):
                # check if we have asked for a specific band...
                alpha_band = f"{alpha}_band"
                if alpha_band in input_files:
                    myBand = input_files[alpha_band]
                else:
                    myBand = 1

                myF_is_ds = not is_path_like(filename)
                if myF_is_ds:
                    myFile = filename
                    filename = None
                else:
                    myFile = open_ds(filename, gdal.GA_ReadOnly)
                if not myFile:
                    raise IOError(f"No such file or directory: '{filename}'")

                myFileNames.append(filename)
                myFiles.append(myFile)
                myBands.append(myBand)
                myAlphaList.append(alpha)
                dt = myFile.GetRasterBand(myBand).DataType
                myDataType.append(gdal.GetDataTypeName(dt))
                myDataTypeNum.append(dt)
                myNDV.append(None if hideNoData else myFile.GetRasterBand(myBand).GetNoDataValue())

                # check that the dimensions of each layer are the same
                myFileDimensions = [myFile.RasterXSize, myFile.RasterYSize]
                if DimensionsCheck:
                    if DimensionsCheck != myFileDimensions:
                        GeoTransformDiffer = True
                        if extent in [Extent.IGNORE, Extent.FAIL]:
                            raise Exception(
                                f"Error! Dimensions of file {filename} ({myFileDimensions[0]:d}, "
                                f"{myFileDimensions[1]:d}) are different from other files "
                                f"({DimensionsCheck[0]:d}, {DimensionsCheck[1]:d}).  Cannot proceed")
                else:
                    DimensionsCheck = myFileDimensions

                # check that the Projection of each layer are the same
                myProjection = myFile.GetProjection()
                if ProjectionCheck:
                    if projectionCheck and ProjectionCheck != myProjection:
                        raise Exception(
                            f"Error! Projection of file {filename} {myProjection} "
                            f"are different from other files {ProjectionCheck}.  Cannot proceed")
                else:
                    ProjectionCheck = myProjection

                # check that the GeoTransforms of each layer are the same
                myFileGeoTransform = myFile.GetGeoTransform(can_return_null=True)
                if extent == Extent.IGNORE:
                    GeoTransformCheck = myFileGeoTransform
                else:
                    Dimensions.append(myFileDimensions)
                    GeoTransforms.append(myFileGeoTransform)
                    if not GeoTransformCheck:
                        GeoTransformCheck = myFileGeoTransform
                    else:
                        my_gt_diff = extent_util.gt_diff(GeoTransformCheck, myFileGeoTransform, eps=compatible_gt_eps,
                                                         diff_support=gt_diff_support)
                        if my_gt_diff not in [GT.SAME, GT.ALMOST_SAME]:
                            GeoTransformDiffer = True
                            if my_gt_diff != GT.COMPATIBLE_DIFF:
                                raise Exception(
                                    f"Error! GeoTransform of file {filename} {myFileGeoTransform} is incompatible "
                                    f"({gt_diff_error[my_gt_diff]}), first file GeoTransform is {GeoTransformCheck}. "
                                    f"Cannot proceed")
                if debug:
                    print(
                        f"file {alpha}: {filename}, dimensions: "
                        f"{DimensionsCheck[0]}, {DimensionsCheck[1]}, type: {myDataType[-1]}")

    # process allBands option
    allBandsIndex = None
    allBandsCount = 1
    if allBands:
        if len(calc) > 1:
            raise Exception("Error! --allBands implies a single --calc")
        try:
            allBandsIndex = myAlphaList.index(allBands)
        except ValueError:
            raise Exception(f"Error! allBands option was given but Band {allBands} not found.  Cannot proceed")
        allBandsCount = myFiles[allBandsIndex].RasterCount
        if allBandsCount <= 1:
            allBandsIndex = None
    else:
        allBandsCount = len(calc)

    if extent not in [Extent.IGNORE, Extent.FAIL] and (
        GeoTransformDiffer or isinstance(extent, GeoRectangle)):
        # mixing different GeoTransforms/Extents
        GeoTransformCheck, DimensionsCheck, ExtentCheck = extent_util.calc_geotransform_and_dimensions(
            GeoTransforms, Dimensions, extent)
        if GeoTransformCheck is None:
            raise Exception("Error! The requested extent is empty. Cannot proceed")
        for i in range(len(myFileNames)):
            temp_vrt_filename, temp_vrt_ds = extent_util.make_temp_vrt(myFiles[i], ExtentCheck)
            myTempFileNames.append(temp_vrt_filename)
            myFiles[i] = None  # close original ds
            myFiles[i] = temp_vrt_ds  # replace original ds with vrt_ds

            # update the new precise dimensions and gt from the new ds
            GeoTransformCheck = temp_vrt_ds.GetGeoTransform()
            DimensionsCheck = [temp_vrt_ds.RasterXSize, temp_vrt_ds.RasterYSize]
        temp_vrt_ds = None

    ################################################################
    # set up output file
    ################################################################

    # open output file exists
    if outfile and os.path.isfile(outfile) and not overwrite:
        if allBandsIndex is not None:
            raise Exception("Error! allBands option was given but Output file exists, must use --overwrite option!")
        if len(calc) > 1:
            raise Exception(
                "Error! multiple calc options were given but Output file exists, must use --overwrite option!")
        if debug:
            print(f"Output file {outfile} exists - filling in results into file")

        myOut = open_ds(outfile, gdal.GA_Update)
        if myOut is None:
            error = 'but cannot be opened for update'
        elif [myOut.RasterXSize, myOut.RasterYSize] != DimensionsCheck:
            error = 'but is the wrong size'
        elif ProjectionCheck and ProjectionCheck != myOut.GetProjection():
            error = 'but is the wrong projection'
        elif GeoTransformCheck and GeoTransformCheck != myOut.GetGeoTransform(can_return_null=True):
            error = 'but is the wrong geotransform'
        else:
            error = None
        if error:
            raise Exception(
                f"Error! Output exists, {error}.  Use the --overwrite option "
                f"to automatically overwrite the existing file")

        myOutB = myOut.GetRasterBand(1)
        myOutNDV = myOutB.GetNoDataValue()
        myOutType = myOutB.DataType

    else:
        if outfile:
            # remove existing file and regenerate
            if os.path.isfile(outfile):
                os.remove(outfile)
            # create a new file
            if debug:
                print(f"Generating output file {outfile}")
        else:
            outfile = ''

        # find data type to use
        if not type:
            # use the largest type of the input files
            myOutType = max(myDataTypeNum)
        else:
            myOutType = type
            if isinstance(myOutType, str):
                myOutType = gdal.GetDataTypeByName(myOutType)

        # create file
        myOutDrv = gdal.GetDriverByName(format)
        myOut = myOutDrv.Create(
            os.fspath(outfile), DimensionsCheck[0], DimensionsCheck[1], allBandsCount,
            myOutType, creation_options)

        # set output geo info based on first input layer
        if not GeoTransformCheck:
            GeoTransformCheck = myFiles[0].GetGeoTransform(can_return_null=True)
        if GeoTransformCheck:
            myOut.SetGeoTransform(GeoTransformCheck)

        if not ProjectionCheck:
            ProjectionCheck = myFiles[0].GetProjection()
        if ProjectionCheck:
            myOut.SetProjection(ProjectionCheck)

        if NoDataValue is None:
            myOutNDV = DefaultNDVLookup[myOutType]  # use the default noDataValue for this datatype
        elif isinstance(NoDataValue, str) and NoDataValue.lower() == 'none':
            myOutNDV = None  # not to set any noDataValue
        else:
            myOutNDV = NoDataValue  # use the given noDataValue

        for i in range(1, allBandsCount + 1):
            myOutB = myOut.GetRasterBand(i)
            if myOutNDV is not None:
                myOutB.SetNoDataValue(myOutNDV)
            if color_table:
                # set color table and color interpretation
                if is_path_like(color_table):
                    color_table = get_color_table(color_table)
                myOutB.SetRasterColorTable(color_table)
                myOutB.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

            myOutB = None  # write to band

        if hideNoData:
            myOutNDV = None

    myOutTypeName = gdal.GetDataTypeName(myOutType)
    if debug:
        print(f"output file: {outfile}, dimensions: {myOut.RasterXSize}, {myOut.RasterYSize}, type: {myOutTypeName}")

    ################################################################
    # find block size to chop grids into bite-sized chunks
    ################################################################

    # use the block size of the first layer to read efficiently
    myBlockSize = myFiles[0].GetRasterBand(myBands[0]).GetBlockSize()
    # find total x and y blocks to be read
    nXBlocks = (int)((DimensionsCheck[0] + myBlockSize[0] - 1) / myBlockSize[0])
    nYBlocks = (int)((DimensionsCheck[1] + myBlockSize[1] - 1) / myBlockSize[1])
    myBufSize = myBlockSize[0] * myBlockSize[1]

    if debug:
        print(f"using blocksize {myBlockSize[0]} x {myBlockSize[1]}")

    # variables for displaying progress
    ProgressCt = -1
    ProgressMk = -1
    ProgressEnd = nXBlocks * nYBlocks * allBandsCount

    ################################################################
    # start looping through each band in allBandsCount
    ################################################################

    for bandNo in range(1, allBandsCount + 1):

        ################################################################
        # start looping through blocks of data
        ################################################################

        # store these numbers in variables that may change later
        nXValid = myBlockSize[0]
        nYValid = myBlockSize[1]

        # loop through X-lines
        for X in range(0, nXBlocks):

            # in case the blocks don't fit perfectly
            # change the block size of the final piece
            if X == nXBlocks - 1:
                nXValid = DimensionsCheck[0] - X * myBlockSize[0]

            # find X offset
            myX = X * myBlockSize[0]

            # reset buffer size for start of Y loop
            nYValid = myBlockSize[1]
            myBufSize = nXValid * nYValid

            # loop through Y lines
            for Y in range(0, nYBlocks):
                ProgressCt += 1
                if 10 * ProgressCt / ProgressEnd % 10 != ProgressMk and not quiet:
                    ProgressMk = 10 * ProgressCt / ProgressEnd % 10
                    from sys import version_info
                    if version_info >= (3, 0, 0):
                        exec('print("%d.." % (10*ProgressMk), end=" ")')
                    else:
                        exec('print 10*ProgressMk, "..",')

                # change the block size of the final piece
                if Y == nYBlocks - 1:
                    nYValid = DimensionsCheck[1] - Y * myBlockSize[1]
                    myBufSize = nXValid * nYValid

                # find Y offset
                myY = Y * myBlockSize[1]

                # create empty buffer to mark where nodata occurs
                myNDVs = None

                # make local namespace for calculation
                local_namespace = {}

                val_lists = defaultdict(list)

                # fetch data for each input layer
                for i, Alpha in enumerate(myAlphaList):

                    # populate lettered arrays with values
                    if allBandsIndex is not None and allBandsIndex == i:
                        myBandNo = bandNo
                    else:
                        myBandNo = myBands[i]
                    myval = gdal_array.BandReadAsArray(myFiles[i].GetRasterBand(myBandNo),
                                                       xoff=myX, yoff=myY,
                                                       win_xsize=nXValid, win_ysize=nYValid)
                    if myval is None:
                        raise Exception(f'Input block reading failed from filename {filename[i]}')

                    # fill in nodata values
                    if myNDV[i] is not None:
                        # myNDVs is a boolean buffer.
                        # a cell equals to 1 if there is NDV in any of the corresponding cells in input raster bands.
                        if myNDVs is None:
                            # this is the first band that has NDV set. we initializes myNDVs to a zero buffer
                            # as we didn't see any NDV value yet.
                            myNDVs = numpy.zeros(myBufSize)
                            myNDVs.shape = (nYValid, nXValid)
                        myNDVs = 1 * numpy.logical_or(myNDVs == 1, myval == myNDV[i])

                    # add an array of values for this block to the eval namespace
                    if Alpha in myAlphaFileLists:
                        val_lists[Alpha].append(myval)
                    else:
                        local_namespace[Alpha] = myval
                    myval = None

                for lst in myAlphaFileLists:
                    local_namespace[lst] = val_lists[lst]

                # try the calculation on the array blocks
                this_calc = calc[bandNo - 1 if len(calc) > 1 else 0]
                try:
                    myResult = eval(this_calc, global_namespace, local_namespace)
                except:
                    print(f"evaluation of calculation {this_calc} failed")
                    raise

                # Propagate nodata values (set nodata cells to zero
                # then add nodata value to these cells).
                if myNDVs is not None and myOutNDV is not None:
                    myResult = ((1 * (myNDVs == 0)) * myResult) + (myOutNDV * myNDVs)
                elif not isinstance(myResult, numpy.ndarray):
                    myResult = numpy.ones((nYValid, nXValid)) * myResult

                # write data block to the output file
                myOutB = myOut.GetRasterBand(bandNo)
                if gdal_array.BandWriteArray(myOutB, myResult, xoff=myX, yoff=myY) != 0:
                    raise Exception('Block writing failed')
                myOutB = None  # write to band

    # remove temp files
    for idx, tempFile in enumerate(myTempFileNames):
        myFiles[idx] = None
        os.remove(tempFile)

    gdal.ErrorReset()
    myOut.FlushCache()
    if gdal.GetLastErrorMsg() != '':
        raise Exception('Dataset writing failed')

    if not quiet:
        print("100 - Done")

    return myOut


def doit(opts):
    kwargs = vars(opts)
    if 'outF' in kwargs:
        kwargs["outfile"] = kwargs.pop('outF')
    return Calc(**kwargs)


class GDALCalc(GDALScript):
    def __init__(self):
        super().__init__()
        self.title = 'Raster calculator with numpy syntax'
        self.description = textwrap.dedent('''\
            Use any basic arithmetic supported by numpy arrays such as +, -, *, and
            along with logical operators such as >.
            Note that all files must have the same dimensions (unless extent option is used),
            but no projection checking is performed (unless projectionCheck option is used).''')
        # add an explicit --help option because the standard -h/--help option is not valid as -h is an alpha option
        self.add_help = '--help'
        self.optfile_arg = "--optfile"

        self.add_example('add two files together',
                         '-A input1.tif -B input2.tif --outfile=result.tif --calc="A+B"')
        self.add_example('average of two layers',
                         '-A input.tif -B input2.tif --outfile=result.tif --calc="(A+B)/2"')
        self.add_example('set values of zero and below to null',
                         '-A input.tif --outfile=result.tif --calc="A*(A>0)" --NoDataValue=0')
        self.add_example('using logical operator to keep a range of values from input',
                         '-A input.tif --outfile=result.tif --calc="A*logical_and(A>100,A<150)"')
        self.add_example('work with multiple bands',
                         '-A input.tif --A_band=1 -B input.tif --B_band=2 '
                         '--outfile=result.tif --calc="(A+B)/2" --calc="A*logical_and(A>100,A<150)"')

    @staticmethod
    def add_alpha_args(parser, is_help):
        if is_help:
            alpha_list = ['A']  # we don't want to make help with all the full alpha list, as it's too long...
        else:
            alpha_list = AlphaList
        for alpha in alpha_list:
            try:
                band = alpha + '_band'
                alpha_arg = '-' + alpha
                band_arg = '--' + band
                parser.add_argument(alpha_arg, action="extend", nargs='*', type=str,
                                    help="input gdal raster file, you can use any letter [a-z, A-Z]",
                                    metavar='filename')
                parser.add_argument(band_arg, action="extend", nargs='*', type=int,
                                    help=f"number of raster band for file {alpha} (default 1)", metavar='n')
            except argparse.ArgumentError:
                pass

    def get_parser(self, argv) -> GDALArgumentParser:
        parser = self.parser
        parser.add_argument("--calc", dest="calc", type=str, required=True, nargs='*', action="extend",
                            help="calculation in numpy syntax using +-/* or any numpy array functions (i.e. log10()). "
                                 "May appear multiple times to produce a multi-band file", metavar="expression")

        is_help = '--help' in argv
        self.add_alpha_args(parser, is_help)

        parser.add_argument("--outfile", dest="outfile", required=True, metavar="filename",
                            help="output file to generate or fill")
        parser.add_argument("--NoDataValue", dest="NoDataValue", type=float, metavar="value",
                            help="output nodata value (default datatype specific value)")
        parser.add_argument("--hideNoData", dest="hideNoData", action="store_true",
                            help="ignores the NoDataValues of the input rasters")
        parser.add_argument("--type", dest="type", type=str, metavar="datatype", choices=GDALDataTypeNames,
                            help="output datatype")
        parser.add_argument("--format", dest="format", type=str, metavar="gdal_format",
                            help="GDAL format for output file")
        parser.add_argument(
            "--creation-option", "--co", dest="creation_options", default=[], action="append", metavar="option",
            help="Passes a creation option to the output format driver. Multiple "
                 "options may be listed. See format specific documentation for legal "
                 "creation options for each format.")
        parser.add_argument("--allBands", dest="allBands", type=str, default="", metavar="[a-z, A-Z]",
                            help="process all bands of given raster [a-z, A-Z]")
        parser.add_argument("--overwrite", dest="overwrite", action="store_true",
                            help="overwrite output file if it already exists")
        parser.add_argument("--debug", dest="debug", action="store_true", help="print debugging information")
        parser.add_argument("--quiet", dest="quiet", action="store_true", help="suppress progress messages")

        parser.add_argument("--color-table", type=str, dest="color_table", help="color table file name")

        group = parser.add_mutually_exclusive_group()
        group.add_argument("--extent", dest="extent",
                           choices=[e.name.lower() for e in Extent],
                           help="how to treat mixed geotrasnforms")
        group.add_argument("--projwin", dest="projwin", type=float, nargs=4, metavar=('ulx', 'uly', 'lrx', 'lry'),
                           help="extent corners given in georeferenced coordinates")

        parser.add_argument("--projectionCheck", dest="projectionCheck", action="store_true",
                            help="check that all rasters share the same projection")

        # parser.add_argument('--namespace', dest='user_namespace', action='extend', nargs='*', type=str)

        return parser

    def doit(self, **kwargs):
        return Calc(**kwargs)

    def augment_kwargs(self, kwargs) -> dict:
        # create the input_files dict from the alpha arguments ('-a' and '--a_band')
        input_files = {}
        input_bands = {}
        for alpha in AlphaList:
            if alpha in kwargs:
                alpha_val = kwargs[alpha]
                del kwargs[alpha]
                if alpha_val is not None:
                    alpha_val = [s.strip('"') for s in alpha_val]
                    input_files[alpha] = alpha_val if len(alpha_val) > 1 else alpha_val[0]
            band_key = alpha + '_band'
            if band_key in kwargs:
                band_val = kwargs[band_key]
                del kwargs[band_key]
                if band_val is not None:
                    input_bands[band_key] = band_val if len(band_val) > 1 else band_val[0]
        kwargs = {**kwargs, **input_files, **input_bands}
        # kwargs['input_files'] = input_files

        return kwargs


def gdal_calcc(argv):
    argv = gdal.GeneralCmdLineProcessor(argv)
    return GDALCalc().main(argv)
