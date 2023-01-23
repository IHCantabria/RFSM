# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

from RFSM_python.utils import *


        
def extract_coastline(shp_coastline,path_DTM,path_DTM_new, cellsize):
    gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
    drv = gdal.Open(path_DTM)
    OutTile = gdal.Warp(path_DTM_new,
                        [drv], 
                        format = 'GTiff',
                        xRes = cellsize, 
                        yRes = cellsize,
                        cutlineDSName=shp_coastline, 
                        cropToCutline=True, 
                        dstNodata = -9999)

    OutTile = None 
    



def line_cost_raster(raster_file,raster_coast_line,path_output):

    raster_CL = gdal.Open(raster_coast_line)
    band0 = raster_CL.GetRasterBand(1)
    data0 = BandReadAsArray(band0)
    data0[data0>=0]= 1

    raster = gdal.Open(raster_file)
    band = raster.GetRasterBand(1)
    data = BandReadAsArray(band)
    data[data0==1]= 1
    data[data0!=1]= 0

    from skimage.segmentation import find_boundaries
    data_2 = data.copy()*0+-9999
    data_2[find_boundaries(data,mode ='inner')]=1

    driver = gdal.GetDriverByName("GTiff")
    dsOut = driver.Create(path_output+'coast_line.tif'
                      ,raster.RasterXSize, raster.RasterYSize, 1, band.DataType)
    CopyDatasetInfo(raster,dsOut)
    bandOut=dsOut.GetRasterBand(1)
    bandOut.Fill(-9999)
    bandOut.SetNoDataValue(-9999)
    BandWriteArray(bandOut, data_2)

    bandOut = None
    dsOut = None
    
    pos = np.where(data_2==1)
    t = 0
    for i in range(len(pos[0])):
        data_2[pos[0][i],pos[1][i]] = t
        t= t+1
    driver = gdal.GetDriverByName("GTiff")
    dsOut = driver.Create(path_output+'coast.tif'
                      ,raster.RasterXSize, raster.RasterYSize, 1, band.DataType)
    CopyDatasetInfo(raster,dsOut)
    bandOut=dsOut.GetRasterBand(1)
    bandOut.Fill(-9999)
    bandOut.SetNoDataValue(-9999)
    BandWriteArray(bandOut, data_2)

    band = None
    raster = None
    bandOut = None
    dsOut = None       

def impact_zones_process(path_project,DTM_LC,cellsize,COTA_ESTUDIO,new_coast_Line=True):
    """ La siguiente función nos permite generar las tablas de ipact zones e identificar las lineas de la costa que serán codición de contorno.
        A continuación se explican cada uno de los parámetros de entrada necesarios y los outputs generados
        
        Parámetros:
        ---------------------
        path_project            : string. path donde se encuentran las carpetas de RFSM para el caso de estudio.
        DTM_LC                  : string. MDT al cual se le han eliminado la zona de mar.
        cellsize                : int. Tamaño de celda del MDT.
        COTA_ESTUDIO            : int. Cota a la que se desea cortar el terreno considerada inundable.
        new_coast_Line          : True or False. En el caso en el que se haya modificado la izid2 no es necesario volver a generar la línea de costa, en ese caso poner en False
       
        Salidas:
        ---------------------
        Table_impact_zone       : dataframe. Tabla de impact zones
        izcoast                 : dataframe. Tabla de impact zones identificadas como costa.
        Files csv de las impact zones y de las impact zones identificadas como impact zone de las costa.
    
    """
    start = time.time()
    izd1_r = path_project+'ascii/check/izid1.asc'
    izd2_r = path_project+'ascii/check/izid2.asc'
    #[ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(DTM_LC)
    if new_coast_Line == True:
        dtm_lc_asc =  gdal.Open(DTM_LC)
        band = dtm_lc_asc.GetRasterBand(1)
        dtm = BandReadAsArray(band)
        dtm[dtm<COTA_ESTUDIO] = -9999
        dtm[dtm>=COTA_ESTUDIO] = 1

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
        
        gdal_polygonize(path_project+'ascii/DTM_OC.tif'+' '+path_project+'shp/DTM_OC.shp -b 1 -f "ESRI Shapefile" DTM_OC IZID')
        
    
    
    gdal_polygonize(izd1_r+' '+path_project+'shp/izid1.shp -b 1 -f "ESRI Shapefile" izid1 IZID')
    gdal_polygonize(izd2_r+' '+path_project+'shp/izid2.shp -b 1 -f "ESRI Shapefile" izid2 IZID')
    
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
    #FC = explode(path_project+'shp/Final_cut_complet.shp')
    FC = FC.explode(ignore_index=True)
    FC = FC.geometry.apply(lambda p: close_holes(p))
    FC.to_file(path_project+'shp/Final_cut_complet_F.shp')

    extract_coastline(path_project+'shp/Final_cut_complet_F.shp',DTM_LC,path_project+'ascii/DTM_LC_2.tif', cellsize)
    rasterize (path_project+'shp/Final_cut_complet_F.shp','FID',path_project+'ascii/DTM_LC_2.tif',path_project+'ascii/CL_ras.tif')
        
    line_cost_raster(path_project+'ascii/DTM_LC_2.tif',path_project+'ascii/CL_ras.tif',path_project+'ascii/')
    gdal_polygonize(path_project+'ascii/coast.tif'+' '+path_project+'shp/coast_IH.shp -b 1 -f "ESRI Shapefile" coast_IH CID')
    
#     createBuffer(path_project+'shp/Final_cut_complet.shp', path_project+'shp/Final_cut_Buffer.shp', cellsize)
    
#     fb_A = gpd.read_file(path_project+'shp/coast_for_buff.shp')
#     fb_B = gpd.read_file(path_project+'shp/coastline_Buffer.shp')
    
#     buffer_inters=gpd.sjoin(fb_A,fb_B , op='intersects')
#     buffer_inters.to_file(path_project+'shp/coast_IH.shp')
      
    
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
    ZE = zonal_stats(path_project+'shp/izcoast.shp',
                     path_project+'ascii/coast_line.tif', 
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

def izcoast_modify(path_project,izcoast_shp,izcoast_table,output):
    """ La siguiente función nos permite generar las tablas de ipact zones e identificar las lineas de la costa que serán codición de contorno.
        A continuación se explican cada uno de los parámetros de entrada necesarios y los outputs generados
        
        Parámetros:
        ---------------------
        path_project            : string. path donde se encuentran las carpetas de RFSM para el caso de estudio.
        izcoast_shp             : string. path donde se encuentra el shape de izcoast modificado-
        izcoast_table           : dataframe. Tabla de izcoast.
        output                  : string. path con el nombre del fichero de la tabla de izcoast modificada.
        
        Salidas:
        ---------------------
        Table_impact_zone       : dataframe. Tabla de impact zones
        izcoast                 : dataframe. Tabla de impact zones identificadas como costa.
        Files csv de las impact zones y de las impact zones identificadas como impact zone de las costa.
    
    """
    izcoast_table_=pd.read_csv(izcoast_table,index_col=0)
    iz_shp=gpd.read_file(izcoast_shp)
    izcoast=izcoast_table_.loc[iz_shp.loc[:,'IZID'].values,:].copy()
    
    izcoast.to_csv(output)