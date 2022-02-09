from RFSM_python.utils import *



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

def preprocess_RFSM(DTM_file,COTA_ESTUDIO,path_project,epsg_n,CoastLine=None):
    path_output = path_project+'ascii/'
    [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(DTM_file)
    
    if CoastLine!=None:
        # Cortamos con la línea de Costa
        extract_coastline(CoastLine,DTM_file,path_output+'DTM_LC.asc',cellsize)
        DTM=np.loadtxt(path_output+'DTM_LC.asc',skiprows=6)
        [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(path_output+'DTM_LC.asc')
    else:
        DTM=np.loadtxt(DTM,skiprows=6)
        [ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value] = header_ascii(DTM_file)
        
    ### Cargamos el DTM
    
    
    xx = np.linspace(xllcorner, xllcorner + (ncols-1)*cellsize, ncols)
    yy = np.linspace(yllcorner, yllcorner + (nrows-1)*cellsize, nrows)
    [XX,YY] = np.meshgrid(xx,yy)
    
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
    gdal_polygonize(path_output+'floodareas.asc'+' '+path_project+'shp/Final_CUT.shp -b 1 -f "ESRI Shapefile" Final_CUT CID')
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