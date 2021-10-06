# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

from RFSM_python.utils import *

def execute_RFSM(path_project,TestDesc,TestID):
    """La siguiente función lanzar la ejecución de RFSM

    Parámetros:
    ---------------------
    path_project           : string. Directorio donde se encuentra las carpetas de configuración de RFSM
    TestDesc               : string. Nombre de la simulación
    TestID                 : int. ID de la simulación

    """
    os.chdir(path_project+'tests/'+TestDesc+'/Input_xml/')
    try:
        test = os.uname()
        if test[0] == "Linux":
            os.system(path_project+'bin/RFSM/Windows7_x86/RFSM_Hydrodynamic.exe '+'input_'+str(TestID)+'.xml')
    except AttributeError:
        print("Assuming windows!")
        os.system(path_project+'bin/RFSM/Windows7_x86/RFSM_Hydrodynamic.exe '+'input_'+str(TestID)+'.xml')


def export_result_RFSM(path_project,TestDesc,Results,src):
    """La siguiente función permite transformar los ficheros de SLR de Matlab en ficheros NETCDF

    Parámetros:
    ---------------------
    path_project           : string. Directorio donde se encuentra las carpetas de configuración de RFSM
    TestDesc               : string. Nombre de la simulación
    TestID                 : int. ID de la simulación
    Results                : string. Número de la simulación y número de la carpeta donde se guardan los resultados en csv
    
    Salidas:
    ---------------------
    File                   : tif. fichero tif de resultados

    """
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
    dsOut = driver.Create(path_project + 'tests/'+TestDesc+'/export/MaxLevel_'+TestDesc+'.tif'
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
    
    gdal_edit((" -stats -a_srs EPSG:"+str(src)+" -a_nodata -9999 " + path_project + 'tests/'+TestDesc+'/export/MaxLevel_'+TestDesc+'.tif').split(" "))