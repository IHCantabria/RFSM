# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

from RFSM_python.utils import *

def tusrBCFlowLevel(table_points,table_inputs,path_project,TestDesc,Table_impact_zone,
                     izcoast,BCSetID,BCTypeID_COAST,
                    raw_q=True, point_X_river = None ,point_Y_river = None ,hidrograma=None):
    
    """La siguiente función nos permite generar los ficheros de condición de contorno de RFSM.
    
       Parámetros:
       ---------------------
       table_points       : pandas dataframe. Dataframe con las coordenadas y características de los puntos donde asignaron las dinámicas
       table_inputs        : pandas dataframe. Dataframe que contiene el temporal reconstruido o hidrograma reconstruido.
       path_project       : string. path donde se encuentra el proyecto de RFSM
       TestDesc           : string. Nombre de la simulación
       Table_impact_zone  : pandas dataframe.  Dataframe con el conjunto de impact zones
       izcoast            : pandas dataframe.  Dataframe con el conjunto de impact zones donde se asigna condición de contorno costera
       BCSetID            : int. Identificador de la simulación
       BCTypeID_COAST     : int. Tipo de condición de contorno # 1 overtopping; # 2 level;
       raw_q              : True or False: Si existe un punto donde se va introducir un caudal en un cauce poner en True
       point_X_river      : float. Si existe un punto donde se va introducir un caudal en un cauce añadir 
       point_Y_river      : float. Si existe un punto donde se va introducir un caudal en un cauce poner en True
       hidrograma         : array. Si existe un punto donde se va introducir un caudal en un cauce añadir hidrograma 
       
        Salidas:
        ---------------------
        Fichero tusrBCFlowLevel necesario para la ejecución de RFSM.
       
       
       
        """
    
    Table_impact_zone_edit =Table_impact_zone.copy()
    Table_impact_zone_edit.loc[:,'BCTypeID'] = 0
    
    table_points = gpd.read_file(table_points)
   
    n_periods = len(table_inputs)
    
    Table_impact_zone_edit.loc[izcoast.index,'BCTypeID'] = BCTypeID_COAST
    
    if raw_q==True:
        
        dist = np.sqrt((point_X_river-Table_impact_zone_edit.iloc[:,2])**2+(point_Y_river-Table_impact_zone_edit.iloc[:,3])**2)
        point_select=np.argmin(dist)
        
        Table_impact_zone_edit.loc[point_select,'BCTypeID'] = 10
        
    Results_TWL = pd.DataFrame(index=np.arange(0,sum(Table_impact_zone_edit.BCTypeID.values>0)*n_periods),columns=['BCSetID', 'BCTypeID', 'IZID', 'Time', 'BCValue'])
    
    Table_impact_zone_edit_2=Table_impact_zone_edit[Table_impact_zone_edit.BCTypeID!=0].copy()
    
    it=0
    for iDZ in tqdm.tqdm(range(len(Table_impact_zone_edit_2))):
        if (Table_impact_zone_edit_2['BCTypeID'].iloc[iDZ]==2) or (Table_impact_zone_edit_2['BCTypeID'].iloc[iDZ]==1):

            x_pointIZID = Table_impact_zone_edit_2[' MidX'].values[iDZ]
            y_pointIZID = Table_impact_zone_edit_2[' MidY'].values[iDZ]
            IZID = Table_impact_zone_edit_2.index[iDZ]

            dist = np.sqrt((x_pointIZID-table_points.CX.values)**2+(y_pointIZID-table_points.CY.values)**2)
            point_select=np.argmin(dist)
            
            cITr = table_inputs.iloc[:,point_select].values
            
            if BCTypeID_COAST ==1:
                cITr = cITr*izcoast.nCells[IZID]*cellsize
            else:
                cITr = cITr-izcoast.minH[IZID]
                cITr[cITr<0]=0
                
            
            Results_TWL.iloc[it:it+n_periods,0] = BCSetID
            Results_TWL.iloc[it:it+n_periods,1] = BCTypeID_COAST
            Results_TWL.iloc[it:it+n_periods,2] = IZID
            Results_TWL.iloc[it:it+n_periods,3] = np.linspace(0,3600*len(cITr),len(cITr))
            Results_TWL.iloc[it:it+n_periods,4] = cITr

            it=it+n_periods

        elif Table_impact_zone_edit_2['BCTypeID'].iloc[iDZ]==10:
            hidrograma = table_inputs.values
            print('Río')
            Results_TWL.iloc[it:it+n_periods,0] = BCSetID
            Results_TWL.iloc[it:it+n_periods,1] = 1
            Results_TWL.iloc[it:it+n_periods,2] = IZID
            Results_TWL.iloc[it:it+n_periods,3] = np.linspace(0,3600*len(hidrograma),len(hidrograma))
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