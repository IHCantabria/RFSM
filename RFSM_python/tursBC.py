# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

from RFSM_python.utils import *

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