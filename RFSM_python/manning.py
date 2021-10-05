# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

from RFSM_python.utils import *

def create_file_manning(path_project,raster_rugos,BCSetID,TestDesc,Table_impact_zone):
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
    tusrlZManning.to_csv(path_project+'ascii/tusrIZManning.csv',index=False)