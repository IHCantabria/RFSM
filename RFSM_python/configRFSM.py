# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

from RFSM_python.utils import *

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