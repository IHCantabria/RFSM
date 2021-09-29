# RFSM
🌎 Mediante los siguientes códigos podremos ejecutar RFSM y configurar todos los datos de partida.
El preproceso se ha sustituido por funciones de python por lo que se puede ejecutar en cualquier entorno.

## Documentación

Contiene el código del modelo RFSM junto con los scripts en python que permiten realizar el preproceso y ejecución del modelo.
| Directorio | Contenidos |
| :--------: | ------- |
| [python](https://github.com/IHCantabria/RFSM/tree/master/RFSM_python) | Las siguientes funciones permiten tanto la configuración como ejecución del modelo, además de obtener los resultados una vez ejecutado el modelo.<br> [RFSM_Ejecution.py](https://github.com/IHCantabria/RFSM/blob/master/RFSM_python/RFSM_funtions.py) contiene las funciones en python para realizar el preproceso, la configuración y la ejecución del modelo. Este fichero permite instalar todas las funciones a través de pip. <br> [ExportTIF.py](https://github.com/IHCantabria/RFSM/blob/master/RFSM_python/ExportTIF.py) fichero ejecutable que permite covertir los resultados de RFSM en formato TIF lo que permite reducir notablemente el tiempo de cómputo  <br> [mosaic_TIF.py](https://github.com/IHCantabria/RFSM/blob/master/RFSM_python/mosaic_TIF.py) fichero ejecutable que permite realizar la unión de los resultados de todas las mallas de configuración del modelo. |
| [exe](https://github.com/IHCantabria/RFSM/tree/master/RFSM_python) | Debido a que diversos usuarios usan Matlab y haber tenido diferentes problemas con la instalación de python, se ha compilado todo el dódigo para ejecutar los ficheros [ExportTIF.py](https://github.com/IHCantabria/RFSM/blob/master/RFSM_python/ExportTIF.py) y [mosaic_TIF.py](https://github.com/IHCantabria/RFSM/blob/master/RFSM_python/mosaic_TIF.py) desde la terminal o desde el propio Matlab.|
| [notebooks](https://github.com/IHCantabria/RFSM/tree/master/notebooks) | contiene un ejemplo en el que se relizan diferentes procesos para simular un caso determinado.|                                                                                                                                                            

## Instalación
Para realizar la instalación de la librería es necesario descargar y descomprimir el fichero.
Es recomendable crear un enviroment de conda para realizar la instalación.
Ejecutaremos las siguientes líneas:

```python
conda create --name RFSM
conda activate RFSM
conda install -c conda-forge geopandas
conda install -c conda-forge gdal
```

Una vez descargado, desde la terminal iremos al directorio donde se sencuentra __setup.py__ y ejecutaremos la siguiente linea:

```python
pip install -e.
```
De esta forma ya estará instalada la librería

## Ejemplos
A continuación se muestran diferentes ejemplos de ejecución del modelo a través de los métodos comentados anteriormente.
### Ejemplo 1
Para las ejecuciones ejecutadas de python se utilizará como ejemplo el notebook [RFSM-Example.ipynb](https://github.com/IHCantabria/RFSM/tree/master/notebooks/RFSM-Example.ipynb)

### Ejemplo 2
Para el uso de los ficheros compilados a través de Matlab se utilizarán la siguientes líneas. Es necesario que los ficheros .exe se encuentren en el mismo directorio en el que se esté ejecutando el código de Matlab.

Cuando se quiere exportar los resultados que se han obtenido con RFSM a raster, se puede utilizar la siguiente expresión una vez ha acabado la simulación
```Matlab
 path_project = 'Directorio donde se encuentra el proyecto'
 TestDesc     = 'TestDesc'
 Results      = 'BCSetID'
 src          = 'Proyección en formato EPSG'
 str  = ['ExportGIF.exe ',path_project,' ',TestDesc,' ',int2str(BCSetID)];
 system(str);
```
Cuando se ejecuta este código genera un fichero que ocupa demasiado y es innecesario, por tanto se elimina.
```Matlab
 path_test = 'path donde se han generado los ficheros de la simulación'
 path_export = fullfile(path_test,'export\');
 delete([path_export,['Result_Level_IZID',int2str(BCSetID),'.tif']]);
```
### Ejemplo 3
Cuando se tienen diferentes mallas y se quiere juntar el resultado en un único ráster se utiliza el siguiente código:
```Matlab
path_project = 'Directorio donde se encuentra el proyecto'
str  = ['mosaic_TIF.exe ',path_project];
system(str);
```

