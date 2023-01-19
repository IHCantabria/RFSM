# RFSM
<img src="https://ihcantabria.com/wp-content/uploads/2020/07/Logo-IHCantabria-Universidad-Cantabria-cmyk.png" alt="drawing" width="500"/>
🌎 Mediante los siguientes códigos podremos ejecutar RFSM y configurar todos los datos de partida.
El preproceso se ha sustituido por funciones de python por lo que se puede ejecutar en cualquier entorno.

## Documentación

Contiene el código del modelo RFSM junto con los scripts en python que permiten realizar el preproceso y ejecución del modelo.
| Directorio | Contenidos |
| :--------: | ------- |
| [python](RFSM_python) | Las funciones contenidas en esta carpeta permiten tanto la configuración como ejecución del modelo, además de obtener los resultados una vez ejecutado el modelo.<br>|
| [notebooks](notebooks) | Contiene un ejemplo en el que se relizan diferentes procesos para simular un caso determinado.<br>|  
| [compile](compile) | En esta carpeta se encuentran los ficheros **.py** que se compilaron para ejecutar desde Matlab. En el *supporting material* de este directorio se encuentran los ficheros ya compilados.<br>| 

## Instalación
Para realizar la instalación de la librería es necesario descargar y descomprimir el fichero.
Es recomendable crear un enviroment de conda para realizar la instalación.
Ejecutaremos las siguientes líneas:

```python
conda env create -f RFSM.yml
conda create --name RFSM --channel=conda-forge geopandas cartopy shapely gdal xarray netcdf4 hdf5 libgdal jupyterlab scikit-image statsmodels seaborn tqdm pyproj fiona rasterio rasterstats
```

Una vez descargado, desde la terminal iremos al directorio donde se sencuentra __setup.py__ y ejecutaremos la siguiente linea:

```python


pip install -e.
```
De esta forma ya estará instalada la librería

## Ejemplos
A continuación se muestran diferentes ejemplos de ejecución del modelo a través de los métodos comentados anteriormente.

### Ejemplo 1
Para las ejecuciones ejecutadas de python se utilizará como ejemplo el notebook [RFSM-Example.ipynb](./notebooks/RFSM-Example.ipynb)

El material necesario puede ser descargado a través de la cuenta de UNICAN en: [RFSM Github](https://unican-my.sharepoint.com/:f:/g/personal/navass_unican_es/Eo43HV8hgUNIsaz4yF_VZVYBsv5YMVO5Zapo4Qdb8GoIeA?e=xA1bAq)

### Ejemplo 2
Generalmente se usa matlab para la ejecución de RFSM, por lo que se han compilado diversos procesos que permiten mejorar el rendimiento en la obtención de las manchas de inundación y juntar todos los resultados de diversas mallas para un evento determinado en un mismo raster.
Para el uso de los ficheros compilados a través de Matlab se utilizarán la siguientes líneas. Es necesario que los ficheros .exe (**para windows**) o -Unix (**para Linux**) se encuentren en el mismo directorio en el que se esté ejecutando el código de Matlab.

Los ejecutables pueden descargarse en el siguiente enlace: [Releases](https://github.com/IHCantabria/RFSM/releases/tag/v1.0.1)

Cuando se quiere exportar los resultados que se han obtenido con RFSM a raster, se puede utilizar la siguiente expresión una vez ha acabado la simulación
```Matlab
 path_project = 'Directorio donde se encuentra el proyecto'
 TestDesc     = 'TestDesc'
 topography   = 'Fichero .asc del MDT'
 izid2_asc    = 'Fichero .asc de izid2'
 Results      = 'BCSetID'
 src          = 'Proyección en formato código EPSG'
 str  = ['ExportGIF.exe ',path_project,' ',TestDesc,' ',topography,' ',izid2_asc,' ',Results,' ',src];
 system(str);
```
Si se está trabajando en equipos con sistema operativo LInux
```Matlab
str  = ['ExportTIF-Unix ',path_project,' ',TestDesc,' ',topography,' ',izid2_asc,' ',Results,' ',src];
```

Cuando se tienen diferentes mallas y se quiere juntar el resultado en un único ráster se utiliza el siguiente código:
```Matlab
path_project = 'Directorio donde se encuentra el proyecto'
str  = ['mosaic_TIF.exe ',path_project];
system(str);
```
Si se está trabajando en equipos con sistema operativo LInux
```Matlab
str  = ['MosaicTIF-Unix ',path_project];
```

# Referencias 
<a id="1">[1]</a> 
Gomes da Silva, P., Coco, G., Garnier, R., & Klein, A. H. F. (2020). 
On the prediction of runup, setup and swash on beaches. 
Earth-Science Reviews, 204(February), 103148. https://doi.org/10.1016/j.earscirev.2020.103148.

# Licencia
Usa exclusivo para miembros de IH Cantabria.

Copyright 2021 Instituto de Hidráulica Ambiental "IHCantabria". Universidad de Cantabria.

