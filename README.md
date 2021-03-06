# RFSM
<img src="https://ihcantabria.com/wp-content/uploads/2020/07/Logo-IHCantabria-Universidad-Cantabria-cmyk.png" alt="drawing" width="500"/>
馃寧 Mediante los siguientes c贸digos podremos ejecutar RFSM y configurar todos los datos de partida.
El preproceso se ha sustituido por funciones de python por lo que se puede ejecutar en cualquier entorno.

## Documentaci贸n

Contiene el c贸digo del modelo RFSM junto con los scripts en python que permiten realizar el preproceso y ejecuci贸n del modelo.
| Directorio | Contenidos |
| :--------: | ------- |
| [python](RFSM_python) | Las funciones contenidas en esta carpeta permiten tanto la configuraci贸n como ejecuci贸n del modelo, adem谩s de obtener los resultados una vez ejecutado el modelo.<br>|
| [notebooks](notebooks) | Contiene un ejemplo en el que se relizan diferentes procesos para simular un caso determinado.<br>|  
| [compile](compile) | En esta carpeta se encuentran los ficheros **.py** que se compilaron para ejecutar desde Matlab. En el *supporting material* de este directorio se encuentran los ficheros ya compilados.<br>| 

## Instalaci贸n
Para realizar la instalaci贸n de la librer铆a es necesario descargar y descomprimir el fichero.
Es recomendable crear un enviroment de conda para realizar la instalaci贸n.
Ejecutaremos las siguientes l铆neas:

```python
conda env create -f RFSM.yml
conda create --name RFSM --channel=conda-forge geopandas cartopy shapely gdal xarray netcdf4 hdf5 libgdal jupyterlab scikit-image statsmodels seaborn tqdm pyproj fiona rasterio rasterstats
```

Una vez descargado, desde la terminal iremos al directorio donde se sencuentra __setup.py__ y ejecutaremos la siguiente linea:

```python


pip install -e.
```
De esta forma ya estar谩 instalada la librer铆a

## Ejemplos
A continuaci贸n se muestran diferentes ejemplos de ejecuci贸n del modelo a trav茅s de los m茅todos comentados anteriormente.

### Ejemplo 1
Para las ejecuciones ejecutadas de python se utilizar谩 como ejemplo el notebook [RFSM-Example.ipynb](./notebooks/RFSM-Example.ipynb)

El material necesario puede ser descargado a trav茅s de la cuenta de UNICAN en: [RFSM Github](https://unican-my.sharepoint.com/:f:/g/personal/navass_unican_es/Eo43HV8hgUNIsaz4yF_VZVYBsv5YMVO5Zapo4Qdb8GoIeA?e=xA1bAq)

### Ejemplo 2
Para el uso de los ficheros compilados a trav茅s de Matlab se utilizar谩n la siguientes l铆neas. Es necesario que los ficheros .exe se encuentren en el mismo directorio en el que se est茅 ejecutando el c贸digo de Matlab.

Cuando se quiere exportar los resultados que se han obtenido con RFSM a raster, se puede utilizar la siguiente expresi贸n una vez ha acabado la simulaci贸n
```Matlab
 path_project = 'Directorio donde se encuentra el proyecto'
 TestDesc     = 'TestDesc'
 Results      = 'BCSetID'
 src          = 'Proyecci贸n en formato EPSG'
 str  = ['ExportGIF.exe ',path_project,' ',TestDesc,' ',int2str(BCSetID)];
 system(str);
```
Cuando se ejecuta este c贸digo genera un fichero que ocupa demasiado y es innecesario, por tanto se elimina.
```Matlab
 path_test = 'path donde se han generado los ficheros de la simulaci贸n'
 path_export = fullfile(path_test,'export\');
 delete([path_export,['Result_Level_IZID',int2str(BCSetID),'.tif']]);
```
### Ejemplo 3
Cuando se tienen diferentes mallas y se quiere juntar el resultado en un 煤nico r谩ster se utiliza el siguiente c贸digo:
```Matlab
path_project = 'Directorio donde se encuentra el proyecto'
str  = ['mosaic_TIF.exe ',path_project];
system(str);
```
# Referencias 
<a id="1">[1]</a> 
Gomes da Silva, P., Coco, G., Garnier, R., & Klein, A. H. F. (2020). 
On the prediction of runup, setup and swash on beaches. 
Earth-Science Reviews, 204(February), 103148. https://doi.org/10.1016/j.earscirev.2020.103148.

# Licencia
Usa exclusivo para miembros de IH Cantabria.

Copyright 2021 Instituto de Hidr谩ulica Ambiental "IHCantabria". Universidad de Cantabria.

