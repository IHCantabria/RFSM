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
| [doc](doc) | En esta carpeta se encuentran los artículos de interés.<br>| 

## Instalación
Para realizar la instalación de la librería es necesario descargar y descomprimir el fichero.
Es recomendable crear un enviroment de conda para realizar la instalación.
Ejecutaremos las siguientes líneas:

```python
conda env create -f RFSM.yml
conda install scikit-image statsmodels seaborn
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
## Referencias 
---
<a id="1">[1]</a> 
Gomes da Silva, P., Coco, G., Garnier, R., & Klein, A. H. F. (2020). 
On the prediction of runup, setup and swash on beaches. 
Earth-Science Reviews, 204(February), 103148. https://doi.org/10.1016/j.earscirev.2020.103148.
---



