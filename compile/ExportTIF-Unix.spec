# -*- mode: python ; coding: utf-8 -*-
# conda create --name CANAL --channel=conda-forge python=3.7 geopandas gdal xarray netcdf4 pysimplegui pyinstaller cartopy pysheds tqdm rasterio fiona 
# conda install seaborn openpyxl svglib
# pip install hydroeval patool
import os
from PyInstaller.utils.hooks import collect_data_files # this is very helpful
from PyInstaller.utils.hooks import collect_dynamic_libs
#from pyproj import _datadir, datadir
#from fiona import _shim, schema
#from osgeo import gdal
import glob, os

env_path = os.environ['CONDA_PREFIX']
dlls = os.path.join(env_path, 'DLLs')
bins = os.path.join(env_path, 'Library', 'bin')


##### include mydir in distribution #######
def extra_datas(mydir):
    def rec_glob(p, files):
        import os
        import glob
        for d in glob.glob(p):
            if os.path.isfile(d):
                files.append(d)
            rec_glob("%s/*" % d, files)
    files = []
    rec_glob("%s/*" % mydir, files)
    extra_datas = []
    for f in files:
        extra_datas.append((f, f, 'DATA'))

    return extra_datas

paths = [
    os.getcwd()]
block_cipher = None
# these binary paths might be different on your installation. 
# modify as needed. 
# caveat emptor


rasterio_imports_paths = glob.glob('/home/salva/anaconda3/CANAL/Lib/site-packages/rasterio/*.py')
rasterio_imports = ['rasterio._shim']

for item in rasterio_imports_paths:
    current_module_filename = os.path.split(item)[-1]
    current_module_filename = 'rasterio.'+current_module_filename.replace('.py', '')
    rasterio_imports.append(current_module_filename)
    
    
_osgeo_pyds = collect_data_files('osgeo', include_py_files=True)
_osgeo_pyds = _osgeo_pyds + collect_data_files('fiona', include_py_files=True)
_osgeo_pyds = _osgeo_pyds + collect_data_files('rasterio', include_py_files=True)

osgeo_pyds = []
for p, lib in _osgeo_pyds:
	if '.pyd' in p or '.pyx' in p or '.pyc' in p:
		osgeo_pyds.append((p, '.'))


binaries = osgeo_pyds


a = Analysis(['ExportTIF.py'],
         pathex=['/mnt/Proyectos/RFSM/compile'], # add all your paths
         binaries=binaries, # add the dlls you may need
         datas = collect_dynamic_libs('rtree'), #this is the important bit for your particular error message
         hiddenimports=rasterio_imports, # double tap
		 hookspath=[],
         runtime_hooks=[],
		 win_no_prefer_redirects=False,
         win_private_assemblies=False,
		 cipher=block_cipher,
         noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='ExportTIF-Unix',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
