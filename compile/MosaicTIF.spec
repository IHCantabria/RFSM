# -*- mode: python ; coding: utf-8 -*-
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


rasterio_imports_paths = glob.glob(r'C:\Anaconda3\envs\CANAL_FIN\Lib\site-packages\rasterio\*.py')
rasterio_imports = ['rasterio._shim']

sklearn_imports_paths = glob.glob(r'C:\Anaconda3\envs\CANAL_FIN\Lib\site-packages\sklearn\*.py')
sklearn_imports = ['sklearn.utils._typedefs','sklearn.neighbors._partition_nodes']

sklearnex_imports_paths = glob.glob(r'C:\Anaconda3\envs\CANAL_FIN\Lib\site-packages\sklearnex\*.py')
sklearnex_imports = []

patoolib_imports_paths = glob.glob(r'C:\Anaconda3\envs\CANAL_FIN\Lib\site-packages\patoolib\*.py')
patoolib_imports = ['patoolib.programs','patoolib.programs.tar','patoolib.programs.p7zip','patoolib.programs.unrar','patoolib.programs.rar']




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


binaries = osgeo_pyds+[
    (os.path.join(bins,'geos.dll'), '.'),
    (os.path.join(bins,'geos_c.dll'), '.'),
    (os.path.join(bins,'spatialindex_c-64.dll'), '.'),
    (os.path.join(bins,'spatialindex-64.dll'),'.'),
]

a = Analysis(['mosaic_TIF.py'],
         pathex=['E:/GitHub/RFSM/compile'], # add all your paths
         binaries=binaries, # add the dlls you may need
         datas=collect_data_files('geopandas', subdir='datasets')+collect_dynamic_libs('rtree'), #this is the important bit for your particular error message
         hiddenimports=rasterio_imports, # double tap
         hookspath=[],
         runtime_hooks=[],
         excludes=[],
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
          name='MosaicTIF',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
