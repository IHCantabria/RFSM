# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files # this is very helpful
from PyInstaller.utils.hooks import collect_dynamic_libs
from pyproj import _datadir, datadir
from fiona import _shim, schema

env_path = os.environ['CONDA_PREFIX']
dlls = os.path.join(env_path, 'DLLs')
bins = os.path.join(env_path, 'Library', 'bin')

paths = [
    os.getcwd()]
block_cipher = None
# these binary paths might be different on your installation. 
# modify as needed. 
# caveat emptor

_osgeo_pyds = collect_data_files('osgeo', include_py_files=True)
_osgeo_pyds = _osgeo_pyds + collect_data_files('fiona', include_py_files=True)

osgeo_pyds = []
for p, lib in _osgeo_pyds:
	if '.pyd' in p or '.pyx' in p or '.pyc' in p:
		osgeo_pyds.append((p, '.'))


binaries = osgeo_pyds+[
    (os.path.join(bins,'geos.dll'), '.'),
    (os.path.join(bins,'geos_c.dll'), '.'),
    (os.path.join(bins,'spatialindex_c-64.dll'), '.'),
    (os.path.join(bins,'spatialindex-64.dll'),'.'),
	(os.path.join(bins,'mkl_intel_thread.1.dll'),'.'),
]

hidden_imports = [
    'ctypes',
    'ctypes.util',
    'fiona',
    'gdal',
    'geos',
    'shapely',
    'shapely.geometry',
    'pyproj',
    'rtree',
    'geopandas.datasets',
    'pytest',
    'pandas._libs.tslibs.timedeltas',
]

a = Analysis(['ExportTIF.py'],
         pathex=['F:\\GRAFCAN\\Scripts'], # add all your paths
         binaries=binaries, # add the dlls you may need
         datas=collect_data_files('geopandas', subdir='datasets')+collect_dynamic_libs('rtree'), #this is the important bit for your particular error message
         hiddenimports=hidden_imports, # double tap
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
          name='ExportGIF',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
