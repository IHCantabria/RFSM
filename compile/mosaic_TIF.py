import numpy as np
import os, glob
from osgeo import osr
import math
import sys
import time
import sys
from osgeo import gdal
#from osgeo_utils.auxiliary.util import GetOutputDriverFor

try:
    from osgeo import gdal
    gdal.TermProgress = gdal.TermProgress_nocb
except ImportError:
    import gdal

__version__ = '$id$'[5:-1]
verbose = 0
quiet = 0


# =============================================================================
def raster_copy( s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
                 t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
                 nodata=None ):

    if nodata is not None:
        return raster_copy_with_nodata(
            s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
            t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
            nodata )

    if verbose != 0:
        print('Copy %d,%d,%d,%d to %d,%d,%d,%d.' \
              % (s_xoff, s_yoff, s_xsize, s_ysize,
             t_xoff, t_yoff, t_xsize, t_ysize ))

    s_band = s_fh.GetRasterBand( s_band_n )
    t_band = t_fh.GetRasterBand( t_band_n )

    data = s_band.ReadRaster( s_xoff, s_yoff, s_xsize, s_ysize,
                              t_xsize, t_ysize, t_band.DataType )
    t_band.WriteRaster( t_xoff, t_yoff, t_xsize, t_ysize,
                        data, t_xsize, t_ysize, t_band.DataType )
        

    return 0
    
# =============================================================================
def raster_copy_with_nodata( s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
                             t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
                             nodata ):
    try:
        import numpy as Numeric
    except ImportError:
        import Numeric
    
    if verbose != 0:
        print('Copy %d,%d,%d,%d to %d,%d,%d,%d.' \
              % (s_xoff, s_yoff, s_xsize, s_ysize,
             t_xoff, t_yoff, t_xsize, t_ysize ))

    s_band = s_fh.GetRasterBand( s_band_n )
    t_band = t_fh.GetRasterBand( t_band_n )

    data_src = s_band.ReadAsArray( s_xoff, s_yoff, s_xsize, s_ysize,
                                   t_xsize, t_ysize )
    data_dst = t_band.ReadAsArray( t_xoff, t_yoff, t_xsize, t_ysize )

    nodata_test = Numeric.equal(data_src,nodata)
    to_write = Numeric.choose( nodata_test, (data_src, data_dst) )
                               
    t_band.WriteArray( to_write, t_xoff, t_yoff )

    return 0
    
# =============================================================================
def names_to_fileinfos( names ):
    """
    Translate a list of GDAL filenames, into file_info objects.
    names -- list of valid GDAL dataset names.
    Returns a list of file_info objects.  There may be less file_info objects
    than names if some of the names could not be opened as GDAL files.
    """
    
    file_infos = []
    for name in names:
        fi = file_info()
        if fi.init_from_name( name ) == 1:
            file_infos.append( fi )

    return file_infos

# *****************************************************************************
class file_info:
    """A class holding information about a GDAL file."""

    def init_from_name(self, filename):
        """
        Initialize file_info from filename
        filename -- Name of file to read.
        Returns 1 on success or 0 if the file can't be opened.
        """
        fh = gdal.Open( filename )
        if fh is None:
            return 0

        self.filename = filename
        self.bands = fh.RasterCount
        self.xsize = fh.RasterXSize
        self.ysize = fh.RasterYSize
        self.band_type = fh.GetRasterBand(1).DataType
        self.projection = fh.GetProjection()
        self.geotransform = fh.GetGeoTransform()
        self.ulx = self.geotransform[0]
        self.uly = self.geotransform[3]
        self.lrx = self.ulx + self.geotransform[1] * self.xsize
        self.lry = self.uly + self.geotransform[5] * self.ysize

        ct = fh.GetRasterBand(1).GetRasterColorTable()
        if ct is not None:
            self.ct = ct.Clone()
        else:
            self.ct = None

        return 1

    def report( self ):
        print('Filename: '+ self.filename)
        print('File Size: %dx%dx%d' \
              % (self.xsize, self.ysize, self.bands))
        print('Pixel Size: %f x %f' \
              % (self.geotransform[1],self.geotransform[5]))
        print('UL:(%f,%f)   LR:(%f,%f)' \
              % (self.ulx,self.uly,self.lrx,self.lry))

    def copy_into( self, t_fh, s_band = 1, t_band = 1, nodata_arg=None ):
        """
        Copy this files image into target file.
        This method will compute the overlap area of the file_info objects
        file, and the target gdal.Dataset object, and copy the image data
        for the common window area.  It is assumed that the files are in
        a compatible projection ... no checking or warping is done.  However,
        if the destination file is a different resolution, or different
        image pixel type, the appropriate resampling and conversions will
        be done (using normal GDAL promotion/demotion rules).
        t_fh -- gdal.Dataset object for the file into which some or all
        of this file may be copied.
        Returns 1 on success (or if nothing needs to be copied), and zero one
        failure.
        """
        t_geotransform = t_fh.GetGeoTransform()
        t_ulx = t_geotransform[0]
        t_uly = t_geotransform[3]
        t_lrx = t_geotransform[0] + t_fh.RasterXSize * t_geotransform[1]
        t_lry = t_geotransform[3] + t_fh.RasterYSize * t_geotransform[5]

        # figure out intersection region
        tgw_ulx = max(t_ulx,self.ulx)
        tgw_lrx = min(t_lrx,self.lrx)
        if t_geotransform[5] < 0:
            tgw_uly = min(t_uly,self.uly)
            tgw_lry = max(t_lry,self.lry)
        else:
            tgw_uly = max(t_uly,self.uly)
            tgw_lry = min(t_lry,self.lry)
        
        # do they even intersect?
        if tgw_ulx >= tgw_lrx:
            return 1
        if t_geotransform[5] < 0 and tgw_uly <= tgw_lry:
            return 1
        if t_geotransform[5] > 0 and tgw_uly >= tgw_lry:
            return 1
            
        # compute target window in pixel coordinates.
        tw_xoff = int((tgw_ulx - t_geotransform[0]) / t_geotransform[1] + 0.1)
        tw_yoff = int((tgw_uly - t_geotransform[3]) / t_geotransform[5] + 0.1)
        tw_xsize = int((tgw_lrx - t_geotransform[0])/t_geotransform[1] + 0.5) \
                   - tw_xoff
        tw_ysize = int((tgw_lry - t_geotransform[3])/t_geotransform[5] + 0.5) \
                   - tw_yoff

        if tw_xsize < 1 or tw_ysize < 1:
            return 1

        # Compute source window in pixel coordinates.
        sw_xoff = int((tgw_ulx - self.geotransform[0]) / self.geotransform[1])
        sw_yoff = int((tgw_uly - self.geotransform[3]) / self.geotransform[5])
        sw_xsize = int((tgw_lrx - self.geotransform[0]) \
                       / self.geotransform[1] + 0.5) - sw_xoff
        sw_ysize = int((tgw_lry - self.geotransform[3]) \
                       / self.geotransform[5] + 0.5) - sw_yoff

        if sw_xsize < 1 or sw_ysize < 1:
            return 1

        # Open the source file, and copy the selected region.
        s_fh = gdal.Open( self.filename )

        return \
            raster_copy( s_fh, sw_xoff, sw_yoff, sw_xsize, sw_ysize, s_band,
                         t_fh, tw_xoff, tw_yoff, tw_xsize, tw_ysize, t_band,
                         nodata_arg )


# =============================================================================
def Usage():
    print('Usage: gdal_merge.py [-o out_filename] [-of out_format] [-co NAME=VALUE]*')
    print('                     [-ps pixelsize_x pixelsize_y] [-separate] [-q] [-v] [-pct]')
    print('                     [-ul_lr ulx uly lrx lry] [-n nodata_value] [-init "value [value...]"]')
    print('                     [-ot datatype] [-createonly] input_files')
    print('                     [--help-general]')
    print('')

# =============================================================================
#
# Program mainline.
#

def gedal_merge( argv=None ):

    global verbose, quiet
    verbose = 0
    quiet = 0
    names = []
    format = 'GTiff'
    out_file = 'out.tif'

    ulx = None
    psize_x = None
    separate = 0
    copy_pct = 0
    nodata = None
    create_options = []
    pre_init = []
    band_type = None
    createonly = 0
    
    gdal.AllRegister()
    if argv is None:
        argv = sys.argv
    argv = gdal.GeneralCmdLineProcessor( argv )
    if argv is None:
        sys.exit( 0 )

    # Parse command line arguments.
    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg == '-o':
            i = i + 1
            out_file = argv[i]

        elif arg == '-v':
            verbose = 1

        elif arg == '-q' or arg == '-quiet':
            quiet = 1

        elif arg == '-createonly':
            createonly = 1

        elif arg == '-separate':
            separate = 1

        elif arg == '-seperate':
            separate = 1

        elif arg == '-pct':
            copy_pct = 1

        elif arg == '-ot':
            i = i + 1
            band_type = gdal.GetDataTypeByName( argv[i] )
            if band_type == gdal.GDT_Unknown:
                print('Unknown GDAL data type: ', argv[i])
                sys.exit( 1 )

        elif arg == '-init':
            i = i + 1
            str_pre_init = argv[i].split()
            for x in str_pre_init:
                pre_init.append(float(x))

        elif arg == '-n':
            i = i + 1
            nodata = float(argv[i])

        elif arg == '-f':
            # for backward compatibility.
            i = i + 1
            format = argv[i]

        elif arg == '-of':
            i = i + 1
            format = argv[i]

        elif arg == '-co':
            i = i + 1
            create_options.append( argv[i] )

        elif arg == '-ps':
            psize_x = float(argv[i+1])
            psize_y = -1 * abs(float(argv[i+2]))
            i = i + 2

        elif arg == '-ul_lr':
            ulx = float(argv[i+1])
            uly = float(argv[i+2])
            lrx = float(argv[i+3])
            lry = float(argv[i+4])
            i = i + 4

        elif arg[:1] == '-':
            print('Unrecognised command option: ', arg)
            Usage()
            sys.exit( 1 )

        else:
            # Expand any possible wildcards from command line arguments
            f = glob.glob( arg )
            if len(f) == 0:
                print('File not found: "%s"' % (str( arg )))
            names += f # append 1 or more files
        i = i + 1

    if len(names) == 0:
        print('No input files selected.')
        Usage()
        sys.exit( 1 )

    Driver = gdal.GetDriverByName(format)
    if Driver is None:
        print('Format driver %s not found, pick a supported driver.' % format)
        sys.exit( 1 )

    DriverMD = Driver.GetMetadata()
    if 'DCAP_CREATE' not in DriverMD:
        print('Format driver %s does not support creation and piecewise writing.\nPlease select a format that does, such as GTiff (the default) or HFA (Erdas Imagine).' % format)
        sys.exit( 1 )

    # Collect information on all the source files.
    file_infos = names_to_fileinfos( names )

    if ulx is None:
        ulx = file_infos[0].ulx
        uly = file_infos[0].uly
        lrx = file_infos[0].lrx
        lry = file_infos[0].lry
        
        for fi in file_infos:
            ulx = min(ulx, fi.ulx)
            uly = max(uly, fi.uly)
            lrx = max(lrx, fi.lrx)
            lry = min(lry, fi.lry)

    if psize_x is None:
        psize_x = file_infos[0].geotransform[1]
        psize_y = file_infos[0].geotransform[5]

    if band_type is None:
        band_type = file_infos[0].band_type

    # Try opening as an existing file.
    gdal.PushErrorHandler( 'CPLQuietErrorHandler' )
    t_fh = gdal.Open( out_file, gdal.GA_Update )
    gdal.PopErrorHandler()
    
    # Create output file if it does not already exist.
    if t_fh is None:
        geotransform = [ulx, psize_x, 0, uly, 0, psize_y]

        xsize = int((lrx - ulx) / geotransform[1] + 0.5)
        ysize = int((lry - uly) / geotransform[5] + 0.5)

        if separate != 0:
            bands = len(file_infos)
        else:
            bands = file_infos[0].bands

        t_fh = Driver.Create( out_file, xsize, ysize, bands,
                              band_type, create_options )
        if t_fh is None:
            print('Creation failed, terminating gdal_merge.')
            sys.exit( 1 )
            
        t_fh.SetGeoTransform( geotransform )
        t_fh.SetProjection( file_infos[0].projection )

        if copy_pct:
            t_fh.GetRasterBand(1).SetRasterColorTable(file_infos[0].ct)
    else:
        if separate != 0:
            bands = len(file_infos)
            if t_fh.RasterCount < bands :
                print('Existing output file has less bands than the number of input files. You should delete it before. Terminating gdal_merge.')
                sys.exit( 1 )
        else:
            bands = min(file_infos[0].bands,t_fh.RasterCount)

    # Do we need to pre-initialize the whole mosaic file to some value?
    if pre_init is not None:
        if t_fh.RasterCount <= len(pre_init):
            for i in range(t_fh.RasterCount):
                t_fh.GetRasterBand(i+1).Fill( pre_init[i] )
        elif len(pre_init) == 1:
            for i in range(t_fh.RasterCount):
                t_fh.GetRasterBand(i+1).Fill( pre_init[0] )

    # Copy data from source files into output file.
    t_band = 1

    if quiet == 0 and verbose == 0:
        gdal.TermProgress( 0.0 )
    fi_processed = 0
    
    for fi in file_infos:
        if createonly != 0:
            continue
        
        if verbose != 0:
            print("")
            print("Processing file %5d of %5d, %6.3f%% completed." \
                  % (fi_processed+1,len(file_infos),
                     fi_processed * 100.0 / len(file_infos)) )
            fi.report()

        if separate == 0 :
            for band in range(1, bands+1):
                fi.copy_into( t_fh, band, band, nodata )
        else:
            fi.copy_into( t_fh, 1, t_band, nodata )
            t_band = t_band+1
            
        fi_processed = fi_processed+1
        if quiet == 0 and verbose == 0:
            gdal.TermProgress( fi_processed / float(len(file_infos))  )
    
    # Force file to be closed.
    t_fh = None
###########################################################################################
def gdal_edit(argv):

    argv = gdal.GeneralCmdLineProcessor(argv)
    if argv is None:
        return -1

    datasetname = None
    srs = None
    ulx = None
    uly = None
    urx = None
    ury = None
    llx = None
    lly = None
    lrx = None
    lry = None
    nodata = None
    unsetnodata = False
    units = None
    xres = None
    yres = None
    unsetgt = False
    unsetstats = False
    stats = False
    setstats = False
    approx_stats = False
    unsetmd = False
    ro = False
    molist = []
    gcp_list = []
    open_options = []
    offset = []
    scale = []
    colorinterp = {}
    unsetrpc = False

    i = 1
    argc = len(argv)
    while i < argc:
        if argv[i] == '-ro':
            ro = True
        elif argv[i] == '-a_srs' and i < len(argv) - 1:
            srs = argv[i + 1]
            i = i + 1
        elif argv[i] == '-a_ullr' and i < len(argv) - 4:
            ulx = float(argv[i + 1])
            i = i + 1
            uly = float(argv[i + 1])
            i = i + 1
            lrx = float(argv[i + 1])
            i = i + 1
            lry = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-a_ulurll' and i < len(argv) - 6:
            ulx = float(argv[i + 1])
            i = i + 1
            uly = float(argv[i + 1])
            i = i + 1
            urx = float(argv[i + 1])
            i = i + 1
            ury = float(argv[i + 1])
            i = i + 1
            llx = float(argv[i + 1])
            i = i + 1
            lly = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-tr' and i < len(argv) - 2:
            xres = float(argv[i + 1])
            i = i + 1
            yres = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-a_nodata' and i < len(argv) - 1:
            nodata = float(argv[i + 1])
            i = i + 1
        elif argv[i] == '-scale' and i < len(argv) -1:
            scale.append(float(argv[i+1]))
            i = i + 1
            while i < len(argv) - 1 and ArgIsNumeric(argv[i+1]):
                scale.append(float(argv[i+1]))
                i = i + 1
        elif argv[i] == '-offset' and i < len(argv) - 1:
            offset.append(float(argv[i+1]))
            i = i + 1
            while i < len(argv) - 1 and ArgIsNumeric(argv[i+1]):
                offset.append(float(argv[i+1]))
                i = i + 1
        elif argv[i] == '-mo' and i < len(argv) - 1:
            molist.append(argv[i + 1])
            i = i + 1
        elif argv[i] == '-gcp' and i + 4 < len(argv):
            pixel = float(argv[i + 1])
            i = i + 1
            line = float(argv[i + 1])
            i = i + 1
            x = float(argv[i + 1])
            i = i + 1
            y = float(argv[i + 1])
            i = i + 1
            if i + 1 < len(argv) and ArgIsNumeric(argv[i + 1]):
                z = float(argv[i + 1])
                i = i + 1
            else:
                z = 0
            gcp = gdal.GCP(x, y, z, pixel, line)
            gcp_list.append(gcp)
        elif argv[i] == '-unsetgt':
            unsetgt = True
        elif argv[i] == '-unsetrpc':
            unsetrpc = True
        elif argv[i] == '-unsetstats':
            unsetstats = True
        elif argv[i] == '-approx_stats':
            stats = True
            approx_stats = True
        elif argv[i] == '-stats':
            stats = True
        elif argv[i] == '-setstats' and i < len(argv)-4:
            stats = True
            setstats = True
            if argv[i + 1] != 'None':
                statsmin = float(argv[i + 1])
            else:
                statsmin = None
            i = i + 1
            if argv[i + 1] != 'None':
                statsmax = float(argv[i + 1])
            else:
                statsmax = None
            i = i + 1
            if argv[i + 1] != 'None':
                statsmean = float(argv[i + 1])
            else:
                statsmean = None
            i = i + 1
            if argv[i + 1] != 'None':
                statsdev = float(argv[i + 1])
            else:
                statsdev = None
            i = i + 1
        elif argv[i] == '-units' and i < len(argv) - 1:
            units = argv[i + 1]
            i = i + 1
        elif argv[i] == '-unsetmd':
            unsetmd = True
        elif argv[i] == '-unsetnodata':
            unsetnodata = True
        elif argv[i] == '-oo' and i < len(argv) - 1:
            open_options.append(argv[i + 1])
            i = i + 1
        elif argv[i].startswith('-colorinterp_')and i < len(argv) - 1:
            band = int(argv[i][len('-colorinterp_'):])
            val = argv[i + 1]
            if val.lower() == 'red':
                val = gdal.GCI_RedBand
            elif val.lower() == 'green':
                val = gdal.GCI_GreenBand
            elif val.lower() == 'blue':
                val = gdal.GCI_BlueBand
            elif val.lower() == 'alpha':
                val = gdal.GCI_AlphaBand
            elif val.lower() == 'gray' or val.lower() == 'grey':
                val = gdal.GCI_GrayIndex
            elif val.lower() == 'undefined':
                val = gdal.GCI_Undefined
            else:
                sys.stderr.write('Unsupported color interpretation %s.\n' % val +
                                 'Only red, green, blue, alpha, gray, undefined are supported.\n')
                return Usage()
            colorinterp[band] = val
            i = i + 1
        elif argv[i][0] == '-':
            sys.stderr.write('Unrecognized option : %s\n' % argv[i])
            return Usage()
        elif datasetname is None:
            datasetname = argv[i]
        else:
            sys.stderr.write('Unexpected option : %s\n' % argv[i])
            return Usage()

        i = i + 1

    if datasetname is None:
        return Usage()

    if (srs is None and lry is None and yres is None and not unsetgt and
            not unsetstats and not stats and not setstats and nodata is None and
            not units and not molist and not unsetmd and not gcp_list and
            not unsetnodata and not colorinterp and
            scale is None and offset is None and not unsetrpc):
        print('No option specified')
        print('')
        return Usage()

    exclusive_option = 0
    if lry is not None:
        exclusive_option = exclusive_option + 1
    if lly is not None:  # -a_ulurll
        exclusive_option = exclusive_option + 1
    if yres is not None:
        exclusive_option = exclusive_option + 1
    if unsetgt:
        exclusive_option = exclusive_option + 1
    if exclusive_option > 1:
        print('-a_ullr, -a_ulurll, -tr and -unsetgt options are exclusive.')
        print('')
        return Usage()

    if unsetstats and stats:
        print('-unsetstats and either -stats or -approx_stats options are exclusive.')
        print('')
        return Usage()

    if unsetnodata and nodata:
        print('-unsetnodata and -nodata options are exclusive.')
        print('')
        return Usage()

    if open_options is not None:
        if ro:
            ds = gdal.OpenEx(datasetname, gdal.OF_RASTER, open_options=open_options)
        else:
            ds = gdal.OpenEx(datasetname, gdal.OF_RASTER | gdal.OF_UPDATE, open_options=open_options)
    # GDAL 1.X compat
    elif ro:
        ds = gdal.Open(datasetname)
    else:
        ds = gdal.Open(datasetname, gdal.GA_Update)
    if ds is None:
        return -1

    if scale:
        if len(scale) == 1:
            scale = scale * ds.RasterCount 
        elif len(scale) != ds.RasterCount:
            print('If more than one scale value is provided, their number must match the number of bands.')
            print('')
            return Usage()
    
    if offset:
        if len(offset) == 1:
            offset = offset * ds.RasterCount
        elif len(offset) != ds.RasterCount:
            print('If more than one offset value is provided, their number must match the number of bands.')
            print('')
            return Usage()
    
    wkt = None
    if srs == '' or srs == 'None':
        ds.SetProjection('')
    elif srs is not None:
        sr = osr.SpatialReference()
        if sr.SetFromUserInput(srs) != 0:
            print('Failed to process SRS definition: %s' % srs)
            return -1
        wkt = sr.ExportToWkt()
        if not gcp_list:
            ds.SetProjection(wkt)

    if lry is not None:
        gt = [ulx, (lrx - ulx) / ds.RasterXSize, 0,
              uly, 0, (lry - uly) / ds.RasterYSize]
        ds.SetGeoTransform(gt)

    elif lly is not None:  # -a_ulurll
        gt = [ulx, (urx - ulx) / ds.RasterXSize, (llx - ulx) / ds.RasterYSize,
              uly, (ury - uly) / ds.RasterXSize, (lly - uly) / ds.RasterYSize]
        ds.SetGeoTransform(gt)

    if yres is not None:
        gt = ds.GetGeoTransform()
        # Doh ! why is gt a tuple and not an array...
        gt = [gt[j] for j in range(6)]
        gt[1] = xres
        gt[5] = yres
        ds.SetGeoTransform(gt)

    if unsetgt:
        # For now only the GTiff drivers understands full-zero as a hint
        # to unset the geotransform
        if ds.GetDriver().ShortName == 'GTiff':
            ds.SetGeoTransform([0, 0, 0, 0, 0, 0])
        else:
            ds.SetGeoTransform([0, 1, 0, 0, 0, 1])

    if gcp_list:
        if wkt is None:
            wkt = ds.GetGCPProjection()
        if wkt is None:
            wkt = ''
        ds.SetGCPs(gcp_list, wkt)

    if nodata is not None:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
    elif unsetnodata:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).DeleteNoDataValue()

    if scale:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetScale(scale[i])

    if offset:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetOffset(offset[i])

    if units:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetUnitType(units)

    if unsetstats:
        for i in range(ds.RasterCount):
            band = ds.GetRasterBand(i + 1)
            for key in band.GetMetadata().keys():
                if key.startswith('STATISTICS_'):
                    band.SetMetadataItem(key, None)

    if stats:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).ComputeStatistics(approx_stats)

    if setstats:
        for i in range(ds.RasterCount):
            if statsmin is None or statsmax is None or statsmean is None or statsdev is None:
                ds.GetRasterBand(i+1).ComputeStatistics(approx_stats)
                min,max,mean,stdev = ds.GetRasterBand(i+1).GetStatistics(approx_stats,True)
                if statsmin is None:
                    statsmin = min
                if statsmax is None:
                    statsmax = max
                if statsmean is None:
                    statsmean = mean
                if statsdev is None:
                    statsdev = stdev
            ds.GetRasterBand(i+1).SetStatistics(statsmin, statsmax, statsmean, statsdev)

    if molist:
        if unsetmd:
            md = {}
        else:
            md = ds.GetMetadata()
        for moitem in molist:
            equal_pos = moitem.find('=')
            if equal_pos > 0:
                md[moitem[0:equal_pos]] = moitem[equal_pos + 1:]
        ds.SetMetadata(md)
    elif unsetmd:
        ds.SetMetadata({})

    for band in colorinterp:
        ds.GetRasterBand(band).SetColorInterpretation(colorinterp[band])

    if unsetrpc:
        ds.SetMetadata(None, 'RPC')

    ds = band = None

    return 0



def mosaic_tifs(path_isla, src):
    files_dir = list()
    files_name = list()
    
    if os.path.exists(path_isla+'TIF')==False:
        os.makedirs(path_isla+'TIF')
    for root, dirs, files in os.walk(path_isla):
        if 'TIF' in root:
            continue
        else:
            for file in files:
                if file.startswith('I') and file.endswith('.tif'):
                    files_dir.append(root+'/'+file)
                    files_name.append(file)
    
    
    for i in np.unique(files_name):
        files_to_mosaic = list()
        for j in files_dir:
            if i in j:
                files_to_mosaic.append(j)

        files_string = " ".join(files_to_mosaic)
        if os.path.exists(path_isla+ '/TIF/' +i):
            os.remove(path_isla+ '/TIF/' +i)
        
        gedal_merge((" -n -9999 -o "+path_isla+ '/TIF/' +i+" " + files_string).split(" "))
        #os.popen(command).read()

        gdal_edit((" -stats -a_srs EPSG:"+str(src)+" -a_nodata 0 " + path_isla+ '/TIF/' +i).split(" "))
        #os.popen(command2).read()
        
        
#if __name__== "__main__":
    # 1 path_project
    # 2 SRC
    # mosaic_tifs(sys.argv[1],sys.argv[2])