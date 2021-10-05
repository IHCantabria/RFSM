# _Autor:_    __Salavador Navas__
# _Revisión:_ __28/09/2021__

from RFSM_python.utils import *
from osgeo import ogr
import math
from scipy.stats import linregress


def _distance(a, b):

    """ Return the distance separating points a and b.

    a and b should each be an (x, y) tuple.

    Warning: This function uses the flat surface formulae, so the output may be
    inaccurate for unprojected coordinates, especially over large distances.

    """

    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return (dx ** 2 + dy ** 2) ** 0.5


def _get_split_point(a, b, dist):

    """ Returns the point that is <<dist>> length along the line a b.

    a and b should each be an (x, y) tuple.
    dist should be an integer or float, not longer than the line a b.

    """

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    m = dy / dx
    c = a[1] - (m * a[0])

    x = a[0] + (dist**2 / (1 + m**2))**0.5
    y = m * x + c
    # formula has two solutions, so check the value to be returned is
    # on the line a b.
    if not (a[0] <= x <= b[0]) and (a[1] <= y <= b[1]):
        x = a[0] - (dist**2 / (1 + m**2))**0.5
        y = m * x + c

    return x, y


def split_line_single(line, length):

    """ Returns two ogr line geometries, one which is the first length
    <<length>> of <<line>>, and one one which is the remainder.

    line should be a ogr LineString Geometry.
    length should be an integer or float.

    """

    line_points = line.GetPoints()
    sub_line = ogr.Geometry(ogr.wkbLineString)

    while length > 0:
        d = _distance(line_points[0], line_points[1])
        if d > length:
            split_point = _get_split_point(line_points[0], line_points[1], length)
            sub_line.AddPoint(line_points[0][0], line_points[0][1])
            sub_line.AddPoint(*split_point)
            line_points[0] = split_point
            break

        if d == length:
            sub_line.AddPoint(*line_points[0])
            sub_line.AddPoint(*line_points[1])
            line_points.remove(line_points[0])
            break

        if d < length:
            sub_line.AddPoint(*line_points[0])
            line_points.remove(line_points[0])
            length -= d

    remainder = ogr.Geometry(ogr.wkbLineString)
    for point in line_points:
        remainder.AddPoint(*point)

    return sub_line, remainder


def split_line_multiple(line, length=None, n_pieces=None):

    """ Splits a ogr wkbLineString into multiple sub-strings, either of
    a specified <<length>> or a specified <<n_pieces>>.

    line should be an ogr LineString Geometry
    Length should be a float or int.
    n_pieces should be an int.
    Either length or n_pieces should be specified.

    Returns a list of ogr wkbLineString Geometries.

    """

    if not n_pieces:
        n_pieces = int(math.ceil(line.Length() / length))
    if not length:
        length = line.length / float(n_pieces)

    line_segments = []
    remainder = line

    for i in range(n_pieces - 1):
        segment, remainder = split_line_single(remainder, length)
        line_segments.append(segment)
    else:
        line_segments.append(remainder)

    return line_segments


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


from matplotlib import path

def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def traza_perfiles_linea(linea_costa_suavizada,SPAC,Ltierra,Lmar,Postierra,plott,path_output):
    
    """La siguiente función nos permite generar lperfiles a partir de una linea dada.
    
       Parámetros:
       ---------------------
       linea_costa_suavizada  : file shp. Fichero de la línea de costa suavizada o con menos curvatura
       SPAC                   : float. separación que se quiere entre perfiles.
       Ltierra                : float. longitud del perfil desde la linea de costa hacia tierra.
       Lmar                   : float. longitud del perfil hacia el mar
       plott                  : True or False.  Si se quiere obtener una figura con los perfiles realizados
       path_output            : string. Path donde se desea guardar el fichero de las secciones realizadas
       Salidas:
       ---------------------
       File Cross_Sections.shp
       
    """
    
    ds = gpd.read_file(linea_costa_suavizada)
    xc = list()
    yc = list()
    for i in ds['geometry'][0].coords:
        xc.append(i[0])
        yc.append(i[1])
    
    if len(xc)>1:
        xc=xc
    if len(yc)>1:
        yc=yc
    xc0=xc
    yc0=yc

    t=np.linspace(0,2*np.pi,2001)
    circuloX=SPAC*np.sin(t)
    circuloY=SPAC*np.cos(t)

    # el primer perfil se sitúa en el primer punto de la costa
    origen=np.array([xc[0],yc[0]]).reshape(1,-1)
    xctmp=np.array(xc)
    yctmp=np.array(yc)
    cont=0
    marcador=1
    while marcador==1:
        # Distancia entre el perfil y el final de la línea de costa
        Lmax=(origen[-1,0]-xctmp[-1])**2+(origen[-1,1]-yctmp[-1])**2
        if Lmax<SPAC**2: # Si no queda suficiente costa para poner otro perfil se acaba
            marcador=0
            cont=cont+1 
            [theta,rho]=cart2pol(xctmp[-1]-origen[0,-1],yctmp[-1]-origen[1,-1])
        else:
            cont=cont+1
            #xctmp=np.concatenate((np.array([origen[cont-1,0]]).reshape(-1,1),np.array(xctmp).reshape(-1,1)))
            #yctmp=np.concatenate((np.array([origen[cont-1,1]]).reshape(-1,1),np.array(yctmp).reshape(-1,1)))
            # Se centra el círculo de búsqueda en el último perfil
            circX=circuloX+origen[cont-1,0]
            circY=circuloY+origen[cont-1,1]
            # Se buscan los puntos de la costa que quedan dentro del círculo
            IN=inpolygon(xctmp,yctmp,circX,circY)
            posIN=np.where(IN==1)[0]
            nIN=len(posIN)
            #Se busca la intersección del círculo con la recta entre el último
            #punto que queda dentro del mismo y el primero que queda fuera
            [tmpx,tmpy]=intersection(circX,circY,xctmp[posIN[-1]:posIN[-1]+2],yctmp[posIN[-1]:posIN[-1]+2])
            #se añade el nuevo perfil a la lista1
            origen = np.vstack((origen,np.array([tmpx,tmpy]).flatten()))
            #perfil
            #xctmp = xctmp[posIN[-1]+1:]
            #yctmp = yctmp[posIN[-1]+1:]

    # Se calcula la perpendicular a la costa para cada perfil
    dx1=np.gradient(origen[:,0])
    dy1=np.gradient(origen[:,1])
    dx=dx1/np.sqrt(dx1**2+dy1**2)
    dy=dy1/np.sqrt(dx1**2+dy1**2)

    if Postierra==1:
        nx=dy
        ny=-dx
    else:
        nx=-dy
        ny=dx

    # Se calculan las posiciones de los extremos de tierra0
    extremo_tierra = np.stack((origen[:,0]-Ltierra*nx, origen[:,1]-Ltierra*ny)).T
    # Se calculan las posiciones de los extremos de mar
    extremo_mar= np.stack((origen[:,0]+Lmar*nx, origen[:,1]+Lmar*ny)).T
    Data = pd.DataFrame(index = np.arange(0,cont) ,columns=['Geometry','xon','yon','xof','yof','nx','ny','xs_id'])

    for i in range(0, cont):
        Data.loc[i, 'Geometry']=LineString([[extremo_tierra[i,0], extremo_tierra[i,1]],[extremo_mar[i,0],extremo_mar[i,1]]])
        Data.loc[i,'xon']=extremo_tierra[i,0].astype(float)
        Data.loc[i,'yon']=extremo_tierra[i,1].astype(float)
        Data.loc[i,'xof']=extremo_mar[i,0].astype(float)
        Data.loc[i,'yof']=extremo_mar[i,1].astype(float)
        #Data.loc[i,'X']=[extremo_tierra[i,0], extremo_mar[i,0]]
        #Data.loc[i,'Y']=[extremo_tierra[i,1], extremo_mar[i,1]]
        nx0=extremo_mar[i,0]-extremo_tierra[i,0]
        ny0=extremo_mar[i,1]-extremo_tierra[i,1]
        modn=np.sqrt(abs(nx0)**2+abs(ny0)**2)
        Data.loc[i,'nx']=nx0/modn
        Data.loc[i,'ny']=ny0/modn
        Data.loc[i,'xs_id'] = i

    if plott==1:   
        plt.plot(xc0,yc0,'-k')
        for i in range(len(Data)):
            xon=Data.loc[i,'xon']
            xof=Data.loc[i,'xof']
            yon=Data.loc[i,'yon']
            yof=Data.loc[i,'yof']
            plt.plot([xon,xof],[yon,yof],'--b')

    df = gpd.GeoDataFrame(Data, geometry='Geometry')
    df.to_file(path_output+'Cross_Sections.shp')
    
    
def correct_perfiles_linea(shape_correct,path_output):
    """La siguiente función nos permite modificar la tabla de atributos si se ha realizado una corrección en las secciones.
    
       Parámetros:
       ---------------------
       shape_correct          : file shp. Fichero con las secciones que se han modificado
       path_output            : string. Path donde se desea guardar el fichero de las secciones realizadas
       Salidas:
       ---------------------
       File Cross_Sections_Final.shp
       
    """
    ds = gpd.read_file(shape_correct)
    for i, shape in enumerate(ds['geometry']):
        xon = shape.coords[0][0]
        xof = shape.coords[1][0]
        yon = shape.coords[0][1]
        yof = shape.coords[1][1]
        
        
        nx0=xof-xon
        ny0=yof-yon
        modn=np.sqrt(abs(nx0)**2+abs(ny0)**2)
        ds.loc[i,'nx']=nx0/modn
        ds.loc[i,'ny']=ny0/modn 
        
    ds.to_file(path_output+'Cross_Sections_Final.shp')
    
    
def extract_elevation_perfil(perfiles,topo_bat,epsg,save_csv,plot,save_fig, path_output):
    
    slope = []

    cross_sections = gpd.read_file(perfiles)

    for ind, row in cross_sections.iterrows():

        XS_ID = row['xs_id']

        start_coords =  list([row.geometry][0].coords)[0]
        end_coords = list([row.geometry][0].coords)[1]

        lon = [start_coords[0]]
        lat = [start_coords[1]]

        n_points = 50

        for i in np.arange(1, n_points+1):
            x_dist = end_coords[0] - start_coords[0]
            y_dist = end_coords[1] - start_coords[1]
            point  = [(start_coords[0] + (x_dist/(n_points+1))*i), (start_coords[1] + (y_dist/(n_points+1))*i)]
            lon.append(point[0])
            lat.append(point[1])

        lon.append(end_coords[0])
        lat.append(end_coords[1])


        df = pd.DataFrame({'Latitude': lat, 
                           'Longitude': lon})

        gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))
        gdf.crs = {'init': 'epsg:'+str(epsg)}

        gdf_pcs = gdf.to_crs(epsg = epsg)

        gdf_pcs['h_distance'] = 0

        for index, row in gdf_pcs.iterrows():
            gdf_pcs['h_distance'].loc[index] = gdf_pcs.geometry[0].distance(gdf_pcs.geometry[index])

        # Extracting the elevations from the DEM     

        gdf_pcs['Elevation'] = 0

        dem = rasterio.open(topo_bat, mode = 'r')

        for index, row in gdf_pcs.iterrows():
            row, col = dem.index(row['Longitude'], row['Latitude'])
            dem_data = dem.read(1)

            gdf_pcs['Elevation'].loc[index] = dem_data[row, col]

        # Extract h_distance (x) and Elevation (y) columns into a Pandas DataFrame

        x_y_data = gdf_pcs[['h_distance', 'Elevation']]
        
        slope.append(-linregress(x_y_data.h_distance.values, x_y_data.Elevation.values).slope)

        if save_csv==True:
            x_y_data.to_csv(path_output + XS_ID + '.csv' )

        if plot==True:


            # Creating plots for each cross sectional profile 
            plt.plot(gdf_pcs['h_distance'], gdf_pcs['Elevation'])
            plt.xlabel('Distance (m)')
            plt.ylabel('Elevation (m)')
            plt.grid(True)
            plt.title(XS_ID)
            if save_fig == True:
                plt.savefig(path_output+ XS_ID + '.png' )
            plt.show()
    return slope


def calculate_mforshore(batimetria, perfile, epsg, path_output):
    """La siguiente función nos obtener la pendiente de la línea de costa a partir de una batimetría en secciones determinadas.
    
       Parámetros:
       ---------------------
       batimetria             : file shp o Tif. Fichero de la batimetría
       perfile                : file shp. Fichero con las secciones a lo largo de la línea de costa
       epsg                   : int. Código EPSG del sistema de coordenadas con el que se está trabajando
       path_output            : string.  Path donde se desea guardar el fichero de secciones actualizados con la pendiente. 
       Salidas:
       ---------------------
       File Cross_Sections_mforshore.shp
       
    """
    
    from scipy.stats import linregress
    
    perf = gpd.read_file(perfile)
    
    if batimetria.endswith('.shp'):
        bat = gpd.read_file(batimetria)

        bat = bat.to_crs(epsg = epsg)

        # El fichero .shp de la batimetría no tiene información de Z en la geometría por tanto es necesario asignarla

        bat_mod = bat.copy()
        for i, shape in enumerate(bat.geometry):
            x_bat = list()
            y_bat = list()
            z_bat = list()
            for j in shape.coords:
                x_bat.append(j[0])
                y_bat.append(j[1])
                z_bat.append(bat.loc[i,'ELEVATION'])

            bat_mod.loc[i,'geometry'] = LineString(np.stack([x_bat, y_bat, z_bat]).T)

        
        perf['mforePer']=0
        for i, shape in enumerate(perf.index):
            x_bat = list()
            y_bat = list()
            z_bat = list()
            points = perf.loc[i,'geometry'].intersection(bat_mod.unary_union)
            for j in points:
                x_bat.append(j.coords[0][0])
                y_bat.append(j.coords[0][1])
                z_bat.append(j.coords[0][2])
            perf.loc[i,'mforePer'] = -linregress(x_bat, z_bat).slope
        perf.to_file(path_output+'Cross_Sections_mforshore.shp')
        
    elif batimetria.endswith('.tif') or batimetria.endswith('.asc'):
        mforshore = extract_elevation_perfil(perfile,
                     batimetria,epsg,False,False,False, path_output)
        perf['mforePer']=0
        perf.loc[:,'mforePer'] = mforshore

        perf.to_file(path_output+'Cross_Sections_mforshore.shp')
        
        
        
def update_mforeshore_TWL(perfiles,puntos_TWL,EPSG,path_output):
    """La siguiente función permite asignar la pendiente del perfil más cercano a los puntos donde se calcula el TWL.
    
       Parámetros:
       ---------------------
       perfiles               : file shp o Tif. Fichero con los perfiles de la línea de costa
       puntos_TWL             : file shp. Fichero con los puntos de TWL
       epsg                   : int. Código EPSG del sistema de coordenadas con el que se está trabajando
       path_output            : string.  Path donde se desea guardar el fichero de secciones actualizados con la pendiente. 
       Salidas:
       ---------------------
       File puntos_TWL_foreshore.shp
       
    """
    import fiona
    from shapely.geometry import shape, Point, LineString
    import pyproj

    gpd1 =  gpd.read_file(perfiles)
    gpd2 = gpd.read_file(puntos_TWL)

    srcProj = pyproj.Proj(init='EPSG:'+str(EPSG))
    dstProj = pyproj.Proj(init='EPSG:'+str(EPSG))

    path1 = perfiles
    path2 = puntos_TWL

    points = fiona.open(path2)
    line = fiona.open(path1)

    points = [ (shape(feat["geometry"]).xy[0][0], shape(feat["geometry"]).xy[1][0]) 
               for feat in points ]

    lines = [ zip(shape(feat["geometry"]).coords.xy[0], shape(feat["geometry"]).coords.xy[1]) 
              for feat in line ]

    proj_lines = [ [] for i in range(len(lines)) ]

    for i, item in enumerate(lines):
        for element in item:
            x = element[0]
            y = element[1]
            x, y = pyproj.transform(srcProj, dstProj, x, y)
            proj_lines[i].append((x, y))

    proj_points = []

    for point in points:
        x = point[0]
        y = point[1]
        x, y = pyproj.transform(srcProj, dstProj, x, y)    
        proj_points.append(Point(x,y))
    gpd2['mforeshore'] = 0
    for k, point in enumerate(proj_points):
        distances = []
        for i, line in enumerate(proj_lines):
            distances.append(LineString(line).distance(point))
        gpd2.loc[k,'mforeshore'] = gpd1.loc[np.argmin(distances),'mforePer']
        
    gpd2.to_file(path_output+'puntos_TWL_foreshore.shp')
            
    