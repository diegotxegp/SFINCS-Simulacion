#!/home/local/bin/python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd

from shapely.geometry import Polygon

#from . import sfincs, utils


# Defining paths and cases
path_main = r'D:\03_SFINCS'
control_case = "CFCC08"
path_case = os.path.join(path_main, control_case)
option = "A"
mesh = "cfcc08_dem_a"
flood_case = "storm_sta"  # 'storm_sta' or 'storm_dyn'
alpha = ""  # empty: no alpha / '_alpha1' or '_alpha2' or '_alpha3' or whatever alpha case you want to simulate

# Carpeta de resultados
presults = os.path.join(r"D:\03_SFINCS\01_SFINCS\Results",f"{control_case}_{option}_{flood_case}{alpha}")
if not os.path.exists(presults):
    os.mkdir(presults)

# Puesta en formato del tif
ptiff = os.path.join(path_case, mesh + '.asc')  # asc file of topography
with rasterio.open(ptiff) as dataset:
    A = dataset.read(1)  # read the raster data
    bounds = dataset.bounds  # get the affine transformation parameters
    XWorldLimits = [bounds.left, bounds.right]
    YWorldLimits = [bounds.bottom , bounds.top]
    width = dataset.width
    height = dataset.height
    cell_extent = dataset.transform[0]

# El tif lo carga empezando por la esquina superior izquierda
# Lo rotamos para empezar por la inferior izquierda
Z = np.flipud(A)
Z[np.isnan(Z)] = np.nanmin(Z)

# Generamos malla
xp = np.arange(XWorldLimits[0], XWorldLimits[1], cell_extent, dtype=np.int32)
yp = np.arange(YWorldLimits[0], YWorldLimits[1], cell_extent, dtype=np.int32)


# Pintamos para verificar
XG, YG = np.meshgrid(xp, yp)
fig, ax = plt.subplots()
pcm = ax.pcolormesh(XG, YG, Z, shading='auto')
ax.set_xlim(XWorldLimits[0], XWorldLimits[1])
ax.set_ylim(YWorldLimits[0], YWorldLimits[1])
pcm.set_clim(-50, 100)
px = [0, 1, 1, 0, 0]
py = [0, 0, 1, 1, 0]
xwl = [XWorldLimits[i] for i in px]
ywl = [YWorldLimits[i] for i in py]
ax.plot(xwl, ywl, 'r-', linewidth=2)

plt.savefig(os.path.join(presults,f"{control_case}_{option}_{flood_case}{alpha}_mesh.jpg"), format='jpg')
#plt.show()


###########################################################################################################################################

buf = gpd.read_file(r"D:\03_SFINCS\CFCC08\CFCC08_coast_buffer_A.shp")
#sfincs.setup_mask_active(include_mask=buf, zmin = 0, zmax = 15)


# límites de zona activa
zmin = 0
zmax = 15

# cargamos condiciones de contorno
#buf = gpd.read_file(path_case + control_case + '_coast_buffer_' + option + '.shp')
coordinates = []
for i in range(len(buf.geometry)):
    coordinates = list(buf.geometry[i].exterior.coords)

    [X, Y] = zip(*coordinates)

X = list(map(round, X))
Y = list(map(round, Y))

#matriz_X = np.tile(X, (1196 // len(X) + 1, 1274 // len(X) + 1))[:1196, :1274]

bufin = {'x': X,
         'y': Y}

print(len(X))
print(len(Y))
print(XG.shape)
print(YG.shape)
print(Z.shape)

msk = np.zeros_like(XG)  # Crear una matriz de ceros del mismo tamaño que XG
for i in range(len(XG)):
    for j in range(len(YG)):
        if zmin <= Z[i, j] <= zmax:
            msk[i, j] = 1

print(xp)
print(XG)
print(X)

print(yp)
print(YG)
print(Y)


np.savetxt(r"D:\03_SFINCS\msk.txt",msk,fmt='%d')

# pintamos para verificar
fig, ax = plt.subplots()
plt.pcolor(XG, YG, msk)
plt.colorbar()

idmsk = np.where(msk == 1)
plt.plot(XG[idmsk], YG[idmsk], 'r*')
#plt.plot(bufin['x'], bufin['y'], 'g-')

plt.savefig(presults + control_case + '_' + option + '_' + flood_case + '_mask.jpg', format='jpg')
plt.show()






"""
def sfincs_make_mask_fast(x, y, z, zlev, *args):
    xy_in = []
    xy_ex = []
    xy_bnd_wl = []
    xy_bnd_out = []

    for arg in args:
        if isinstance(arg, str):
            if arg.lower() == 'includepolygon':
                xy_in = args[args.index(arg) + 1]
            elif arg.lower() == 'excludepolygon':
                xy_ex = args[args.index(arg) + 1]
            elif arg.lower() == 'waterlevelboundarypolygon':
                xy_bnd_wl = args[args.index(arg) + 1]
            elif arg.lower() == 'outflowboundarypolygon':
                xy_bnd_out = args[args.index(arg) + 1]

    # Set some defaults
    if xy_in:
        for ip in range(len(xy_in)):
            if 'zmin' not in xy_in[ip]:
                xy_in[ip]['zmin'] = None
            if 'zmax' not in xy_in[ip]:
                xy_in[ip]['zmax'] = None

            if xy_in[ip]['zmin'] is None:
                xy_in[ip]['zmin'] = -99999.0
            if xy_in[ip]['zmax'] is None:
                xy_in[ip]['zmax'] = 99999.0

    if xy_ex:
        for ip in range(len(xy_ex)):
            if 'zmin' not in xy_ex[ip]:
                xy_ex[ip]['zmin'] = None
            if 'zmax' not in xy_ex[ip]:
                xy_ex[ip]['zmax'] = None

            if xy_ex[ip]['zmin'] is None:
                xy_ex[ip]['zmin'] = -99999.0
            if xy_ex[ip]['zmax'] is None:
                xy_ex[ip]['zmax'] = 99999.0

    if xy_bnd_wl:
        for ip in range(len(xy_bnd_wl)):
            if 'zmin' not in xy_bnd_wl[ip]:
                xy_bnd_wl[ip]['zmin'] = None
            if 'zmax' not in xy_bnd_wl[ip]:
                xy_bnd_wl[ip]['zmax'] = None

            if xy_bnd_wl[ip]['zmin'] is None:
                xy_bnd_wl[ip]['zmin'] = -99999.0
            if xy_bnd_wl[ip]['zmax'] is None:
                xy_bnd_wl[ip]['zmax'] = 99999.0

    if xy_bnd_out:
        for ip in range(len(xy_bnd_out)):
            if 'zmin' not in xy_bnd_out[ip]:
                xy_bnd_out[ip]['zmin'] = None
            if 'zmax' not in xy_bnd_out[ip]:
                xy_bnd_out[ip]['zmax'] = None

            if xy_bnd_out[ip]['zmin'] is None:
                xy_bnd_out[ip]['zmin'] = -99999.0
            if xy_bnd_out[ip]['zmax'] is None:
                xy_bnd_out[ip]['zmax'] = 99999.0

    # Global
    msk = np.ones_like(z)
    msk[z < zlev[0]] = 0
    msk[z > zlev[1]] = 0
    msk[np.isnan(z)] = 0

    # Include polygons
    if xy_in:
        for ip in range(len(xy_in)):
            if len(xy_in[ip]['x']) > 1:
                xp = xy_in[ip]['x']
                yp = xy_in[ip]['y']
                inp = InPolygon(x, y, xp, yp) & (z >= xy_in[ip]['zmin']) & (z <= xy_in[ip]['zmax'])
                msk[inp] = 1

    # Exclude polygons
    if xy_ex:
        for ip in range(len(xy_ex)):
            if len(xy_ex[ip]['x']) > 1:
                xp = xy_ex[ip]['x']
                yp = xy_ex[ip]['y']
                inp = InPolygon(x, y, xp, yp) & (z >= xy_ex[ip]['zmin']) & (z <= xy_ex[ip]['zmax'])
                msk[inp] = 0

    if xy_bnd_wl or xy_bnd_out:
        # Now first find cells that are potential boundary cells (i.e. any point that is active and has at least one inactive neighbor)
        msk2 = np.zeros((x.shape[0] + 2, x.shape[1] + 2))
        msk2[1:-1, 1:-1] = msk
        msk4 = np.zeros((4, x.shape[0], x.shape[1]))
        msk4[0, :, :] = msk2[:-2, 1:-1]
        msk4[1, :, :] = msk2[2:, 1:-1]
        msk4[2, :, :] = msk2[1:-1, :-2]
        msk4[3, :, :] = msk2[1:-1, 2:]
        msk4 = np.min(msk4, axis=0)  # msk4 is now an nmax*mmax array with zeros for cells that have an inactive neighbor
        mskbnd = np.zeros_like(x)
        mskbnd[(msk == 1) & (msk4 == 0)] = 1  # array with potential boundary cells

        # Water level boundaries
        if xy_bnd_wl:
            for ip in range(len(xy_bnd_wl)):
                print('Calculando bnd', ip, 'de', len(xy_bnd_wl))
                if len(xy_bnd_wl[ip]['x']) > 1:
                    xp = xy_bnd_wl[ip]['x']
                    yp = xy_bnd_wl[ip]['y']
                    inp = InPolygon(x, y, xp, yp) & (mskbnd == 1) & (z >= xy_bnd_wl[ip]['zmin']) & (
                                z <= xy_bnd_wl[ip]['zmax'])
                    msk[inp] = 2

        # Outflow boundaries
        if xy_bnd_out:
            for ip in range(len(xy_bnd_out)):
                if len(xy_bnd_out[ip]['x']) > 1:
                    xp = xy_bnd_out[ip]['x']
                    yp = xy_bnd_out[ip]['y']
                    inp = InPolygon(x, y, xp, yp) & (mskbnd == 1) & (z >= xy_bnd_out[ip]['zmin']) & (
                                z <= xy_bnd_out[ip]['zmax'])
                    msk[inp] = 3

    return msk
"""