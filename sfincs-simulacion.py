#!/home/local/bin/python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal

import rasterio
from rasterio.mask import mask
import subprocess

# import hydromt and setup logging
import hydromt
from hydromt.log import setuplog

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils
from hydromt_sfincs import plots

import scipy
import shutil
from datetime import datetime, timedelta

from SFINCSManning import manning

from DiegoLibrary import get_crs, asc2tif

################################################ MODIFICAR AQUÍ ######################################################################################
######################################################################################################################################################

# Datos del caso
mdt_asc = r"D:\03_SFINCS\CFCC08\cfcc08_dem_a.asc"
flood_case = "storm_sta"  # 'storm_sta' or 'storm_dyn'
_alpha = ""  # empty: no alpha / '_alpha1' or '_alpha2' or '_alpha3' or whatever alpha case you want to simulate

buffer_shp = r"D:\03_SFINCS\CFCC08\CFCC08_coast_buffer_A.shp"
own_manning = True # Si True, generar nuestro propio Manning. De lo contrario, usar los valores Manning por defecto
lucascorine_tif = "D:\LucasCorine_30m_2019.tif"

zmin = 0
zmax = 15

input_dynamics = r"D:\03_SFINCS\CFCC08\input_dynamics\Input_SFINCS_storm_sta_A.mat"

dtout       = 43200
dtmaxout    = 43200
trsout      = 0
dtrstout    = 0
dtwnd       = 0
alpha       = 0.5
huthresh    = 0.001
manningg     = 0.15
manning_land= 0.15
manning_sea = 0.02
rgh_lev_land= -0.001
rhoa        = 0
rhow        = 0
advection   = 0
gapres      = 0
btfilter    = 0
viscosity   = 0
cdnrb       = 0
cdwnd       = 0
cdval       = 0


#######################################################################################################################################################

manning_asc = ""

if own_manning == True:
    manning_asc = os.path.splitext(mdt_asc)[0] + "_manning.asc"

user = os.getcwd()

translate = fr"{user}\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\osgeo\gdal_translate.exe"
polygonize = fr'{user}\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\gdal_polygonize.py'

path_case = os.path.dirname(mdt_asc) # Dir principal del caso
path_main = os.path.dirname(path_case)

control_case = os.path.splitext(os.path.basename(mdt_asc))[0][0:6].upper()
option = os.path.splitext(os.path.basename(mdt_asc))[0][-1].upper()

crs = get_crs(buffer_shp)


# Carpeta de resultados de SFINCS
presults = os.path.join(path_case,"SFINCS-results",f"{control_case}_{option}_{flood_case}{_alpha}")
if not os.path.exists(presults):
    os.makedirs(presults)


"""
Guarda en un fichero los parámetros usados para ejecutar RFSM
"""
def parameters2txt(file_txt):
    with open(file_txt, "w") as f:
        f.write(f"mdt = {mdt_asc}\n")
        f.write(f"case_name = {control_case}\n")
        f.write(f"option = {option}\n")
        f.write(f"flood_case = {flood_case}\n")
        f.write(f"alpha = {_alpha}\n")
        f.write(f"buffer = {buffer_shp}\n")
        f.write(f"lucascorine_tif = {lucascorine_tif}\n")
        f.write(f"""dtout       = {dtout}
                    dtmaxout    = {dtmaxout}
                    trsout      = {trsout}
                    dtrstout    = {dtrstout}
                    dtwnd       = {dtwnd}
                    alpha       = {alpha}
                    huthresh    = {huthresh}
                    manningg     = {manningg}
                    manning_land= {manning_land}
                    manning_sea = {manning_sea}
                    rgh_lev_land= {rgh_lev_land}
                    rhoa        = {rhoa}
                    rhow        = {rhow}
                    advection   = {advection}
                    gapres      = {gapres}
                    btfilter    = {btfilter}
                    viscosity   = {viscosity}
                    cdnrb       = {cdnrb}
                    cdwnd       = {cdwnd}
                    cdval       = {cdval}\n""")


"""
Genera el fichero YAML para SFINCS.
"""
def generate_yaml():

    global crs

    mdt_tif = asc2tif(mdt_asc, crs, translate)

    name_mdt = os.path.basename(mdt_tif)
    namemdt = os.path.splitext(name_mdt)[0]

    name_buffer = os.path.basename(buffer_shp)
    namebuffer = os.path.splitext(name_buffer)[0]

    crs = crs.to_epsg()

    if manning_asc != "":
        manning_tif = asc2tif(manning_asc, crs, translate)
        name_manning = os.path.basename(manning_tif)
        namemanning = os.path.splitext(name_manning)[0]

        m = f"""{namemanning}:
            path: {name_manning}
            data_type: GeoDataFrame
            driver: vector
            crs: {crs}
            meta:
                category: roughness"""
    else:
        manning_tif = None
        name_manning = ""
        namemanning = ""
        m = ""


    yml_str = f"""
    meta:
        root: {path_case}
    
    {namemdt}:
        path: {name_mdt}
        data_type: RasterDataset
        driver: raster
        crs: {crs}
        meta:
            category: topobathy

    {namebuffer}:
        path: {name_buffer}
        data_type: GeoDataFrame
        driver: vector
        crs: {crs}
        meta:
            category: topobathy

    {m}
     """
    
    yaml = os.path.splitext(mdt_tif)[0]+".yaml"
    with open(yaml, mode="w") as f:
        f.write(yml_str)

    print("YAML created")

    return yaml, mdt_tif, manning_tif, crs


""""
Genera la malla de la simulación
"""
def generate_grid(sf, mdt_tif):

    with rasterio.open(mdt_tif) as f:
        A = f.read(1)  # read the raster data
        bounds = f.bounds  # get the affine transformation parameters
        XWorldLimits = [bounds.left, bounds.right]
        YWorldLimits = [bounds.bottom , bounds.top]
        width = f.width
        height = f.height
        cell_extent = f.transform[0]

    crs = get_crs(mdt_tif)
    epsg = crs.to_epsg()

    sf.setup_grid(
    x0=bounds.left,
    y0=bounds.bottom,
    dx=cell_extent,
    dy=cell_extent,
    nmax=width,
    mmax=height,
    rotation=0,
    epsg=epsg,
    )


"""
Forcing
"""
def forcing(sf):
    sf.setup_config(
    **{
        "dtout": f"{dtout}",
        "dtmaxout": f"{dtmaxout}",
        "dtrstout": f"{dtrstout}",
        "trsout": f"{trsout}",
        "dtwnd": f"{dtwnd}",
        "alpha": f"{alpha}",
        "huthresh": f"{huthresh}",
        "manning": f"{manningg}",
        "manning_land": f"{manning_land}",
        "manning_sea": f"{manning_sea}",
        "rgh_lev_land": f"{rgh_lev_land}",
        "rhoa": f"{rhoa}",
        "rhow": f"{rhow}",
        "advection": f"{advection}",
        "gapres": f"{gapres}",
        "btfilter": f"{btfilter}",
        "viscosity": f"{viscosity}",
        "cdnrb": f"{cdnrb}",
        "cdwnd": f"{cdwnd}",
        "cdval": f"{cdval}",
    }
    )

    bc1 = scipy.io.loadmat(input_dynamics)
    x, y = sfincs_write_boundary_points(bc1)
    t, bzs = sfincs_write_boundary_conditions(bc1)

    # add to Geopandas dataframe as needed by HydroMT
    pnts = gpd.points_from_xy(x, y)
    index = [i for i in range(1,len(x)+1)] # NOTE that the index should start at one
    bnd = gpd.GeoDataFrame(index=index, geometry=pnts, crs=sf.crs)

    time = secs2datetime(start=utils.parse_datetime(sf.config["tstart"]), period_list = t)

    bzspd = pd.DataFrame(index=time, columns=index, data=bzs)

    sf.setup_waterlevel_forcing(timeseries=bzspd, locations=bnd)

    sf.forcing.keys()

    #plot_forcing(bzspd, t)



def secs2datetime(start, period_list):
    fechas = []
    for step in period_list:
        fechas.append(start + timedelta(seconds=int(step)))
    return pd.DatetimeIndex(fechas)

"""
Obtiene x e y de input_dynamics para generar sfincs.bnd
"""
def sfincs_write_boundary_points(bc1):

    x_array = np.array(list(bc1['input']['x']))
    x = np.array([xxx[0][0] for arr in x_array for xxx in arr]).tolist()

    y_array = np.array(list(bc1['input']['y']))
    y = np.array([yyy[0][0] for arr in y_array for yyy in arr]).tolist()

    return x, y


"""
Obtiene el tiempo t y el time-water-level de input_dynamics para generar sfincs.bzs
"""
def sfincs_write_boundary_conditions(bc1):

    t = bc1['input']['timeV'][0, 0]
    t = [item for sublist in t for item in sublist]

    inflowV=np.array(list(bc1['input']['inflowV']))
    twl = np.array([arr[0] for sublist in inflowV for arr in sublist])

    twl = twl.transpose()

    return t, twl


def plot_forcing(bzspd, t):

    x = range(1,len(bzspd)+1)

    pgraphics = os.path.join(presults, "charts")

    if not os.path.exists(pgraphics):
        os.mkdir(pgraphics)

    for i in range(1, len(bzspd)+1):
        y = bzspd[i]

        plt.plot(t, y, marker='o', linestyle='-', color='b', label='index')
        plt.xlabel('time')
        plt.xticks(rotation = 45)
        plt.subplots_adjust(bottom=0.2)
        plt.ylabel('waterlevel [m+ref]')
        plt.title('SFINCS waterlevel forcing (bzs)')
        plt.legend(['twl'], loc='upper left')
        plt.savefig(os.path.join(pgraphics ,f"waterlevel-{i}.jpg"), format='jpg')
        #plt.show()

"""
Se construye el modelo
"""
def build_model(yaml, mdt_tif, buffer_shp, manning_tif):
    sf = SfincsModel(data_libs=[yaml], root = presults, mode="w+")

    logger = setuplog("prepare data catalog", log_level=10)
    data_catalog = hydromt.DataCatalog(data_libs=[yaml], logger=logger)

    generate_grid(sf, mdt_tif)

    name_mdt = os.path.basename(mdt_tif)
    namemdt = os.path.splitext(name_mdt)[0]
    da = data_catalog.get_rasterdataset(namemdt)

    datasets_dep = [{'elevtn': da}]
    sf.setup_dep(datasets_dep=datasets_dep)

    sf.setup_mask_active(zmin=zmin, zmax=zmax, fill_area=0.0, reset_mask=True)

    gdf_include = sf.data_catalog.get_geodataframe(buffer_shp)
    sf.setup_mask_bounds(btype="waterlevel", include_mask=gdf_include, reset_bounds=True, all_touched=True)

    if manning_asc != "":
        da_manning = xr.open_dataarray(manning_tif)
        datasets_rgh = [{"manning": da_manning}]
        sf.setup_manning_roughness(
            datasets_rgh=datasets_rgh,
            rgh_lev_land=0,  # the minimum elevation of the land
        )

    forcing(sf)

    sf.write()

    dir_list = os.listdir(sf.root)
    print(dir_list)

    return sf


"""
Execution of model with sfincs.exe
"""
def run_model():

    cur_dir = os.getcwd()

    os.chdir(presults)

    shutil.copyfile(os.path.join(path_main,"run.bat"), os.path.join(presults, "run.bat"))

    os.system(os.path.join(presults,"run.bat"))

    os.chdir(cur_dir)

"""
Recorta y convierte el fichero NC a TIFF
"""
def nc2cuttif(dir, epsg):

    # NC TO TIFF

    pm = dir
    region_epsg = epsg

    # Busca todos los ficheros .nc dentro del directorio indicado
    for ruta_actual, directorios, archivos in os.walk(pm):
            for archivo in archivos:
                if archivo == 'sfincs_map.nc':
                    print("Convirtiendo .nc a .tif...")

                    nc = os.path.join(ruta_actual, archivo)

                    ds = xr.open_dataset(nc,mask_and_scale=True)

                    xx = sorted(np.unique(ds.x.data))
                    yy = sorted(np.unique(ds.y.data))
                    hmax = ds.hmax.data[0,:,:]
                    time = ds.timemax.data

                    if not len(xx)==hmax.shape[1]:
                        da = xr.Dataset(data_vars=dict(
                            hmax=(["y","x"], hmax.T),
                        ),
                        coords=dict(
                            x=(["x"], xx),
                            y=(["y"], yy),
                        ),
                        attrs=dict(description="Weather related data."),)
                        da = da.rio.write_crs("EPSG:"+str(region_epsg))
                    else:
                        da = xr.Dataset(data_vars=dict(
                        hmax=(["y","x"], hmax),
                        ),
                        coords=dict(
                            x=(["x"], xx),
                            y=(["y"], yy),
                        ),
                        attrs=dict(description="Weather related data."),)
                        da = da.rio.write_crs("EPSG:"+str(region_epsg))
                        
                    #da.to_netcdf(os.path.join(ruta_actual,'sfincs_map_hmax.nc'))
                    da.rio.to_raster(os.path.join(ruta_actual,r"sfincs_map_hmax.tif"))

                    print(os.path.join(ruta_actual,r"sfincs_map_hmax.tif"))


                    ## TIFF TO CUT TIFF
                    print("Recortando .tif...")

                    pout=ruta_actual

                    archivo = os.path.join(os.path.dirname(nc),r"sfincs_map_hmax.tif")

                    nombre_archivo = os.path.splitext(os.path.basename(archivo))[0]
                    name = str(nombre_archivo+"_recortado")
                    output_raster = os.path.join(pout, name)

                    try:
                        # Abrir el archivo raster con GDAL
                        dataset = gdal.Open(archivo, gdal.GA_ReadOnly)
                        
                        if dataset is None:
                            raise Exception(f"No se pudo abrir el archivo: {archivo}")

                        # Calcular las estadísticas del raster
                        dataset.GetRasterBand(1).ComputeStatistics(False)

                        # Obtener el valor mínimo
                        minimo = dataset.GetRasterBand(1).GetMinimum()
                        minimo = float(minimo)

                        # Crear una copia del raster y establecer los valores iguales al mínimo a NULL
                        driver = gdal.GetDriverByName("GTiff")
                        out_dataset = driver.CreateCopy(output_raster + ".tif", dataset, 0)
                        out_band = out_dataset.GetRasterBand(1)
                        out_band.SetNoDataValue(minimo)
                        out_band.FlushCache()

                        # Cerrar los datasets
                        dataset = None
                        out_dataset = None

                        print(output_raster + ".tif")

                    except Exception as e:
                        print(f"Error al procesar el archivo: {archivo}")
                        print(str(e))

    print("¡.TIF RECORTADOS!")


"""
Ejecuta la simulación de SFINCS
"""
def simulation_sfincs():

    yaml, mdt_tif, manning_tif, epsg = generate_yaml()
    sf = build_model(yaml, mdt_tif, buffer_shp, manning_tif)
    run_model()
    nc2cuttif(presults, epsg)

    dir_list = os.listdir(sf.root)
    print(dir_list)


def main_SFINCS():
    crs = get_crs(buffer_shp)
    manning.generation_manning_file(mdt_asc, lucascorine_tif, polygonize, translate, crs)
    simulation_sfincs()
    parameters2txt(os.path.join(presults,"parameters-SFINCS.txt"))


if __name__ == "__main__":
    main_SFINCS()