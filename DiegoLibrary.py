import os
import subprocess
import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr
from pyproj import CRS

"""
Devuelve la resolución de un mdt.asc
"""
def resolution_asc(dem_asc):
    with open(dem_asc, 'r') as file:
        for line in file:
            if 'cellsize' in line:
                # Elimina los espacios en blanco y divide la línea en palabras
                words = line.strip().split()
                res = int(words[1])
                return res
            
"""
Consigue el CRS de un mapa SHP o TIFF
@input shp/tif
@output CRS
"""
def get_crs(file_in):

    if os.path.splitext(file_in)[1] == ".shp":
        shp = gpd.read_file(file_in)
        return shp.crs
    
    elif os.path.splitext(file_in)[1] == ".tif":
        with rasterio.open(file_in) as f:
            return f.crs
    
    else:
        print("File not suitable")
        return
            

"""
Conversor de formato ascii (.asc) a formato shapefile (.shp)
"""
def asc2shp(asc_in, polygonize, crs_ref):

    print("asc to shp")

    shp_out_file = os.path.splitext(asc_in)[0]+".shp"

    command = ["python3", f"{polygonize}", asc_in, '-f', 'ESRI Shapefile', shp_out_file]
    subprocess.call(command)

    shp_out = gpd.read_file(shp_out_file)
    shp_out.to_file(shp_out_file,crs=crs_ref)

    return shp_out_file
            
"""
Conversor de formato ASCII a formato TIFF

@input ASCII (.asc)

@return TIFF path
"""
def asc2tif(asc_in, crs, translate):

    print("ASCII to TIFF")
    tif_out = os.path.splitext(asc_in)[0]+".tif"
    args = f"{translate} -of \"GTiff\" {asc_in} {tif_out}"
    subprocess.call(args, stdout=None, stderr=None, shell=False)


    with rasterio.open(tif_out, 'r+') as f:
            f.crs = crs

    print("TIFF created")

    return tif_out
            
"""
Convierte un mapa ASC a NETCDF

@input ASCII (.asc)
@input crs

@return NETCDF path (.nc)
"""
def asc2nc(asc_in, crs):
    with rasterio.open(asc_in) as asc:
        data = asc.read(1)  # Lee la banda del raster

        # Obtén información necesaria del archivo ASC
        transform = asc.transform
        nodata = asc.nodata

        # Crea un objeto DataArray de xarray
        data_array = xr.DataArray(data, dims=('y', 'x'), coords={'x': asc.bounds.left + asc.res[0] * (0.5 + np.arange(asc.width)),
                                                                 'y': asc.bounds.top - asc.res[1] * (0.5 + np.arange(asc.height))})

        crs = crs

        # Agrega atributos a la variable
        data_array.attrs['transform'] = transform.to_gdal()
        data_array.attrs['crs'] = crs.to_string()
        data_array.attrs['_FillValue'] = nodata

        # Crea un conjunto de datos Dataset de xarray
        dataset = xr.Dataset({'data': data_array})

        nc_out = os.path.splitext(asc_in)[0] + ".nc"

        # Guarda el conjunto de datos en formato NetCDF
        dataset.to_netcdf(nc_out)