#!/home/local/bin/python3
# -*- coding: utf-8 -*-

import geopandas as gpd
import numpy as np

import os

import rasterio
from rasterio.mask import mask
from osgeo import gdal

from DiegoLibrary import resolution_asc, asc2shp


mdt = None
lucascorine_tif = None
crs = None

polygonize = None
translate = None


"""
Constructor
"""
def init(mdt1, lucascorine_tif1, polygonize1, translate1, crs1):

    global mdt, lucascorine_tif, polygonize, translate, crs

    mdt = mdt1
    lucascorine_tif = lucascorine_tif1
    crs = crs1

    polygonize = polygonize1
    translate = translate1



"""
Recorta la imagen TIFF acorde al área indicada, aplicando un pequeño buffer de una celda
"""
def extract_by_mask(tif_in, shp_in):
    # Abre el archivo raster y el archivo shapefile
    tif = rasterio.open(tif_in)
    shp = gpd.read_file(shp_in)

    res = resolution_asc(mdt)
    shp = shp.buffer(res, join_style=2)

    # Obtén la geometría de la máscara del shapefile
    mask_geometry = shp.geometry.unary_union

    # Recorta el archivo raster utilizando la geometría del shapefile como máscara
    cropped_image, cropped_transform = mask(tif, [mask_geometry], crop=True)

    # Actualiza los metadatos del archivo raster recortado
    cropped_meta = tif.meta.copy()
    cropped_meta.update({
        'transform': cropped_transform,
        'height': cropped_image.shape[1],
        'width': cropped_image.shape[2]
    })

    # Guarda el resultado en un archivo TIFF
    tif_out = os.path.join(os.path.dirname(shp_in), os.path.basename(tif_in)+"_masked.tif")
    with rasterio.open(tif_out, 'w', **cropped_meta) as dst:
        dst.write(cropped_image)

    # Cierra los datasets
    tif.close()
    shp = None

    print("Mask done")

    return tif_out


"""
Resample
"""
def resample(tif_in):
    # Abrir el archivo de entrada
    dataset = gdal.Open(tif_in)

    res = resolution_asc(mdt)

    # Crear una copia del archivo de entrada con la nueva resolución
    tif_out = os.path.splitext(tif_in)[0]+f"_{res}.tif"
    gdal.Warp(tif_out, dataset, format="GTiff", xRes=res, yRes=res, dstNodata=0, resampleAlg=gdal.GRIORA_NearestNeighbour, targetAlignedPixels=True)

    # Cerrar el dataset
    dataset = None

    print("Resolution changed")

    return tif_out


"""
Reclasifica los valores del mapa máscara a unos nuevos indicados. Futuro: Habrá que cambiarlo para leer los nuevos valores por un fichero .txt.
"""
def reclassify(tif_in, shp_in):

    # Abrir el archivo tif
    with rasterio.open(tif_in) as src:
        # Leer la matriz de datos
        data = src.read(1)

        # Definir las nuevas clases y los valores de reclasificación
        clases = {
            0.15: [1, 1],
            0.2: [2, 8],
            0.127: [9, 15],
            0.1: [16, 21],
            0.12: [22, 26],
            0.05: [27, 31]
        }

        # Crear una matriz de ceros con las mismas dimensiones que los datos
        reclasificado = np.zeros_like(data, dtype=np.float32)

        # Recorrer cada clase y reclasificar los valores dentro del rango
        for clase, rango in clases.items():
            reclasificado[(data >= rango[0]) & (data <= rango[1])] = clase

        reclassified_manning = os.path.splitext(shp_in)[0]+"_manning.asc"

        np.savetxt(reclassified_manning, reclasificado, fmt='%.3f')

        return reclassified_manning


"""
Función principal
"""
def generation_manning_file(mdt, lucascorine_tif, polygonize, translate, crs):
    
    init(mdt, lucascorine_tif, polygonize, translate, crs)
    mdt_shp = asc2shp(mdt, polygonize, crs)
    masked_tif = extract_by_mask(lucascorine_tif, mdt_shp)
    resampled_tif = resample(masked_tif)
    reclassified_manning = reclassify(resampled_tif, mdt_shp)