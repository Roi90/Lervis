"""
Este archivo contiene funciones para la primera carga de datos en la BBDD.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""

from Functions.API import extraccion_por_categorias
import pandas as pd

#metadatos_publicaciones = descarga_por_batches(1000)
aaa = extraccion_por_categorias()
print(len(aaa))
#print(metadatos_publicaciones)

