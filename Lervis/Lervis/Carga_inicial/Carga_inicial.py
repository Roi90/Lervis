"""
Este archivo contiene funciones para la primera carga de datos en la BBDD.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""

from Functions.API_metadata import extraccion_por_categorias
from Functions.PDF_descarga import PDF_descarga
import pandas as pd

#metadatos_publicaciones = descarga_por_batches(1000)
#aaa = extraccion_por_categorias()
#print(len(aaa))
#print(metadatos_publicaciones)

truco = pd.read_excel('publicaciones_arxiv.xlsx')
#print(truco.head())


for publicacion in truco.itertuples():
    PDF_descarga(publicacion.url_pdf, publicacion.id)
    # SOLO DESCARGO 1 PARA TEST CON EL OCR Y DOCLING
    break

