"""
Este archivo contiene funciones para la primera carga de datos en la BBDD.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""

from Functions.API import extraer_publicaciones_arxiv

df = extraer_publicaciones_arxiv('cs.AI', max_resultados=10)
df.to_excel('publicaciones_arxiv.xlsx', index=False)
print(df)
