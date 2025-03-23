"""
Este archivo contiene funciones para extraer datos de la API de arXiv.
Proporciona métodos para buscar artículos, obtener detalles de los autores y
filtrar resultados por fecha de publicación.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
from Static_data import categorias_arxiv
import requests
from Functions.Loggers import extraccion_metadatos_log
import feedparser
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def extraer_publicaciones_arxiv(categoria, max_resultados=1000, ordenar_por='submittedDate', orden_descendente=True):
    """
    Extrae publicaciones de arXiv.org usando su API.
    
    Parámetros:
    -----------
    categoria : str
        Categoría de arXiv (ej. 'cs.AI', 'physics.gen-ph', 'math')
    max_resultados : int
        Número máximo de resultados a devolver (por defecto 10)
    ordenar_por : str
        Campo por el cual ordenar los resultados ('submittedDate', 'relevance', 'lastUpdatedDate')
    orden_descendente : bool
        Si es True, ordena en orden descendente (más reciente primero)
        
    Retorna:
    --------
    DataFrame de pandas con las publicaciones encontradas
    """
    logger = extraccion_metadatos_log()

    logger.debug(f"Iniciando búsqueda de publicaciones en arXiv para categoría: {categoria}")
    logger.debug(f"Parámetros: max_resultados={max_resultados}, ordenar_por={ordenar_por}, orden_descendente={orden_descendente}")
    
    # Construir la URL de la consulta
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'cat:{categoria}'
    sort_by = ordenar_por
    sort_order = 'descending' if orden_descendente else 'ascending'
    
    # Construir la URL completa
    url = f"{base_url}search_query={search_query}&max_results={max_resultados}&sortBy={sort_by}&sortOrder={sort_order}"
    logger.debug(f"URL de consulta: {url}")
    
    # Realizar la petición
    try:
        logger.debug("Enviando solicitud HTTP...")
        response = requests.get(url)
        # Verificar si hubo errores en la petición
        response.raise_for_status()  
        logger.debug(f"Respuesta recibida. Código de estado: {response.status_code}")
        
        # Parsear el feed con feedparser
        logger.debug("Parseando feed de respuesta...")
        feed = feedparser.parse(response.content)
        
        # Comprobar si se obtuvieron resultados
        if len(feed.entries) == 0:
            logger.warning(f"No se encontraron publicaciones para la categoría '{categoria}'.")
            return pd.DataFrame()
        
        logger.debug(f"Se encontraron {len(feed.entries)} publicaciones.")
        
        # Extraer información relevante
        publicaciones = []
        for i, entry in enumerate(feed.entries):
            logger.debug(f"Procesando entrada {i+1}/{len(feed.entries)}: {entry.title[:50]}...")
            
            # Extraer autores
            autores = ", ".join([author.name for author in entry.authors])
            
            # Extraer categorías
            categorias = ", ".join([tag["term"] for tag in entry.tags])

            publicacion = {
                "titulo": entry.title,
                "autores": autores.split(', '),
                "resumen": str(entry.summary).replace('\n', ' ').strip(),
                "fecha_publicacion": entry.published,
                # Se mantienen con el codigo de arxiv, ya que hay publicaciones que aparecen en otras categorias fuera de computer science
                "categorias_lista": categorias,
                # Convertir URL de abstract a URL de PDF
                "url_pdf": entry.id.replace("abs", "pdf"),  
                # El ultimo valor del ID
                "identificador_arxiv" : entry.id.split("/")[-1],
                "categoria_principal": categoria
            }
            
            publicaciones.append(publicacion)
        
        # Crear DataFrame
        df = pd.DataFrame(publicaciones)
        logger.debug(f"DataFrame creado exitosamente con {len(df)} filas y {len(df.columns)} columnas.")
        
        return df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al realizar la petición HTTP: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error inesperado durante la extracción: {e}", exc_info=True)
        return pd.DataFrame()
    
def extraccion_por_categorias(max_resultados=1000):
    """
    Ejecuta la función extraer_publicaciones_arxiv iterando por todas las categorías definidas en categorias_arxiv.
        
    Parámetros:
    -----------
    max_resultados : int
        Número máximo de resultados a devolver por categoría (por defecto 1000)
    
    Retorna:
    --------
    DataFrame de pandas con los metadatos de las publicaciones descargadas de todas las categorías.
    """
    # Definicion del logger
    logger = extraccion_metadatos_log()
    df_lst = []
    for i in tqdm(categorias_arxiv.keys()):
        print(f' Extrayendo la categoria: {categorias_arxiv[i]}...')
        # Descargar metadatos de la categoria
        logger.debug(f"Descargando metadatos de la categoria {categorias_arxiv[i].upper()}...")
        # DF con los metadatos de la categoria
        metadatos_categoria = extraer_publicaciones_arxiv(i, max_resultados) # VALOR DE PRUEBA DESPLIEGUE EN 1000 -------------------!!!
        df_lst.append(metadatos_categoria)
        #print(metadatos_categoria)
    
    # Concatenacion
    df_metadata_total = pd.concat(df_lst, ignore_index=True)
    return df_metadata_total
