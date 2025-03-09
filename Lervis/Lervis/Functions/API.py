"""
Este archivo contiene funciones para extraer datos de la API de arXiv.
Proporciona métodos para buscar artículos, obtener detalles de los autores y
filtrar resultados por fecha de publicación.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import requests
from Functions.Loggers import extraccion_metadatos_log
import feedparser
import pandas as pd
from datetime import datetime

def extraer_publicaciones_arxiv(categoria, max_resultados=10, ordenar_por='submittedDate', orden_descendente=True):
    """
    Extrae publicaciones de arXiv.org usando su API de consulta basada en feeds Atom.
    
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

    logger.info(f"Iniciando búsqueda de publicaciones en arXiv para categoría: {categoria}")
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
        response.raise_for_status()  # Verificar si hubo errores en la petición
        logger.debug(f"Respuesta recibida. Código de estado: {response.status_code}")
        
        # Parsear el feed con feedparser
        logger.debug("Parseando feed de respuesta...")
        feed = feedparser.parse(response.content)
        
        # Comprobar si se obtuvieron resultados
        if len(feed.entries) == 0:
            logger.warning(f"No se encontraron publicaciones para la categoría '{categoria}'.")
            return pd.DataFrame()
        
        logger.info(f"Se encontraron {len(feed.entries)} publicaciones.")
        
        # Extraer información relevante
        publicaciones = []
        for i, entry in enumerate(feed.entries):
            logger.debug(f"Procesando entrada {i+1}/{len(feed.entries)}: {entry.title[:50]}...")
            

            publicacion = {
                "titulo": entry.title,
                "autores": entry.authors,
                "resumen": entry.summary,
                "fecha_publicacion": entry.published,
                "categorias": entry.tags,
                "url_pdf": entry.id.replace("abs", "pdf"),  # Convertir URL de abstract a URL de PDF
                "url_abstract": entry.id,
                "id" : entry.id.split("/")[-1] # El ultimo valor del ID
            }
            
            publicaciones.append(publicacion)
        
        # Crear DataFrame
        df = pd.DataFrame(publicaciones)
        logger.info(f"DataFrame creado exitosamente con {len(df)} filas y {len(df.columns)} columnas.")
        
        return df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al realizar la petición HTTP: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error inesperado durante la extracción: {e}", exc_info=True)
        return pd.DataFrame()