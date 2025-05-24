from Static_data import categorias_arxiv
import requests
from Functions.Loggers import crear_logger
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Se define el logger para todo el modulo
logger = crear_logger('Extraccion_arxiv', 'extraccion_arxiv.log')


def extraer_publicaciones_arxiv(categoria, max_resultados=1, ordenar_por='submittedDate', orden_descendente=True):
    """
    Extrae publicaciones académicas desde la API de arXiv.org según una categoría especificada.

    Realiza una consulta HTTP a la API de arXiv y transforma la respuesta en un DataFrame estructurado con los metadatos relevantes.

    Args:
        categoria (str): Categoría de arXiv (por ejemplo, 'cs.AI').
        max_resultados (int): Número máximo de publicaciones a recuperar. Por defecto es 1.
        ordenar_por (str): Criterio de ordenación . Por defecto es 'submittedDate'.
        orden_descendente (bool, optional): Si True, ordena los resultados de forma descendente. Por defecto es True.

    Returns:
        DataFrame: DataFrame con las publicaciones encontradas, incluyendo título, autores, resumen,
        fecha de publicación, categorías, URL del PDF, identificador de arXiv y categoría principal.

    Raises:
        requests.exceptions.RequestException: Si ocurre un error al hacer la solicitud HTTP.
        Exception: Si ocurre un error inesperado durante el procesamiento de los datos.
    """

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
    
def extraccion_por_categorias(conn, categorias_id_dict ,max_resultados=1):
    """
    Extrae publicaciones de arXiv iterando por múltiples categorías y devuelve los resultados nuevos.

    Esta función ejecuta `extraer_publicaciones_arxiv` para cada categoría presente en `categorias_id_dict`,
    recupera los metadatos de las publicaciones y filtra aquellas ya presentes en la base de datos
    para evitar duplicados.

    Args:
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.
        categorias_id_dict (dict): Diccionario donde las claves son los identificadores de categorías de arXiv.
        max_resultados (int, optional): Número máximo de resultados a extraer por categoría. Por defecto es 1.

    Returns:
        DataFrame: DataFrame con los metadatos de las publicaciones nuevas (no insertadas previamente en la BBDD).

    Logs:
        - Informa del inicio y fin de la extracción por cada categoría.
        - Reporta errores en la conexión o consulta a la base de datos.
    """
    df_lst = []
    for i in tqdm(categorias_id_dict.keys()):
        #print(f' Extrayendo la categoria: {categorias_arxiv[i]}...')
        # Descargar metadatos de la categoria
        logger.debug(f"Descargando metadatos de la categoria {categorias_id_dict[i]}...")
        # DF con los metadatos de la categoria
        metadatos_categoria = extraer_publicaciones_arxiv(i, max_resultados)
        df_lst.append(metadatos_categoria)

    # Concatenacion
    df_metadata_total = pd.concat(df_lst, ignore_index=True)
    # ----- Se eliminan posibles duplicados ------
    try:
        id_insertados_lst = consulta_id_arxiv(conn)
    except Exception as e:
        logger.error(f"Error al consultar la base de datos: {e}")
        id_insertados_lst = []
    # Se filtra el DataFrame para eliminar los registros que ya están en la base de datos
    df_metadata_total = df_metadata_total[~df_metadata_total['identificador_arxiv'].isin(id_insertados_lst)]

    return df_metadata_total


# ----- Funciones para filtrado en la extraccion ------
def consulta_id_arxiv(conn) -> list:
    """
    Consulta la base de datos y obtiene todos los identificadores únicos de publicaciones de arXiv.

    Esta función accede a la tabla publicaciones de la base de datos y devuelve una lista
    con todos los valores únicos de la columna identificador_arxiv.

    Args:
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.

    Returns:
        list: Lista de strings, cada uno representando un identificador de arXiv único.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT identificador_arxiv FROM publicaciones;")
        result = cur.fetchall()
    return list(set([row['identificador_arxiv'] for row in result]))


