"""
Este archivo contiene funciones para conectarse a una base de datos PostgreSQL.
Proporciona métodos para crear un motor de conexión.

Autor: Roi Pereira Fiuza
"""

from sqlalchemy import create_engine
from Static_data import categorias_arxiv
import pandas as pd
import numpy as np

import psycopg
from psycopg.rows import dict_row

# ---------------------------TO DO: Crear un archivo con las variables para seguridad
# --------------------------- CREAR OBJETOS PARA MANEJAR LA BBDD
# --------------------------- CREAR FUNCIONES EN POSTGRES PARA TRASLADAR EL CALCULO A LA BBDD.
def conn_bbdd():
    """
    Crea y devuelve un motor de conexión a la base de datos PostgreSQL.
    La función utiliza una URL de conexión predefinida para conectarse a una base de datos PostgreSQL
    y crea un motor de conexión utilizando SQLAlchemy.
    Returns:
        engine (sqlalchemy.engine.base.Engine): Motor de conexión a la base de datos.
    """
    # ---------- GUARDAR SECRETOS EN UN ARHCIVO EXTERNO A ESTE ENTORNO
    # URL de conexión
    DATABASE_URL = "postgresql://postgres:Quiksilver90!@localhost:5432/Lervis"
    # Crear motor de conexión
    #engine = create_engine(DATABASE_URL)
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)

    return conn

def carga_dimension_categorias(conn):
    """
    Carga la dimensión de categorías en la base de datos usando psycopg.
    Args:
        conn (psycopg.Connection): Conexión a la base de datos.
    Returns:
        dict: Diccionario con los códigos de categoría como claves y los IDs de la base de datos como valores.
    """
    categoria_dict = {}
    
    # Convertir los datos a DataFrame
    df = pd.DataFrame({
        "codigo_categoria": list(categorias_arxiv.keys()),
        "categoria": list(categorias_arxiv.values()),
    })
    
    # Inserción de datos en la tabla 'categoria'
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute(
                """
                INSERT INTO categoria (codigo_categoria, categoria)
                VALUES (%s, %s);
                """, (row["codigo_categoria"], row["categoria"])
            )
        conn.commit()
    
    # Extracción de datos para generar el diccionario
    with conn.cursor() as cur:
        cur.execute("SELECT id, codigo_categoria FROM categoria;")
        for row in cur.fetchall():
            categoria_dict[row['codigo_categoria']] = row['id'] 
    
    return categoria_dict

def carga_hechos_publicaciones(conn, df: pd.DataFrame):
    """
    Carga los hechos de publicaciones en la base de datos.
    Esta función toma un DataFrame `df` que contiene datos de publicaciones y los inserta en una tabla llamada 'publicaciones'
    en la base de datos especificada por el parámetro `engine`.
    Args:
        engine (sqlalchemy.engine.Engine): Conexión a la base de datos donde se insertarán los datos.
        df (pd.DataFrame): DataFrame que contiene los datos de publicaciones a insertar.
    Returns:
        dict: Diccionario con los identificadores de arXiv como claves y los IDs de la base de datos como valores.
    """
    publicaciones_dict = {}

    columnas_para_insercion = ['titulo', 'autores','fecha_publicacion',
                          'categoria_principal', 'categorias_lista', 'url_pdf', 'identificador_arxiv']

     # Inserción de datos en la tabla 'publicaciones' usando psycopg
    with conn.cursor() as cur:
        for _, row in df[columnas_para_insercion].iterrows():
            cur.execute(
                """
                INSERT INTO publicaciones (titulo, autores, fecha_publicacion, categoria_principal,
                                           categorias_lista, url_pdf, identificador_arxiv)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (row['titulo'], row['autores'], row['fecha_publicacion'],
                 row['categoria_principal'], row['categorias_lista'], row['url_pdf'], row['identificador_arxiv'])
            )
        conn.commit()

    # Extracción de datos (id, identificador_arxiv) desde la tabla 'publicaciones'
    with conn.cursor() as cur:
        cur.execute("SELECT id, identificador_arxiv FROM publicaciones;")
        for row in cur.fetchall():
            publicaciones_dict[row['identificador_arxiv']] = row['id']
    
    return publicaciones_dict

def normalizador_id_categoria_BBDD(df: pd.DataFrame, diccionario: dict):
    """
    Normaliza los IDs de las categorías en el DataFrame utilizando un diccionario de mapeo.
    Este paso se realiza para la normalizacion de las dimensiones en la tabla de hechos publicaciones.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos a normalizar.
        diccionario (dict): Diccionario que mapea códigos de categorías a IDs de la base de datos.

    Returns:
        pd.DataFrame: DataFrame con los IDs de categorías normalizados.
    """

    columnas_esperadas = ['titulo', 'autores','fecha_publicacion','resumen',
                          'categoria_principal', 'categorias_lista', 'url_pdf', 'identificador_arxiv']
    
    # Validamos que el DataFrame tiene las columnas esperadas
    if not all(col in df.columns for col in columnas_esperadas):
        raise ValueError(f"Las columnas del DataFrame no coinciden con las esperadas: {columnas_esperadas}")
    
    try:
        # Normaliza la columna 'categoria_principal' usando el diccionario
        df['categoria_principal'] = df['categoria_principal'].map(diccionario)

    except KeyError as e:
        raise KeyError(f"Categoría no encontrada en el diccionario de mapeo: {e}")

    return df[columnas_esperadas]

def normalizador_id_embeddings_BBDD(df: pd.DataFrame, diccionario: dict):
    """
    Normaliza los IDs de las categorías en el DataFrame utilizando un diccionario de mapeo.
    Este paso se realiza para la normalizacion de las dimensiones en la tabla de hechos publicaciones.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos a normalizar.
        diccionario (dict): Diccionario que mapea códigos de categorías a IDs de la base de datos.

    Returns:
        pd.DataFrame: DataFrame con los IDs de categorías normalizados.
    """
    columnas_esperadas = ['id_publicaciones', 'contenido','contenido_emb_dense',
                          'contenido_emb_sparse', 'resumen', 'resumen_emb_dense', 'resumen_emb_sparse']
    
    # Validamos que el DataFrame tiene las columnas esperadas
    if not all(col in df.columns for col in columnas_esperadas):
        raise ValueError(f"Las columnas del DataFrame no coinciden con las esperadas: {columnas_esperadas}")
    
    try:
        # Normaliza la columna 'categoria_principal' usando el diccionario
        df['id_publicaciones'] = df['id_publicaciones'].map(diccionario)

    except KeyError as e:
        raise KeyError(f"Categoría no encontrada en el diccionario de mapeo: {e}")

    return df[columnas_esperadas]

def similitud_coseno(embedding_denso: np.array, engine, similitud:float):
    """
    Calcula la similitud del coseno entre dos vectores.
    Esta función toma dos vectores y calcula la similitud del coseno entre ellos.
    Returns:
        float: Similitud del coseno entre los dos vectores.
    """
    # Queries para recuperar en funcion del embedding denso en funcion de dos parametros el embedding y el umbral de similitud
    # En donde se recuperan los 5 documentos mas similares y se ordenan en funcion de las dos similitudes calculadas.
    query = """
        SELECT id_publicaciones, contenido, resumen,
               1 - public.cosine_distance(contenido_emb_dense, :embedding_denso) AS similitud_contenido,
               1 - public.cosine_distance(resumen_emb_dense, :embedding_denso) AS similitud_resumen
        FROM embeddings
        WHERE 1 - public.cosine_distance(contenido_emb_dense, :embedding_denso) > :similitud
        OR 1 - public.cosine_distance(resumen_emb_dense, :embedding_denso) > :similitud
        ORDER BY GREATEST(similitud_contenido, similitud_resumen) DESC
        LIMIT 5;
    """
    documentos_recuperados = pd.read_sql(query, con=engine, params={"embedding_denso": embedding_denso.astype(np.float32).tolist(), "similitud": similitud})

    if documentos_recuperados.empty:
        return "No se si te he entendido bien, pero ¿Podrías detallar un poco mas tu pregunta?"

        # Devolver el resumen del documento más relevante
    return documentos_recuperados.iloc[0]['resumen']

    
    

