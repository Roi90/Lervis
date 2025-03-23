"""
Este archivo contiene funciones para conectarse a una base de datos PostgreSQL.
Proporciona métodos para crear un motor de conexión.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""

from sqlalchemy import create_engine
from Static_data import categorias_arxiv
import pandas as pd

def engine_bbdd():
    """
    Crea y devuelve un motor de conexión a la base de datos PostgreSQL.
    La función utiliza una URL de conexión predefinida para conectarse a una base de datos PostgreSQL
    y crea un motor de conexión utilizando SQLAlchemy.
    Returns:
        engine (sqlalchemy.engine.base.Engine): Motor de conexión a la base de datos.
    """


    # URL de conexión
    DATABASE_URL = "postgresql://postgres:Quiksilver90!@localhost:5432/Lervis"

    # Crear motor de conexión
    engine = create_engine(DATABASE_URL)

    return engine

def carga_dimension_categorias(engine):
    """
    Carga la dimensión de categorías en la base de datos.
    Esta función toma un diccionario global `categorias_arxiv` que contiene códigos de categorías y sus descripciones,
    y los inserta en una tabla llamada 'categoria' en la base de datos especificada por el parámetro `engine`.
    Args:
        engine (sqlalchemy.engine.Engine): Conexión a la base de datos donde se insertarán los datos.
    Returns:
        dict: Diccionario con los códigos de categoría como claves y los IDs de la base de datos como valores.
    """
    categoria_dict = {}
    df_structure = {
        "codigo_categoria" : categorias_arxiv.keys(),
        "categoria" : categorias_arxiv.values(),
    }

    df = pd.DataFrame(df_structure)
    # Insercion de datos
    df.to_sql('categoria', con=engine, if_exists='append', index=False)

    query = "SELECT * FROM categoria"

    # Extraccion de datos
    categorias_df = pd.read_sql(query, con=engine)

    for row in categorias_df.itertuples():
        categoria_dict[row.codigo_categoria] = row.id
    
    return categoria_dict

def carga_hechos_publicaciones(engine, df: pd.DataFrame):
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

    # Insercion de datos evitando la columna resumen
    df[columnas_para_insercion].to_sql('publicaciones', con=engine, if_exists='append', index=False)

    query = "SELECT id, identificador_arxiv FROM publicaciones"
    
    # Extraccion de datos
    publicaciones_df = pd.read_sql(query, con=engine)

    for row in publicaciones_df.itertuples():
        publicaciones_dict[row.identificador_arxiv] = row.id
    
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

    

