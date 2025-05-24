"""
Este archivo contiene funciones para conectarse a una base de datos PostgreSQL.
Proporciona métodos para crear un motor de conexión.

Autor: Roi Pereira Fiuza
"""
import json
import os
from Static_data import categorias_arxiv
import pandas as pd
import numpy as np
from psycopg import sql
import psycopg
from psycopg.rows import dict_row
from Functions.Loggers import crear_logger

logger = crear_logger('funciones_BDDD', 'funciones_BDDD.log')

# Extraccion de user y pass  
with open(r'C:\Users\Usuario\OneDrive\UOC\TFG\Lervis\Lervis\passwords.json', 'r') as e:
    data = json.load(e)
user = data['user']
password = data['password']

def conn_bbdd():
    """
    Establece una conexión con una base de datos PostgreSQL y devuelve el objeto de conexión.

    Utiliza psycopg para crear una conexión a la base de datos definida por DATABASE_URL,
    aplicando row_factory=dict_row para obtener los resultados como diccionarios.

    Returns:
        psycopg.Connection: Objeto de conexión activo a la base de datos PostgreSQL.

    Raises:
        Exception: Si ocurre un error durante la conexión, se registra y se lanza de nuevo.
    """
    # ---------- GUARDAR SECRETOS EN UN ARHCIVO EXTERNO A ESTE ENTORNO
    # URL de conexión
    DATABASE_URL = f"postgresql://{user}:{password}@localhost:5432/Lervis"
    # Crear motor de conexión
    #engine = create_engine(DATABASE_URL)
    try:
        # Crear conexión a la base de datos usando psycopg
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise


    return conn
# --------------------------- Carga de datoss en la BBDD
def carga_dimension_categorias(conn):
    """
    Inserta las categorías de arXiv en la tabla categoria de la base de datos y retorna un diccionario de mapeo.

    Esta función carga los datos desde el diccionario categorias_arxiv a la base de datos PostgreSQL,
    y luego construye un diccionario que mapea cada código de categoría a su ID correspondiente en la tabla.

    Args:
        conn (psycopg.Connection): Objeto de conexión a la base de datos PostgreSQL.

    Returns:
        dict: Diccionario donde las claves son códigos de categoría (str) y los valores son los IDs (int) en la tabla categoria.

    Raises:
        Exception: Si ocurre un error al recuperar los datos tras la inserción, la excepción se lanza nuevamente.
    """
    categoria_dict = {}
    
    # Convertir los datos a DataFrame
    df = pd.DataFrame({
        "codigo_categoria": list(categorias_arxiv.keys()),
        "categoria": list(categorias_arxiv.values()),
    })
    try:
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
    except Exception as e:
        logger.error(f"Error al insertar datos en la tabla 'categoria': {e}")
        
    try:
        # Extracción de datos para generar el diccionario
        with conn.cursor() as cur:
            cur.execute("SELECT id, codigo_categoria FROM categoria;")
            for row in cur.fetchall():
                categoria_dict[row['codigo_categoria']] = row['id'] 
    except Exception as e:
        logger.error(f"Error al recuperar datos de la tabla 'categoria': {e}")
        raise
    return categoria_dict

def dict_catetorias(conn):
    """
    Recupera un diccionario que mapea códigos de categoría a sus respectivos IDs desde la base de datos.

    Esta función consulta la tabla categoria de la base de datos y construye un diccionario donde 
    cada clave es un código de categoría y su valor es el ID correspondiente.

    Args:
        conn (psycopg.Connection): Objeto de conexión a la base de datos PostgreSQL.

    Returns:
        dict: Diccionario con claves como códigos de categoría (str) y valores como IDs (int).

    Raises:
        Exception: Si ocurre un error durante la consulta SQL.
    """
    categoria_dict = {}
    try:
        # Extracción de datos para generar el diccionario
        with conn.cursor() as cur:
            cur.execute("SELECT id, codigo_categoria FROM categoria;")
            for row in cur.fetchall():
                categoria_dict[row['codigo_categoria']] = row['id'] 
    except Exception as e:
        logger.error(f"Error al recuperar datos de la tabla 'categoria': {e}")
        raise
    return categoria_dict

def carga_hechos_publicaciones(conn, df: pd.DataFrame):
    """
    Inserta los datos de publicaciones en la base de datos y devuelve un diccionario de identificadores.

    Esta función toma un DataFrame que contiene información sobre publicaciones científicas extraídas de arXiv
    y las inserta en la tabla publicaciones de la base de datos. 
    
    Posteriormente, recupera y devuelve un diccionario que relaciona cada identificador de arXiv con su ID asignado en la base de datos.

    Args:
        conn (psycopg.Connection): Objeto de conexión a la base de datos PostgreSQL.
        df (DataFrame): DataFrame que contiene los datos de publicaciones a insertar. Debe contener las columnas:
        'titulo', 'autores', 'fecha_publicacion', 'categoria_principal',
        'categorias_lista', 'url_pdf', 'identificador_arxiv'.

    Returns:
        dict: Diccionario donde las claves son identificadores de arXiv (str) y los valores son IDs (int)
              generados en la tabla 'publicaciones'.

    Raises:
        Exception: Si ocurre un error durante la inserción o recuperación de datos en la base de datos.
    """
    publicaciones_dict = {}

    columnas_para_insercion = ['titulo', 'autores','fecha_publicacion',
                          'categoria_principal', 'categorias_lista', 'url_pdf', 'identificador_arxiv']
    try:
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
    except Exception as e:
        logger.error(f"Error al insertar datos en la tabla 'publicaciones': {e}")
    try:
        # Extracción de datos (id, identificador_arxiv) desde la tabla 'publicaciones'
        with conn.cursor() as cur:
            cur.execute("SELECT id, identificador_arxiv FROM publicaciones;")
            for row in cur.fetchall():
                publicaciones_dict[row['identificador_arxiv']] = row['id']
    except Exception as e:
        logger.error(f"Error al recuperar datos de la tabla 'publicaciones': {e}")
    
    return publicaciones_dict

def carga_hechos_chunks_embeddings(df: pd.DataFrame, engine):
    """
    Inserta los datos de embeddings por chunks en la base de datos.

    Esta función toma un DataFrame que contiene los embeddings generados por fragmento (chunk)
    y los inserta en la tabla embeddings_chunks. Se usa el DataFrame, ya que facilita la insercion
    de los vetores dispersos HSTORE.

    Args:
        df (DataFrame): DataFrame con los datos de embeddings a insertar. Debe contener todas las columnas
        requeridas por la tabla 'embeddings_chunks'.
        engine (sqlalchemy.engine.Engine): Motor de conexión a la base de datos PostgreSQL.

    Returns:
        None

    Raises:
        Exception: Si ocurre un error durante la inserción de los datos.
    """ 
    try:
        # Insertar el DataFrame en la base de datos, tabla 'embeddings'
        return df.to_sql('embeddings_chunks', con=engine, if_exists='append', index=False)
    except Exception as e:
        logger.error(f"Error al insertar datos en la tabla 'embeddings_chunks': {e} - id {df['id_publicaciones']}")
        

def carga_hechos_resumen_embeddings(df: pd.DataFrame, engine):
    """
    Inserta los datos de embeddings por resumen en la base de datos.

    Esta función toma un DataFrame que contiene los embeddings generados a partir de resúmenes
    y los inserta en la tabla embeddings_resumen. Se usa el DataFrame, ya que facilita la inserción
    de los vectores dispersos HSTORE.

    Args:
        df (DataFrame): DataFrame con los datos de embeddings a insertar. Debe contener todas las columnas
        requeridas por la tabla 'embeddings_resumen'.
        engine (sqlalchemy.engine.Engine): Motor de conexión a la base de datos PostgreSQL.

    Returns:
        None

    Raises:
        Exception: Si ocurre un error durante la inserción de los datos.
    """
    try:
        # Insertar el DataFrame en la base de datos, tabla 'embeddings'
        return df.to_sql('embeddings_resumen', con=engine, if_exists='append', index=False)
    except  Exception as e:
        logger.error(f"Error al insertar datos en la tabla 'embeddings_resumen': {e} - id {df['id_publicaciones']}")

def carga_doc_enriquecido(documento_enriquecido, identificador_arxiv,  conn):
    """
    Actualiza el documento enriquecido en la base de datos.

    Esta función actualiza el campo documento_completo de la tabla publicaciones para el 
    registro correspondiente al identificador de arXiv proporcionado.

    Args:
        documento_enriquecido (str): Documento enriquecido a insertar en la base de datos.
        identificador_arxiv (str): Identificador único de la publicación en arXiv.
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.

    Returns:
        None

    Raises:
        Exception: Si ocurre un error durante la actualización de los datos.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE publicaciones
                        SET documento_completo = %s
                        WHERE identificador_arxiv = %s;""", (documento_enriquecido, identificador_arxiv))
            conn.commit()
    except Exception as e:
        logger.error(f"Error al insertar datos en la tabla 'publicaciones': {e} - id {identificador_arxiv}")
        

def normalizador_id_categoria_BBDD(df: pd.DataFrame, diccionario: dict):
    """
    Normaliza los IDs de las categorías en el DataFrame usando un diccionario de mapeo.

    Esta función transforma los valores de la columna categoria_principal del DataFrame
    utilizando un diccionario que relaciona códigos de categoría con sus correspondientes IDs
    en la base de datos. 
    
    Se utiliza para asegurar la integridad referencial al cargar datos en la tabla de hechos 
    publicaciones.

    Args:
        df (DataFrame): DataFrame con las publicaciones que contienen códigos de categorías.
        diccionario (dict): Diccionario que mapea códigos de categoría (str) a IDs de la BBDD (int).

    Returns:
        DataFrame: DataFrame con la columna categoria_principal normalizada por IDs,
        y limitado a las columnas relevantes para su posterior inserción.

    Raises:
        ValueError: Si el DataFrame no contiene todas las columnas esperadas.
        Exception: Si ocurre un error durante la normalización de la columna.
    """

    columnas_esperadas = ['titulo', 'autores','fecha_publicacion','resumen',
                          'categoria_principal', 'categorias_lista', 'url_pdf', 'identificador_arxiv']
    
    # Validamos que el DataFrame tiene las columnas esperadas
    if not all(col in df.columns for col in columnas_esperadas):
        raise ValueError(f"Las columnas del DataFrame no coinciden con las esperadas: {columnas_esperadas}")
    
    try:
        # Normaliza la columna 'categoria_principal' usando el diccionario
        df['categoria_principal'] = df['categoria_principal'].map(diccionario)

    except Exception as e:
        logger.error(f"Error al normalizar la columna 'categoria_principal': {e}")
        raise

    return df[columnas_esperadas]

def normalizador_id_embeddings_BBDD(df: pd.DataFrame, diccionario: dict):
    """
    Normaliza los IDs de publicaciones en el DataFrame usando un diccionario de mapeo.

    Esta función reemplaza los valores en la columna id_publicaciones por los correspondientes IDs
    obtenidos de la base de datos. Se utiliza para asegurar que los embeddings se asocien correctamente
    con sus registros en la tabla de hechos de publicaciones.

    Args:
        df (DataFrame): DataFrame con los datos de embeddings a normalizar. Debe contener las columnas 
        esperadas para chunks y resúmenes.
        diccionario (dict): Diccionario que mapea identificadores originales a IDs de la BBDD.

    Returns:
        DataFrame: DataFrame con la columna id_publicaciones normalizada por IDs.

    Raises:
        ValueError: Si el DataFrame no contiene todas las columnas esperadas.
        Exception: Si ocurre un error durante la normalización de la columna.
    """
    columnas_esperadas = ['id_publicaciones', 'contenido','contenido_emb_dense',
                          'contenido_emb_sparse', 'resumen', 'resumen_emb_dense', 'resumen_emb_sparse']
    
    # Validamos que el DataFrame tiene las columnas esperadas
    if not all(col in df.columns for col in columnas_esperadas):
        raise ValueError(f"Las columnas del DataFrame no coinciden con las esperadas: {columnas_esperadas}")
    
    try:
        # Normaliza la columna 'categoria_principal' usando el diccionario
        df['id_publicaciones'] = df['id_publicaciones'].map(diccionario)

    except Exception as e:
        logger.error(f"Error al normalizar la columna 'id_publicaciones': {e}")
        raise


    return df[columnas_esperadas]

    

# ---------------------------- Recuperacion documentos y formateo para contexto

def similitud_coseno_chunks(embedding_denso: np.array, conn, umbral_distancia, n_docs)-> dict:
    """
    Recupera los fragmentos (chunks) más similares a un vector de embedding denso usando similitud del coseno.

    Esta función consulta la base de datos para encontrar los fragmentos de texto (chunks) cuyo vector de embedding 
    denso tiene mayor similitud con respecto a un vector de entrada. 

    Args:
        embedding_denso (np.array): Vector de embedding denso a comparar.
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.
        umbral_distancia (float): Umbral máximo de distancia del coseno permitido para considerar un chunk como relevante.
        n_docs (int): Número máximo de documentos a recuperar.

    Returns:
        list: Lista de diccionarios con los chunks recuperados que cumplen con el umbral de similitud,
        ordenados de mayor a menor similitud. Cada entrada incluye id, id_publicaciones, chunk_emb_sparse,
        chunk y la similitud_contenido.

    Raises:
        Exception: Si ocurre un error durante la ejecución de la consulta SQL.
    """

    embedding = embedding_denso.tolist()
    try:
        with conn.cursor() as cur:
            
            cur.execute("""
                        SELECT id, id_publicaciones, chunk_emb_sparse, chunk,
                            chunk_emb_dense <=> %s::vector AS similitud_contenido
                        FROM embeddings_chunks
                        WHERE (chunk_emb_dense <=> %s::vector) <= %s
                        ORDER BY chunk_emb_dense <=> %s::vector ASC
                        LIMIT %s;
                        """, (embedding, embedding, umbral_distancia, embedding, n_docs))
            # Recuperar los resultados
            documentos_recuperados = cur.fetchall()
            logger.debug(f"Consultando BBDD chunks (SIMILITUD COSENO) - {umbral_distancia}, Docs recuperados - {len(documentos_recuperados)}")
            return documentos_recuperados
    except Exception as e:
        logger.error(f"Error al consultar la base de datos: {e}")
        return []
        

def similitud_coseno_resumen(embedding_denso: np.array, conn, umbral_distancia, n_docs)-> dict:
    """
    Recupera los resúmenes más similares a un vector de embedding denso usando similitud del coseno.

    Esta función consulta la tabla embeddings_resumen para encontrar los resúmenes de documentos 
    cuyo vector de embedding denso presenta la menor distancia del coseno respecto al vector de entrada.

    Args:
        embedding_denso (np.array): Vector de embedding denso del resumen de entrada.
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.
        umbral_distancia (float): Umbral máximo de distancia coseno para considerar un resumen como relevante.
        n_docs (int): Número máximo de resúmenes/documentos a recuperar.

    Returns:
        list: Lista de diccionarios con los chunks recuperados que cumplen con el umbral de similitud,
        ordenados de mayor a menor similitud. Cada entrada incluye id, id_publicaciones, chunk_emb_sparse,
        chunk y la similitud_contenido.

    Raises:
        Exception: Si ocurre un error durante la consulta a la base de datos.
    """

    embedding = embedding_denso.tolist()
    try:
        with conn.cursor() as cur:
            
            cur.execute("""
                        SELECT id, id_publicaciones, resumen_emb_sparse, resumen,
                            resumen_emb_dense <=> %s::vector AS similitud_contenido
                        FROM embeddings_resumen
                        WHERE (resumen_emb_dense <=> %s::vector) <= %s
                        ORDER BY resumen_emb_dense <=> %s::vector ASC
                        LIMIT %s;
                        """, (embedding, embedding, umbral_distancia, embedding, n_docs))
            # Recuperar los resultados
            documentos_recuperados = cur.fetchall()
            logger.debug(f"Consultando BBDD resumenes (SIMILITUD COSENO) - {umbral_distancia}, Docs recuperados - {len(documentos_recuperados)}")
            return documentos_recuperados
    except Exception as e:
        logger.error(f"Error al consultar la base de datos: {e}")
        return []
    
def hstore_a_dict(hstore_str: str) -> dict:
    """
    Convierte una cadena en formato HSTORE a un diccionario.

    Esta función toma una cadena con formato HSTORE y la convierte a un diccionario con claves 
    tipo str y valores tipo float.

    Args:
        hstore_str (str): String en formato HSTORE, con pares clave/valor separados por '=>'.

    Returns:
        dict: Diccionario Python con los valores convertidos. Devuelve un diccionario vacío si falla la conversión.

    Raises:
        Exception: Solo se registra el error. No se lanza, para evitar la interrupción del flujo.
    """
    try:
        hstore_dict = dict(
                            map(lambda x: (x[0], float(x[1])),
                            [hstore.replace('"', '').split('=>') for hstore in hstore_str.split(', ')])  # Convertir a float
        )
    except Exception as e:
        logger.error(f"Error al convertir HSTORE a dict: {e}")
        # Si no se puede convertir, devolver un diccionario vacío
        return {}

    return hstore_dict

def reranking(documentos_recuperados, embedding_disperso):
    """
    Reordena documentos basándose en similitud de contenido y vocabulario.

    Esta función transforma los vectores dispersos HSTORE de los documentos recuperados a diccionarios,
    calcula la similitud de vocabulario como la intersección de claves entre los embeddings,
    y luego ordena los documentos por similitud de contenido y vocabulario.

    Args:
        documentos_recuperados (list of dict): Lista de documentos recuperados desde la base de datos.
        embedding_disperso (dict): Embedding disperso del prompt de búsqueda.

    Returns:
        DataFrame: DataFrame ordenado, primero por similitud de contenido (ascendente)
        y luego por similitud de vocabulario (descendente).

    Raises:
        Exception: Se registra cualquier error encontrado durante el reordenamiento.
    """
    try:
        df_temp = pd.DataFrame(documentos_recuperados)
        # Transformacion de la columna hstore a dict
        df_temp['chunk_emb_sparse'] = df_temp['chunk_emb_sparse'].apply(lambda x: hstore_a_dict(x))
        # Creacion de la columna mediante la interseccion del conjunto de ambas claves

        df_temp['similitud_vocabulario'] = df_temp['chunk_emb_sparse'].apply(lambda x: len(set(x.keys()).intersection(set(embedding_disperso.keys()))))
        # Ordeno el DF por similitud de contenido y similitud de vocabulario
        df_temp = df_temp.sort_values(by=['similitud_contenido','similitud_vocabulario'], ascending=[True, False])
        return df_temp
    except Exception as e:
        logger.error(f"Error al reordenar los documentos: {e}")
        return df_temp

def formatear_metadata(doc):

    """
    Formatea los metadatos de una publicación científica en un bloque de texto legible.

    Esta función toma un diccionario con información de una publicación de arXiv y devuelve
    una cadena formateada en estilo Markdown con título, ID, autores, fecha, URL y resumen.

    Args:
        doc (dict): Diccionario con los campos necesarios, incluyendo:
            - titulo (str)
            - identificador_arxiv (str)
            - categoria (str)
            - autores (str)
            - fecha_publicacion (str)
            - url_pdf (str)
            - resumen (str)

    Returns:
        str: Texto formateado en estilo Markdown, orientado para la lectura del LLM.
    """

    return (
        "\n**----------------------------**\n"
        f"**Titulo**: {doc['titulo']}\n"
        f"**Arxiv ID**: {doc['identificador_arxiv']}\n"
        f"**Categoria**: {doc['categoria']}\n"
        f"**Autores**:\n" + "\n".join(f"- {a}" for a in doc['autores'].split(",")) + "\n"
        f"**Fecha de publicacion**: {doc['fecha_publicacion']}\n"
        f"**URL de la publicacion**: {doc['url_pdf']}\n"
        f"**Resumen**: {doc['resumen']}\n"
        "**----------------------------**\n"
    )


def formato_contexto_doc_recuperados(urls_usados, conn, df: pd.DataFrame, num_docs: int = 2) -> str:
    """
    Esta funcion formatea los documentos recuperados de la base de datos en la recuperacion de
    documentos.

    Esta función toma los identificadores de los documentos de un DataFrame,
    recupera sus metadatos y resúmenes desde la base de datos, y genera un texto formateado
    para ser usado como contexto en el LLM.

    Se evita la duplicación utilizando el conjunto urls_usados.

    Args:
        urls_usados (set): Conjunto que almacena las URLs ya formateadas para evitar duplicaciones.
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.
        df (DataFrame): DataFrame ordenado con los documentos recuperados y sus puntuaciones.
        num_docs (int): Número de documentos a incluir en el contexto. Por defecto 2.

    Returns:
        str: Texto formateado con los metadatos de los documentos seleccionados. Si ocurre un error,
             se devuelve una cadena vacía.
    """

    documentos_formateados = ''

    try:
        # Subconjunto de documentos recuperados
        df_top_docs = df.head(num_docs)
        # Extraer los IDs de los documentos
        id_publicaciones = list(set(df_top_docs['id_publicaciones']))
        # Formateo para correcta ejecucion de cur.execute
        params = tuple(id_publicaciones)
        # Formateo para poder introducir una tupla (%s, %s, %s...)
        placeholders_id_publicaciones = sql.SQL(', ').join([sql.Placeholder() for _ in id_publicaciones])
        
        with conn.cursor() as cur:
            query = sql.SQL("""SELECT 
                        titulo,
                        CAT.categoria,
                        autores,
                        fecha_publicacion,
                        url_pdf,
                        identificador_arxiv,
                        ER.resumen
                        FROM publicaciones
                        LEFT JOIN categoria as CAT
                            ON publicaciones.categoria_principal = CAT.id
                        LEFT JOIN embeddings_resumen as ER
                            ON publicaciones.id = ER.id_publicaciones
                        WHERE publicaciones.id in ({})""").format(placeholders_id_publicaciones)
            cur.execute(query, params)
            # Recuperar los resultados (Lista de diccionarios)
            documentos_recuperados = cur.fetchall()
        # Formateo e insercion de los chunks y metadatos para el contexto
        docs_insertados = 0
        for doc in documentos_recuperados:
            # Se usa la secuencia del DF que esta ordenado
            url   = doc['url_pdf'].strip()

            # Compruebo que no se ha insertado para evitar duplicacion de metadatos.
            if url not in urls_usados:
                if docs_insertados == num_docs:
                    break
                    
                else:
                    # Nuevo documento: metadata
                    urls_usados.add(url)
                    documentos_formateados += formatear_metadata(doc)
                    docs_insertados += 1

        return documentos_formateados
    except Exception as e:
        logger.error(f"Error al formatear los documentos recuperados: {e}")
        return ""

def temporalidad_a_SQL(conn, temporalidad: tuple):
    """
    Genera y ejecuta una consulta SQL para contar publicaciones según una condición temporal.

    Esta función interpreta la estructura de la temporalidad extraída del input del usuario
    y construye dinámicamente una consulta SQL para recuperar el número de publicaciones
    en la base de datos que cumplen con las condiciones de tiempo (mes, año o ambas).

    Args:
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.
        temporalidad (tuple): Tupla con tres elementos:
            - tipo (str): Puede ser Combinada, Mes, Anio o EXP.
            - valores (dict o list o int): Información temporal según el tipo.
            - texto_usuario (str): Texto original proporcionado por el usuario.

    Returns:
        int or None: Número de publicaciones que cumplen con la condición temporal,
        o None si ocurre un error en la consulta.

    Raises:
        No se lanzan errores. Se registran mediante logger y se devuelve `None` en caso de fallo.
    """

    if temporalidad is not None:
        
        if temporalidad[0] == 'Combinada':
            try:
                # Extraigo los meses y años del diccionario
                meses_list = temporalidad[1]['Meses']   #  [1, 2, 3]
                anios_list = temporalidad[1]['Anios']   #  [2023, 2024]

                # Genero los Composed de "%s, %s, %s" para cada lista
                mes_ph  = sql.SQL(', ').join(sql.Placeholder() for _ in meses_list)
                anio_ph = sql.SQL(', ').join(sql.Placeholder() for _ in anios_list)

                # Uso de sql.SQL + .format() para que los placeholders puedan ser usados por psycopg
                query = sql.SQL("""
                    SELECT COUNT(*) AS conteo_publicaciones
                    FROM publicaciones
                    WHERE MES  IN ({meses})
                    AND ANIO IN ({anios});
                """).format( 
                    meses=mes_ph,  # Insercion de los placeholders de meses
                    anios=anio_ph # Insercion de los placeholders de anios
                )
                # Preparo la tupla de parámetros con los valores reales
                params = tuple(meses_list) + tuple(anios_list)
                logger.debug(f"Consultando BBDD (TEMPORALIDAD) {temporalidad[0]}")
                with conn.cursor() as cur:
                    cur.execute(query, params) # En esta llamada se rellenan los placeholders
                    result = cur.fetchone()
                    return result['conteo_publicaciones']
            except Exception as e:
                logger.error(f"Error al consultar la base de datos: {e} en {temporalidad}")
                return None
        
        elif temporalidad[0] == 'Mes': # Fomato en lista

            try:

                meses_list = temporalidad[1]
                # Genero los Composed de "%s, %s, %s" para la lista
                mes_ph  = sql.SQL(', ').join(sql.Placeholder() for _ in meses_list)

                # Uso de sql.SQL + .format() para que los placeholders puedan ser usados por psycopg
                query = sql.SQL("""
                    SELECT COUNT(*) AS conteo_publicaciones
                    FROM publicaciones
                    WHERE MES  IN ({meses});
                """).format( 
                    meses=mes_ph,  # Insercion de los placeholders de meses
                )
                # Preparo la tupla de parámetros con los valores reales
                params = tuple(meses_list)
                logger.debug(f"Consultando BBDD (TEMPORALIDAD) {temporalidad[0]}")
                with conn.cursor() as cur:
                    cur.execute(query, params) # En esta llamada se rellenan los placeholders
                    result = cur.fetchone()
                    return result['conteo_publicaciones']
            except Exception as e:
                logger.error(f"Error al consultar la base de datos: {e} en {temporalidad}")
                return None
            
        elif temporalidad[0] == 'Anio': # Fomato en lista

            try:
                anios_list = temporalidad[1]
                # Genero los Composed de "%s, %s, %s" para la lista
                anios_ph  = sql.SQL(', ').join(sql.Placeholder() for _ in anios_list)

                # Uso de sql.SQL + .format() para que los placeholders puedan ser usados por psycopg
                query = sql.SQL("""
                    SELECT COUNT(*) AS conteo_publicaciones
                    FROM publicaciones
                    WHERE ANIO  IN ({anios});
                """).format( 
                    anios=anios_ph,  # Insercion de los placeholders de anios
                )
                # Preparo la tupla de parámetros con los valores reales
                params = tuple(anios_list) 
                logger.info(f"Consultando BBDD (TEMPORALIDAD) {temporalidad[0]}")
                with conn.cursor() as cur:
                    cur.execute(query, params) # En esta llamada se rellenan los placeholders
                    result = cur.fetchone()
                    return result['conteo_publicaciones']
            except Exception as e:
                logger.error(f"Error al consultar la base de datos: {e} en {temporalidad}")
                return None
        
        # Expresiones temporales   
        elif temporalidad[0] == 'EXP': # Fomato en lista
            try:
                exp_dias = int(temporalidad[1])
                # Genero los Composed de "%s, %s, %s"
                #exp_ph  = sql.SQL(', ').join(exp_list)

                # Uso de sql.SQL + .format() para que los placeholders puedan ser usados por psycopg
                # Se usan los dias para sustraer a la fecha de la consulta.
                query = """
                    SELECT COUNT(*) AS conteo_publicaciones
                    FROM publicaciones
                    WHERE fecha_publicacion = CURRENT_DATE - %s;
                """
                logger.info(f"Consultando BBDD (TEMPORALIDAD) {temporalidad[0]}")
                with conn.cursor() as cur:
                    cur.execute(query, (exp_dias,)) # Creacion de tupla por psycopg
                    result = cur.fetchone()
                    return result['conteo_publicaciones']
            except Exception as e:
                logger.error(f"Error al consultar la base de datos: {e} en {temporalidad}")
                return None


    return None

def recuperar_documento_por_arxiv_id(arxiv_id: str, conn):
    """
    Recupera el título y el texto completo del documento enriquecido desde la base de datos utilizando su identificador de arXiv.

    Esta función consulta la base de datos para encontrar un documento cuyo identificador coincida con el valor
    proporcionado.

    Args:
        arxiv_id (str): Identificador único del documento en arXiv.
        conn (psycopg.Connection): Conexión activa a la base de datos PostgreSQL.

    Returns:
        tuple:
            - str: Título del documento.
            - str: Texto completo del documento si se encuentra, o un mensaje indicando que no existe.

    Raises:
        No lanza excepciones. Si ocurre un error, lo registra en el logger y devuelve None.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT titulo, documento_completo
                FROM publicaciones
                WHERE identificador_arxiv = %s
                AND documento_completo IS NOT NULL;
            """, (arxiv_id,))
            row = cur.fetchone()
            if row:
                return row['titulo'] ,row['documento_completo']
            else:
                return '',f'No existe ese id {arxiv_id} en la base de datos'
    except Exception as e:
        logger.error(f"Error al recuperar documento por arXiv: {e}")
        return None
    
