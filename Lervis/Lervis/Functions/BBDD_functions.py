"""
Este archivo contiene funciones para conectarse a una base de datos PostgreSQL.
Proporciona métodos para crear un motor de conexión.

Autor: Roi Pereira Fiuza
"""
import re
from Static_data import categorias_arxiv
import pandas as pd
import numpy as np
from psycopg import sql
import psycopg
from psycopg.rows import dict_row

from Functions.Embeddings import carga_BAAI, embedding

# ---------------------------TO DO: Crear un archivo con las variables para seguridad

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
# --------------------------- Carga de datoss en la BBDD
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

def carga_hechos_chunks_embeddings(df: pd.DataFrame, engine):
    """
    Inserta los datos de embeddings en la base de datos utilizando SQLAlchemy y pandas.
    Se usa SQLAlchemy, debido a complicaciones en la insercion para el HSTORE.
    
    Args:
        engine (sqlalchemy.engine.Engine): Conexión a la base de datos.
        datos_embeddings_lst (list): Lista de diccionarios con los datos de embeddings.
        
    Returns:
        None
    """        
    # Insertar el DataFrame en la base de datos, tabla 'embeddings'
    df.to_sql('embeddings_chunks', con=engine, if_exists='append', index=False)

def carga_hechos_resumen_embeddings(df: pd.DataFrame, engine):
    """
    Inserta los datos de embeddings en la base de datos utilizando SQLAlchemy y pandas.
    Se usa SQLAlchemy, debido a complicaciones en la insercion para el HSTORE.
    
    Args:
        engine (sqlalchemy.engine.Engine): Conexión a la base de datos.
        datos_embeddings_lst (list): Lista de diccionarios con los datos de embeddings.
        
    Returns:
        None
    """
        
    # Insertar el DataFrame en la base de datos, tabla 'embeddings'
    df.to_sql('embeddings_resumen', con=engine, if_exists='append', index=False)

def carga_doc_enriquecido(documento_enriquecido, id,  conn):

    id = int(id)
    with conn.cursor() as cur:
        cur.execute("""UPDATE publicaciones
                    SET documento_completo = %s
                    WHERE id = %s;""", (documento_enriquecido, id))
        conn.commit()

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

# ---------------------------- Recuperacion documentos y formateo para contexto

def similitud_coseno(embedding_denso: np.array, conn)-> dict:
    """
    Calcula la similitud del coseno entre un vector y los vectores almacenados en la base de datos.
    Esta función toma un vector y calcula la similitud del coseno con los vectores de contenido y resumen
    de los documentos almacenados en la base de datos.

    Args:
        embedding_denso (np.array): Vector de embedding a comparar.
        conn (psycopg.Connection): Conexión a la base de datos.

    Returns:
        list: Lista de documentos recuperados con sus similitudes.
    """

    embedding_denso_float64 = embedding_denso.tolist()
    with conn.cursor() as cur:

        cur.execute("""
                    SELECT id_publicaciones, chunk_emb_sparse, chunk,
                        chunk_emb_dense <=> %s::vector AS similitud_contenido
                    FROM embeddings_chunks
                    ORDER BY chunk_emb_dense <=> %s::vector DESC
                    LIMIT 5;
                    """, (embedding_denso_float64, embedding_denso_float64))
        # Recuperar los resultados
        documentos_recuperados = cur.fetchall()
        #print(f"Total de registros en embeddings_chunks: {documentos_recuperados[0]}")
        

        if not documentos_recuperados:
            return "No se si te he entendido bien, pero ¿Podrías detallar un poco mas tu pregunta?"

        # Devolver el resumen del documento más relevante
        return documentos_recuperados
    
def hstore_a_dict(hstore_str: str) -> dict:
    """
    Convierte una cadena de texto en formato HSTORE a un diccionario de Python.
    
    Args:
        hstore_str (str): Cadena HSTORE a convertir.
        
    Returns:
        dict: Diccionario resultante.
    """
   
    hstore_dict = dict(
                        map(lambda x: (x[0], float(x[1])),
                        [hstore.replace('"', '').split('=>') for hstore in hstore_str.split(', ')])  # Convertir a float
    )

    return hstore_dict

def reranking(documentos_recuperados, embedding_disperso):
    """
    Reordena los documentos recuperados basándose en la similitud de vocabulario y contenido.
    
    Args:
        documentos_recuperados (lista de diccionarios): Lista de documentos recuperados con sus similitudes.
        
    Returns:
        pd.DataFrame: DataFrame ordenado con los documentos re-rankeados.
    """
    
    df_temp = pd.DataFrame(documentos_recuperados)
    # Transformacion de la columna hstore a dict
    df_temp['chunk_emb_sparse'] = df_temp['chunk_emb_sparse'].apply(lambda x: hstore_a_dict(x))
    # Creacion de la columna mediante la interseccion del conjunto de ambas claves

    df_temp['similitud_vocabulario'] = df_temp['chunk_emb_sparse'].apply(lambda x: len(set(x.keys()).intersection(set(embedding_disperso.keys()))))
    # Ordeno el DF por similitud de contenido y similitud de vocabulario
    df_temp = df_temp.sort_values(by=['similitud_contenido','similitud_vocabulario'], ascending=False)
    
    return df_temp

def formato_contexto_doc_recuperados(conn, df: pd.DataFrame, num_docs: int = 3) -> str:
    """
    Genera un contexto a partir de los documentos más relevantes.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los documentos recuperados.
        num_docs (int): Número de documentos a incluir en el contexto.
        
    Returns:
        str: Contexto generado a partir de los documentos más relevantes.
    """
    documentos_formateados = ''

    if num_docs > len(df):
        print('El número de documentos solicitados es mayor que el número de documentos recuperados')
    else:
        # Subconjunto de documentos recuperados
        df_top_docs = df.head(num_docs)
        # Extraer los IDs de los documentos
        ids = list(df_top_docs['id_publicaciones'])
        # Formateo para poder introducir una tupla (%s, %s, %s...)
        placeholders = sql.SQL(', ').join([sql.Placeholder() for _ in ids])

        with conn.cursor() as cur:
            query = sql.SQL("""SELECT 
                        titulo,
                        CAT.categoria,
                        autores,
                        fecha_publicacion,
                        url_pdf,
                        ER.resumen
                        FROM publicaciones
                        LEFT JOIN categoria as CAT
                            ON publicaciones.categoria_principal = CAT.id
                        LEFT JOIN embeddings_resumen as ER
                            ON publicaciones.id = ER.id_publicaciones
                        WHERE publicaciones.id in ({})""").format(placeholders)
            cur.execute(query, ids)
            # Recuperar los resultados (Lista de diccionarios)
            documentos_recuperados = cur.fetchall()
        # generacion del formato de la respuesta
        for doc in documentos_recuperados:
            # Titulo
            documentos_formateados += f"\nTitulo: {doc['titulo']}\n"
            # Categoria
            documentos_formateados += f"Categoria: {doc['categoria']}\n"
            # Autores
            documentos_formateados += f"Autores: {doc['autores']}\n".replace("{", "").replace("}", "")
            # Fecha de publicacion
            documentos_formateados += f"Fecha de publicacion: {doc['fecha_publicacion']}\n"
            # URL PDF
            documentos_formateados += f"URL PDF: {doc['url_pdf']}\n"
            # Resumen
            documentos_formateados += f"Resumen: {doc['resumen']}\n"
            # Separador entre documentos
            documentos_formateados += "-" * 80 + "\n"
        return documentos_formateados


def temporalidad_a_SQL(conn, temporalidad: tuple):
    """
    Convierte la temporalidad extraida del input del usuario a una consulta SQL.
    Args:
        temporalidad (tuple): Tuple que contiene la temporalidad extraida del input del usuario. (Ej: ('Tipo de temporalidad', diccionario con los valores, user_input))
    Returns:
        str: Consulta SQL generada a partir de la temporalidad.
    """
    print(temporalidad)
    if temporalidad != None:
        # Identificacion de agregados
        expresiones_count = ["cuantas", "numero de", "total de", "cantidad de", "que cantidad","suma de"]
        
        expresiones_agregados = ["promedio de", "media de"]

        regex_count = r'\b(?:' + '|'.join(expresiones_count) + r')\b'

        if re.findall(regex_count, temporalidad[2]):
            
            if temporalidad[0] == 'Combinada':
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

                with conn.cursor() as cur:
                    cur.execute(query, params) # En esta llamada se rellenan los placeholders
                    result = cur.fetchone()
                    return result
            
            elif temporalidad[0] == 'Mes': # Fomato en lista

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
                with conn.cursor() as cur:
                    cur.execute(query, params) # En esta llamada se rellenan los placeholders
                    result = cur.fetchone()
                    return result
                
            elif temporalidad[0] == 'Anio': # Fomato en lista

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
                with conn.cursor() as cur:
                    cur.execute(query, params) # En esta llamada se rellenan los placeholders
                    result = cur.fetchone()
                    return result
            
            # Expresiones temporales   
            elif temporalidad[0] == 'EXP': # Fomato en lista
                
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
    
                with conn.cursor() as cur:
                    cur.execute(query, (exp_dias,)) # Creacion de tupla por psycopg
                    result = cur.fetchone()
                    return result    


        return None

# ------ TESTEO
#conn =  conn_bbdd()
#model = carga_BAAI()
#txt = 'Overall, the image appears to be a collection of different animal images, with a variety of colors and textures.The image is a set of six bar graphs that show the number of people who have been diagnosed with cancer. '
#htore ='''"5"=>"0.04694", "13"=>"0.01945", "42"=>"0.0347", "70"=>"0.006874"'''
#print(hstore_a_dict(htore))

#embedding_denso, embedding_disperso = embedding(txt,model)
#valor = similitud_coseno(embedding_denso, conn_bbdd())
#df_temp = reranking(valor,embedding_disperso)
#print(df_temp)
#doc_format = formato_contexto_doc_recuperados(conn, df_temp, num_docs=3)

#print(doc_format)