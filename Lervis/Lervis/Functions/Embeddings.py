
import time
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from Functions.Loggers import crear_logger
from sklearn.metrics.pairwise import cosine_similarity

logger = crear_logger('Embedding', 'Embedding.log')
logger_chunks = crear_logger('Embedding_chunks_ETL', 'Embedding_chunks_ETL.log')
logger_Doc_Enri = crear_logger('Embedding_chunk_Doc_Enri_ETL', 'Embedding_chunk_Doc_Enri_ETL.log')

def embedding(text, model):
    """
    Genera embeddings densos y dispersos para un texto dado utilizando un modelo de embeddings.

    Esta función utiliza el modelo proporcionado para calcular los vectores de representación del texto
    en dos formatos: uno denso  y otro disperso.

    Parámetros:
        text (str): Texto de entrada sobre el cual se generarán los embeddings.
        model: Modelo compatible con retorno de vectores densos y dispersos.

    Retorna:
        tuple: Una tupla que contiene:
            - numpy.ndarray: Embedding denso del texto.
            - default dict: Embedding disperso representado como un diccionario donde se meustran los tokens
            con mayor relevancia y los que no estan como clave poseen el valor default definido.

    Raises:
        Exception: Si ocurre un error durante la generación de los embeddings.
    """

    try:
        inicio = time.time()

        num_tokens = len(model.tokenizer(text)['input_ids'])

        embeddings = model.encode(text, return_dense=True,return_sparse=True,return_colbert_vecs=False,batch_size=12,max_length=6000)

        fin = time.time() 
        
        embeddings_denso = embeddings['dense_vecs']
        embedding_disperso = embeddings['lexical_weights']

        

        # Suma booleanos por la condicion
        len_dense_embeddings = len(embeddings['dense_vecs'])
        len_sparse_embeddings = len(embeddings['lexical_weights'])
        len_text = len(text)
        

        duracion_segundos = fin - inicio

        logger.debug(f"Caracteres Text - {len_text}, Input tokens - {num_tokens}, Duracion segundos - {duracion_segundos:.2f}, Longitud Denso - {len_dense_embeddings}, Longitud Disperso - {len_sparse_embeddings}")
        
        return embeddings_denso,embedding_disperso
    except Exception as e:
        logger.error(f"Error al crear los embeddings: {e}")

def embedding_ETL(text, embedding_documento_enriquecido, id_documento, model):
    """
    Genera embeddings densos y dispersos para un texto dado, y calcula la similitud con un documento enriquecido.

    Esta función genera representaciones vectoriales del texto tanto en formato denso como disperso.
    Además, calcula la similitud coseno entre el embedding generado y un embedding de referencia
    correspondiente a un documento previamente enriquecido.

    Parámetros:
        text (str): Texto de entrada para generar el embedding.
        embedding_documento_enriquecido (np.ndarray): Embedding denso del documento completo para comparación.
        id_documento (str o int): Identificador del documento (usado para trazabilidad).
        model: Modelo compatible con retorno de vectores densos y dispersos.

    Retorna:
        tuple:
            - numpy.ndarray: Vector de embedding denso del texto.
            - dict: Embedding disperso representado como diccionario léxico.

    Raises:
        Exception: Si ocurre un error durante la generación de los embeddings o el cálculo de similitud.
    """

    try:
        inicio = time.time()

        num_tokens = len(model.tokenizer(text)['input_ids'])

        embeddings = model.encode(text, return_dense=True,return_sparse=True,return_colbert_vecs=False,batch_size=12,max_length=6000)

        fin = time.time() 
        
        embeddings_denso = embeddings['dense_vecs']
        embedding_disperso = embeddings['lexical_weights']

        # Calculo la similitud semantica con el documento, el valor semantico del chunk.
        sim_semantica = cosine_similarity(embeddings_denso.reshape(1, -1), embedding_documento_enriquecido.reshape(1, -1))[0][0]

        # Suma booleanos por la condicion
        len_dense_embeddings = len(embeddings['dense_vecs'])
        len_sparse_embeddings = len(embeddings['lexical_weights'])
        len_text = len(text)
        

        duracion_segundos = fin - inicio

        logger_chunks.debug(f"Caracteres Text - {len_text}, Input tokens - {num_tokens}, Duracion segundos - {duracion_segundos:.2f}, Longitud Denso - {len_dense_embeddings}, Longitud Disperso - {len_sparse_embeddings}, Similitud Doc Enriquecido - {sim_semantica}, Id Doc - {id_documento}")
        
        return embeddings_denso,embedding_disperso
    except Exception as e:
        logger_chunks.error(f"Error al crear los embeddings: {e}")

def embedding_ETL_DOC_ENRI(text,id, model):
    """
    Genera embeddings densos y dispersos para fragmentos (chunks) de un documento enriquecido.

    Esta función utiliza un modelo de embeddings para transformar el texto de un documento
    en vectores densos y dispersos. Registra métricas útiles para trazabilidad.

    Parámetros:
        text (str): Texto del documento a embebir.
        id (str o int): Identificador del documento (usado para trazabilidad en logs).
        model: Modelo compatible con retorno de vectores densos y dispersos.

    Retorna:
        tuple:
            - numpy.ndarray: Embedding denso del documento.
            - dict: Embedding disperso representado como diccionario léxico.

    Raises:
        Exception: Si ocurre un error durante la generación de embeddings.
    """

    try:
        inicio = time.time()

        num_tokens = len(model.tokenizer(text)['input_ids'])

        embeddings = model.encode(text, return_dense=True,return_sparse=True,return_colbert_vecs=False,batch_size=12,max_length=6000)

        fin = time.time() 
        
        embeddings_denso = embeddings['dense_vecs']
        embedding_disperso = embeddings['lexical_weights']

        

        # Suma booleanos por la condicion
        len_dense_embeddings = len(embeddings['dense_vecs'])
        len_sparse_embeddings = len(embeddings['lexical_weights'])
        len_text = len(text)
        

        duracion_segundos = fin - inicio

        logger_Doc_Enri.debug(f"Caracteres Text - {len_text}, Input tokens - {num_tokens}, Duracion segundos - {duracion_segundos:.2f}, Longitud Denso - {len_dense_embeddings}, Longitud Disperso - {len_sparse_embeddings}, Id Doc - {id}")
        
        return embeddings_denso,embedding_disperso
    except Exception as e:
        logger_Doc_Enri.error(f"Error al crear los embeddings: {e}")

def embedding_evaluator(text, model):
    """
    Genera embeddings densos y dispersos para un texto dado utilizando un modelo de embeddings. Especificamente
    esta funcion se ha creado con la finalidad de segmentar el proceso de embedding del input del usuario y la salida
    del LLM para realizar metricas entre estos.

    Esta función utiliza el modelo proporcionado para calcular los vectores de representación del texto
    en dos formatos: uno denso  y otro disperso.

    Parámetros:
        text (str): Texto de entrada sobre el cual se generarán los embeddings.
        model: Modelo compatible con retorno de vectores densos y dispersos.

    Retorna:
        tuple: Una tupla que contiene:
            - numpy.ndarray: Embedding denso del texto.
            - default dict: Embedding disperso representado como un diccionario donde se meustran los tokens
            con mayor relevancia y los que no estan como clave poseen el valor default definido.

    Raises:
        Exception: Si ocurre un error durante la generación de los embeddings.
    """

    try:

        embeddings = model.encode(text, return_dense=True,return_sparse=True,return_colbert_vecs=False,batch_size=12,max_length=6000)
        
        embeddings_denso = embeddings['dense_vecs']
        embedding_disperso = embeddings['lexical_weights']

        
        return embeddings_denso,embedding_disperso
    except Exception as e:
        logger.error(f"EVALUATOR Error al crear los embeddings: {e}")

def carga_BAAI():
    """
    Carga el modelo de embeddings BGE-M3 de BAAI utilizando la biblioteca FlagEmbedding.

    Esta función carga el modelo BAAI/bge-m3 en modo FP16 para optimizar el rendimiento.

    Es útil para tareas de generación de embeddings densos y dispersos. 
    
    En caso de éxito, se registra un mensaje en el logger.

    Returns:
        BGEM3FlagModel: Objeto del modelo cargado listo para generar embeddings.

    Raises:
        Exception: Si ocurre un error durante la carga del modelo.
    """

    try:
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)   
        logger.info(f"Modelo 'BAAI/bge-m3' cargado con exito.")

        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo 'BAAI/bge-m3': {e}")
        raise


