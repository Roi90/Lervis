
from datetime import time
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from Functions.Loggers import crear_logger


logger = crear_logger('Embedding', 'Embedding.log')

def embedding(text, model):
    """
    Genera embeddings para el texto dado utilizando el modelo y el tokenizador especificados.
    Args:
        text (str): El texto de entrada a ser embebido.
        model (transformers.PreTrainedModel): El modelo preentrenado para generar embeddings.
        tokenizer (transformers.PreTrainedTokenizer): El tokenizador asociado con el modelo.
    Returns:
        numpy.ndarray: Los embeddings del texto de entrada.
    """

    try:
        inicio = time.time()

        num_tokens = len(model.tokenizer(text)['input_ids'])

        embeddings = model.encode(text, return_dense=True,return_sparse=True,return_colbert_vecs=False,batch_size=12,max_length=6000)

        fin = time.time() 
        
        embeddings_denso = embeddings['dense_vecs']
        embedding_disperso = embeddings['lexical_weights']

        # Calculo la norma L2 para ver que de lejos esta del origen de mis datos
        norma_L2 = np.linalg.norm(embeddings['dense_vecs'])

        if norma_L2 <= 0.1:
            logger.warning(f"Norma L2 embeddings BAJA - {norma_L2}, texto - {text}")

        len_dense_embeddings = len(embeddings['dense_vecs'])
        len_sparse_embeddings = len(embeddings['lexical_weights'])
        len_text = len(text)
        

        duracion_segundos = fin - inicio

        logger.debug(f"Caracteres Text - {len_text}, Input tokens - {num_tokens}, Duracion segundos - {duracion_segundos:.2f}, Longitud Denso - {len_dense_embeddings}, Longitud Disperso - {len_sparse_embeddings}")
        
        return embeddings_denso,embedding_disperso
    except Exception as e:
        logger.error(f"Error al crear los embeddings: {e}")

    

    

def carga_BAAI():
    """
        Carga el modelo y el tokenizer de BAAI BGEM3.
        Esta función carga el modelo "BAAI/bge-m3" utilizando la biblioteca FlagEmbedding.
        Además, registra un mensaje de éxito una vez que el modelo se ha cargado correctamente.
        Returns:
            model (BGEM3FlagModel): El modelo BGEM3 cargado.
        """
    try:
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)   
        logger.info(f"Modelo 'BAAI/bge-m3' cargado con exito.")

        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo 'BAAI/bge-m3': {e}")
        raise
