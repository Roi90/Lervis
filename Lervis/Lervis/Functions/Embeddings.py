
from FlagEmbedding import BGEM3FlagModel
from Functions.Loggers import BAAI_log

# PENDIENTE DE DECIDIR SI GENERO CHUNKS O DEJO EL EMBEDDING EN SU TOTALIDAD
# HAY QUE ANALIZAR

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

    logger = BAAI_log()

    embeddings = model.encode(text, return_dense=True,return_sparse=True,return_colbert_vecs=False,batch_size=12,max_length=6000)
    logger.debug(f"Embeddings creados con exito")

    embeddings_denso = embeddings['dense_vecs']
    embedding_disperso = embeddings['lexical_weights']

    return embeddings_denso, embedding_disperso

def carga_BAAI():
    """
        Carga el modelo y el tokenizer de BAAI BGEM3.
        Esta función carga el modelo "BAAI/bge-m3" utilizando la biblioteca FlagEmbedding.
        Además, registra un mensaje de éxito una vez que el modelo se ha cargado correctamente.
        Returns:
            model (BGEM3FlagModel): El modelo BGEM3 cargado.
        """

    logger = BAAI_log()

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)   
    logger.info(f"Modelo 'BAAI/bge-m3' cargado con éxito.")

    return model

