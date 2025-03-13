
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from Functions.Loggers import Nomic_log

# PENDIENTE DE DECIDIR SI GENERO CHUNKS O DEJO EL EMBEDDING EN SU TOTALIDAD
# HAY QUE ANALIZAR

def carga_nomic():

    logger = Nomic_log() 

    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    logger.info(f"Modelo nomic-embed-text cargado con éxito.")

    return embed_model
