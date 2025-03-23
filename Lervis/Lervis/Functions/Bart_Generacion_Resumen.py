from transformers import pipeline
from Functions.Loggers import BART_log
import torch
from transformers import AutoTokenizer

def carga_BART():
    """
    Carga el modelo BART para la generación de resúmenes.
    Esta función inicializa y carga el modelo BART (facebook/bart-large-cnn) 
    utilizando la biblioteca Hugging Face Transformers para la tarea de 
    generación de resúmenes. También configura un logger para registrar 
    el estado de la carga del modelo.
    Returns:
        summarizer_bart: Un objeto de la clase pipeline configurado para 
                         la tarea de summarization con el modelo BART.
        tokenizer_bart: Un objeto de la clase AutoTokenizer configurado 
                        con el modelo BART.
    """

    logger = BART_log() 

    summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer_bart = AutoTokenizer.from_pretrained("facebook/bart-large-cnn") 
    logger.info(f"Modelo facebook/bart-large-cnn cargado con éxito.") 

    return summarizer_bart, tokenizer_bart

def max_token_fragmentacion(texto_completo: str, tokenizer: AutoTokenizer, max_tokens=1024):
    """
    Divide un texto dado en fragmentos basados en un número máximo de tokens.
    Args:
        max_tokens (int, opcional): El número máximo de tokens por fragmento. Por defecto es 1024.
        texto_completo (str): El texto completo a tokenizar y dividir.
        tokenizer (AutoTokenizer): El tokenizador a utilizar para codificar y decodificar el texto.
    Returns:
        list: Una lista de fragmentos de texto, cada uno con hasta `max_tokens` tokens.
    """

    tokens = tokenizer.encode(texto_completo)

    fragmentos = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    textos_fragmentados = [tokenizer.decode(fragmento, skip_special_tokens=True) for fragmento in fragmentos]

    return textos_fragmentados




