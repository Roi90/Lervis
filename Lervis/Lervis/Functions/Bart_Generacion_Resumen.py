from transformers import pipeline
from Functions.Loggers import BART_log
#from transformers import AutoTokenizer

def carga_BART():

    logger = BART_log() 

    summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")
    #tokenizer_bart = AutoTokenizer.from_pretrained("facebook/bart-large-cnn") 
    logger.info(f"Modelo facebook/bart-large-cnn cargado con éxito.") 

    return summarizer_bart
