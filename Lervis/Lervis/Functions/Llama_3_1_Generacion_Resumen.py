from transformers import pipeline
from Functions.Loggers import Llama31_resumen_log
import torch
from transformers import AutoTokenizer
from rouge_score import rouge_scorer


from langchain_ollama import OllamaLLM 

def carga_Llama3_1():
    """
    Carga el modelo Llama3.1 para la generación de resúmenes.
    Esta función inicializa y carga el modelo Llama3.1 
    utilizando la biblioteca langchain_ollama para la tarea de 
    generación de resúmenes. También configura un logger para registrar 
    el estado de la carga del modelo.
    Returns:
        model: Un objeto de la clase OllamaLLM configurado para 
               la tarea de summarization con el modelo Llama3.1.
    """

    logger = Llama31_resumen_log() 

    model = OllamaLLM(model ="llama3.1")  
    logger.info(f"Modelo llama3.1 cargado con éxito.") 
    return model

def generar_resumen(model: OllamaLLM, texto: str):
    """
    Genera un resumen del texto proporcionado utilizando el modelo Llama3.1.
    Args:
        model (OllamaLLM): El modelo Llama3.1 cargado para la generación de resúmenes.
        texto (str): El texto que se desea resumir.
    Returns:
        str: El resumen generado del texto proporcionado.
    """

    accion = f"""Act as a scientist specialized in the field of research. 
    Your task is to provide a clear, concise, and technical summary, like an abstract, of approximately 1500 words of the following text. 
    Focus on the key points and omit any unnecessary or irrelevant details. 
    The summary should be direct, precise, and easy to understand for an academic or specialized audience. 
    Do not mention the authors or references. Use formal and technical language appropriate for a scientific paper.: {texto}"""


    return model.invoke(accion)



#text = """
#We introduce Florence-2 , a novel vision foundation model with a unified, prompt-based representation for a variety of computer vision and 
#vision-language tasks. While existing large vision models excel in transfer learning, they struggle to perform a diversity of tasks with simple instructions,
#a capability that implies handling the complexity of various spatial hierarchy and semantic granularity. Florence-2 was designed to take text-prompt as task
#instructions and generate desirable results in text forms, whether it be captioning, object detection, grounding or segmentation. This multi-task learning
#setup demands largescale, high-quality annotated data. To this end, we codeveloped FLD-5B that consists of 5.4 billion comprehensive visual annotations
#on 126 million images, using an iterative strategy of automated image annotation and model refinement. We adopted a sequence-to-sequence structure to 
#train Florence-2 to perform versatile and comprehensive vision tasks. Extensive evaluations on numerous tasks demonstrated Florence-2 to be a strong vision 
#foundation model contender with unprecedented zero-shot and fine-tuning capabilities.
#"""

#llm = carga_Llama3_1()

#resumen = generar_resumen(llm,text)

#scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

#scores = scorer.score(text, resumen)

#print(f'Rouge 1\n\tPrecision : {round(scores["rouge1"][0],2)}\n\tRecall : {round(scores["rouge1"][1],2)}\n\tF1 Score : {round(scores["rouge1"][2],2)}')
#print(f'Rouge 2\n\tPrecision : {round(scores["rouge2"][0],2)}\n\tRecall : {round(scores["rouge2"][1],2)}\n\tF1 Score : {round(scores["rouge2"][2],2)}')
#Sprint(f'Rouge L\n\tPrecision : {round(scores["rougeL"][0],2)}\n\tRecall : {round(scores["rougeL"][1],2)}\n\tF1 Score : {round(scores["rougeL"][2],2)}')
