"""
Este archivo contiene funciones para generar anotaciones de imagenes
con el modelo Florence-2.

Autor: Roi Pereira Fiuza
Fecha: 11/03/2024
"""

from Functions.Loggers import crear_logger
from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import torch
import time

logger = crear_logger('Florence2', 'florence2.log')

def Carga_FLorence2_modelo():
    """
    Carga el modelo de anotacion de imagenes Florence-2 y su procesador correspondiente.

    Esta función descarga y configura el modelo Florence-2 de Microsoft, preparado para ejecutarse 
    en GPU. También carga su procesador asociado para el preprocesamiento de datos de entrada (imagen y texto).

    Returns:
        tuple: Una tupla que contiene:
            - str: Identificador del modelo cargado.
            - transformers.PreTrainedModel: Instancia del modelo Florence-2 listo para inferencia.
            - transformers.PreTrainedProcessor: Procesador asociado para preparar las entradas.
    
    Logs:
        - Mensaje de éxito al cargar el modelo.
    """

    model_id='microsoft/Florence-2-large'
    logger.info(f"Modelo {model_id} cargado con exito.")

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to('cuda')
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model_id, model, processor



def Florence2_detailed_annotation(model, processor, image: Image, task_prompt='<MORE_DETAILED_CAPTION>', text_input=None):
    """
    Genera una anotación detallada para una imagen utilizando el modelo Florence-2.

    Esta función utiliza el modelo para generar descripciones enriquecidas de una imagen. 

    Parámetros:
        model: Modelo para generación de anotaciones.
        processor: Procesador asociado al modelo.
        image (PIL.Image): Imagen que se desea anotar.
        task_prompt (str, opcional): Prompt que define el tipo de anotación a generar. Por defecto '<MORE_DETAILED_CAPTION>'.
        text_input (str, opcional): Texto adicional que puede complementar el prompt. Por defecto None.

    Retorna:
        dict: Diccionario con la anotación generada por el modelo, estructurada por tipo de prompt (clave = task_prompt).

    Raises:
        Exception: En caso de fallo durante la inferencia o el postprocesamiento de la anotación.
    """  

    try:
    
        
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        timestamp = time.time()

        #logger.debug("Preparando inputs para Florence2...")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)

        #logger.debug("Generando IDs...")
        generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        )
    # logger.debug("Decodificando salida...")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        #logger.debug(f"Texto generado por Florence2: {generated_text}...")

        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        #print('-----PARSED ANSWER----\n',parsed_answer)
        timestamp_elapsed = time.time()
        duracion = timestamp_elapsed - timestamp
        logger.debug(f"Duracion segundos - {duracion:.2f}, Caracteres generados - {len(parsed_answer.get('<MORE_DETAILED_CAPTION>', ''))} , Alto Imagen - {image.height}, Ancho Imagen - {image.width}")

        return parsed_answer
    except Exception as e:
        logger.error(f'Error al generar la anotacion: {e}')
