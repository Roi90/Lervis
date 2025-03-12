"""
Este archivo contiene funciones para generar anotaciones de imagenes
con el modelo Florence-2.

Autor: Roi Pereira Fiuza
Fecha: 11/03/2024
"""

from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
from pathlib import Path
import torch
import time
from Functions.Loggers import Florence_log


def Carga_FLorence2_modelo(model_id='microsoft/Florence-2-large'):
    """
    Carga el modelo de Florence-2 y el procesador de texto.

    Parámetros:
    -----------
    model_id : str
        Identificador del modelo de Florence-2.

    Retorna:
    --------
    tuple
        Una tupla que contiene el identificador del modelo, el modelo y el procesador.
    """
    logger = Florence_log()

    logger.info(f"Modelo {model_id} cargado con éxito.")

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to('cuda')
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model_id, model, processor




def Florence2_detailed_annotation(model, processor, image: Image, task_prompt='<MORE_DETAILED_CAPTION>', text_input=None):

    logger = Florence_log()

    #model_id = 'microsoft/Florence-2-large'
    #model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to('cuda')
    #processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    
    
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    timestamp = time.time()
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    timestamp_elapsed = time.time()
    logger.info(f"Tiempo de inferencia: {timestamp_elapsed - timestamp} segundos)")

    return parsed_answer