"""
Este archivo contiene funciones para inyectar los PDF en el modelo Docling, para
segmentar el PDF mediante el OCR.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import time
from pathlib import Path
import pandas as pd
from Functions.Loggers import crear_logger
from datasets import Dataset
from tqdm import tqdm
from PIL import Image
import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.export import generate_multimodal_pages
from docling.utils.utils import create_hash

logger = crear_logger('Docling_OCR', 'Docling_OCR.log')
# Definir resolución (ajústala si es necesario)
IMAGE_RESOLUTION_SCALE = 1.0

def transform_bytes_to_image(examples):
        """
        Convierte una imagen representada en bytes en un objeto de imagen RGB de PIL.

        Esta función toma un diccionario que contiene las dimensiones y los datos en bytes de una imagen
        y los transforma en un objeto de imagen utilizando la librería PIL (Python Imaging Library).

        Args:
            examples (dict): Diccionario que debe contener las siguientes claves:
                - image.width (int): Ancho de la imagen.
                - image.height (int): Alto de la imagen.
                - image.bytes (bytes): Contenido de la imagen en formato de bytes en modo RGB.

        Returns:
            dict o None: El mismo diccionario con la clave adicional image que contiene un objeto PIL.Image.
            Devuelve None si ocurre un error durante la conversión.
        """
        try:
            examples["image"] = Image.frombytes('RGB', (examples["image.width"], examples["image.height"]), examples["image.bytes"], 'raw')
            return examples
        except Exception as e:
            logger.error(f"Error al convertir los bytes de la imagen: {e}")
            examples["image"] = None
            return None
        


def Carga_Docling_OCR():
    """
    Configura y carga un pipeline OCR para la conversión de documentos PDF usando Docling.

    Esta función inicializa un objeto DocumentConverter que está configurado para:
    - Procesar archivos en formato PDF.
    - Generar imágenes de las páginas procesadas.
    - Mantener la escala original de las imágenes.

    Returns:
        DocumentConverter: Objeto configurado para realizar OCR sobre documentos PDF.

    Raises:
        Exception: Si ocurre un error durante la configuración del pipeline OCR.
    """
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 1  # Escala de la imagen (resolución)
        pipeline_options.generate_page_images = True  # Generar imágenes de las páginas
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        logger.info("Docling cargado correctamente.")
        return doc_converter
    except Exception as e:
        logger.error(f"Error al cargar el pipeline de OCR: {e}")
        raise



def Archivo_to_OCR(input_doc_path: str, doc_converter: DocumentConverter) -> Dataset:
    """
    Procesa un documento PDF con OCR utilizando Docling y devuelve un Dataset estructurado con los resultados.

    Esta función convierte un documento PDF en texto segmentado página por página, incluyendo metadatos de la imagen,
    contenido textual y estructuras detectadas por el OCR.

    Parámetros:
        input_doc_path (str): Ruta al documento PDF que se desea procesar.
        doc_converter (DocumentConverter): Objeto configurado de Docling para convertir documentos PDF mediante OCR.

    Retorna:
        Dataset: Objeto Dataset que contiene la información estructurada del documento,
                 incluyendo texto, metadatos e imágenes por página.

    Raises:
        Exception: Si ocurre algún error durante el procesamiento del documento.
    """
    try:
        # Archivo input
        input_doc_path = Path(input_doc_path)
        if not input_doc_path.exists():
            logger.error(f"El archivo {input_doc_path} no existe.")
            return None

        start_time = time.time()
        conv_res = doc_converter.convert(input_doc_path)  # Se pasa el PDF en memoria en lugar de un archivo

        # Extraer contenido del OCR
        rows = []
        
        for (content_text, content_md, content_dt, page_cells, page_segments, page) in tqdm(generate_multimodal_pages(conv_res)):
            dpi = page._default_image_scale * 72
            rows.append(
                {
                    "document": "Archivo en Memoria",
                    "hash": conv_res.input.document_hash,
                    "page_hash": create_hash(conv_res.input.document_hash + ":" + str(page.page_no - 1)),
                    "image": {
                        "width": page.image.width,
                        "height": page.image.height,
                        "bytes": page.image.tobytes(),
                    },
                    "cells": page_cells,
                    "contents": content_text,
                    "contents_md": content_md,
                    "contents_dt": content_dt,
                    "segments": page_segments,
                    "extra": {
                        "page_num": page.page_no + 1,
                        "width_in_points": page.size.width,
                        "height_in_points": page.size.height,
                        "dpi": dpi,
                    },
                }
            )

        # Convertir a DataFrame
        df = pd.json_normalize(rows)

        # # Se convierte el Dataframe al formato de Dataset, 
        # formato optimizado cuando se utilizan herramientas de Huggin Face
        dataset = Dataset.from_pandas(df)

        end_time = time.time()
        duracion = end_time - start_time
        # Reconstruccion de las imagenes dentro del tipo dataset.
        try:
            dataset = dataset.map(transform_bytes_to_image)
        except Exception as e:
            logger.error(f'Error al recontruir las imagenes en el dataset: {e}')

        logger.info(f"Documento descargado y segmentado en {duracion:.2f} segundos.")
        logger.debug(f"Duracion OCR segundos - {duracion:.2f}, Paginas - {len(dataset)}")
        
    
        # Se debe limpiar la cache para que no de error por las limitaciones de la GPU
        #torch.cuda.empty_cache()
        #print('Cache limpiada de la GPU')
        return dataset
    except Exception as e:
        logger.error(f"Error al procesar el documento con OCR: {e} - Doc Path: {input_doc_path}")
        
