"""
Este archivo contiene funciones para inyectar los PDF en el modelo Docling, para
segmentar el PDF mediante el OCR, usar Florence-2 para generar las anotaciones
de los diagramas, generando asi el documento enriquecido.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
from tqdm import tqdm
import tempfile
from PIL import Image
from Functions.Florence_2_anotacion import Florence2_detailed_annotation
from pathlib import Path
from Functions.Loggers import crear_logger

logger = crear_logger('Doc_Enriquecido', 'Doc_Enriquecido.log')

def enriquecimiento_doc(dataset, F2_model, F2_processor):
    try:
        texto_enriquecido = ''
        # Iteracion por cada pagina del documento
        for i in tqdm(range(len(dataset)), desc= 'Escaneando imagenes del documento...'):
            # Iteracion por cada segmento de cada pagina
            for segments in dataset[i]['segments']:
                if segments['label']  == 'header':
                    texto_enriquecido += f'{segments["text"]}'

                elif  segments['label'] == 'footnote': 
                    texto_enriquecido += f'{segments["text"]}'

                elif segments['label'] == 'picture' or segments['label'] == 'table':
                    # Coordenadas para recortar la imagen
                    bbox = segments['bbox']
                    

                    # Extraccion de la resolucion de la imagen (Pagina entera del PDF)
                    image_width, image_height = dataset[i]['image'].size
                    # Calculo de las coordenadas de la imagen a extraer mediante las dimensiones
                    # de la imagen total del PDF
                    left = int(bbox[0] * image_width)
                    top = int(bbox[1] * image_height)
                    right = int(bbox[2] * image_width)
                    bottom = int(bbox[3] * image_height)
                    # Se recorta la imagen para solo centrarse en el diagrama obtenido por los segmentos
                    cropped_image = dataset[i]['image'].crop((left, top, right, bottom))
                    
                    # Archivo temporal para la imagen
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
                        # Guardado como archivo temporal.
                        cropped_image.save(temp_img_file.name)
                        # Inyeccion imagen a Florence-2 (Anotacion de imagenes)
                        temp_imagen = Image.open(temp_img_file.name)
                        anotacion = Florence2_detailed_annotation(image=temp_imagen, model=F2_model, processor=F2_processor)['<MORE_DETAILED_CAPTION>']
                        # Eliminación del archivo temporal
                        temp_img_file.close()  # Cerramos el archivo temporal para que se pueda eliminar
                        Path(temp_img_file.name).unlink()  # Eliminamos el archivo usando el nombre

                    texto_enriquecido += anotacion
                else:
                    texto_enriquecido += f'\n\n {segments["text"]}'
        return texto_enriquecido
    except Exception as e:
        logger.error(f"Error al enriquecer el documento: {e}")
        return None

