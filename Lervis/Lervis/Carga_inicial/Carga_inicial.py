"""
Este archivo contiene funciones para la primera carga de datos en la BBDD.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""

from Functions.API_metadata import extraccion_por_categorias
from Functions.Docling_OCR import Archivo_to_OCR, transform_bytes_to_image, Carga_Docling_OCR
from Functions.PDF_descarga import PDF_descarga
from Functions.Florence_2_anotacion import Florence2_detailed_annotation, Carga_FLorence2_modelo
import pandas as pd
from pathlib import Path
from PIL import Image
import tempfile
from tqdm import tqdm
import copy

# Carga de modelos
F2_model_id, F2_model, F2_processor = Carga_FLorence2_modelo(model_id='microsoft/Florence-2-large')
doc_converter = Carga_Docling_OCR()

# DF de los metadatos para la extraccion
#metadatos_publicaciones = descarga_por_batches(1000)

truco = pd.read_excel('publicaciones_arxiv.xlsx')
#print(truco.head())


for publicacion in truco.itertuples():

    # FUNCION DESCARGAR PDF
    path, nombre_archivo = PDF_descarga(publicacion.url_pdf, publicacion.id)
    # SOLO DESCARGO 1 PARA TEST CON EL OCR Y DOCLING
    break

# Dataset con el documento segmentado

# PENDIENTE DE GENERAR EL CODIGO QUE ITERE SOBRE CADA PAGINA Y CONVIERTA LA IMAGEN EN TEXTO
test = Archivo_to_OCR(path, doc_converter=doc_converter)
test = test.map(transform_bytes_to_image)

texto_enriquecido = ''
# Iteracion por cada pagina del documento
for i in tqdm(range(len(test)), desc= 'Escaneando imagenes del documento...'):
    # Iteracion por cada segmento de cada pagina
    for segments in test[i]['segments']:
        if segments['label']  == 'header' or segments['label'] == 'footnote':
            texto_enriquecido += f'{segments["text"]}'

        elif  segments['label'] == 'footnote': 
            texto_enriquecido += f'\n {segments["text"]} \n'

        elif segments['label'] == 'picture' or segments['label'] == 'table':
            # Coordenadas para recortar la imagen
            bbox = segments['bbox']
            # Indice del objeto en el conjunto de todos los objetos del documento entero.
            # Fundamental para que la anotacion se inyecte en esta posicion.
            index_in_doc = segments['index_in_doc']

            # Extraccion de la resolucion de la imagen (Pagina entera del PDF)
            image_width, image_height = test[i]['image'].size
            # Calculo de las coordenadas de la imagen a extraer mediante las dimensiones
            # de la imagen total del PDF
            left = int(bbox[0] * image_width)
            top = int(bbox[1] * image_height)
            right = int(bbox[2] * image_width)
            bottom = int(bbox[3] * image_height)
            # Se recorta la imagen para solo centrarse en el diagrama obtenido por los segmentos
            cropped_image = test[i]['image'].crop((left, top, right, bottom))
            
            # Archivo temporal
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
            #print(f"Después de la modificación: {segments}")
        else:
            texto_enriquecido += f'{segments["text"]} \n'
    if i == 2:
        break
    
# Conversion de imagenes para realizar las anotaciones
#test = test.map(transform_bytes_to_image)
print('AQUI EMPIEZA')
#text_test = [' '.join(x) for x in texto_enriquecido_lst]

print(texto_enriquecido)


#for i in tqdm(range(len(test)), desc= 'Escaneando imagenes del documento...'):
 #   for segments in test[i]['segments']:
  #       if segments['index_in_doc'] in segmentos_a_actualizar_lst.keys():
             




