"""
Este archivo contiene funciones para la primera carga de datos en la BBDD, incluyendo el proceso
de transformacion de los datos.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import pandas as pd
from Functions.API_metadata import extraccion_por_categorias
from Functions.Docling_OCR import Archivo_to_OCR, Carga_Docling_OCR
from Functions.PDF_descarga import PDF_descarga_temp
from Functions.Florence_2_anotacion import  Carga_FLorence2_modelo
from Functions.Enriquecimiento_documento import enriquecimiento_doc
from Functions.Embeddings import carga_nomic
from Functions.Bart_Generacion_Resumen import carga_BART

# -----------------
target_por_categoria = 1
# -----------------

#  Carga de modelos
F2_model_id, F2_model, F2_processor = Carga_FLorence2_modelo()
doc_converter = Carga_Docling_OCR()
embeder = carga_nomic()
summarizer_bart = carga_BART()

# DF de los metadatos para la extraccion
metadatos_publicaciones = extraccion_por_categorias(target_por_categoria)

# Listas para crear posteriormente el DF.

id_lst = []
titulo_lst = []
autores_lst = []
fecha_puclicacion_lst = []
resumen_enriquecido = []
documento_enriquecido = []
embedding_lst = []

# Procesamiento y transformacion
for publicacion in metadatos_publicaciones.itertuples():
    # Descargamos el PDF en un archivo temporal
    path = PDF_descarga_temp(publicacion.url_pdf)    
    # Segmentacion del documento en formato dataset, con las 
    # imagenes reconstruidas para facilitar el trabajo a Florence-2
    documento_dataset = Archivo_to_OCR(path, doc_converter=doc_converter)

    # Transformacion de imagen y tabla a texto enriqueciendo el documento
    documento_enriquecido = enriquecimiento_doc(documento_dataset, F2_model, F2_processor)
    # Embedding de todo el documento
    embedding = embeder._get_text_embedding(documento_enriquecido)

    print(summarizer_bart(documento_enriquecido, max_length=1000, min_length=30, do_sample=False)[0]['summary_text'])
    # Dividir el texto en chunks
    #print(documento_enriquecido)

    break


stage_df = pd.DataFrame()

             




