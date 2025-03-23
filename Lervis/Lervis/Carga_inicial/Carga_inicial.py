"""
Este archivo contiene funciones para la primera carga de datos en la BBDD, incluyendo el proceso
de transformacion de los datos.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import os
import pandas as pd
from Functions.API_metadata import extraccion_por_categorias
from Functions.Docling_OCR import Archivo_to_OCR, Carga_Docling_OCR
from Functions.PDF_descarga import PDF_descarga_temp
from Functions.Florence_2_anotacion import  Carga_FLorence2_modelo
from Functions.Enriquecimiento_documento import enriquecimiento_doc
from Functions.Embeddings import carga_BAAI, embedding
from Functions.Bart_Generacion_Resumen import carga_BART, max_token_fragmentacion
from Functions.BBDD_config import engine_bbdd, carga_dimension_categorias, normalizador_id_BBDD

# Configurar la variable de entorno para la arquitectura CUDA de la RTX 3050 Laptop GPU
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

# -----------------
target_por_categoria = 1
# -----------------

#  ----------------- Carga de modelos
# OCR
doc_converter = Carga_Docling_OCR()
# Anotacion de imagenes
F2_model_id, F2_model, F2_processor = Carga_FLorence2_modelo()
# Embeddings
#modelo_BAAI = carga_BAAI()
# Generador de resumenes
summarizer_bart, tokenizer_bart = carga_BART()
# -----------------

#  ----------------- Insercion dimensiones y motor de bbdd
engine = engine_bbdd()
# Diccionario para transformar los datos en funcion del id generado en la carga en la BBDD
#categorias_id_dict = carga_dimension_categorias(engine)
# -----------------

# DF de los metadatos para la extraccion
metadatos_publicaciones = extraccion_por_categorias(target_por_categoria)
# Transformacion de los codigos de categorias en el id de la tabla CATEGORIA
#metadatos_publicaciones = normalizador_id_BBDD(metadatos_publicaciones, categorias_id_dict)
# Eliminado de duplicados por posible extraccion desde distintas categorias
metadatos_publicaciones = metadatos_publicaciones.drop_duplicates('identificador_arxiv')

# Insercion de los metadatos en la BBDD - Tabla: Publicaciones
#metadatos_publicaciones.to_sql('publicaciones', con=engine, if_exists='append', index=False)


# Descarga, enriquecimiento del documento, generacion de resumenes y embeddings
for publicacion in metadatos_publicaciones.itertuples():

    # Descarga el PDF en un archivo temporal
    path = PDF_descarga_temp(publicacion.url_pdf)  

    # Segmentacion del documento en formato dataset, con las 
    # imagenes reconstruidas para facilitar el trabajo a Florence-2
    documento_dataset = Archivo_to_OCR(path, doc_converter=doc_converter)

    # Transformacion de imagen y tabla a texto enriqueciendo el documento
    documento_enriquecido = enriquecimiento_doc(documento_dataset, F2_model, F2_processor)

    # Embedding de todo el documento (Denso y disperso) gracias a la funcionalidad de BAAI
#    embedding_denso, embedding_disperso = embedding(documento_enriquecido, model=modelo_BAAI)
    # Resumen enriquecido
    resumen_enriquecido = summarizer_bart(documento_enriquecido, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']
    fragmentos_lst = max_token_fragmentacion(resumen_enriquecido)
    print(fragmentos_lst[0])
    break
    # Dividir el texto en chunks
    #print(documento_enriquecido)

#    break
#documento_dataset = Archivo_to_OCR(r'C:\Users\Usuario\OneDrive\UOC\TFG\Lervis\Lervis\2503.04725v1.pdf', doc_converter=doc_converter)
#documento_enriquecido = enriquecimiento_doc(documento_dataset, F2_model, F2_processor)
#embedding_denso, embedding_disperso = embedding(documento_enriquecido, model=modelo_BAAI)
#print(embedding_denso)
#rint(embedding_disperso)

# ----------------------------------------------RESUMEN QUEDA PENDIENTE
#print(summarizer_bart(documento_enriquecido, max_length=1000, min_length=30, do_sample=False)[0]['summary_text'])





             




