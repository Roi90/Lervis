"""
Este archivo contiene funciones para la primera carga de datos en la BBDD, incluyendo el proceso
de transformacion de los datos.

Autor: Roi Pereira Fiuza
"""

import os
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine

from Functions.API_metadata import extraccion_por_categorias
from Functions.Docling_OCR import Archivo_to_OCR, Carga_Docling_OCR
from Functions.PDF_descarga import PDF_descarga_temp
from Functions.Florence_2_anotacion import  Carga_FLorence2_modelo
from Functions.Enriquecimiento_documento import enriquecimiento_doc
from Functions.Embeddings import carga_BAAI, embedding
from Functions.BBDD_functions import conn_bbdd, carga_dimension_categorias, carga_hechos_publicaciones,\
normalizador_id_categoria_BBDD, carga_hechos_chunks_embeddings,carga_hechos_resumen_embeddings, carga_doc_enriquecido

# Configurar la variable de entorno para la arquitectura CUDA de la RTX 3050 Laptop GPU
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
# Utiliza la primera GPU disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

# -----------------
target_por_categoria = 1
# -----------------

#  ----------------- Carga de modelos
# OCR
doc_converter = Carga_Docling_OCR()
# Anotacion de imagenes
F2_model_id, F2_model, F2_processor = Carga_FLorence2_modelo()
# Embeddings
modelo_BAAI = carga_BAAI()

# motor de bbdd
conn = conn_bbdd()
# engine para inyectar como df (Problemas de insercion de hstore y vector)
engine = create_engine('postgresql://postgres:Quiksilver90!@localhost:5432/Lervis')

# Diccionario para transformar los datos en funcion del id generado en la carga en la BBDD
categorias_id_dict = carga_dimension_categorias(conn)

# DF de los metadatos para la extraccion
metadatos_publicaciones = extraccion_por_categorias(target_por_categoria)

# Normalizacion de los codigos de categorias en el id de la tabla CATEGORIA
metadatos_publicaciones = normalizador_id_categoria_BBDD(metadatos_publicaciones, categorias_id_dict)

# Eliminado de duplicados por posible extraccion desde distintas categorias
metadatos_publicaciones = metadatos_publicaciones.drop_duplicates('identificador_arxiv')

# Insercion de los metadatos en la BBDD - Tabla: Publicaciones
publicaciones_id_dict = carga_hechos_publicaciones(conn, metadatos_publicaciones)
#metadatos_publicaciones.to_sql('publicaciones', con=engine, if_exists='append', index=False)

# Descarga, enriquecimiento del documento y embeddings
for publicacion in metadatos_publicaciones.itertuples():
    
    # Descarga el PDF en un archivo temporal
    path = PDF_descarga_temp(publicacion.url_pdf)  

    # Segmentacion del documento en formato dataset, con las 
    # imagenes reconstruidas para facilitar el trabajo a Florence-2
    documento_dataset = Archivo_to_OCR(path, doc_converter=doc_converter)

    # Transformacion de imagen y tabla a texto enriqueciendo el documento
    documento_enriquecido = enriquecimiento_doc(documento_dataset, F2_model, F2_processor)

    # Carga del documento enriquecido en la BBDD
    carga_doc_enriquecido(documento_enriquecido, publicaciones_id_dict[publicacion.identificador_arxiv],  conn)

    # Chunking del documento en partes de 2000 caracteres y 300 de solapamiento
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

    chunks = text_splitter.split_text(documento_enriquecido)

    for c in chunks:
        # Embedding de todo el documento (Denso y disperso) gracias a la funcionalidad de BAAI
        chunk_embedding_denso, chunk_embedding_disperso = embedding(c, model=modelo_BAAI)

        datos_contenido = {
        # Normalizo la id_publicaciones mediante el dict extraido anteriormente
       'id_publicaciones': publicaciones_id_dict[publicacion.identificador_arxiv],
       'chunk': c,
       # Se convierte el vector denso np.array a lista que PostgreSQL guardara como vector.
       'chunk_emb_dense': chunk_embedding_denso.astype(np.float32).tolist(),
       # Se convierte el vector disperso defaultdict a dict que PostgreSQL guardara como hstore.
       'chunk_emb_sparse': {str(key): str(value) for key, value in chunk_embedding_disperso.items()},
        }
        # Conversion a DataFrame para la carga en la BBDD
        df_temp = pd.DataFrame([datos_contenido])
        # Inyeccion del chunk en la BBDD
        carga_hechos_chunks_embeddings(df_temp, engine)

    # Embedding del abstract (Denso y disperso)
    res_embedding_denso, res_embedding_disperso = embedding(publicacion.resumen, model=modelo_BAAI)

    datos_resumen = {
        # Normalizo la id_publicaciones mediante el dict extraido anteriormente
       'id_publicaciones': publicaciones_id_dict[publicacion.identificador_arxiv],
       'resumen': publicacion.resumen,
       'resumen_emb_dense': res_embedding_denso.astype(np.float32).tolist(),
       'resumen_emb_sparse': {str(key): str(value) for key, value in res_embedding_disperso.items()}
    }
    df_temp_resumen = pd.DataFrame([datos_resumen])
    carga_hechos_resumen_embeddings(df_temp_resumen, engine)
    break

#embeddings_df = pd.DataFrame(datos_embeddings_lst)

#embeddings_df.to_sql('embeddings', con=engine, if_exists='append', index=False)

# ----------------------------------------------TESTEO
#documento_dataset = Archivo_to_OCR(r'C:\Users\Usuario\OneDrive\UOC\TFG\Lervis\Lervis\2008.05746v1.pdf', doc_converter=doc_converter)
#documento_enriquecido = enriquecimiento_doc(documento_dataset, F2_model, F2_processor)
#embedding_denso, embedding_disperso = embedding(documento_enriquecido, model=modelo_BAAI)
#print(embedding_denso)
#print(embedding_disperso)






             




