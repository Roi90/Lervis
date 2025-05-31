"""
Este archivo contiene funciones para extraer datos de la API de arXiv de forma diaria.
Proporciona métodos para buscar artículos, obtener detalles de los autores y
filtrar resultados por fecha de publicación.

Autor: Roi Pereira Fiuza
"""
import time
import os
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine

from Functions.PDF_descarga import PDF_descarga_temp
from Functions.Loggers import crear_logger
from Functions.API_metadata import extraccion_por_categorias
from Functions.Docling_OCR import Archivo_to_OCR, Carga_Docling_OCR
from Functions.Florence_2_anotacion import  Carga_FLorence2_modelo
from Functions.Enriquecimiento_documento import enriquecimiento_doc
from Functions.Embeddings import carga_BAAI, embedding_ETL, embedding_ETL_DOC_ENRI
from Functions.BBDD_functions import conn_bbdd, dict_catetorias, carga_hechos_publicaciones,\
normalizador_id_categoria_BBDD, carga_hechos_chunks_embeddings,carga_hechos_resumen_embeddings, carga_doc_enriquecido

logger = crear_logger('ETL_diaria', 'ETL_diaria.log')

# Configurar la variable de entorno para la arquitectura CUDA de la RTX 3050 Laptop GPU
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
# Utiliza la primera GPU disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

# -----------------
target_por_categoria = 3
# -----------------
try:
    inicio = time.time()
    #  ----------------- Carga de modelos
    # OCR
    doc_converter = Carga_Docling_OCR()
    # Anotacion de imagenes
    F2_model_id, F2_model, F2_processor = Carga_FLorence2_modelo()
    # Embeddings
    modelo_BAAI = carga_BAAI()
    fin = time.time()
    duracion = fin - inicio
    logger.info(f'Modelos cargados con exito - Duracion segundos: {duracion:.2f}')
except Exception as e:
    logger.error(f'Error en la carga de modelos: {e}')


try:
    inicio = time.time()
    # motor de bbdd
    conn = conn_bbdd()
    # engine para inyectar como df (Problemas de insercion de hstore y vector)
    engine = create_engine('postgresql://postgres:Quiksilver90!@localhost:5432/Lervis')
    fin = time.time()
    duracion = fin - inicio
    logger.info(f'Conexion a la BBDD establecida con exito - Duracion segundos: {duracion:.2f}')
except Exception as e:
    logger.error(f'Error en la configuracion de la conexion de la BBDD: {e}')
try:
    inicio = time.time()
    # Diccionario para transformar los datos en funcion del id generado en la carga en la BBDD
    categorias_id_dict = dict_catetorias(conn)
    fin = time.time()
    duracion = fin - inicio
    logger.info(f'Diccionario de categorias cargado con exito - Numero de categorias: {len(categorias_id_dict)} - Duracion segundos: {duracion:.2f}')
except Exception as e:
    logger.error(f'Error en la insercion de la dimension categorias: {e}')
try:
    inicio = time.time()
    # DF de los metadatos para la extraccion
    metadatos_publicaciones = extraccion_por_categorias(conn, categorias_id_dict, max_resultados=target_por_categoria)
    print(metadatos_publicaciones)
    fin = time.time()
    duracion = fin - inicio
    logger.info(f'Consulta API y generacion del df para descarga de documentos con exito - Numero de publicaciones : {len(metadatos_publicaciones)} - Duracion segundos: {duracion:.2f}')
except Exception as e:
    logger.error(f'Error en la consulta API y generacion del df para descarga de documentos: {e}')


try:
    inicio = time.time()
    # Normalizacion de los codigos de categorias en el id de la tabla CATEGORIA
    metadatos_publicaciones = normalizador_id_categoria_BBDD(metadatos_publicaciones, categorias_id_dict)

    # Eliminado de duplicados por posible extraccion desde distintas categorias
    metadatos_publicaciones = metadatos_publicaciones.drop_duplicates('identificador_arxiv')

    # Insercion de los metadatos en la BBDD - Tabla: Publicaciones
    publicaciones_id_dict = carga_hechos_publicaciones(conn, metadatos_publicaciones)

    fin = time.time()
    duracion = fin - inicio
    #metadatos_publicaciones.to_sql('publicaciones', con=engine, if_exists='append', index=False)
    logger.info(f'Normalizacion de datos e insercion de metadatos en la tabla de publicaciones con exito - Numero de publicaciones sin duplicados insertadas: {len(metadatos_publicaciones)} - Duracion segundos: {duracion:.2f}')
except Exception as e:
    logger.error(f'Error en la normalizacion de metadatos e insercion de datos en la tabla de publicaciones: {e}')

total_embeddings = 0
total_fragmentos = 0
total_publicaciones = 0
try:
    inicio_pipeline = time.time()
    # Descarga, enriquecimiento del documento y embeddings
    for publicacion in metadatos_publicaciones.itertuples():
        
            
            total_publicaciones += 1
            try:
                inicio = time.time()
                # Descarga el PDF en un archivo temporal
                path = PDF_descarga_temp(publicacion.url_pdf)
                fin = time.time()
                duracion = fin - inicio
                logger.debug(f'Archivo temporal generado con exito de {publicacion.url_pdf} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
            except Exception as e:
                logger.error(f'Error en la descarga del PDF y generacion de archivo temporal: {e} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                break

            try:
                inicio = time.time()
                # Segmentacion del documento en formato dataset, con las 
                # imagenes reconstruidas para facilitar el trabajo a Florence-2
                documento_dataset = Archivo_to_OCR(path, doc_converter=doc_converter)
                fin = time.time()
                duracion = fin - inicio
                logger.debug(f'Segmentacion del documento con exito - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
            except Exception as e:
                logger.error(f'Error en la segmentacion del documento (DOCLING): {e} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                break

            try:
                inicio = time.time()
                # Transformacion de imagen y tabla a texto enriqueciendo el documento
                documento_enriquecido = enriquecimiento_doc(documento_dataset, F2_model, F2_processor)
                fin = time.time()
                duracion = fin - inicio
                logger.debug(f'Enriquecimiento del documento con exito - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
            except Exception as e:
                logger.error(f'Error en el enriquecimiento del documento: {e} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                break
            try:
                inicio = time.time()
                # Carga del documento enriquecido en la BBDD
                carga_doc_enriquecido(documento_enriquecido, publicacion.identificador_arxiv,  conn)
                # Genero chunks del documento mas grande para realizar un embedding medio, dado que los documentos
                # superan la cantidad de tokens que el modelo puede ingerir
                try:
                    text_splitter_doc_enri = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)
                    chunks_doc_enri = text_splitter_doc_enri.split_text(documento_enriquecido)
                    doc_chunks = []
                    for doc_chunk in chunks_doc_enri:
                        embedding_doc_enri_denso, _ = embedding_ETL_DOC_ENRI(doc_chunk, publicacion.identificador_arxiv, model=modelo_BAAI)
                        doc_chunks.append(embedding_doc_enri_denso)
                    embedding_doc_enri_media_denso = np.mean(doc_chunks, axis=0)
                except Exception as e:
                    logger.error(f'Error en la creacion del embedding del documento enriquecido: {e}- Publicacion: {publicacion.identificador_arxiv}')
                
                fin = time.time()
                duracion = fin - inicio
                logger.debug(f'Carga del documento enriquecido en la BBDD con exito - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
            except Exception as e:
                logger.error(f'Error en la carga en la BBDD del documento enriquecido: {e} - Publicacion: {publicacion.identificador_arxiv}')
                break


            try:
                inicio = time.time()
                # Chunking del documento en partes de 3000 caracteres y 300 de solapamiento
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

                chunks = text_splitter.split_text(documento_enriquecido)
                fin = time.time()
                duracion = fin - inicio
                logger.debug(f'Chunking del documento enriquecido con exito - Numero de chunks: {len(chunks)} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
            except Exception as e:
                logger.error(f'Error al realizar chunks del documento enriquecido: {e} - Publicacion: {publicacion.identificador_arxiv}')
                break
            
            for c in chunks:
                try:
                    inicio = time.time()
                    # Embedding de todo el documento (Denso y disperso) gracias a la funcionalidad de BAAI
                    chunk_embedding_denso, chunk_embedding_disperso = embedding_ETL(c,embedding_doc_enri_media_denso, publicacion.identificador_arxiv ,model=modelo_BAAI )
                    total_fragmentos += 1
                    total_embeddings += 1
                    fin = time.time()
                    duracion = fin - inicio
                    logger.debug(f'Embedding del chunk con exito - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                except Exception as e:
                    logger.error(f'Error al generar embedding del chunk: {e} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                    break
                
                try:
                    inicio = time.time()
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
                    fin = time.time()
                    duracion = fin - inicio
                    logger.debug(f'Insercion del chunk y sus embeddings en la BBDD con exito - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                except Exception as e:
                    logger.error(f'Error al insertar el chunk y sus embeddings en la BBDD: {e} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                    break

            try:
                inicio = time.time()
                # Embedding del abstract (Denso y disperso)
                res_embedding_denso, res_embedding_disperso = embedding_ETL(publicacion.resumen,embedding_doc_enri_media_denso, publicacion.identificador_arxiv, model=modelo_BAAI)
                total_embeddings += 1
                fin = time.time()
                duracion = fin - inicio
                logger.debug(f'Embedding del resumen con exito - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
            except Exception as e:
                logger.error(f'Error al generar embedding del resumen: {e} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                break

            try:
                inicio = time.time()
                datos_resumen = {
                    # Normalizo la id_publicaciones mediante el dict extraido anteriormente
                'id_publicaciones': publicaciones_id_dict[publicacion.identificador_arxiv],
                'resumen': publicacion.resumen,
                'resumen_emb_dense': res_embedding_denso.astype(np.float32).tolist(),
                'resumen_emb_sparse': {str(key): str(value) for key, value in res_embedding_disperso.items()}
                }
                df_temp_resumen = pd.DataFrame([datos_resumen])
                carga_hechos_resumen_embeddings(df_temp_resumen, engine)
                fin = time.time()
                duracion = fin - inicio
                logger.debug(f'Insercion del resumen y sus embeddings en la BBDD con exito - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
            except Exception as e:
                logger.error(f'Error al insertar el resumen y sus embeddings en la BBDD: {e} - Duracion segundos: {duracion:.2f} - Publicacion: {publicacion.identificador_arxiv}')
                break
    fin_pipeline = time.time()
    duracion_pipeline = fin_pipeline - inicio_pipeline
    logger.debug(f'ETL inicial finalizado con exito - Duracion segundos: {duracion_pipeline:.2f} - Total Publicaciones: {total_publicaciones} - Total Chunks: {total_fragmentos} - Total Embeddings: {total_embeddings}')
except Exception as e:
    logger.error(f'Error en el pipeline de extraccion: {e} -Duracion segundos: {duracion_pipeline:.2f} - Total Publicaciones: {total_publicaciones} - Total Chunks: {total_fragmentos} - Total Embeddings: {total_embeddings}')
    

       
