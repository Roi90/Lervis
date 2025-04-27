"""
Este archivo contiene el desarrollo del Chatbot con Llama 3.1 mediante Ollama.

Autor: Roi Pereira Fiuza
"""
import json
import requests
import re
import subprocess
import pandas as pd
from datetime import datetime,timedelta
from ollama import chat

import requests
from Functions.Embeddings import embedding, carga_BAAI
from Functions.BBDD_functions import conn_bbdd
from Functions.Loggers import Llama31_chatbot_log
from Functions.BBDD_functions import similitud_coseno, reranking, formato_contexto_doc_recuperados,temporalidad_a_SQL, conn_bbdd
from Functions.MarianMT_traductor import carga_modelo_traductor, translate_text

def eliminacion_acentos(user_input: str) -> str:
    """
    Elimina los acentos de las vocales en una cadena de texto dada.
    Esta función reemplaza las vocales acentuadas (tanto en minúsculas como en mayúsculas)
    por sus equivalentes sin acento. Por ejemplo, 'á' se reemplaza por 'a', 'É' se reemplaza
    por 'E', etc.
    Args:
        user_input (str): La cadena de texto de entrada que puede contener vocales acentuadas.
    Returns:
        str: La cadena de texto resultante con las vocales acentuadas reemplazadas por sus
        equivalentes sin acento.
    """

    # Definir la expresión regular para reemplazar las vocales acentuadas
    user_input = re.sub(r'[áàäâ]', 'a', user_input)
    user_input = re.sub(r'[éèëê]', 'e', user_input)
    user_input = re.sub(r'[íìïî]', 'i', user_input)
    user_input = re.sub(r'[óòöô]', 'o', user_input)
    user_input = re.sub(r'[úùüû]', 'u', user_input)

    # También reemplazar las vocales en mayúsculas
    user_input = re.sub(r'[ÁÀÄÂ]', 'A', user_input)
    user_input = re.sub(r'[ÉÈËÊ]', 'E', user_input)
    user_input = re.sub(r'[ÍÌÏÎ]', 'I', user_input)
    user_input = re.sub(r'[ÓÒÖÔ]', 'O', user_input)
    user_input = re.sub(r'[ÚÙÜÛ]', 'U', user_input)

    # Todo a minusculas
    user_input = user_input.lower()

    return user_input

def limitador_contexto(input_context: str, max_chars: int = 200000) -> str:
    """
    Recorta el contexto si supera un número máximo de caracteres.
    Mantiene los últimos caracteres más recientes.
    """
    if len(input_context) > max_chars:
        return input_context[-max_chars:]
    return input_context

def actualizacion_informacion_inicial():
    """
    Actualiza la información inicial del chatbot.
    Esta función se puede utilizar para modificar el contexto inicial que el chatbot utiliza para responder a las preguntas.
    """
    # Variable para almacenar la información inicial actualizada
    info_incial = ''
    conn =conn_bbdd()

    
    # Consulta SQL para obtener la información sobre las categorías y el conteo de publicaciones
    query = """
        SELECT 
               MIN(publicaciones.fecha_publicacion) as fecha_minima,
               MAX(publicaciones.fecha_publicacion) as fecha_maxima,
               COUNT(*) as conteo_categorias
        FROM publicaciones
        LEFT JOIN categoria
            ON publicaciones.categoria_principal = categoria.id
    """
    
    # Crear un cursor para ejecutar la consulta
    with conn.cursor() as cur:
        cur.execute(query)
        categorias = cur.fetchall()
    
    ahora = datetime.utcnow().strftime("%d/%m/%Y %H:%M")
    info_incial += f'{ahora} -- Categorias disponibles en la base de datos--\n\n'
    
    # Iterar sobre los resultados obtenidos de la consulta
    #for row in categorias:
    #    info_incial += f"Categoria: {row['categoria']} - Cantidad: {row['conteo_categorias']}\n"
        
    info_incial += f"Fecha minima de publicacion: {categorias[0]['fecha_minima']}\n"
    info_incial += f"Fecha maxima de publicacion: {categorias[0]['fecha_maxima']}\n"
    info_incial += f"Cantidad total de categorias: {categorias[0]['conteo_categorias']}\n"
    info_incial += f"Pagina web de ArXiv: https://www.arxiv.org/\n"
        
    return info_incial

def deteccion_temporalidad(user_input: str):

    # Diccionario pensado para realizar consultas por año y mes
    mes_anio_dict = {
        'Meses': [],
        'Anios': [],
    }

    # eliminacion acentos y 
    user_input = eliminacion_acentos(user_input)

    # Diccionario para meses
    meses_dict = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, 
        "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, 
        "noviembre": 11, "diciembre": 12
    }
    regex_mes = r'\b(?:' + '|'.join(meses_dict.keys()) + r')\b'

    # Diccionario para años
    anios_dict = {str(año): año for año in range(2000, 2026)}

    regex_anio = r'\b(?:' + '|'.join(anios_dict.keys()) + r')\b'

    # Diccionario para expresiones temporales
    expresiones_temporales_dict = {
    "hoy": lambda: timedelta(days=0),
    "ayer": lambda: timedelta(days=1),
    "antes de ayer": lambda: timedelta(days=2),
    "ultimos 2 dias": lambda: timedelta(days=3),
    "ultimos 3 dias": lambda: timedelta(days=3),
    "ultimos 5 dias": lambda: timedelta(days=3),
    "ultimos 7 dias": lambda: timedelta(days=7),
    "ultimos 15 dias": lambda: timedelta(days=15),
    "hace un dia": lambda: timedelta(days=1),
    "hace un mes": lambda: timedelta(days=30),
    "el mes pasado": lambda: timedelta(days=30),
    "este año": lambda: timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
    "el año pasado": lambda: timedelta(days=(datetime(datetime.now().year - 1, 1, 1) - datetime(datetime.now().year - 1, 12, 31)).days),
    "ultimos 2 meses": lambda: timedelta(days=60),
    "ultimos 3 meses": lambda: timedelta(days=90),
    "ultimo trimestre": lambda: timedelta(days=90),
    "ultimos 6 meses": lambda: timedelta(days=180),
    "ultimos 2 años": lambda: timedelta(days=730)
    }
    regex_expresiones_temporales = r'\b(?:' + '|'.join(expresiones_temporales_dict.keys()) + r')\b'

    if re.findall(regex_mes, user_input):
        meses_encontrados = re.findall(regex_mes, user_input)
        meses_lst = [meses_dict[m] for m in meses_encontrados]
        mes_anio_dict['Meses'] = meses_lst
        
    if re.findall(regex_anio, user_input):
        años_encontrados = re.findall(regex_anio, user_input)
        años_lst = [anios_dict[a] for a in años_encontrados] 
        mes_anio_dict['Anios']= años_lst

    # Devuelve el valor en dias de la expresion encontrada
    if re.findall(regex_expresiones_temporales, user_input):
        expresiones_encontradas = re.findall(regex_expresiones_temporales, user_input)
        exp_encontradas = [expresiones_temporales_dict[e]().days for e in expresiones_encontradas]
        return ('EXP',exp_encontradas[0], user_input)
    
    if len(mes_anio_dict['Meses']) > 0 and len(mes_anio_dict['Anios']) > 0:
        # Consulta anio y mes
        return ('Combinada',mes_anio_dict, user_input)
    elif len(mes_anio_dict['Meses']) > 0:
        # Consulta solo mes
        return ('Mes', mes_anio_dict['Meses'], user_input)
    elif len(mes_anio_dict['Anios']) > 0:
        # Consulta solo anio
        return ('Anio', mes_anio_dict['Anios'], user_input)
    else:
        # no se encuentra temporalidad en el input del usuario
        return None

def deteccion_intencion(user_input: str) -> str:
    system_prompt = (
    "Eres un clasificador cuya única función es decidir entre dos intenciones:\n\n"
    "- consultar → El usuario quiere consultar, buscar, recuperar o encontrar información en la base de datos. "
    "Usa 'consultar' si detectas verbos como consultar, buscar, encontrar, obtener, recuperar, listar, acceder, "
    "investigar, explorar o similares.\n"
    "- hablar → El usuario sólo quiere una respuesta conversacional sin necesidad de acceder a la base de datos.\n\n"
    "Debes RESPONDER **SIEMPRE** estrictamente con un JSON como uno de estos:\n"
    "{ \"intencion\": \"consultar\" }\n"
    "o\n"
    "{ \"intencion\": \"hablar\" }\n\n"
    "¡SIN NADA MÁS Y SIN ACENTOS NI COMENTARIOS! Únicamente responde con la clave \"intencion\"."
)
    full_prompt = f"{system_prompt}\n\nUsuario: {user_input}\nClasificación:"

    try:
        response = chat(
            model="llama3.1",
            messages=[{"role": "user", "content": full_prompt}],
            stream=False  # ❗️NO streaming
        )

        texto_respuesta = response["message"]["content"]

        # Parseamos el JSON de la respuesta
        respuesta_json = json.loads(texto_respuesta)
        print(f"🟢 Intencion: {respuesta_json}")
        return respuesta_json.get("intencion", "hablar")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"❌ Error detectando intención: {e}")
        return "hablar"


def Llama3_1_API(prompt):
    # Utiliza la API local.
    stream = chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for part in stream:
        yield part["message"]["content"]


def RAG_chat_V2(urls_usados, user_input:str, context: str, logger, conn, embedding_model, ahora,traductor_model, traductor_tokenizer) -> tuple[str, str]:

    if deteccion_intencion(user_input) == 'consultar':
        temporalidad = deteccion_temporalidad(user_input)
        print("\n\nTEMPORALIDAD----->",temporalidad, flush=True)
        # Se detecta temporalidad en el input del usuario.
        if temporalidad is not None:
            
            try:
                #user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                consulta = temporalidad_a_SQL(conn,temporalidad)
                # Apendizo al contexto
                #context += f'\n{ahora} Usuario: {user_input}'
                context += f"\nLervis: Conteo de publicaciones en la base de datos para {temporalidad[1]}: {consulta}"
                full_prompt = f"""
                Eres un experto en publicaciones academicas de Arxiv.\n
                Se conciso y claro en tus respuestas.\n
                Tono profesional y amable.\n
                Usa el contexto lo maximo posible para responder.\n
                **CONTESTA UNICAMENTE ESPAÑOL**\n\n   
                             
                {context}
                """
                return full_prompt.strip()#, context
            
            except Exception as e:
                print(f"En la lectura de temporalidad: {e}")
                logger.error(f"En la lectura de temporalidad: {e}")
                return "Disculpame, me estaba haciendo unos huevos fritos, ¿Podrías repetir la pregunta?", context
        
        # No Se detecta temporalidad en el input del usuario.
        else:
            try:# Se traudce el input para mejores resultados.
                user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                # Dado que se consulta, se genera el embedding
                embedding_denso, embedding_disperso = embedding(user_input, model=embedding_model)
                # Recuperacion de documentos mediante similitud del coseno
                docs_recuperado = similitud_coseno(embedding_denso, conn, 0.3, 5)
                df_temp = reranking(docs_recuperado, embedding_disperso)
                # El numero de documentos generados en el contexto es en funcion de num_docs
                doc_formateado = formato_contexto_doc_recuperados(urls_usados, conn, df_temp, num_docs=2)
                # Se apendiza el valor de la consulta al contexto
                #context += f'\n{ahora} Usuario: {user_input}'
                # Validacion de que no esta vacio
                if doc_formateado:
                    context += f"\n\nRealiza una breve explicacion sobre los documentos:\n\n{doc_formateado}"
                # ---------------------------------- LLM - Llama 3.1 ----------------------------------
                full_prompt = f"""  
                Eres un experto en publicaciones academicas de Arxiv.\n
                 Se conciso y claro en tus respuestas.\n
                Tono profesional y amable.\n
                Usa el contexto lo maximo posible para responder.\n
                **CONTESTA UNICAMENTE ESPAÑOL**\n\n               
                {context}
                """
                return full_prompt.strip()#, context
        
            except Exception as e:
                logger.error(f"Error al crear el embedding: {e}")
                return "Disculpame, me estaba sirviendo un cafe, ¿Podrías repetir la pregunta?", context

    else: # Chatear con el usuario
        
        # ---------------------------------- LLM - Llama 3.1 ----------------------------------
        #context += f'\n{ahora} Usuario: {user_input}'
        full_prompt = f"""  
        Eres un experto en publicaciones academicas de Arxiv.\n
        Se conciso y claro en tus respuestas.\n
        Tono profesional y amable.\n
        Usa el contexto lo maximo posible para responder.\n
        **CONTESTA UNICAMENTE ESPAÑOL**\n\n                
        {context}
        """
        return full_prompt.strip()#, context
