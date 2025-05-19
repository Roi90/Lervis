"""
Este archivo contiene el desarrollo del Chatbot con Llama 3.1 mediante Ollama.

Autor: Roi Pereira Fiuza
"""
import json
import re
import pandas as pd
from datetime import datetime,timedelta
import time
from ollama import chat

from Functions.Embeddings import embedding, carga_BAAI
from Functions.BBDD_functions import conn_bbdd
from Functions.Loggers import crear_logger
from Functions.BBDD_functions import similitud_coseno_chunks, similitud_coseno_resumen, reranking, formato_contexto_doc_recuperados,temporalidad_a_SQL, conn_bbdd , recuperar_documento_por_arxiv_id
from Functions.MarianMT_traductor import carga_modelo_traductor, translate_text

logger = crear_logger('Funciones_Chatbot', 'Funciones_Chatbot.log')
logger_llama = crear_logger('Llama3_1', 'Llama3_1.log')
logger_llama_intencion = crear_logger('Llama3_1_intencion', 'Llama3_1_intencion.log')

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
    try:
        conn =conn_bbdd()
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        

    
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
    info_incial += f"Numero total de publicaciones: {categorias[0]['conteo_categorias']}\n"
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
    "Usa 'consultar' si detectas verbos como consultar, buscar, encontrar, obtener, recuperar, listar, investigar, explorar o similares.\n"
    "- hablar → Siempre que no encuentres los verbos de consultar sera hablar.\n\n"
    "Debes RESPONDER **SIEMPRE** estrictamente con un JSON como uno de estos:\n"
    "{ \"intencion\": \"consultar\" }\n"
    "o\n"
    "{ \"intencion\": \"hablar\" }\n\n"
    "¡SIN NADA MÁS Y SIN ACENTOS NI COMENTARIOS! Únicamente responde con la clave \"intencion\"."
)
    #full_prompt = f"{system_prompt}\n\nUsuario: {user_input}\nClasificación:"

    try:
        response = chat(
            model="llama3.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ],
            stream=False 
        )

        texto_respuesta = response["message"]["content"]
        # transformacion a segundos
        duracion_segundos = response['total_duration'] / 1e9
        # Parseamos el JSON de la respuesta
        try:
            respuesta_json = json.loads(texto_respuesta)
            intencion = respuesta_json.get("intencion")
            if intencion  in ["consultar", "hablar"]:
                logger_llama_intencion.debug(f"Input usuario - {user_input}, Intencion detectada - {respuesta_json}")
                logger_llama_intencion.debug(f"Tokens de entrada - {response['prompt_eval_count']}, Tokens generados - {response['eval_count']},  Duracion Segundos - {duracion_segundos:.2f}")
                return intencion
            else:
                logger_llama_intencion.error(f"Respuesta inesperada del modelo: {texto_respuesta},Input usuario {user_input}")
                # En caso de respuesta inesperada, se habla
                return "hablar"
        except json.JSONDecodeError:
            logger_llama_intencion.error(f"Error al decodificar JSON: {texto_respuesta}")
            # En caso de error en el JSON, se habla
            return "hablar"

    except Exception as e:
        logger_llama_intencion.error(f"Error detectando intención: {e}")
        # En caso de error, se habla
        return "hablar"


def Llama3_1_API(system_prompt, context_prompt, user_prompt):
    try:
        # Utiliza la API local.
        stream = chat(
            model="llama3.1",
            
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": context_prompt},
            {"role": "user", "content": user_prompt}
        ],
            stream=True,
            options={'temperature': 0.3, # Se busca profesionalidad y certeza
                     "num_ctx": 5000, # Ventana de tokens que puede manejar el modelo como entrada
                     "num_predict": 800} # Número máximo de tokens que generará el modelo como salida
            

        )

        for part in stream:
            if "message" in part:
                if part.get("done", False):
                    # Verificacion de que el streaming ha terminado
                    try:
                        duracion_segundos = part['total_duration'] / 1e9
                        if duracion_segundos > 0:
                            token_segundo = part['eval_count'] / duracion_segundos
                        else:
                            token_segundo = 0
                        logger_llama.debug(f"Tokens de entrada - {part['prompt_eval_count']}, Tokens generados - {part['eval_count']},  Duracion Segundos - {duracion_segundos:.2f}, Token por segundo - {token_segundo:.2f}")
                    except Exception as e:
                        logger_llama.error(f"Error extrayendo metricas del modelo: {e}")
                
                yield part["message"]["content"]
 
                
    except Exception as e:
        logger_llama.error(f"Error en la API de Llama3.1: {e}")
        yield "Lo siento, hubo un error al procesar tu solicitud."


def detectar_identificador_arxiv(user_input: str):
    """
    Detecta si hay un identificador de arXiv válido en el texto del usuario.

    Args:
        user_input (str): Texto introducido por el usuario.

    Returns:
        str | None: El identificador de arXiv si se encuentra, o None.
    """
    patron_arxiv = r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"

    match = re.search(patron_arxiv, user_input, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def RAG_chat_V2(urls_usados, user_input:str, context: str, logger, conn, embedding_model, traductor_model, traductor_tokenizer):

    context_prompt = context
    user_prompt = user_input

    arxiv_detectado = detectar_identificador_arxiv(user_input)
    if arxiv_detectado:
        logger.debug(f"Arxiv id detectado: {arxiv_detectado}, User input {user_input}")
        try:
           titulo_documento, documento_completo = recuperar_documento_por_arxiv_id(arxiv_detectado,conn)
        except Exception as e:
            logger.error(f'Error al recuperar el documento mediante identificador de arxiv: {e}')

        system_prompt = f"""
            Genera un resumen de cada apartado a partir del documento proporcionado, resaltando puntos claves y las conclusiones. Con formato Markdown.
            **CONTESTA UNICAMENTE ESPAÑOL**  
            """
        
        context_prompt += f"\n\n Titulo del documento: {titulo_documento}\n\nDocumento proporcionado:\n\n{documento_completo[:4000]}" # Esto se limita dada la limitacion computacional.
        
        return system_prompt, context_prompt, user_prompt

    if deteccion_intencion(user_input) == 'consultar':
        temporalidad = deteccion_temporalidad(user_input)
        # Se detecta temporalidad en el input del usuario.
        if temporalidad is not None:
            
            try:
                logger.debug(f"Temporalidad detectada: {temporalidad}")
                #user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                consulta = temporalidad_a_SQL(conn,temporalidad)
                # Apendizo al contexto
                #context += f'\n{ahora} Usuario: {user_input}'
                context_prompt += f"\nLervis: Conteo de publicaciones en la base de datos para {temporalidad[1]}: {consulta}"
                system_prompt = f"""
                Eres un experto en publicaciones academicas de Arxiv en la categoria CS, Ciencias de la computacion.
                Se conciso y claro en tus respuestas.
                Tono profesional y amable.
                Usa el contexto lo maximo posible para responder.
                **CONTESTA UNICAMENTE ESPAÑOL**  
                """
                #{context}
                return system_prompt, context_prompt, user_prompt
            
            except Exception as e:
                logger.error(f"En la lectura de temporalidad: {e} - temporalidad: {temporalidad}")
                system_prompt = "Responde brevemente y en español."
                return system_prompt, "Error al recuperar contexto.", "Disculpa, ¿podrías repetir la pregunta?"
        
        # No Se detecta temporalidad en el input del usuario.
        else:
            try:# Se traduce el input para mejores resultados.
                user_input_en = translate_text(traductor_model, traductor_tokenizer, user_input)
                # Dado que se consulta, se genera el embedding
                embedding_denso, embedding_disperso = embedding(user_input_en, model=embedding_model)
                # Recuperacion de documentos mediante similitud del coseno (Distancia para simplificarlo)
                docs_recuperados_chunk = similitud_coseno_chunks(embedding_denso, conn, 0.45, 5)
                docs_recuperados_resumen = similitud_coseno_resumen(embedding_denso, conn, 0.5, 5)
                if docs_recuperados_chunk:
                    #print(docs_recuperado)
                    df_temp = reranking(docs_recuperados_chunk, embedding_disperso)
                    # El numero de documentos generados en el contexto es en funcion de num_docs
                    doc_formateado = formato_contexto_doc_recuperados(urls_usados, conn, df_temp, num_docs=2)
                    print(doc_formateado)
                    # Validacion de que no esta vacio
                    context_prompt += f"\n\nDOCUMENTOS RECUPERADOS:\n\n{doc_formateado}"
                    system_prompt = f"""
                    **CONTESTA UNICAMENTE EN ESPAÑOL** 
                    **Responde únicamente utilizando el contexto bajo 'DOCUMENTOS RECUPERADOS'.**
                    **Respeta el formato pero traducelo al español**
                    NO inventes publicaciones. Si no hay información suficiente, indícalo.
                    
                    """
                    #{context}
                    return system_prompt, context_prompt, user_prompt
                
                elif docs_recuperados_resumen:

                    df_temp = reranking(docs_recuperados_resumen, embedding_disperso)
                    # El numero de documentos generados en el contexto es en funcion de num_docs
                    doc_formateado = formato_contexto_doc_recuperados(urls_usados, conn, df_temp, num_docs=2)
                    #print(doc_formateado)
                    # Validacion de que no esta vacio
                    context_prompt += f"\n\nDOCUMENTOS RECUPERADOS:\n\n{doc_formateado}"
                    system_prompt = f"""
                    **CONTESTA UNICAMENTE EN ESPAÑOL** 
                    **Responde únicamente utilizando el contexto bajo 'DOCUMENTOS RECUPERADOS'.**
                    **Respeta el formato pero traducelo al español**
                    NO inventes publicaciones. Si no hay información suficiente, indícalo.
                    """
                    #{context}
                    return system_prompt, context_prompt, user_prompt
                else:
                    system_prompt = """
                    No se han encontrado resultados relevantes. Intenta ser más específico o proporciona un identificador arXiv si lo tienes.
                    **CONTESTA UNICAMENTE ESPAÑOL**
                    """
                    return system_prompt, context_prompt, user_prompt
        
            except Exception as e:
                logger.error(f"Error en la consulta de RAG_chat_V2: {e} - temporalidad: {temporalidad}")
                system_prompt = "Responde brevemente y en español."
                return system_prompt, "Error al recuperar contexto.", "Disculpa, ¿podrías repetir la pregunta?"
                

    else: # Chatear con el usuario
        
        system_prompt = f"""  
        Eres un experto en publicaciones academicas de Arxiv en la categoria CS, Ciencias de la computacion.
        Se conciso y claro en tus respuestas.
        Tono profesional y amable.
        Usa el contexto lo maximo posible para responder.
        **CONTESTA UNICAMENTE ESPAÑOL**               
        """
        return system_prompt, context_prompt, user_prompt

