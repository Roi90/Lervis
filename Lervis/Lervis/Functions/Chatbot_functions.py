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
    Elimina los acentos de las vocales en una cadena de texto y la convierte a minúsculas.

    Esta función reemplaza las vocales acentuadas tanto en minúsculas como en mayúsculas
    por sus equivalentes sin acento, y transforma todo el texto a minúsculas.

    Args:
        user_input (str): Cadena de texto que puede contener vocales acentuadas.

    Returns:
        str: Texto sin acentos y en minúsculas.
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

def limitador_contexto(input_context: list, max_men: int = 50) -> list:
    """
    Limita la longitud del historial de contexto eliminando los mensajes más antiguos.

    Si el número de mensajes excede el límite definido por max_men, 
    los mensajes más antiguos se eliminan para mantener el tamaño permitido.

    Args:
        input_context (list): Lista de mensajes que representa el historial del contexto.
        max_men (int, optional): Número máximo de mensajes permitidos. Por defecto es 50.

    Returns:
        list: Lista de mensajes truncada si se excede el máximo permitido.
    """
    exceso = len(input_context) - max_men
    if exceso > 0:
        return input_context[exceso:]
    return input_context

def actualizacion_informacion_inicial():
    """
    Actualiza la información contextual inicial utilizada por el LLM.

    Esta función se conecta a la base de datos y genera un resumen actualizado
    con la fecha mínima y máxima de publicación, el número total de publicaciones,
    y un enlace a la página principal de ArXiv. 

    Returns:
        str: Cadena de texto con la información inicial formateada.
    
    Raises:
        Exception: Si ocurre un error al conectar o consultar la base de datos.
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
    """
    Detecta y clasifica referencias temporales presentes en el input del usuario.

    Analiza el texto para identificar meses, años o expresiones temporales comunes. 
    
    Clasifica la detección como una de las siguientes categorías:
    - Combinada: Detecta meses y años.
    - Mes: Detecta meses.
    - Anio: Detecta años.
    - EXP: Detecta una expresión temporal relativa, como ayer o últimos 3 días.
    - None: Si no se encuentra ninguna referencia temporal.

    Args:
        user_input (str): Texto de entrada del usuario.

    Returns:
        tuple o None: Tupla con la categoría detectada, los valores asociados y el input original.
                      None si no se detecta ninguna temporalidad.
    """

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
    """
    Clasifica la intención del usuario en consultar o hablar, usando el LLM.

    Esta función envía una instrucción al LLM para determinar si el input del usuario
    tiene intención de:
    - Consultar: cuando desea recuperar información de una base de datos.
    - Hablar: cuando no se detectan verbos relacionados con la consulta.

    El modelo debe responder exclusivamente con un JSON: 
    { "intencion": "consultar" } o { "intencion": "hablar" }

    Si la respuesta es inválida o el modelo falla, se asume por defecto que la intención es 'hablar'.

    Args:
        user_input (str): Texto introducido por el usuario.

    Returns:
        str: 'consultar' o 'hablar', según lo detectado por el modelo.
    """

    system_prompt = (
    "Eres un clasificador cuya única función es decidir entre dos intenciones:\n\n"

    "- consultar = El usuario quiere consultar, buscar información en la base de datos. "
    "Usa 'consultar' si detectas verbos como consultar o buscar.\n"
    "- hablar = Siempre que no encuentres los verbos de consultar sera hablar.\n\n"
    "Debes RESPONDER **SIEMPRE** estrictamente con un JSON como uno de estos:\n"
    "{ \"intencion\": \"consultar\" }\n"
    "o\n"
    "{ \"intencion\": \"hablar\" }\n\n"
    "¡SIN NADA MÁS Y SIN ACENTOS NI COMENTARIOS! Únicamente responde con la clave \"intencion\"."
)
    
#buscar, encontrar, obtener, recuperar, listar, investigar, explorar
    try:
        response = chat(
            model="llama3.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ],
            stream=False,
            options= {'temperature': 0.1, 
                     "num_ctx":  1000, 
                     "num_predict": 100}
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


def Llama3_1_API(system_prompt, context, user_prompt, rag_flow):
    """
    Genera una respuesta usando el LLM , ajustando los parámetros según el flujo RAG detectado.

    Esta función configura dinámicamente los parámetros de temperatura, número de tokens de contexto 
    y tokens de predicción en función del tipo de flujo (rag_flow) detectado. 
    
    Luego, realiza una llamada  a la API local de Ollama, del LLM en modo streaming.

    Args:
        system_prompt (str): Instrucción inicial del sistema que establece el comportamiento del modelo.
        context (list): Lista de mensajes previos como historial de conversación.
        user_prompt (str): Mensaje actual del usuario.
        rag_flow (str): Tipo de flujo de generación. Puede ser:
            - id_arxiv: Resumen detallado de documentos completos.
            - consultar_temp: Consulta basada en temporalidad.
            - consultar_docs_chunk: Recuperación basada en chunks.
            - consultar_docs_resumen: Recuperación basada en resúmenes.
            - hablar: Interacción generica de conversacion.
            - recuperacion_vacia: Cuando no hay documentos encontrados en base a
            la similitud.

    Yields:
        str: Fragmentos de texto generados por el modelo en tiempo real (streaming).

    Raises:
        Exception: Si se produce un error durante la llamada al modelo.
    """
    try:
        # Valores default para ser sobre escritos en funcion del flow
        options= {'temperature': 0.3, 
                     "num_ctx":  2048, 
                     "num_predict": 2000} 

        if rag_flow == 'id_arxiv':

            options['temperature'] = 0.7 #Profesional pero mas abierto a mayores variaciones
            options['num_ctx'] =  10000 # Gran ventana dado los grandes documentos
            options['num_predict'] = 5000 # Generacion extensa dado el detalle buscado

        elif rag_flow == 'consultar_temp':

            options['temperature'] = 0.1 # super estricto
            options['num_ctx'] = 100 #contexto minimo ya que la informacion introducida es minima
            options['num_predict'] = 100 # generacion corta, dada la respuesta que se busca.
        
        elif rag_flow == 'consultar_docs_chunk':

            options['temperature'] = 0.4 # Mayor creatividad
            options['num_ctx'] = 5000 # Mayor contexto dado que se recuperan 2 documentos con sus metadatos
            options['num_predict'] = 2000 # Suficiente para informar, pero sintetica

        elif rag_flow == 'consultar_docs_resumen':

            options['temperature'] = 0.4 # Mayor creatividad
            options['num_ctx'] = 5000 # Mayor contexto dado que se recuperan 2 documentos con sus metadatos
            options['num_predict'] = 2000 # Suficiente para informar, pero sintetica

        elif rag_flow == 'hablar':

            options['temperature'] = 0.9 # Maxima creatividad para la conversacion
            options['num_ctx'] = 500 # Contexto para una conversacion, apuntando a las interacciones previas
            options['num_predict'] = 1000 # Salida concisa

        elif rag_flow == 'recuperacion_vacia':

            options['temperature'] = 0.9 # Maxima creatividad para la conversacion
            options['num_ctx'] = 500 # Contexto para una conversacion, apuntando a las interacciones previas
            options['num_predict'] = 1000 # Salida concisa



        # Utiliza la API local.
        stream = chat(
            model="llama3.1",
            
            messages=[{"role": "system", "content": system_prompt}] + context + [{"role": "user", "content": user_prompt}],
            stream=True,
            options=options           
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
    Detecta si hay un identificador de arXiv válido dentro del texto proporcionado por el usuario.

    Utiliza expresiones regulares para buscar identificadores válidos de arXiv.

    Args:
        user_input (str): Texto introducido por el usuario que podría contener un identificador de arXiv.

    Returns:
        str o None: Identificador de arXiv encontrado sin el prefijo arXiv, o None si no se encuentra ninguno.
    """
    patron_arxiv = r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)"
    # Ejemplo válido: 2101.12345v2 o arxiv:2101.12345
        # (?:arxiv:)? =  Grupo opcional para el prefijo 'arxiv:'
        # \d{4} = 4 dígitos del año, como 2101
        # \. =  Punto literal
        # \d{4,5} =  Número del artículo (4 o 5 dígitos)
        # (?:v\d+)? =  Version opcional como v2 o v12


    match = re.search(patron_arxiv, user_input, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def RAG_chat_V2(urls_usados, user_input:str, context, logger, conn, embedding_model, traductor_model, traductor_tokenizer):

    """
    Procesa una entrada de usuario utilizando un flujo RAG para generar una respuesta
    personalizada basada en distintos escenarios: identificador de arXiv, consulta con temporalidad, recuperación de
    documentos por embeddings, o conversación general.

    Args:
        urls_usados (set): Conjunto de URL ya utilizadas en el contexto para evitar duplicación de metadatos.
        user_input (str): Entrada textual del usuario.
        context (list): Lista de mensajes previos en el chat, que sirven como contexto.
        logger: Logger para registrar eventos y errores.
        conn: Conexión a la base de datos PostgreSQL.
        embedding_model: Modelo de embeddings.
        traductor_model: Modelo de traducción.
        traductor_tokenizer: Tokenizador del modelo de traducción.

    Returns:
        tuple:
            - system_prompt (str): Instrucción inicial del sistema para guiar al modelo de lenguaje.
            - context (list): Contexto actualizado que será usado en la generación del LLM.
            - rag_flow (str): Tipo de flujo RAG aplicado, que puede ser:
                - id_arxiv: Identificador de arXiv detectado.
                - consultar_temp: Consulta basada en temporalidad.
                - consultar_docs_chunk: Consulta con recuperación por chunks.
                - consultar_docs_resumen: Consulta con recuperación por resúmenes.
                - recuperacion_vacia: No se recuperaron documentos relevantes.
                - hablar: Interacción de conversación general.
    """

    rag_flow = ''
    
    arxiv_detectado = detectar_identificador_arxiv(user_input)
    if arxiv_detectado:
        logger.debug(f"Arxiv id detectado: {arxiv_detectado}, User input {user_input}")
        try:
           titulo_documento, documento_completo = recuperar_documento_por_arxiv_id(arxiv_detectado,conn)
        except Exception as e:
            logger.error(f'Error al recuperar el documento mediante identificador de arxiv: {e}')

        system_prompt = f"""
            Genera un resumen en profundidad del documento proporcionado, resaltando puntos claves y las conclusiones.
            **CONTESTA UNICAMENTE ESPAÑOL**  
            """
        context.append({
            "role": "assistant",
            "content": f"\n\n Titulo del documento: {titulo_documento}\n\nDocumento proporcionado:\n\n{documento_completo[:10000]}"
        })

        #context_prompt += f"\n\n Titulo del documento: {titulo_documento}\n\nDocumento proporcionado:\n\n{documento_completo[:10000]}" # Esto se limita dada la limitacion computacional.
        
        rag_flow = 'id_arxiv'
        return system_prompt, context, rag_flow #user_prompt, rag_flow

    if deteccion_intencion(user_input) == 'consultar':
        temporalidad = deteccion_temporalidad(user_input)
        # Se detecta temporalidad en el input del usuario.
        if temporalidad is not None:
            
            try:
                logger.debug(f"Temporalidad detectada: {temporalidad}")
                #user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                consulta = temporalidad_a_SQL(conn,temporalidad)
                # Apendizo al contexto
                context.append({
                    "role": "assistant",
                    "content": f"Conteo de publicaciones en la base de datos para {temporalidad[1]}: {consulta}"
                })
                
                system_prompt = f"""
                Contesta de forma concisa el conteo de publicaciones facilitado.
                **CONTESTA UNICAMENTE ESPAÑOL**   
                """
                rag_flow = 'consultar_temp'

                return system_prompt, context, rag_flow #user_prompt, rag_flow
            
            except Exception as e:
                logger.error(f"En la lectura de temporalidad: {e} - temporalidad: {temporalidad}")
                #system_prompt = "Responde brevemente y en español."
                #return system_prompt, "Error al recuperar contexto.", "Disculpa, ¿podrías repetir la pregunta?"
                system_prompt = "Responde brevemente y en español."
    
                context.append({
                    "role": "assistant",
                    "content": "Ha ocurrido un error al procesar la temporalidad. ¿Podrías repetir la pregunta?"
                })
                
                rag_flow = 'consultar_temp'
                return system_prompt, context, rag_flow
        
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
                    #context_prompt += f"\n\nDOCUMENTOS RECUPERADOS:\n\n{doc_formateado}"

                    context.append({
                        "role": "assistant",
                        "content": f"\n\nDOCUMENTOS RECUPERADOS:\n\n{doc_formateado}"
                    })
                    system_prompt = f"""
                    Eres un modelo de lenguaje que debe generar respuestas **basadas únicamente** en el texto bajo la sección 'DOCUMENTOS RECUPERADOS', **respetando exactamente su formato**.

                    Tareas a realizar:
                    1. **Lee cuidadosamente cada documento o apartado bajo 'DOCUMENTOS RECUPERADOS'.**
                    2. **Para cada documento o sección clave, genera un resumen breve y claro en formato Markdown.**
                    3. El formato debe seguir esta estructura:

                    ### Documento   
                    - **Título:** [Si está presente; si no, indicar 'No disponible']
                    - **Arxiv ID**: [Si está presente; si no, indicar 'No disponible']
                    - **Autores:** [Si está presente; si no, indicar 'No disponible']  
                    - **Fecha:** [Si está presente; si no, indicar 'No disponible']  
                    - **URL de la publicacion**: [Si está presente; si no, indicar 'No disponible'] 
                    - **Resumen:** [Escribe un resumen en 2-3 líneas sobre el contenido clave del documento]  

                    4. Si algún campo está ausente en el documento original, **indícalo explícitamente** como 'No disponible'.
                    5. No agregues información externa. Usa **exclusivamente** los 'DOCUMENTOS RECUPERADOS'.

                    **CONTESTA ÚNICAMENTE EN ESPAÑOL**
                    """
                    rag_flow = 'consultar_docs_chunk'
                    return system_prompt, context, rag_flow #user_prompt, rag_flow
                
                elif docs_recuperados_resumen:

                    df_temp = reranking(docs_recuperados_resumen, embedding_disperso)
                    # El numero de documentos generados en el contexto es en funcion de num_docs
                    doc_formateado = formato_contexto_doc_recuperados(urls_usados, conn, df_temp, num_docs=2)
                    print(doc_formateado)
                    # Validacion de que no esta vacio
                    #context_prompt += f"\n\nDOCUMENTOS RECUPERADOS:\n\n{doc_formateado}"
                    context.append({
                        "role": "assistant",
                        "content": f"\n\nDOCUMENTOS RECUPERADOS:\n\n{doc_formateado}"
                    })
                    system_prompt = f"""
                    Eres un modelo de lenguaje que debe generar respuestas **basadas únicamente** en el texto bajo la sección 'DOCUMENTOS RECUPERADOS', **respetando exactamente su formato**.

                    Tareas a realizar:
                    1. **Lee cuidadosamente cada documento o apartado bajo 'DOCUMENTOS RECUPERADOS'.**
                    2. **Para cada documento o sección clave, genera un resumen breve y claro en formato Markdown.**
                    3. El formato debe seguir esta estructura:

                    ### Documento   
                    - **Título:** [Si está presente; si no, indicar 'No disponible']
                    - **Arxiv ID**: [Si está presente; si no, indicar 'No disponible']
                    - **Autores:** [Si está presente; si no, indicar 'No disponible']  
                    - **Fecha:** [Si está presente; si no, indicar 'No disponible']  
                    - **URL de la publicacion**: [Si está presente; si no, indicar 'No disponible'] 
                    - **Resumen:** [Escribe un resumen en 2-3 líneas sobre el contenido clave del documento]   

                    4. Si algún campo está ausente en el documento original, **indícalo explícitamente** como 'No disponible'.
                    5. No agregues información externa. Usa **exclusivamente** los 'DOCUMENTOS RECUPERADOS'.

                    **CONTESTA ÚNICAMENTE EN ESPAÑOL**
                    """
                    rag_flow = 'consultar_docs_resumen'

                    return system_prompt, context, rag_flow#user_prompt, rag_flow
                else:
                    system_prompt = """Te llamas Lervis. Eres un experto en publicaciones academicas de Arxiv en la categoria CS, Ciencias de la computacion.
                    Indica al usuario que no se han encontrado documentos relacionados con la pregunta.
                    **CONTESTA UNICAMENTE ESPAÑOL**"""
                    rag_flow = 'recuperacion_vacia'
                    return system_prompt, context, rag_flow #user_prompt, rag_flow
        
            except Exception as e:
                logger.error(f"Error en la consulta de RAG_chat_V2: {e} - temporalidad: {temporalidad}")
                #system_prompt = "Responde brevemente y en español."
                #return system_prompt, "Error al recuperar contexto.", "Disculpa, ¿podrías repetir la pregunta?"
                system_prompt = "Responde brevemente y en español."
    
                context.append({
                    "role": "assistant",
                    "content": "Ha ocurrido un error al procesar la temporalidad. ¿Podrías repetir la pregunta?"
                })
                
                rag_flow = 'consultar'
                return system_prompt, context, rag_flow
                

    else: # Chatear con el usuario
        
        system_prompt = f"""  
        Te llamas Lervis. Eres un experto en publicaciones academicas de Arxiv en la categoria CS, Ciencias de la computacion.
        Usa el contexto lo maximo posible para responder.
        **CONTESTA UNICAMENTE ESPAÑOL**               
        """
        rag_flow = 'hablar'
        return system_prompt, context, rag_flow #user_prompt, rag_flow

