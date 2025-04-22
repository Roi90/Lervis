"""
Este archivo contiene el desarrollo del Chatbot con Llama 3.1 mediante Ollama.

Autor: Roi Pereira Fiuza
"""
import json
import re
import subprocess
import pandas as pd
from datetime import datetime,timedelta
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

def limitador_contexto(input_context: str, max_interacciones: int = 20) -> str:
    """
    Actualiza el contexto parcial del chatbot manteniendo un historial de interacciones reciente.
    Esta función agrega la última interacción del usuario y la respuesta del chatbot al contexto existente.
    Si el número total de líneas en el contexto supera un límite, se eliminan las interacciones más antiguas
    para mantener solo las más recientes.
    Args:
        input_context (str): El contexto actual del chatbot, que contiene el historial de interacciones.
        user_input (str): El mensaje más reciente enviado por el usuario.
        result (str): La respuesta generada por el chatbot para el mensaje del usuario.
    Returns:
        str: El contexto actualizado que incluye la nueva interacción y mantiene un historial limitado.
    """ 
    # Se divide el contexto por interacciones
    contexto_particionado = input_context.split('**************************')
    if len(contexto_particionado) > max_interacciones:
        # Se eliminan las interacciones más antiguas
        contexto_particionado = contexto_particionado[-max_interacciones:]
    # Reconstruccion del contexto
    contexto = "**************************".join(contexto_particionado).strip()
    
    return contexto

def actualizacion_informacion_inicial():
    """
    Actualiza la información inicial del chatbot.
    Esta función se puede utilizar para modificar el contexto inicial que el chatbot utiliza para responder a las preguntas.
    """
    # Variable para almacenar la información inicial actualizada
    info_incial = ''
    conn =conn_bbdd()
    contextos = []
    
    # Consulta SQL para obtener la información sobre las categorías y el conteo de publicaciones
    query = """
        SELECT categoria.categoria,
               categoria.codigo_categoria,
               COUNT(*) as conteo_categorias
        FROM publicaciones
        LEFT JOIN categoria
            ON publicaciones.categoria_principal = categoria.id
        GROUP BY categoria.categoria, categoria.codigo_categoria
    """
    
    # Crear un cursor para ejecutar la consulta
    with conn.cursor() as cur:
        cur.execute(query)
        categorias = cur.fetchall()
    
    contextos.append('CATEGORIAS ARXIV')
    
    # Iterar sobre los resultados obtenidos de la consulta
    for row in categorias:
        contexto = f"Category: {row['categoria']} -Category code : {row['codigo_categoria']} - Total Publications: {row['conteo_categorias']}"
        contextos.append(contexto)
    
    contextos.append(f'Total number of categories: {len(categorias)}')
    contextos.append(f'arXiv website: https://www.arxiv.org/')
    
    # Unir todos los contextos en un solo string con saltos de línea
    info_incial = "\n".join(contextos)
    
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

    logger = Llama31_chatbot_log()

    system_prompt = (
        "Eres un clasificador cuya única función es decidir entre dos intenciones:\n"
        "- consultar  → el usuario quiere consultar la base de datos.\n"
        "- hablar      → el usuario quiere sólo una respuesta conversacional.\n\n"
        "RESPONDE **SIEMPRE** estrictamente con un JSON así:\n"
        "{ \"intencion\": \"consultar\" }\n"
        "o\n"
        "{ \"intencion\": \"hablar\" }\n"
        "¡SIN NADA MAS Y SIN ACENTOS NI NADA!"
        "Unicamente usa la clave 'intencion' en el JSON.\n\n"
    )

    full_prompt = f"{system_prompt}\n\nUsuario: {user_input}\nClasificación:"

    # Linea de comandos para ejecutar el modelo de lenguaje Llama 3.1 mediante Ollama
    cmd = [
        "ollama", "run", "llama3.1:latest",
        full_prompt,
        "--format", "json",
        "--nowordwrap"
    ]
    
    proc = subprocess.run( # Interactuamos con Ollama mediante subprocess
        cmd,
        capture_output=True,
        text=True,# Devuelve texto y no bytes            
        encoding="utf-8", # codificacion utf-8 para la decodificacion del texto      
        errors="replace"  # En el caso de que haya errores de transformacion de los bytes a texto, se reemplazan por un caracter de reemplazo.
    )


    if proc.returncode != 0:
        print("Error al ejecutar Ollama:", proc.stderr)
        logger.error(f"Error al ejecutar Ollama: {proc.stderr}")
        return f"ERROR {proc.returncode}"

    data = json.loads(proc.stdout)['intencion']
    
    return data

def chat(embedding_model):

    try:
        logger = Llama31_chatbot_log()
        conn = conn_bbdd()
        print("Lervis: Bienvenido a Lervis")
        context = "Lervis: Bienvenido a Lervis"
    except Exception as e:
        logger.error(f"Iniciando el logger y la conexion a la BBDD: {e}")

    try:
        info_inicial = actualizacion_informacion_inicial()
    except Exception as e:
        logger.error(f"Error al actualizar la informacion inicial: {e}")
    
    # Traductor español - ingles
    traductor_model, traductor_tokenizer = carga_modelo_traductor()

    while True:
        template = f"""
        Eres Lervis, un asistente experto en ciencias de la computación, especializado en analizar publicaciones académicas de arXiv.

        Tu tarea es responder de forma clara y breve a la pregunta del usuario, usando solo el contexto proporcionado si está disponible.
       
        --- Contexto disponible ---
        {context}

        --- Información adicional ---
        {info_inicial}

        --- Pregunta del usuario ---
        """
    
        # Input del usuario
        user_input = input("Usuario: ") 
        # Variable de detecion de temporalidad
        if deteccion_intencion(user_input) == 'consultar':
            temporalidad = deteccion_temporalidad(user_input)
            # Se detecta temporalidad en el input del usuario.
            if temporalidad is not None:
                try:
                    consulta = temporalidad_a_SQL(conn,temporalidad)
                    # Se apendiza el valor de la consulta al contexto
                    context += f"\n\nPublicaciones encontradas para esas fechas:({consulta})" 
                   
                   # ---------------------------------- LLM - Llama 3.1 ----------------------------------
                    full_prompt = f"{template}\n\nUsuario: {user_input}\nRespuesta:"
                    # Linea de comandos para ejecutar el modelo de lenguaje Llama 3.1 mediante Ollama
                    cmd = [
                        "ollama", "run", "llama3.1:latest",
                        full_prompt,
                        #"--format", "json",
                        #"--nowordwrap"
                    ]
                    proc = subprocess.run( # Interactuamos con Ollama mediante subprocess
                        cmd,
                        capture_output=True, # Comando que captura el output del comando que se ejecuta en la terminal
                        text=True,# Devuelve texto y no bytes            
                        encoding="utf-8", # codificacion utf-8 para la decodificacion del texto      
                        errors="replace"  # En el caso de que haya errores de transformacion de los bytes a texto, se reemplazan por un caracter de reemplazo.
                    )
                    if proc.returncode != 0:
                        print("Error al ejecutar Ollama:", proc.stderr)
                        logger.error(f"Error al ejecutar Ollama: {proc.stderr}")
                        return f"ERROR {proc.returncode}"
                    #respuesta = parser_salida_LLM(proc.stdout)
                    #respuesta = json.loads(proc.stdout)['respuesta']
                    respuesta = proc.stdout
                    # --------------------------------------------------------------------

                    context = actualizacion_contexto(context, user_input, respuesta)

                except Exception as e:
                    logger.error(f"En la lectura de temporalidad: {e}")
                    print('Lervis: Disculpame, me estaba haciendo unos huevos fritos, ¿Podrías repetir la pregunta?')
                    continue
            # No Se detecta temporalidad en el input del usuario.
            else:
                try:
                    # Traduccion al ingles para mejor resultado de similitud
                    user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                    # Embedding de la consulta del usuario
                    embedding_denso, embedding_disperso = embedding(user_input, model=embedding_model)
                    # Recuperacion de documentos mediante similitud del coseno
                    docs_recuperado = similitud_coseno(embedding_denso, conn, 0.3, 5)
                    df_temp = reranking(docs_recuperado, embedding_disperso)
                    # El numero de documentos generados en el contexto es en funcion de num_docs
                    docs_formateados = formato_contexto_doc_recuperados(context, conn, df_temp, num_docs=3)
                    # Se apendiza el valor de la consulta al contexto
                    context += f"{docs_formateados}"
                    #print(context)
                    # ---------------------------------- LLM - Llama 3.1 ----------------------------------
                    full_prompt = f"{template}\n\nUsuario: {user_input}\nRespuesta:"
                    # Linea de comandos para ejecutar el modelo de lenguaje Llama 3.1 mediante Ollama
                    cmd = [
                        "ollama", "run", "llama3.1:latest",
                        full_prompt,
                        #"--format", "json",
                        #"--nowordwrap"
                    ]
                    proc = subprocess.run( # Interactuamos con Ollama mediante subprocess
                        cmd,
                        capture_output=True,
                        text=True,# Devuelve texto y no bytes            
                        encoding="utf-8", # codificacion utf-8 para la decodificacion del texto      
                        errors="replace"  # En el caso de que haya errores de transformacion de los bytes a texto, se reemplazan por un caracter de reemplazo.
                    )
                    if proc.returncode != 0:
                        print("Error al ejecutar Ollama:", proc.stderr)
                        logger.error(f"Error al ejecutar Ollama: {proc.stderr}")
                        return f"ERROR {proc.returncode}"
                    #print("SALIDA CONSULTA: ", proc.stdout)
                    #respuesta = parser_salida_LLM(proc.stdout)
                    respuesta = proc.stdout
                    #respuesta = json.loads(proc.stdout)['respuesta']
                    # -------------------------------------------------------------------- 
                    print("Lervis: ", respuesta)
                    context = actualizacion_contexto(context, user_input, docs_formateados)
                    continue
                except Exception as e:
                    logger.error(f"Error al crear el embedding: {e}")
                    print('Lervis: Disculpame, me estaba sirviendo un cafe, ¿Podrías repetir la pregunta?')
                    continue
        else: # Chatear con el usuario
            
            # ---------------------------------- LLM - Llama 3.1 ----------------------------------
            full_prompt = f"{template}\n\nUsuario: {user_input}\nRespuesta:"
            # Linea de comandos para ejecutar el modelo de lenguaje Llama 3.1 mediante Ollama
            cmd = [
                "ollama", "run", "llama3.1:latest",
                full_prompt,
                #"--format", "json",
                #"--nowordwrap"
            ]
            proc = subprocess.run( # Interactuamos con Ollama mediante subprocess
                cmd,
                capture_output=True,
                text=True,# Devuelve texto y no bytes            
                encoding="utf-8", # codificacion utf-8 para la decodificacion del texto      
                errors="replace"  # En el caso de que haya errores de transformacion de los bytes a texto, se reemplazan por un caracter de reemplazo.
            )
            if proc.returncode != 0:
                print("Error al ejecutar Ollama:", proc.stderr)
                logger.error(f"Error al ejecutar Ollama: {proc.stderr}")
                return f"ERROR {proc.returncode}"
            #print("SALIDA HABLAR: ", proc.stdout)
            #respuesta = parser_salida_LLM(proc.stdout)
            #respuesta = json.loads(proc.stdout)['respuesta']
            respuesta = proc.stdout
            print("Lervis: ", respuesta)
            context = actualizacion_contexto(context, user_input, respuesta)
            # --------------------------------------------------------------------

def RAG_chat(urls_usados, user_input:str, context: str, info_inicial:str, logger, conn, embedding_model, traductor_model,traductor_tokenizer, ahora) -> tuple[str, str]:


   
    if deteccion_intencion(user_input) == 'consultar':
        temporalidad = deteccion_temporalidad(user_input)
        # Se detecta temporalidad en el input del usuario.
        if temporalidad is not None:
            
            try:
                #user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                consulta = temporalidad_a_SQL(conn,temporalidad)
                # Apendizo al contexto
                context += f'\n{ahora} Usuario: {user_input}'
                context += f"\nLervis (sistema): Conteo de publicaciones en la base de datos para {temporalidad[1]}: {consulta}"
                # ---------------------------------- LLM - Llama 3.1 ----------------------------------
                full_prompt = f"""  
                Eres Lervis, un asistente experto en ciencias de la computacion, especializado en el análisis de artículos académicos de arXiv.\n\n              
                {context}
                """
                # Linea de comandos para ejecutar el modelo de lenguaje Llama 3.1 mediante Ollama
                cmd = [
                    "ollama", "run", "llama3.1:latest",
                    full_prompt,
                ]
                proc = subprocess.run( # Interactuamos con Ollama mediante subprocess
                    cmd,
                    capture_output=True, # Comando que captura el output del comando que se ejecuta en la terminal
                    text=True,# Devuelve texto y no bytes            
                    encoding="utf-8", # codificacion utf-8 para la decodificacion del texto      
                    errors="replace"  # En el caso de que haya errores de transformacion de los bytes a texto, se reemplazan por un caracter de reemplazo.
                )
                if proc.returncode != 0:
                    print("Error al ejecutar Ollama:", proc.stderr)
                    logger.error(f"Error al ejecutar Ollama: {proc.stderr}")
                    return f"ERROR {proc.returncode}", context
                respuesta = proc.stdout
                # --------------------------------------------------------------------
                context += f"\nLervis: {respuesta.strip()}\n\n **************************\n\n"
                # Control del tamanio del texto
                context = limitador_contexto(context)
                print(context)
                return respuesta.strip(), context

            except Exception as e:
                logger.error(f"En la lectura de temporalidad: {e}")
                return "Disculpame, me estaba haciendo unos huevos fritos, ¿Podrías repetir la pregunta?", context
        
        # No Se detecta temporalidad en el input del usuario.
        else:
            try:
                # Traduccion al ingles para mejor resultado de similitud
                #user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                # Embedding de la consulta del usuario
                embedding_denso, embedding_disperso = embedding(user_input, model=embedding_model)
                # Recuperacion de documentos mediante similitud del coseno
                docs_recuperado = similitud_coseno(embedding_denso, conn, 0.3, 5)
                df_temp = reranking(docs_recuperado, embedding_disperso)
                # El numero de documentos generados en el contexto es en funcion de num_docs
                doc_formateado = formato_contexto_doc_recuperados(urls_usados, conn, df_temp, num_docs=2)
                # Se apendiza el valor de la consulta al contexto
                context += f'\n{ahora} Usuario: {user_input}'
                # Validacion de que no esta vacio
                if doc_formateado:
                    context += f"\n\nDocumentos recuperados:\n\n{doc_formateado}"
                # ---------------------------------- LLM - Llama 3.1 ----------------------------------
                full_prompt = f"""
                Eres Lervis, un asistente experto en ciencias de la computacion, especializado en el análisis de artículos académicos de arXiv.\n\n 
                {context}
                """
                # Linea de comandos para ejecutar el modelo de lenguaje Llama 3.1 mediante Ollama
                cmd = [
                    "ollama", "run", "llama3.1:latest",
                    full_prompt,
                ]
                proc = subprocess.run( # Interactuamos con Ollama mediante subprocess
                    cmd,
                    capture_output=True,
                    text=True,# Devuelve texto y no bytes            
                    encoding="utf-8", # codificacion utf-8 para la decodificacion del texto      
                    errors="replace"  # En el caso de que haya errores de transformacion de los bytes a texto, se reemplazan por un caracter de reemplazo.
                )
                if proc.returncode != 0:
                    print("Error al ejecutar Ollama:", proc.stderr)
                    logger.error(f"Error al ejecutar Ollama: {proc.stderr}")
                    return f"ERROR {proc.returncode}"

                respuesta = proc.stdout
                # -------------------------------------------------------------------- 
                context = limitador_contexto(context)
                print(context)
                return respuesta.strip(), context
            except Exception as e:
                logger.error(f"Error al crear el embedding: {e}")
                return "Disculpame, me estaba sirviendo un cafe, ¿Podrías repetir la pregunta?", context

    else: # Chatear con el usuario
        
        # ---------------------------------- LLM - Llama 3.1 ----------------------------------
        context += f'\n{ahora} Usuario: {user_input}'
        full_prompt = f"""
        Eres Lervis, un asistente experto en ciencias de la computacion, especializado en el análisis de artículos académicos de arXiv.\n\n   
        {context}
        """
        # Linea de comandos para ejecutar el modelo de lenguaje Llama 3.1 mediante Ollama
        cmd = [
            "ollama", "run", "llama3.1:latest",
            full_prompt,
        ]
        proc = subprocess.run( # Interactuamos con Ollama mediante subprocess
            cmd,
            capture_output=True,
            text=True,# Devuelve texto y no bytes            
            encoding="utf-8", # codificacion utf-8 para la decodificacion del texto      
            errors="replace"  # En el caso de que haya errores de transformacion de los bytes a texto, se reemplazan por un caracter de reemplazo.
        )
        if proc.returncode != 0:
            print("Error al ejecutar Ollama:", proc.stderr)
            logger.error(f"Error al ejecutar Ollama: {proc.stderr}")
            return f"ERROR {proc.returncode}", context
        
        respuesta = proc.stdout        
        # --------------------------------------------------------------------
        context = limitador_contexto(context)
        print(context)
        return respuesta.strip(), context

#input_user = 'Total de publicaciones que han abril 2025'
#tupla = deteccion_temporalidad(input_user)
#print(tupla)
#calculos =  temporalidad_a_SQL(conn_bbdd(), tupla)
#print(calculos)

#entrada = "Que publicaciones hay en abril 2025"
#print(entrada)
#print(deteccion_intencion(entrada))  # probablemente 'retrieve'
#chat(carga_BAAI()) # Cargamos el modelo de embeddings BAAI y lo pasamos al chatbot