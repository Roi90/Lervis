"""
Este archivo contiene el desarrollo del Chatbot con Llama 3.1 mediante Ollama.

Autor: Roi Pereira Fiuza
"""
import re
import pandas as pd
from datetime import datetime,timedelta
from langchain_ollama import OllamaLLM
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import  ChatPromptTemplate 
from Functions.Embeddings import carga_BAAI, embedding
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

def actualizacion_contexto(input_context: str, user_input: str, result: str, max_interacciones: int = 50) -> str:
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
    contexto = input_context + f"**********\nUsuario: {user_input}\nLervis: {result}"
    
    # Se divide el contexto por interacciones
    contexto_particionado = contexto.split('**********')
    if len(contexto_particionado) > max_interacciones:
        # Se eliminan las interacciones más antiguas
        contexto_particionado = contexto_particionado[-max_interacciones:]
    # Reconstruccion del contexto
    contexto = "**********".join(contexto_particionado)
    
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
        contexto = f"Categoria: {row['categoria']} - Codigo Categoria: {row['codigo_categoria']} - Total publicaciones: {row['conteo_categorias']}"
        contextos.append(contexto)
    
    contextos.append(f'Numero total de categorias: {len(categorias)}')
    contextos.append(f'Pagina web de arxiv: https://www.arxiv.org/')
    contextos.append(f'Fecha y hora actual: {datetime.now()}')
    
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






def chat(info_inicial: str, embedding_model, chain):

    logger = Llama31_chatbot_log()
    conn = conn_bbdd()
    print("Lervis: Bienvenido a Lervis")
    context = ""

    while True:
        user_input = input("Usuario: ") # Input del usuario
        # Traductor español - ingles
        traductor_model, traductor_tokenizer = carga_modelo_traductor()

        # Palabra clave para salir del chat en un futuro se identificara el cierre del browser por parte .
        if user_input == "exit": 
            print('Lervis: Un placer ayudarte, hasta pronto!')
            break
        elif context == "":
            # Si no hay contexto, se utiliza la información inicial como contexto
            # Se utiliza el invoke para asi anñadir el contexto que va creciendo en cada iteracion
            result = chain.invoke({"context": context,
                                    "question":user_input ,
                                    "info_incial": info_inicial}) 
            # Se imprime la respuesta del modelo
            print("Lervis: ", result)
            # Se actualiza el contexto recordando 50 interacciones
            context = actualizacion_contexto(context, user_input, result)
        else:

            try:
                # Traduccion al ingles para mejor resultado de similitud
                user_input = translate_text(traductor_model, traductor_tokenizer, user_input)
                # Embedding de la consulta del usuario
                embedding_denso, embedding_disperso = embedding(user_input, model=embedding_model)
                # Recuperacion de documentos mediante similitud del coseno
                docs_recuperado = similitud_coseno(embedding_denso, conn)
                df_temp = reranking(docs_recuperado, embedding_disperso)
                docs_formateados = formato_contexto_doc_recuperados(conn, df_temp, num_docs=3)

                result = chain.invoke({"context": docs_formateados,
                                    "question":user_input ,
                                    "info_incial": info_inicial}) 
                print("Lervis: ", result)
                context = actualizacion_contexto(context, user_input, docs_formateados)
                continue
            except Exception as e:
                logger.error(f"Error al crear el embedding: {e}")
                print('Lervis: Disculpame, me estaba sirviendo un cafe, ¿Podrías repetir la pregunta?')
                continue

            
             

info_inicial = actualizacion_informacion_inicial()

#-------------------------------------- Definimos el template para el prompt
template = """
Te llamas Lervis y eres un asistente experto en ciencias de la computación y un chat conversacional, especializado en analizar y resumir publicaciones académicas de arXiv.
Mantén un tono amable, conciso y estructurado. Solo saluda si el contexto esta vacio.

Aqui esta el  contexto:
{context}
--- Información adicional ---
{info_incial}
Pregunta:
{question}
Respuesta:

"""
modelo_BAAI = carga_BAAI()

modelo = OllamaLLM(model ="llama3.1") 
# Query al modelo con el template
prompt = ChatPromptTemplate.from_template(template) 
# Estos chains reflejan el flujo de interaccion entre el modelo y su entrada, en este caso el prompt del usuario que se inyecta en el modelo y se obtiene una respuesta
chain = prompt | modelo 

chat(info_inicial, modelo_BAAI, chain=chain)

#input_user = 'Total de publicaciones que han salido ayer'
#tupla = deteccion_temporalidad(input_user)
#print(tupla[2])
#calculos =  temporalidad_a_SQL(conn_bbdd(), tupla)
#print(calculos)
