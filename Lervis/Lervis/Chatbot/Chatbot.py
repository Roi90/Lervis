"""
Este archivo contiene el desarrollo del Chatbot con Llama 3.1 mediante Ollama.

Autor: Roi Pereira Fiuza
"""
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import  ChatPromptTemplate 
from Functions.Embeddings import carga_BAAI, embedding
from Functions.BBDD_functions import conn_bbdd
from Functions.Loggers import Llama31_chatbot_log
from Functions.BBDD_functions import similitud_coseno
from Functions.MarianMT_traductor import carga_modelo_traductor, translate_text
from langdetect import detect

def actualizacion_contexto(input_context: str, user_input: str, result: str):
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
    contexto = input_context
    # Añadir la última interacción al contexto
    contexto += f"\nUsuario: {user_input}\nLervis: {result}"
    # Se divide el contexto por interacciones
    contexto_particionado = contexto.split('\n')
    if len(contexto_particionado) > 50:
        # Se eliminan las interacciones más antiguas
        contexto = contexto_particionado[-5:]
    # Reconstruccion del contexto
    contexto = "\n".join(contexto)
    
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
    
    # Unir todos los contextos en un solo string con saltos de línea
    info_incial = "\n".join(contextos)
    
    return info_incial

def chat(info_inicial: str, embedding_model):

    logger = Llama31_chatbot_log()
    conn = conn_bbdd()
    idioma_detectado = ''
    print("Lervis: Bienvenido a Lervis")
    context = ""

    while True:
        user_input = input("Usuario: ") # Input del usuario
        # Deteccion del idioma
        nuevo_idioma = detect(user_input)
        
        # Si el idioma ha cambiado, cargar el modelo de traducción correspondiente
        if nuevo_idioma != idioma_detectado:
            idioma_detectado = nuevo_idioma
            traductor_model, traductor_tokenizer = carga_modelo_traductor(idioma_detectado)

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
                embedding_denso, _ = embedding(user_input, model=embedding_model)
                # Recuperacion de documentos mediante similitud del coseno
                resumen_recuperado = similitud_coseno(embedding_denso, conn)
                result = chain.invoke({"context": context,
                                    "question":resumen_recuperado[2] ,
                                    "info_incial": info_inicial}) 
                print("Lervis: ", result)
                context = actualizacion_contexto(context, user_input, resumen_recuperado)
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

chat(info_inicial, modelo_BAAI)
