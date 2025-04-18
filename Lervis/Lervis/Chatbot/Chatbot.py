from Functions.Chatbot_functions import actualizacion_informacion_inicial, chat
from langchain_ollama import OllamaLLM
from langchain_core.prompts import  ChatPromptTemplate 
from Functions.Embeddings import carga_BAAI


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

chat(info_inicial, modelo_BAAI, chain)

#input_user = 'Total de publicaciones que han salido ayer'
#tupla = deteccion_temporalidad(input_user)
#print(tupla[2])
#calculos =  temporalidad_a_SQL(conn_bbdd(), tupla)
#print(calculos)
