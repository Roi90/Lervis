from datetime import datetime
from Functions.Chatbot_functions import  Llama3_1_API, RAG_chat_V2, actualizacion_informacion_inicial, eliminacion_acentos, limitador_contexto
from Functions.Embeddings import carga_BAAI
from Functions.Loggers import Llama31_chatbot_log
from Functions.BBDD_functions import conn_bbdd
from Functions.MarianMT_traductor import carga_modelo_traductor


logger = Llama31_chatbot_log()
conn = conn_bbdd()
modelo_BAAI = carga_BAAI()
info_inicial = actualizacion_informacion_inicial()
traductor_model, traductor_tokenizer = carga_modelo_traductor()

urls_usados   = set()

if __name__ == '__main__':
    # Contexto inicial al arrancar la APP
    #context = "Eres Lervis, un asistente experto en ciencias de la computacion, especializado en el análisis de artículos académicos de arXiv."
       
    while True:
        user_input = eliminacion_acentos(input("User: "))
        ahora = datetime.utcnow().strftime("%d/%m/%Y %H:%M")

        full_prompt, context = RAG_chat_V2(
            user_input=user_input,
            context=context_nuevo,
            logger=logger,
            conn=conn,
            embedding_model=modelo_BAAI,
            ahora=ahora,
            urls_usados=urls_usados,
            traductor_model=traductor_model,
            traductor_tokenizer=traductor_tokenizer
        )

        # Aquí generamos la respuesta token por token
        respuesta_final = ""
        for token in Llama3_1_API(full_prompt):
            if token.startswith("<--FIN-->"):
                respuesta_final = token.replace("<--FIN-->", "")
            else:
                print(token, end="", flush=True)

        # Actualizamos el contexto con la respuesta del modelo
        context += f"\nLervis: {respuesta_final}\n\n **************************\n\n"
        context_nuevo = limitador_contexto(context)
        