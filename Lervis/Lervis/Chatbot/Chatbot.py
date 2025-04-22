from datetime import datetime
from Functions.Chatbot_functions import  RAG_chat, actualizacion_informacion_inicial, eliminacion_acentos
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
    
       
    while True:
        user_input=eliminacion_acentos(input("User: "))
        # Variable para introducir el tiempo y dia de la conversacion
        ahora = datetime.utcnow().strftime("%d/%m/%Y %H:%M")
        respuesta, context =RAG_chat(
            user_input=user_input,
            context=context,
            info_inicial=info_inicial,
            logger=logger,
            conn=conn,
            embedding_model=modelo_BAAI,
            traductor_model =traductor_model,
            traductor_tokenizer=traductor_tokenizer,
            ahora=ahora,
            urls_usados=urls_usados)

        