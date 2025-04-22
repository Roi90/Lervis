from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, session
import os
import sys

sys.path.append(  # añade una ruta a las rutas que Python usa para buscar módulos
    os.path.abspath(               # convierte a ruta absoluta...
        os.path.join(             # ...una unión de rutas:
            os.path.dirname(__file__),  # path del archivo actual (app.py)
            '..','Chatbot' # Carpeta donde se declara la funcion.
        )
    )
)
from Chatbot import RAG_chat, logger, conn, modelo_BAAI, info_inicial, traductor_model, traductor_tokenizer, urls_usados

app = Flask(__name__) # variable que contiene la app Flask


app.secret_key = "123123123123123"

# Ruta del landing page
@app.route('/') 
def index():
    return render_template('landing_page.html') 

# Ruta del chat
@app.route('/chat')  
def chat_page():
    return render_template('RAG_chat.html') 

# Ruta para ejecutar el reseteo del contexto
@app.route('/reset', methods=['POST'])
def reset():
    session.pop('context', None)
    session.pop('last_context_time', None)
    return jsonify({'status': 'ok'})

# Ruta que ejecuta las funciones del chat
@app.route('/chat', methods=['POST']) 
def chat():

    user_input = request.json['message']
    # Obtener hora actual y hora guardada (si existe)
    ahora = datetime.utcnow()
    last_time = session.get('last_context_time')

    # Comprobar si han pasado más de 15 minutos
    if last_time:
        last_time = datetime.fromisoformat(last_time)
        if ahora - last_time > timedelta(minutes=5):
            session['context'] = "Bienvenido a Lervis"

    # Obtener el contexto actual (nuevo o ya existente)
    context = session.get('context', "Bienvenido a Lervis")
    
    
    respuesta, nuevo_contexto = RAG_chat(
        user_input=user_input,
        context=context,
        info_inicial=info_inicial,
        logger=logger,
        conn=conn,
        embedding_model=modelo_BAAI,
        traductor_model=traductor_model,
        traductor_tokenizer=traductor_tokenizer,
        ahora=ahora,
        urls_usados=urls_usados

    )

    session['context'] = nuevo_contexto
    return jsonify({'response': respuesta})

# Solo si ejecutas directamente el archivo, lanza el servidor
if __name__ == '__main__':
    app.run(debug=True)