from datetime import datetime, timedelta
from flask import Flask, Response, request, render_template, stream_with_context, session
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy



from Functions.Chatbot_functions import  Llama3_1_API, RAG_chat_V2, actualizacion_informacion_inicial,  limitador_contexto
from Functions.Embeddings import carga_BAAI
from Functions.Loggers import crear_logger
from Functions.BBDD_functions import conn_bbdd
from Functions.MarianMT_traductor import carga_modelo_traductor

logger = crear_logger('Flask', 'Flask.log')
conn = conn_bbdd()
info_inicial = actualizacion_informacion_inicial()
modelo_BAAI = carga_BAAI()
info_inicial = actualizacion_informacion_inicial()
traductor_model, traductor_tokenizer = carga_modelo_traductor()

logger = crear_logger('app', 'app.log')

urls_usados   = set()
# variable que contiene la app Flask
app = Flask(__name__) 

# ---------------- Configuracion para la BBDD para guardar el contexto ----------------
app.secret_key = "1990TESTINGPROJECT" # clave secreta para la app Flask
# Configuracion de la sesion en la BBDD
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'postgresql://postgres:Quiksilver90!@localhost:5432/Lervis'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- Configuracion de la sesion en Flask ----------------
app.config['SESSION_TYPE'] = 'sqlalchemy'
app.config['SESSION_SQLALCHEMY'] = db
# Nombre de la tabla donde se guardarán las sesiones; se crea automáticamente
app.config['SESSION_SQLALCHEMY_TABLE'] = 'flask_sessions'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=2)
Session(app)

with app.app_context():
    db.create_all()

# Ruta del landing page
@app.route('/') 
def index():
    return render_template('landing_page.html') 

# Ruta del chat
@app.route('/chat_page')  
def chat_page():
    return render_template('RAG_chat.html') 

# Funcion de reseteo de contexto
@app.route('/reset', methods=['POST'])
def reset():
    session.pop('context', None)
    logger.debug(f"Contexto reseteado - session_id {session.sid}")
    return ('', 204)

# Ruta que ejecuta las funciones del chat
@app.route('/chat', methods=['POST']) 
def chat():

    ahora = datetime.utcnow().strftime("%d/%m/%Y %H:%M")
    # Toda la informacion que nos devuelve la accion POST (escribir en el chat)
    data = request.json
    user_input = data['message']

    context = session.get('context', f'{info_inicial}\n \n{ahora} - Usuario: {user_input}')

    #context += f"\n\n{ahora} - Usuario: {user_input}"
    context = limitador_contexto(context)

    print("\n🟡 CONTEXTO ANTES DEL RAG_chat_V2:")
    print(context)

    try:
        # Manejo del contexto dentro de la función dependiendo de las funciones invocadas
        full_prompt = RAG_chat_V2(
            urls_usados=urls_usados,
            user_input=user_input,
            context=context,
            logger=logger,
            conn=conn,
            embedding_model=modelo_BAAI,
            traductor_model=traductor_model,
            traductor_tokenizer=traductor_tokenizer
        )
        
    except Exception as e:
        logger.error(f"Error en RAG_chat_V2: {e}")
        # Si hay un error en la funcion RAG_chat_V2 se devuelve un mensaje de error al usuario
        return Response("Perdon, se me cruzaron los cables, ¿Podrías repetirlo?")
        
    def generate():
        for token in Llama3_1_API(full_prompt):
            yield token 

    return Response(
        # Respuesta por tokens, es decir, es stream
        stream_with_context(generate()),
        content_type='text/plain',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.route('/save_context', methods=['POST'])
def save_context():
    data = request.json
    input_usuario = data['mensaje_usuario']
    respuesta_lervis = data['respuesta_lervis']
    #print(f"🔵 Respuesta recibida en /save_context: {respuesta_lervis}")

    ahora = datetime.utcnow().strftime("%d/%m/%Y %H:%M")
    # Si no hay contexto se crea uno nuevo junto con la informacion inicial
    contexto_nuevo = session.get('context', f'{info_inicial}\n\n{ahora} - Lervis: Bienvenido a Lervis')

    # Se apendiza el input del usuario y la respuesta de Lervis al contexto
    contexto_nuevo += f"\n\n{ahora} - Usuario: {input_usuario}"
    contexto_nuevo += f"\n\n{ahora} - Lervis: {respuesta_lervis}"
    session['context'] = limitador_contexto(contexto_nuevo)
    session.modified = True
    return '', 204


if __name__ == '__main__':
    print("🟢 Iniciando la app Flask...🟢")
    logger.debug(f"Iniciando la app Flask ")
    app.run(debug=False, threaded=True)