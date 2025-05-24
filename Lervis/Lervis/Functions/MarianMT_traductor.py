import time
from transformers import MarianMTModel, MarianTokenizer
from Functions.Loggers import crear_logger

logger = crear_logger('Traductor', 'Traductor.log')

def carga_modelo_traductor(src_lang="es"):
    """
    Carga el modelo y el tokenizador MarianMT para traducir del idioma de origen al inglés.

    Utiliza modelos de Helsinki-NLP para traducción automática. Por defecto, traduce del español al inglés,
    pero se puede especificar otro idioma de origen compatible.

    Parámetros:
        src_lang (str): Código del idioma de origen, por defecto Español (es).

    Retorna:
        tuple:
            - model (MarianMTModel): Modelo de traducción cargado.
            - tokenizer (MarianTokenizer): Tokenizador correspondiente al modelo.

    Raises:
        Exception: Si ocurre un error durante la carga del modelo o el tokenizador.
    """
    try:
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{"en"}'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error al cargar el modelo de traducción: {e}")
        raise

def translate_text(model, tokenizer, text):
    """
    Traduce un texto desde el idioma de origen al inglés utilizando un modelo MarianMT.

    Esta función tokeniza el texto, lo traduce utilizando el modelo proporcionado y decodifica el resultado.
    También registra métricas como duración, longitud de texto y ratio de compresión.

    Parámetros:
        model (MarianMTModel): Modelo de traducción previamente cargado.
        tokenizer (MarianTokenizer): Tokenizador correspondiente al modelo.
        text (str): Texto en idioma de origen que se desea traducir.

    Retorna:
        str: Texto traducido al inglés. Si ocurre un error, se devuelve el texto original.

    Raises:
        Exception: Si ocurre un error durante la traducción, se captura y se registra, devolviendo el texto original.

    """
    try:
        start_time = time.time()
                
        intput_text_length = len(text)
        # Tokeniza el texto
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))

        # Decodifica el resultado
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        output_text_length =  len(result)

        end_time = time.time()
        duracion = end_time - start_time

        # Dado que es de español a ingles se espera que se compacte el texto, por lo que el ratio ha de ser menor a 1
        ratio = intput_text_length / output_text_length 
        categoria = 'Esperada'
        if ratio > 1:
            logger.warning(f" Ratio > 1 Texto original - {text}, Texto traducido - {result}")
            categoria = 'No Esperada'
            
        
        logger.debug(f"Duracion segundos - {duracion:.2f}, Caracteres input - {intput_text_length}, Caracteres output - {output_text_length}, Ratio - {ratio:.2f}, Categoria - {categoria}")
        return result
    
    except Exception as e:
        logger.error(f"Error al traducir el texto: {e}")
        return text
