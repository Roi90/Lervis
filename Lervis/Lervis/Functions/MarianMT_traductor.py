import time
from transformers import MarianMTModel, MarianTokenizer
from Functions.Loggers import crear_logger

logger = crear_logger('Traductor', 'Traductor.log')

def carga_modelo_traductor(src_lang="es"):
    """
    Carga el modelo y el tokenizador para la traducción de un idioma a otro.
    
    Args:
        src_lang (str): Idioma de origen.
        tgt_lang (str): Idioma de destino.
        
    Returns:
        model: Modelo de traducción cargado.
        tokenizer: Tokenizador cargado.
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



# -----TESTEO

# Traducir ejemplo
#texto_original = "Razonamiento antes de la acción e imaginar posibles resultados (es decir, modelos del mundo) son esenciales para los agentes incorporados que operan en entornos abiertos y complejos. Sin embargo, trabajos previos incorporan solo una de estas habilidades en un agente de extremo a extremo o integran múltiples modelos especializados en un sistema de agente, lo que limita la eficiencia de aprendizaje y la generalización de la política."
#idioma_detectado = detect(texto_original)
#model, tokenizer = carga_modelo_traductor(idioma_detectado)
#texto_traducido = translate_text(model, tokenizer, texto_original)
#print(f'Texto traducido: {texto_traducido}')
