from transformers import MarianMTModel, MarianTokenizer


def carga_modelo_traductor(src_lang="es"):
    """
    Carga el modelo y el tokenizador para la traducción de un idioma a otro.
    
    Args:
        src_lang (str): Idioma de origen (código ISO 639-1).
        tgt_lang (str): Idioma de destino (código ISO 639-1).
        
    Returns:
        model: Modelo de traducción cargado.
        tokenizer: Tokenizador cargado.
    """
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{"en"}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    return model, tokenizer

def translate_text(model, tokenizer, text):
    
    # Tokeniza el texto
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))

    # Decodifica el resultado
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    return result



# -----TESTEO

# Traducir ejemplo
#texto_original = "Razonamiento antes de la acción e imaginar posibles resultados (es decir, modelos del mundo) son esenciales para los agentes incorporados que operan en entornos abiertos y complejos. Sin embargo, trabajos previos incorporan solo una de estas habilidades en un agente de extremo a extremo o integran múltiples modelos especializados en un sistema de agente, lo que limita la eficiencia de aprendizaje y la generalización de la política."
#idioma_detectado = detect(texto_original)
#model, tokenizer = carga_modelo_traductor(idioma_detectado)
#texto_traducido = translate_text(model, tokenizer, texto_original)
#print(f'Texto traducido: {texto_traducido}')
