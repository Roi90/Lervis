import requests

def obtener_embeddings(texto):
    url = "http://localhost:11434/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "bge-m3:latest",
        "input": texto  # Enviar el texto directamente
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Verifica si la respuesta es exitosa (código 200)
        embeddings = response.json()  # Asume que la respuesta es JSON
        return embeddings['']
    except requests.exceptions.RequestException as e:
        print(f"Error al generar embeddings: {e}")

# Prueba con algún texto
obtener_embeddings("Este es el texto que quiero convertir en embeddings.")
