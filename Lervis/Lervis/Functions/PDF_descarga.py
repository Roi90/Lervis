"""
Este archivo contiene funciones para descargar los PDF de las publicaciones de arXiv.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import requests
from Functions.Loggers import crear_logger
import tempfile

logger = crear_logger('Descarga_PDF', 'Descarga_PDF.log')

def PDF_descarga_temp(URL: str):
    """
    Descarga un archivo PDF desde una URL y lo guarda en un archivo temporal.

    Esta función realiza una solicitud HTTP a la URL proporcionada, descarga el contenido del PDF
    y lo guarda en un archivo temporal que no se elimina automáticamente al cerrarse.

    Parámetros:
        URL (str): URL del archivo PDF que se desea descargar.

    Retorna:
        str | None: Ruta del archivo temporal creado, o None si ocurre un error.

    Raises:
        RequestException: Si la solicitud HTTP falla.
        Exception: Si ocurre un error al guardar el archivo temporal.
    """
    
    try:
        # Hacer la solicitud GET a la URL
        response = requests.get(URL)
        response.raise_for_status()  # Verifica si hubo errores en la petición

        # Crear un archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        
        # Guardar el contenido del PDF en el archivo temporal
        with open(temp_file.name, 'wb') as file:
            file.write(response.content)

        logger.debug(f"Archivo PDF descargado con éxito: {temp_file.name}")
        return temp_file.name  # Retorna la ruta del archivo temporal

    except requests.exceptions.RequestException as e:
        logger.error(f"Error al realizar la petición HTTP: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        return None