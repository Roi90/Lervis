"""
Este archivo contiene funciones para descargar los PDF de las publicaciones de arXiv.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import requests
from Functions.Loggers import Descarga_PDF_log
import tempfile

def PDF_descarga(URL: str, nombre_archivo:str) -> None:
    """
    Descarga un archivo PDF desde una URL y lo guarda en el sistema de archivos.

    Parámetros:
    -----------
    URL : str
        La URL desde donde se descargará el archivo PDF.
    nombre_archivo : str
        El nombre con el que se guardará el archivo PDF.

    Retorna:
    --------
    str :
        Path para el archivo descargado
    """
    ruta_target = f"Temp_files/{nombre_archivo}.pdf"

    logger = Descarga_PDF_log()

    try:
        # Hacer una solicitud GET a la URL
        response = requests.get(URL)
        # Verificar si hubo errores en la petición
        response.raise_for_status()
        # Verificar que la solicitud fue exitosa
        logger.debug(f"Respuesta recibida. Código de estado: {response.status_code}")
        # Guardar el contenido de la respuesta (el archivo PDF)
        with open(ruta_target, 'wb') as file:
            file.write(response.content)
        logger.info(f"Archivo PDF descargado con éxito: {nombre_archivo}")
        return ruta_target, nombre_archivo
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al realizar la petición HTTP: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado durante la extracción: {e}", exc_info=True)
        return None
    

def PDF_descarga_temp(URL: str) -> str:
    """
    Descarga un archivo PDF desde una URL y lo guarda en un archivo temporal.

    Parámetros:
    -----------
    URL : str
        La URL desde donde se descargará el archivo PDF.

    Retorna:
    --------
    str :
        Path del archivo PDF temporal.
    """
    logger = Descarga_PDF_log()
    
    try:
        # Hacer la solicitud GET a la URL
        response = requests.get(URL)
        response.raise_for_status()  # Verifica si hubo errores en la petición

        # Crear un archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        
        # Guardar el contenido del PDF en el archivo temporal
        with open(temp_file.name, 'wb') as file:
            file.write(response.content)

        logger.info(f"Archivo PDF descargado con éxito: {temp_file.name}")
        return temp_file.name  # Retorna la ruta del archivo temporal

    except requests.exceptions.RequestException as e:
        logger.error(f"Error al realizar la petición HTTP: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        return None