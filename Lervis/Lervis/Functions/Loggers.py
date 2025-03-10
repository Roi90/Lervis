"""
Este archivo contiene funciones para declarar y configurar los distintos loggers que hay.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import logging

def extraccion_metadatos_log() -> logging.Logger:
    """
    Crea y configura un logger para la extracción de metadatos.
    Si el logger ya tiene handlers, no los duplica.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger('extraccion_metadatos')
    
    # Verifica si el logger ya tiene handlers para evitar duplicados
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Crea un manejador de consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.DEBUG)
        
        # Define el formato del log
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        consola_handler.setFormatter(formateador)
        
        # Agrega el manejador al logger
        logger.addHandler(consola_handler)
    
    return logger