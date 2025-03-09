"""
Este archivo contiene funciones para declarar y configurar los distintos loggers que hay.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import logging

def extraccion_metadatos_log():
    # Crea un logger con el nombre especificado
    logger = logging.getLogger('extraccion_metadatos.log')
    
    # Establece el nivel de logging (puede ser DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.INFO)
    
    # Crea un manejador de consola para mostrar los logs en la terminal
    consola_handler = logging.StreamHandler()
    consola_handler.setLevel(logging.DEBUG)
    
    # Crea un formateador para mostrar la información con un formato específico
    formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    consola_handler.setFormatter(formateador)
    
    # Agrega el manejador de consola al logger
    logger.addHandler(consola_handler)
    
    return logger