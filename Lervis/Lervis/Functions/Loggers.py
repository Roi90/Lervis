"""
Este archivo contiene funciones para declarar y configurar los distintos loggers que hay.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import logging
import os

def crear_logger(nombre: str, archivo_log: str) -> logging.Logger:

    """
    Crea y configura un logger único para el proyecto.
    
    Parámetros:
    ------------
    nombre : str
        Nombre del logger.
    archivo_log : str
        Nombre del archivo .log donde guardar los logs.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger(nombre)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Formato del log
        formateador = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')

        # Handler consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.DEBUG)
        consola_handler.setFormatter(formateador)
        logger.addHandler(consola_handler)

        # Handler archivo
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # sube un nivel
        log_dir = os.path.join(base_dir, 'Logs')
        os.makedirs(log_dir, exist_ok=True)

        log_file_path = os.path.join(log_dir, archivo_log)

        try:
            # Creamos headers para posteriormente facilitar la lectura del log
            if not os.path.exists(log_file_path):
                # Cuando el archivo no existe, lo creamos y añadimos los headers
                with open(log_file_path, 'w') as f:
                    f.write('timestamp|name|level|message\n')
            archivo_handler = logging.FileHandler(log_file_path)
            archivo_handler.setLevel(logging.DEBUG)
            archivo_handler.setFormatter(formateador)
            logger.addHandler(archivo_handler)
        except Exception as e:
            logger.error(f"Error al configurar el FileHandler: {e}")

    return logger
