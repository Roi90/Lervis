"""
Este archivo contiene funciones para declarar y configurar los distintos loggers que hay.

Autor: Roi Pereira Fiuza
Fecha: 08/03/2024
"""
import logging
import os

def extraccion_metadatos_log() -> logging.Logger:
    """
    Crea y configura un logger para la extracción de metadatos.
    Si el logger ya tiene handlers, no los duplica.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger('Descarga_PDF')
    
    # Verifica si el logger ya tiene handlers para evitar duplicados
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)  # Cambiado a DEBUG para que registre todos los mensajes
        
        # Consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.DEBUG)  # Nivel de consola ajustado a DEBUG
        
        # Formato del log
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        consola_handler.setFormatter(formateador)
        

        logger.addHandler(consola_handler)
        
        # Verifica si el directorio 'Logs/' existe, si no lo crea
        log_dir = 'Logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Crea la carpeta Logs si no existe
        
        # Configura el manejador de archivo
        log_file_path = os.path.join(log_dir, 'extraccion_metadatos.log')
        
        try:
            archivo_handler = logging.FileHandler(log_file_path)
            archivo_handler.setLevel(logging.INFO) 
            archivo_handler.setFormatter(formateador)
            logger.addHandler(archivo_handler)
        except Exception as e:
            logger.error(f"Error al configurar el FileHandler: {e}")
    
    return logger

def Descarga_PDF_log() -> logging.Logger:
    """
    Crea y configura un logger para la extracción de metadatos.
    Si el logger ya tiene handlers, no los duplica.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger('Descarga_PDF')
    
    # Verifica si el logger ya tiene handlers para evitar duplicados
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Cambiado a DEBUG para que registre todos los mensajes
        
        # Consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.INFO)  # Nivel de consola ajustado a DEBUG
        
        # Formato del log
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        consola_handler.setFormatter(formateador)
        

        logger.addHandler(consola_handler)
        
        # Verifica si el directorio 'Logs/' existe, si no lo crea
        log_dir = 'Logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Crea la carpeta Logs si no existe
        
        # Configura el manejador de archivo
        log_file_path = os.path.join(log_dir, 'descarga_PDF.log')
        
        try:
            archivo_handler = logging.FileHandler(log_file_path)
            archivo_handler.setLevel(logging.DEBUG) 
            archivo_handler.setFormatter(formateador)
            logger.addHandler(archivo_handler)

        except Exception as e:
            logger.error(f"Error al configurar el FileHandler: {e}")
    
    return logger

def Docling_log() -> logging.Logger:
    """
    Crea y configura un logger para la extracción de metadatos.
    Si el logger ya tiene handlers, no los duplica.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger('OCR_Docling')
    
    # Verifica si el logger ya tiene handlers para evitar duplicados
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Cambiado a DEBUG para que registre todos los mensajes
        
        # Consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.INFO)  # Nivel de consola ajustado a DEBUG
        
        # Formato del log
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        consola_handler.setFormatter(formateador)
        

        logger.addHandler(consola_handler)
        
        # Verifica si el directorio 'Logs/' existe, si no lo crea
        log_dir = 'Logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Crea la carpeta Logs si no existe
        
        # Configura el manejador de archivo
        log_file_path = os.path.join(log_dir, 'OCR_Docling.log')
        
        try:
            archivo_handler = logging.FileHandler(log_file_path)
            archivo_handler.setLevel(logging.DEBUG) 
            archivo_handler.setFormatter(formateador)
            logger.addHandler(archivo_handler)

        except Exception as e:
            logger.error(f"Error al configurar el FileHandler: {e}")
    
    return logger

def Florence_log() -> logging.Logger:
    """
    Crea y configura un logger para la extracción de metadatos.
    Si el logger ya tiene handlers, no los duplica.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger('Florence2')
    
    # Verifica si el logger ya tiene handlers para evitar duplicados
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Cambiado a DEBUG para que registre todos los mensajes
        
        # Consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.INFO)  # Nivel de consola ajustado a DEBUG
        
        # Formato del log
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        consola_handler.setFormatter(formateador)
        

        logger.addHandler(consola_handler)
        
        # Verifica si el directorio 'Logs/' existe, si no lo crea
        log_dir = 'Logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Crea la carpeta Logs si no existe
        
        # Configura el manejador de archivo
        log_file_path = os.path.join(log_dir, 'Florence2.log')
        
        try:
            archivo_handler = logging.FileHandler(log_file_path)
            archivo_handler.setLevel(logging.DEBUG) 
            archivo_handler.setFormatter(formateador)
            logger.addHandler(archivo_handler)

        except Exception as e:
            logger.error(f"Error al configurar el FileHandler: {e}")
    
    return logger

def BAAI_log() -> logging.Logger:
    """
    Crea y configura un logger para la extracción de metadatos.
    Si el logger ya tiene handlers, no los duplica.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger('BAAI')
    
    # Verifica si el logger ya tiene handlers para evitar duplicados
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Cambiado a DEBUG para que registre todos los mensajes
        
        # Consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.INFO)  # Nivel de consola ajustado a DEBUG
        
        # Formato del log
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        consola_handler.setFormatter(formateador)
        

        logger.addHandler(consola_handler)
        
        # Verifica si el directorio 'Logs/' existe, si no lo crea
        log_dir = 'Logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Crea la carpeta Logs si no existe
        
        # Configura el manejador de archivo
        log_file_path = os.path.join(log_dir, 'BAAI.log')
        
        try:
            archivo_handler = logging.FileHandler(log_file_path)
            archivo_handler.setLevel(logging.DEBUG) 
            archivo_handler.setFormatter(formateador)
            logger.addHandler(archivo_handler)

        except Exception as e:
            logger.error(f"Error al configurar el FileHandler: {e}")
    
    return logger

def BART_log() -> logging.Logger:
    """
    Crea y configura un logger para la extracción de metadatos.
    Si el logger ya tiene handlers, no los duplica.
    
    Retorna:
    --------
    logging.Logger: Logger configurado
    """
    logger = logging.getLogger('BART')
    
    # Verifica si el logger ya tiene handlers para evitar duplicados
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Cambiado a DEBUG para que registre todos los mensajes
        
        # Consola
        consola_handler = logging.StreamHandler()
        consola_handler.setLevel(logging.INFO)  # Nivel de consola ajustado a DEBUG
        
        # Formato del log
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        consola_handler.setFormatter(formateador)
        

        logger.addHandler(consola_handler)
        
        # Verifica si el directorio 'Logs/' existe, si no lo crea
        log_dir = 'Logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Crea la carpeta Logs si no existe
        
        # Configura el manejador de archivo
        log_file_path = os.path.join(log_dir, 'BART.log')
        
        try:
            archivo_handler = logging.FileHandler(log_file_path)
            archivo_handler.setLevel(logging.DEBUG) 
            archivo_handler.setFormatter(formateador)
            logger.addHandler(archivo_handler)

        except Exception as e:
            logger.error(f"Error al configurar el FileHandler: {e}")
    
    return logger