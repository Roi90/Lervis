import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Evaluar el rendimiento de los embeddings: 
#   tiempo ejecucion respecto al token de entrada
#   longitud de los embeddings densos y dispersos respecto al token de entrada


# Obtener el directorio donde está este archivo (.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta  al log
log_path = os.path.join(BASE_DIR, '..', 'Logs', 'Embedding.log')

# Cargar el log
df = pd.read_csv(log_path)


