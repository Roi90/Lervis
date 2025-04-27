import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Obtener el directorio donde está este archivo (.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta  al log
log_path = os.path.join(BASE_DIR, '..', 'Logs', 'Embedding.log')

# Cargar el log
df = pd.read_csv(log_path)
