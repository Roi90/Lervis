import os
import sys
# # Añadir la raíz del proyecto al sys.path para poder hacer importaciones como Functions.API_metadata
sys.path.insert(0, os.path.abspath('../..'))

project = 'Lervis'
copyright = '2025, Roi Pereira Fiuza'
author = 'Roi Pereira Fiuza'
release = '1.0'


extensions = [
    'sphinx.ext.autodoc',     # Para extraer docstrings de tus funciones/clases
    'sphinx.ext.napoleon',    # Para dar soporte a los formatos de docstring tipo Google o NumPy
]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
