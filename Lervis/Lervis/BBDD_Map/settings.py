import os   
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

load_dotenv()

#DB_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep
#DB_FILE = os.getenv("DB", "sistema.db")

engine = None

# Configurar conexiones entre SQLAlchemy y SQLite3 DB API
#engine = os.getenv("DATABASE_URL", "postgresql://postgres:Quiksilver90!@localhost:5432/Lervis")
engine = create_engine(f"postgresql://postgres:Quiksilver90!@localhost:5432/Lervis")

if "engine" in globals():
    # Crear sesión con el engine de base de datos
    Session = sessionmaker(bind=engine)
    session = Session()
    # Crear base declarativa
    Base = declarative_base()