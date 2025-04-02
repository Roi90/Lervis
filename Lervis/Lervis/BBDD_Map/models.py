"Clases para mapear las tablas de la base de datos PostgreSQL a objetos Python utilizando SQLAlchemy."

from sqlalchemy import Column, BigInteger, String, Date, Text,  ForeignKey
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy import and_
from BBDD_Map.settings import session

from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Publicaciones(Base):
    """Modelo de la tabla publicaciones en PostgreSQL"""

    __tablename__ = "publicaciones"

    id = Column(BigInteger, primary_key=True, autoincrement=True)  # Equivalente a GENERATED ALWAYS AS IDENTITY
    titulo = Column(Text, nullable=False)
    autores = Column(Text, nullable=False)
    fecha_publicacion = Column(Date, nullable=False)
    categoria_principal = Column(BigInteger, nullable=False)
    categorias_lista = Column(Text, nullable=False)
    url_pdf = Column(Text, nullable=False)
    identificador_arxiv = Column(Text, nullable=False, unique=True)  # Constraint UNIQUE en la DB

    def __repr__(self):
        return f"<Publicaciones(id={self.id}, titulo={self.titulo}, autores={self.autores}, \
                fecha_publicacion={self.fecha_publicacion}, categoria_principal={self.categoria_principal}, \
                categorias_lista={self.categorias_lista}, url_pdf={self.url_pdf}, identificador_arxiv={self.identificador_arxiv})>"

    def __str__(self):
        return f"Publicación: {self.titulo} (Autor(es): {self.autores}) - Fecha: {self.fecha_publicacion}, \
    Categoría Principal: {self.categoria_principal}, Categorías Lista: {self.categorias_lista}, \
    URL PDF: {self.url_pdf}, Identificador Arxiv: {self.identificador_arxiv}"

class Embeddings(Base):
    """Modelo de la tabla embeddings en PostgreSQL"""

    __tablename__ = "embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)  # IDENTITY en PostgreSQL
    id_publicaciones = Column(BigInteger, ForeignKey("publicaciones.id"), nullable=False, unique=True)  # Clave foránea y UNIQUE
    contenido = Column(Text, nullable=False)
    contenido_emb_dense = Column(Vector, nullable=False)  # Tipo VECTOR (extensión pgvector)
    contenido_emb_sparse = Column(HSTORE, nullable=False)  # Tipo HSTORE (pares clave-valor)
    resumen = Column(Text, nullable=False)
    resumen_emb_dense = Column(Vector, nullable=False)
    resumen_emb_sparse = Column(HSTORE, nullable=False)

    def __repr__(self):
        return f"<Embeddings(id={self.id}, id_publicaciones={self.id_publicaciones}, contenido={self.contenido[:50]}..., resumen={self.resumen[:50]}...)>"

    def __str__(self):
        return f"Embeddings para Publicación {self.id_publicaciones} - Contenido: {self.contenido[:50]}..., Resumen: {self.resumen[:50]}..."

class Categoria(Base):
    """Modelo de la tabla categoria en PostgreSQL"""

    __tablename__ = "categoria"

    id = Column(BigInteger, primary_key=True, autoincrement=True)  # Equivalente a GENERATED ALWAYS AS IDENTITY
    categoria = Column(Text, nullable=False)
    codigo_categoria = Column(Text, nullable=False, unique=True)  # Constraint UNIQUE en la DB

    def __repr__(self):
        return f"<Categoria(id={self.id}, categoria={self.categoria}, codigo_categoria={self.codigo_categoria})>"

    def __str__(self):
        return f"Categoría: {self.categoria} (Código: {self.codigo_categoria})"
    

#pub = session.query(Publicaciones.categoria_principal).filter_by(id=32).all()
resultados = session.query(Publicaciones).filter(and_(Publicaciones.id > 30, Publicaciones.id < 36)).all()
print(resultados[0].titulo)
