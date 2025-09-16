import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROJECT_ROOT = BASE_DIR  # proyecto-flask-guayabas/
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

class Config:
    # Seguridad
    SECRET_KEY = os.getenv('SECRET_KEY', 'super-secret-key')

    # Modelo y rutas
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'src', 'data', 'DataSetGuayabas')
    
    # Modelos ResNet50 (principal)
    RESNET50_MODEL_PATH = os.path.join(MODELS_DIR, 'guayaba_resnet50.h5')
    RESNET50_CLASS_INDICES_PATH = os.path.join(MODELS_DIR, 'resnet50_class_indices.json')
    RESNET50_METADATA_PATH = os.path.join(MODELS_DIR, 'resnet50_metadata.json')
    
    # Fallback sklearn
    SKLEARN_MODEL_PATH = os.path.join(MODELS_DIR, 'guayaba_sklearn_model.pkl')
    SKLEARN_SCALER_PATH = os.path.join(MODELS_DIR, 'guayaba_scaler.pkl')
    
    # Para compatibilidad con código anterior
    MODEL_PATH = RESNET50_MODEL_PATH
    CLASS_INDICES_PATH = RESNET50_CLASS_INDICES_PATH

    STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')
    UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')

    # Imagen
    INPUT_SIZE = (224, 224)
    CLASSES = ['Anthracnose', 'healthy_guava']  # Solo 2 clases: enferma y sana

    # Base de datos (nombre solicitado: guayaba_db)
    DB_DIR = os.path.join(PROJECT_ROOT, 'db')
    os.makedirs(DB_DIR, exist_ok=True)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URI',
        f"sqlite:///{os.path.join(DB_DIR, 'guayaba_db.sqlite3')}"
    )

def init_dirs():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)