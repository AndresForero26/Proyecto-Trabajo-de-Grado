import os
import json
from pathlib import Path

# Variables de entorno
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configuración
tf.keras.backend.clear_session()
tf.config.run_functions_eagerly(True)

print("=== Entrenamiento Modelo Guayabas ===")
print("Librerías importadas correctamente")

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'src' / 'data' / 'DataSetGuayabas'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)

print(f"Proyecto: {PROJECT_ROOT}")
print(f"Dataset: {DATA_DIR}")
print(f"Modelos: {MODELS_DIR}")