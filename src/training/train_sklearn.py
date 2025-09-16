import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

print("=== Entrenamiento con Scikit-Learn (Alternativa) ===")

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'src' / 'data' / 'DataSetGuayabas'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)

print(f"Dataset: {DATA_DIR}")

def load_and_preprocess_images(data_dir, target_size=(64, 64)):
    """Cargar y preprocesar imágenes para scikit-learn"""
    images = []
    labels = []
    class_names = ['Anthracnose', 'healthy_guava']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
            
        print(f"Procesando {class_name}...")
        count = 0
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    # Cargar y redimensionar imagen
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(target_size)
                    
                    # Convertir a array y normalizar
                    img_array = np.array(img).flatten()
                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"  Procesadas {count} imágenes de {class_name}")
                        
                except Exception as e:
                    print(f"  Error procesando {img_path}: {e}")
                    continue
        
        print(f"  Total {class_name}: {count} imágenes")
    
    return np.array(images), np.array(labels), class_names

# Cargar datos de entrenamiento
print("\nCargando datos de entrenamiento...")
train_dir = DATA_DIR / 'train'
X, y, class_names = load_and_preprocess_images(train_dir)

print(f"\nDatos cargados:")
print(f"  Forma de X: {X.shape}")
print(f"  Forma de y: {y.shape}")
print(f"  Clases: {class_names}")

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDivisión de datos:")
print(f"  Entrenamiento: {X_train.shape[0]} muestras")
print(f"  Validación: {X_val.shape[0]} muestras")

# Entrenar modelo Random Forest
print("\nEntrenando modelo Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# Evaluar en validación
print("\nEvaluando modelo...")
y_pred = model.predict(X_val)
accuracy = model.score(X_val, y_val)

print(f"\nPrecisión en validación: {accuracy:.4f}")
print(f"\nReporte de clasificación:")
print(classification_report(y_val, y_pred, target_names=class_names))
print(f"\nMatriz de confusión:")
print(confusion_matrix(y_val, y_pred))

# Guardar modelo
model_path = MODELS_DIR / 'guayaba_sklearn_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Guardar metadatos
metadata = {
    'class_names': class_names,
    'target_size': [64, 64],
    'accuracy': float(accuracy),
    'model_type': 'RandomForest'
}

metadata_path = MODELS_DIR / 'sklearn_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en: {model_path}")
print(f"✅ Metadatos guardados en: {metadata_path}")
print(f"\n🎉 Entrenamiento completado!")
print(f"📊 Precisión final: {accuracy:.4f}")

# También guardar índices de clases en el formato esperado por la aplicación
class_indices = {name: idx for idx, name in enumerate(class_names)}
indices_path = MODELS_DIR / 'class_indices.json'
with open(indices_path, 'w') as f:
    json.dump(class_indices, f, indent=2)
    
print(f"✅ Índices de clases guardados en: {indices_path}")