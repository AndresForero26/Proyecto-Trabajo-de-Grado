import os
from pathlib import Path

print("=== Entrenamiento Simplificado ===")

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'src' / 'data' / 'DataSetGuayabas'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)

print(f"Proyecto: {PROJECT_ROOT}")
print(f"Dataset: {DATA_DIR}")

# Verificar dataset
train_dir = DATA_DIR / 'train'
if not train_dir.exists():
    print("ERROR: No se encontró el directorio de entrenamiento")
    exit(1)

classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
print(f"Clases encontradas: {classes}")

for cls in classes:
    count = len(list((train_dir / cls).glob('*')))
    print(f"  {cls}: {count} imágenes")

print("Verificación completada. Ahora puedes ejecutar el entrenamiento completo.")