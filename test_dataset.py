import os
from pathlib import Path

print("Iniciando entrenamiento simplificado...")

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent  # proyecto-flask-guayabas/
DATA_DIR = PROJECT_ROOT / 'src' / 'data' / 'DataSetGuayabas'

print(f"Proyecto: {PROJECT_ROOT}")
print(f"Dataset: {DATA_DIR}")

# Verificar que existe el dataset
train_dir = DATA_DIR / 'train'
val_dir = DATA_DIR / 'val'
test_dir = DATA_DIR / 'test'

print(f"\nVerificando directorios:")
print(f"Train existe: {train_dir.exists()}")
print(f"Val existe: {val_dir.exists()}")
print(f"Test existe: {test_dir.exists()}")

if train_dir.exists():
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    print(f"Clases encontradas: {classes}")
    
    for cls in classes:
        count = len(list((train_dir / cls).glob('*')))
        print(f"  {cls}: {count} imágenes")

print("Verificación completada!")