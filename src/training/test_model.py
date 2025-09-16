#!/usr/bin/env python3
"""
Script para probar el modelo avanzado con una imagen.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

# Agregar el directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importar el servicio del modelo
from src.services.model_service import model_predict, ensure_model_ready

def test_prediction():
    """Probar predicción con una imagen de ejemplo."""
    
    # Cargar modelo
    print("🔧 Cargando modelo avanzado...")
    success = ensure_model_ready()
    
    if not success:
        print("❌ Error al cargar modelo")
        return
    
    print("✅ Modelo cargado exitosamente")
    
    # Buscar una imagen de test
    test_dir = Path(__file__).parent.parent / "data" / "DataSetGuayabas" / "test"
    
    # Buscar primera imagen disponible
    test_image = None
    for class_dir in ["Anthracnose", "healthy_guava"]:
        class_path = test_dir / class_dir
        if class_path.exists():
            # Buscar archivos PNG y JPG
            for pattern in ["*.png", "*.jpg", "*.jpeg"]:
                for img_file in class_path.glob(pattern):
                    test_image = img_file
                    expected_class = class_dir
                    break
                if test_image:
                    break
        if test_image:
            break
    
    if not test_image:
        print("❌ No se encontró imagen de test")
        return
    
    print(f"🖼️  Probando con imagen: {test_image}")
    print(f"🎯 Clase esperada: {expected_class}")
    
    # Hacer predicción
    try:
        result = model_predict(str(test_image))
        
        if result is None:
            print("❌ Error: model_predict devolvió None")
            return
        
        print("\n📊 RESULTADO:")
        print(f"   Predicción: {result['label']}")
        print(f"   Confianza: {result['confidence']:.4f}")
        print(f"   Modelo usado: {result['model_type']}")
        print(f"   Probabilidades:")
        for class_name, prob in result['probs'].items():
            print(f"     {class_name}: {prob:.4f}")
        
        # Verificar si la predicción es correcta
        if result['label'].lower() == expected_class.lower():
            print("✅ ¡Predicción correcta!")
        else:
            print("❌ Predicción incorrecta")
            
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()