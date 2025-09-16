#!/usr/bin/env python3
"""
Script para probar el modelo avanzado con una imagen específica.
"""

import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from src.services.model_service import model_predict, ensure_model_ready, last_error

def test_model_prediction():
    """Probar el modelo con una imagen específica."""
    print("🚀 Probando modelo avanzado...")
    
    # Verificar que el modelo esté listo
    if not ensure_model_ready():
        print(f"❌ Error: {last_error()}")
        return
    
    print("✅ Modelo cargado exitosamente")
    
    # Imagen de prueba
    test_image = Path(__file__).parent / "src" / "data" / "DataSetGuayabas" / "test" / "healthy_guava" / "99_unsharp_clahe_augmented_6.png"
    
    if not test_image.exists():
        print(f"❌ Imagen de prueba no encontrada: {test_image}")
        return
    
    print(f"📸 Probando con imagen: {test_image.name}")
    
    # Hacer predicción
    result = model_predict(str(test_image))
    
    if result:
        print(f"\n🎯 RESULTADO:")
        print(f"   Modelo usado: {result['model_type']}")
        print(f"   Predicción: {result['pred_label']}")
        print(f"   Confianza: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"   Probabilidades:")
        print(f"     Anthracnose: {result['prob_anthracnose']:.4f} ({result['prob_anthracnose']*100:.2f}%)")
        print(f"     healthy_guava: {result['prob_healthy']:.4f} ({result['prob_healthy']*100:.2f}%)")
        print("✅ Predicción completada exitosamente")
    else:
        print(f"❌ Error en predicción: {last_error()}")

if __name__ == "__main__":
    test_model_prediction()