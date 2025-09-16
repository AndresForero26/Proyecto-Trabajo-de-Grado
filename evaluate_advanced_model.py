#!/usr/bin/env python3
"""
Script para evaluar el modelo avanzado de scikit-learn y comparar con modelos anteriores.
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns

# Importar la clase AdvancedFeatureExtractor
sys.path.append(str(Path(__file__).parent / "src" / "training"))
from train_advanced_sklearn import AdvancedFeatureExtractor

# Configuración
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "src" / "data" / "DataSetGuayabas" / "test"

CLASSES = ['Anthracnose', 'healthy_guava']

def load_advanced_model():
    """Cargar el modelo avanzado y sus componentes."""
    print("🚀 Cargando modelo avanzado...")
    
    model_path = MODELS_DIR / "guayaba_advanced_sklearn_model.pkl"
    scaler_path = MODELS_DIR / "guayaba_advanced_scaler.pkl"
    extractor_path = MODELS_DIR / "guayaba_feature_extractor.pkl"
    metadata_path = MODELS_DIR / "advanced_sklearn_metadata.json"
    
    # Verificar que existen los archivos
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
    if not extractor_path.exists():
        raise FileNotFoundError(f"Feature extractor no encontrado: {extractor_path}")
    
    # Cargar componentes
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    
    # Cargar metadatos si existen
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    print("✅ Modelo avanzado cargado exitosamente")
    
    return model, scaler, feature_extractor, metadata

def load_test_data(feature_extractor):
    """Cargar datos de test usando el feature extractor."""
    print("📂 Cargando datos de prueba...")
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️  Directorio no encontrado: {class_dir}")
            continue
        
        print(f"   Procesando clase: {class_name}")
        
        # Buscar archivos de imagen
        image_files = (list(class_dir.glob('*.jpg')) + 
                      list(class_dir.glob('*.jpeg')) + 
                      list(class_dir.glob('*.png')) + 
                      list(class_dir.glob('*.JPG')) + 
                      list(class_dir.glob('*.PNG')))
        
        processed = 0
        for img_path in image_files:
            try:
                features = feature_extractor.extract_all_features(img_path)
                X.append(features)
                y.append(class_idx)
                processed += 1
                
                if processed % 50 == 0:
                    print(f"     Procesadas {processed} imágenes...")
                    
            except Exception as e:
                print(f"     Error procesando {img_path.name}: {e}")
                continue
        
        print(f"     ✅ Total procesadas: {processed} imágenes")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"📊 Datos de prueba cargados:")
    print(f"   Forma de X_test: {X.shape}")
    print(f"   Forma de y_test: {y.shape}")
    print(f"   Distribución de clases: {np.bincount(y)}")
    
    return X, y

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluar el modelo y generar métricas."""
    print("🔍 Evaluando modelo avanzado...")
    
    # Normalizar características
    X_test_scaled = scaler.transform(X_test)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Reporte detallado
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def compare_with_previous_models():
    """Comparar con modelos anteriores."""
    print("📊 Comparando con modelos anteriores...")
    
    comparison = {
        'RandomForest (anterior)': {
            'accuracy': 0.82,
            'precision': 0.8385,
            'recall': 0.7776,
            'f1_score': 0.7925,
            'training_time': 'No disponible',
            'inference_speed': 'No disponible'
        }
    }
    
    return comparison

def plot_confusion_matrix(cm, save_path):
    """Crear gráfico de matriz de confusión."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Matriz de Confusión - Modelo Avanzado')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_results(results, metadata):
    """Imprimir resultados de evaluación."""
    print("\n" + "="*60)
    print("📊 REPORTE DETALLADO - MODELO AVANZADO")
    print("="*60)
    
    print(f"\n🎯 MÉTRICAS GENERALES:")
    print(f"   Exactitud (Accuracy):     {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precisión (Macro):        {results['precision_macro']:.4f} ({results['precision_macro']*100:.2f}%)")
    print(f"   Recall (Macro):           {results['recall_macro']:.4f} ({results['recall_macro']*100:.2f}%)")
    print(f"   F1-Score (Macro):         {results['f1_macro']:.4f} ({results['f1_macro']*100:.2f}%)")
    
    print(f"\n📈 MÉTRICAS PONDERADAS:")
    print(f"   Precisión (Weighted):     {results['precision_weighted']:.4f} ({results['precision_weighted']*100:.2f}%)")
    print(f"   Recall (Weighted):        {results['recall_weighted']:.4f} ({results['recall_weighted']*100:.2f}%)")
    print(f"   F1-Score (Weighted):      {results['f1_weighted']:.4f} ({results['f1_weighted']*100:.2f}%)")
    
    # Métricas por clase
    print(f"\n📋 MÉTRICAS POR CLASE:")
    report = results['classification_report']
    for class_name in CLASSES:
        if class_name in report:
            class_metrics = report[class_name]
            print(f"       {class_name}:")
            print(f"      Precisión:  {class_metrics['precision']:.4f} ({class_metrics['precision']*100:.2f}%)")
            print(f"      Recall:     {class_metrics['recall']:.4f} ({class_metrics['recall']*100:.2f}%)")
            print(f"      F1-Score:   {class_metrics['f1-score']:.4f} ({class_metrics['f1-score']*100:.2f}%)")
            print(f"      Soporte:    {class_metrics['support']} muestras")
            print("")
    
    # Matriz de confusión
    cm = results['confusion_matrix']
    print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"                 {CLASSES[0]:>12} {CLASSES[1]:>12}")
    print(f"-" * 41)
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name:>12}          {cm[i][0]:>3}          {cm[i][1]:>3}")
    
    # Información del modelo
    if metadata:
        print(f"\n⚡ INFORMACIÓN DEL MODELO:")
        print(f"   Tipo: {metadata.get('model_type', 'N/A')}")
        print(f"   Características: {metadata.get('n_features', 'N/A')} features")
        if 'training_time' in metadata:
            print(f"   Tiempo entrenamiento: {metadata['training_time']:.2f} segundos")
        if 'inference_time' in metadata:
            print(f"   Tiempo inferencia: {metadata['inference_time']:.3f} segundos")
        if 'inference_speed' in metadata:
            print(f"   Velocidad: {metadata['inference_speed']:.1f} muestras/segundo")

def compare_models(current_results, previous_results):
    """Comparar modelo actual con anteriores."""
    print(f"\n🔄 COMPARACIÓN CON MODELOS ANTERIORES:")
    print(f"{'Modelo':<25} {'Exactitud':<12} {'Precisión':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 73)
    
    # Modelo actual
    print(f"{'Sklearn Avanzado':<25} {current_results['accuracy']:<11.4f} {current_results['precision_macro']:<11.4f} {current_results['recall_macro']:<11.4f} {current_results['f1_macro']:<11.4f}")
    
    # Modelos anteriores
    for model_name, metrics in previous_results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<11.4f} {metrics['precision']:<11.4f} {metrics['recall']:<11.4f} {metrics['f1_score']:<11.4f}")
    
    # Mejoras
    rf_accuracy = previous_results['RandomForest (anterior)']['accuracy']
    improvement = (current_results['accuracy'] - rf_accuracy) / rf_accuracy * 100
    
    print(f"\n📈 MEJORAS:")
    print(f"   Exactitud mejorada en: {improvement:+.2f}% respecto a RandomForest")
    if improvement > 0:
        print("   ✅ El modelo avanzado supera al anterior")
    else:
        print("   ⚠️  El modelo anterior tenía mejor rendimiento")

def main():
    """Función principal."""
    try:
        # Cargar modelo avanzado
        model, scaler, feature_extractor, metadata = load_advanced_model()
        
        # Cargar datos de test
        X_test, y_test = load_test_data(feature_extractor)
        
        # Evaluar modelo
        results = evaluate_model(model, scaler, X_test, y_test)
        
        # Guardar matriz de confusión
        cm_path = PROJECT_ROOT / "advanced_confusion_matrix.png"
        plot_confusion_matrix(results['confusion_matrix'], cm_path)
        
        # Imprimir resultados
        print_results(results, metadata)
        
        # Comparar con modelos anteriores
        previous_results = compare_with_previous_models()
        compare_models(results, previous_results)
        
        print(f"\n📊 Matriz de confusión guardada en: {cm_path}")
        print(f"🎉 Evaluación del modelo avanzado completada!")
        print(f"📊 Exactitud final: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()