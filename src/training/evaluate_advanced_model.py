#!/usr/bin/env python3
"""
Evaluación completa del modelo avanzado sklearn.
Genera métricas de rendimiento, matriz de confusión y comparaciones.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Agregar el directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.services.feature_extractor import AdvancedFeatureExtractor

def load_model_and_components():
    """Cargar modelo, scaler y feature extractor"""
    models_dir = PROJECT_ROOT / "models"
    
    try:
        # Cargar modelo
        with open(models_dir / "guayaba_advanced_sklearn_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Cargar scaler
        with open(models_dir / "guayaba_advanced_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Cargar feature extractor
        with open(models_dir / "guayaba_feature_extractor.pkl", 'rb') as f:
            feature_extractor = pickle.load(f)
        
        print("✅ Modelo y componentes cargados exitosamente")
        return model, scaler, feature_extractor
    
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado - {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None, None

def load_test_data():
    """Cargar datos de test"""
    data_dir = PROJECT_ROOT / "src" / "data" / "DataSetGuayabas" / "test"
    
    if not data_dir.exists():
        print(f"❌ Directorio de test no encontrado: {data_dir}")
        return None, None, None
    
    feature_extractor = AdvancedFeatureExtractor()
    
    X_test = []
    y_test = []
    image_paths = []
    
    classes = ['Anthracnose', 'healthy_guava']
    
    print("📊 Cargando datos de test...")
    
    for class_idx, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Directorio no encontrado: {class_dir}")
            continue
        
        # Buscar imágenes
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        
        print(f"   {class_name}: {len(image_files)} imágenes")
        
        for img_path in image_files:
            try:
                # Extraer características
                features = feature_extractor.extract_all_features(str(img_path))
                X_test.append(features)
                y_test.append(class_idx)
                image_paths.append(str(img_path))
            except Exception as e:
                print(f"⚠️  Error procesando {img_path}: {e}")
    
    if len(X_test) == 0:
        print("❌ No se pudieron cargar imágenes de test")
        return None, None, None
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"✅ Dataset de test cargado: {len(X_test)} muestras")
    return X_test, y_test, image_paths

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluar modelo y calcular métricas"""
    print("\n🔍 EVALUANDO MODELO...")
    
    # Escalar características
    X_test_scaled = scaler.transform(X_test)
    
    # Hacer predicciones
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    inference_time = time.time() - start_time
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Velocidad de inferencia
    samples_per_second = len(y_test) / inference_time
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'inference_time': inference_time,
        'samples_per_second': samples_per_second
    }

def plot_confusion_matrix(y_test, y_pred, classes=['Anthracnose', 'healthy_guava']):
    """Crear matriz de confusión"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión - Modelo Avanzado', fontsize=14, weight='bold')
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Predicción', fontsize=12)
    
    # Agregar estadísticas
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    plt.figtext(0.15, 0.02, f'Exactitud Total: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                fontsize=10, weight='bold')
    
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_test, y_pred_proba):
    """Crear curva ROC"""
    plt.figure(figsize=(8, 6))
    
    # ROC para cada clase
    classes = ['Anthracnose', 'healthy_guava']
    colors = ['#FF6B6B', '#4ECDC4']
    
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        # Convertir a problema binario
        y_binary = (y_test == i).astype(int)
        y_scores = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Línea de referencia
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC por Clase', fontsize=14, weight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def create_metrics_summary(metrics):
    """Crear resumen de métricas"""
    print("\n" + "="*60)
    print("📊 MÉTRICAS DE RENDIMIENTO")
    print("="*60)
    print(f"🎯 Exactitud (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"🎯 Precisión (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"🎯 Recall (Sensibilidad): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"🎯 F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print(f"\n⚡ RENDIMIENTO:")
    print(f"   Tiempo de inferencia: {metrics['inference_time']:.3f} segundos")
    print(f"   Velocidad: {metrics['samples_per_second']:.1f} muestras/segundo")
    print(f"   Tiempo por muestra: {metrics['inference_time']*1000/len(metrics['y_pred']):.1f} ms")

def compare_with_baseline():
    """Comparar con modelo baseline si existe"""
    try:
        models_dir = PROJECT_ROOT / "models"
        
        # Buscar metadatos del modelo básico
        if (models_dir / "sklearn_metadata.json").exists():
            import json
            with open(models_dir / "sklearn_metadata.json", 'r') as f:
                baseline_data = json.load(f)
            
            print(f"\n📈 COMPARACIÓN CON MODELO BASELINE:")
            print("="*50)
            print(f"Modelo Básico (RandomForest):")
            print(f"   Exactitud: {baseline_data.get('test_accuracy', 'N/A')}")
            print(f"   F1-Score: {baseline_data.get('test_f1_score', 'N/A')}")
            
        # Buscar metadatos del modelo avanzado
        if (models_dir / "advanced_sklearn_metadata.json").exists():
            import json
            with open(models_dir / "advanced_sklearn_metadata.json", 'r') as f:
                advanced_data = json.load(f)
            
            print(f"\nModelo Avanzado (Ensemble):")
            print(f"   Exactitud: {advanced_data.get('test_accuracy', 'N/A')}")
            print(f"   F1-Score: {advanced_data.get('test_f1_score', 'N/A')}")
            
    except Exception as e:
        print(f"⚠️  No se pudo cargar comparación baseline: {e}")

def main():
    """Función principal"""
    print("🚀 EVALUACIÓN DEL MODELO AVANZADO SKLEARN")
    print("="*60)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Cargar modelo
    model, scaler, feature_extractor = load_model_and_components()
    if model is None:
        print("❌ No se pudo cargar el modelo")
        return
    
    # Cargar datos de test
    X_test, y_test, image_paths = load_test_data()
    if X_test is None:
        print("❌ No se pudieron cargar los datos de test")
        return
    
    # Evaluar modelo
    metrics = evaluate_model(model, scaler, X_test, y_test)
    
    # Mostrar métricas
    create_metrics_summary(metrics)
    
    # Crear visualizaciones
    print(f"\n📊 Generando visualizaciones...")
    
    # Matriz de confusión
    fig1 = plot_confusion_matrix(y_test, metrics['y_pred'])
    
    # Curvas ROC
    fig2 = plot_roc_curve(y_test, metrics['y_pred_proba'])
    
    # Guardar gráficos
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    fig1.savefig(results_dir / "confusion_matrix_advanced.png", dpi=300, bbox_inches='tight')
    fig2.savefig(results_dir / "roc_curves_advanced.png", dpi=300, bbox_inches='tight')
    
    print(f"✅ Gráficos guardados en: {results_dir}")
    
    # Comparación con baseline
    compare_with_baseline()
    
    # Reporte detallado
    print(f"\n📋 REPORTE DETALLADO:")
    print("="*40)
    class_names = ['Anthracnose', 'healthy_guava']
    report = classification_report(y_test, metrics['y_pred'], 
                                 target_names=class_names, digits=4)
    print(report)
    
    # Mostrar gráficos
    plt.show()
    
    print(f"\n🎉 ¡EVALUACIÓN COMPLETADA!")

if __name__ == "__main__":
    main()