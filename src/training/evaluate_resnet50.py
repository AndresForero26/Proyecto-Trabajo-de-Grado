#!/usr/bin/env python3
"""
Evaluación del modelo ResNet50 para clasificación de guayabas.
Incluye métricas detalladas, matriz de confusión y comparación de eficiencia.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "DataSetGuayabas"
MODELS_DIR = PROJECT_ROOT / "models"

# Paths de modelos
RESNET50_MODEL_PATH = MODELS_DIR / "guayaba_resnet50.h5"
SKLEARN_MODEL_PATH = MODELS_DIR / "guayaba_sklearn_model.pkl"
RESNET50_METADATA_PATH = MODELS_DIR / "resnet50_metadata.json"

def load_resnet50_model():
    """Cargar modelo ResNet50."""
    if not RESNET50_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo ResNet50 no encontrado: {RESNET50_MODEL_PATH}")
    
    print(f"📥 Cargando modelo ResNet50...")
    model = tf.keras.models.load_model(RESNET50_MODEL_PATH)
    print(f"✅ Modelo cargado: {model.count_params():,} parámetros")
    return model

def load_sklearn_model():
    """Cargar modelo scikit-learn para comparación."""
    if not SKLEARN_MODEL_PATH.exists():
        print(f"⚠️  Modelo scikit-learn no encontrado: {SKLEARN_MODEL_PATH}")
        return None
    
    import pickle
    with open(SKLEARN_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def create_test_generator():
    """Crear generador de datos de test."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=(224, 224),  # ResNet50 input size
        batch_size=32,
        class_mode='binary',
        shuffle=False  # Importante para mantener orden
    )
    
    print(f"📊 Dataset de test:")
    print(f"   Muestras: {test_generator.samples}")
    print(f"   Clases: {list(test_generator.class_indices.keys())}")
    print(f"   Índices: {test_generator.class_indices}")
    
    return test_generator

def evaluate_resnet50(model, test_generator):
    """Evaluar modelo ResNet50."""
    print(f"\n🎯 EVALUANDO RESNET50")
    print("="*40)
    
    start_time = time.time()
    
    # Predicciones
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    
    # Para clasificación binaria
    if predictions.shape[1] == 1:
        y_pred_proba = predictions.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred_proba = predictions[:, 1]  # Probabilidad de clase positiva
        y_pred = np.argmax(predictions, axis=1)
    
    # Labels reales
    y_true = test_generator.classes
    
    # Tiempo de inferencia
    inference_time = time.time() - start_time
    samples_per_second = len(y_true) / inference_time
    
    # Métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Reporte de clasificación
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    results = {
        'model_type': 'ResNet50',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'inference_time': inference_time,
        'samples_per_second': samples_per_second,
        'total_samples': len(y_true),
        'predictions': y_pred_proba.tolist()
    }
    
    return results

def evaluate_sklearn_model(test_generator):
    """Evaluar modelo scikit-learn para comparación."""
    sklearn_model = load_sklearn_model()
    if sklearn_model is None:
        return None
    
    print(f"\n🔬 EVALUANDO SCIKIT-LEARN (Comparación)")
    print("="*45)
    
    # Preparar datos para sklearn
    from PIL import Image
    
    X_test = []
    y_true = []
    
    # Obtener paths de imágenes
    test_generator.reset()
    start_time = time.time()
    
    for i in range(len(test_generator.filenames)):
        # Cargar imagen
        img_path = DATA_DIR / "test" / test_generator.filenames[i]
        img = Image.open(img_path).convert('RGB').resize((64, 64))  # Sklearn usa 64x64
        img_array = np.array(img).flatten() / 255.0
        
        X_test.append(img_array)
        
        # Obtener label real
        if 'Anthracnose' in test_generator.filenames[i]:
            y_true.append(0)  # Anthracnose = 0
        else:
            y_true.append(1)  # healthy_guava = 1
    
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    # Predicciones
    y_pred_proba = sklearn_model.predict_proba(X_test)[:, 1]
    y_pred = sklearn_model.predict(X_test)
    
    inference_time = time.time() - start_time
    samples_per_second = len(y_true) / inference_time
    
    # Métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'model_type': 'scikit-learn',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'inference_time': inference_time,
        'samples_per_second': samples_per_second,
        'total_samples': len(y_true),
        'predictions': y_pred_proba.tolist()
    }
    
    return results

def print_results(results, model_name):
    """Imprimir resultados de evaluación."""
    print(f"\n📊 RESULTADOS - {model_name.upper()}")
    print("="*50)
    print(f"🎯 Exactitud (Accuracy): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"🎯 Precisión (Precision): {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"🎯 Exhaustividad (Recall): {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"🎯 F1-Score: {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    
    print(f"\n⚡ EFICIENCIA:")
    print(f"   Tiempo total: {results['inference_time']:.3f} segundos")
    print(f"   Muestras procesadas: {results['total_samples']}")
    print(f"   Velocidad: {results['samples_per_second']:.1f} muestras/segundo")
    print(f"   Tiempo por muestra: {1000/results['samples_per_second']:.1f} ms")
    
    # Matriz de confusión
    cm = results['confusion_matrix']
    print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"{'':>15} {'Anthracnose':>12} {'healthy_guava':>12}")
    print("-" * 41)
    print(f"{'Anthracnose':>15} {cm[0][0]:>12} {cm[0][1]:>12}")
    print(f"{'healthy_guava':>15} {cm[1][0]:>12} {cm[1][1]:>12}")

def compare_models(resnet50_results, sklearn_results):
    """Comparar resultados de ambos modelos."""
    if sklearn_results is None:
        print(f"\n⚠️  No se puede comparar: modelo scikit-learn no disponible")
        return
    
    print(f"\n🆚 COMPARACIÓN DE MODELOS")
    print("="*50)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    print(f"{'Métrica':<15} {'ResNet50':<12} {'Scikit-Learn':<15} {'Mejora':<10}")
    print("-" * 55)
    
    for metric in metrics:
        resnet_val = resnet50_results[metric]
        sklearn_val = sklearn_results[metric]
        improvement = ((resnet_val - sklearn_val) / sklearn_val) * 100
        
        print(f"{metric.capitalize():<15} {resnet_val:.4f}{'':<6} {sklearn_val:.4f}{'':<9} {improvement:+.1f}%")
    
    print(f"\n⚡ EFICIENCIA:")
    resnet_speed = resnet50_results['samples_per_second']
    sklearn_speed = sklearn_results['samples_per_second']
    speed_ratio = sklearn_speed / resnet_speed
    
    print(f"{'Modelo':<15} {'Velocidad (fps)':<15} {'Tiempo/muestra':<15}")
    print("-" * 45)
    print(f"{'ResNet50':<15} {resnet_speed:.1f}{'':<10} {1000/resnet_speed:.1f} ms")
    print(f"{'Scikit-Learn':<15} {sklearn_speed:.1f}{'':<10} {1000/sklearn_speed:.1f} ms")
    print(f"\n🏃 Scikit-Learn es {speed_ratio:.1f}x más rápido que ResNet50")

def plot_comparison(resnet50_results, sklearn_results=None):
    """Crear gráficos de comparación."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Métricas de ResNet50
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values_resnet = [resnet50_results[m] for m in metrics]
    
    if sklearn_results:
        values_sklearn = [sklearn_results[m] for m in metrics]
        
        # Gráfico de barras comparativo
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, values_resnet, width, label='ResNet50', color='#2E8B57')
        ax1.bar(x + width/2, values_sklearn, width, label='Scikit-Learn', color='#4169E1')
        ax1.set_xlabel('Métricas')
        ax1.set_ylabel('Valor')
        ax1.set_title('Comparación de Métricas')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.capitalize() for m in metrics])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Velocidad de inferencia
        speeds = [resnet50_results['samples_per_second'], sklearn_results['samples_per_second']]
        models = ['ResNet50', 'Scikit-Learn']
        colors = ['#2E8B57', '#4169E1']
        
        ax2.bar(models, speeds, color=colors)
        ax2.set_ylabel('Muestras por segundo')
        ax2.set_title('Velocidad de Inferencia')
        ax2.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for i, v in enumerate(speeds):
            ax2.text(i, v + max(speeds)*0.01, f'{v:.1f}', ha='center', va='bottom')
    
    else:
        # Solo ResNet50
        ax1.bar(metrics, values_resnet, color='#2E8B57')
        ax1.set_xlabel('Métricas')
        ax1.set_ylabel('Valor')
        ax1.set_title('Métricas ResNet50')
        ax1.set_xticklabels([m.capitalize() for m in metrics])
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Velocidad
        ax2.bar(['ResNet50'], [resnet50_results['samples_per_second']], color='#2E8B57')
        ax2.set_ylabel('Muestras por segundo')
        ax2.set_title('Velocidad de Inferencia')
        ax2.grid(True, alpha=0.3)
    
    # Matriz de confusión ResNet50
    cm_resnet = resnet50_results['confusion_matrix']
    im1 = ax3.imshow(cm_resnet, interpolation='nearest', cmap='Blues')
    ax3.set_title('Matriz de Confusión - ResNet50')
    
    classes = ['Anthracnose', 'healthy_guava']
    tick_marks = np.arange(len(classes))
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(classes)
    ax3.set_yticklabels(classes)
    
    # Agregar números en la matriz
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax3.text(j, i, cm_resnet[i, j], ha="center", va="center", color="black")
    
    ax3.set_ylabel('Clase Real')
    ax3.set_xlabel('Clase Predicha')
    
    # Distribución de confianza
    predictions = resnet50_results['predictions']
    ax4.hist(predictions, bins=30, alpha=0.7, color='#2E8B57', edgecolor='black')
    ax4.set_xlabel('Confianza del Modelo')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Confianza - ResNet50')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_path = MODELS_DIR / 'resnet50_evaluation_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Gráficos guardados en: {plot_path}")

def save_evaluation_results(resnet50_results, sklearn_results=None):
    """Guardar resultados de evaluación."""
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'resnet50': resnet50_results
    }
    
    if sklearn_results:
        results['sklearn'] = sklearn_results
        results['comparison'] = {
            'accuracy_improvement': ((resnet50_results['accuracy'] - sklearn_results['accuracy']) / sklearn_results['accuracy']) * 100,
            'speed_ratio': sklearn_results['samples_per_second'] / resnet50_results['samples_per_second']
        }
    
    # Guardar resultados
    results_path = MODELS_DIR / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        # Convertir arrays de numpy a listas para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"💾 Resultados guardados en: {results_path}")

def main():
    """Función principal de evaluación."""
    print("🎯 EVALUACIÓN DE MODELO RESNET50")
    print("="*50)
    
    try:
        # Cargar modelo ResNet50
        resnet50_model = load_resnet50_model()
        
        # Crear generador de test
        test_generator = create_test_generator()
        
        # Evaluar ResNet50
        resnet50_results = evaluate_resnet50(resnet50_model, test_generator)
        print_results(resnet50_results, "ResNet50")
        
        # Evaluar scikit-learn para comparación
        sklearn_results = evaluate_sklearn_model(test_generator)
        if sklearn_results:
            print_results(sklearn_results, "Scikit-Learn")
            compare_models(resnet50_results, sklearn_results)
        
        # Crear gráficos
        plot_comparison(resnet50_results, sklearn_results)
        
        # Guardar resultados
        save_evaluation_results(resnet50_results, sklearn_results)
        
        print(f"\n🎉 EVALUACIÓN COMPLETADA!")
        print(f"🏆 Modelo recomendado: ResNet50")
        if sklearn_results:
            accuracy_improvement = ((resnet50_results['accuracy'] - sklearn_results['accuracy']) / sklearn_results['accuracy']) * 100
            print(f"📈 Mejora en exactitud: +{accuracy_improvement:.1f}%")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()