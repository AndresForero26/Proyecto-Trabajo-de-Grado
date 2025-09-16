#!/usr/bin/env python3
"""
Entrenamiento con modelo avanzado usando scikit-learn.
Simula características de deep learning con extracción de features complejas,
data augmentation y ensemble methods.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from skimage import feature, filters, segmentation, measure, morphology
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

# Importar el extractor de características
from src.services.feature_extractor import AdvancedFeatureExtractor

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Subir a proyecto-flask-guayabas
DATA_DIR = PROJECT_ROOT / "src" / "data" / "DataSetGuayabas"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Parámetros
IMG_SIZE = (224, 224)  # Mismo que ResNet50 para comparación
CLASSES = ['Anthracnose', 'healthy_guava']

def load_dataset_with_augmentation(data_dir, feature_extractor, use_augmentation=True):
    """Cargar dataset con extracción de características y data augmentation."""
    print("📊 Cargando dataset con extracción de características avanzadas...")
    
    X = []
    y = []
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"   ⚠️  Directorio {split} no existe: {split_dir}")
            continue
            
        print(f"   Procesando {split}...")
        
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"   ⚠️  Directorio de clase {class_name} no existe: {class_dir}")
                continue
            
            # Buscar archivos de imagen con diferentes extensiones
            image_files = (list(class_dir.glob('*.jpg')) + 
                          list(class_dir.glob('*.jpeg')) + 
                          list(class_dir.glob('*.png')) + 
                          list(class_dir.glob('*.JPG')) + 
                          list(class_dir.glob('*.PNG')))
            
            print(f"     {class_name}: {len(image_files)} imágenes encontradas en {class_dir}")
            
            for img_path in image_files:
                try:
                    # Cargar y procesar imagen original
                    features = feature_extractor.extract_all_features(img_path)
                    X.append(features)
                    y.append(class_idx)
                    
                    # Data augmentation limitado solo para entrenamiento
                    if split == 'train' and use_augmentation and len(image_files) < 100:
                        try:
                            image = Image.open(img_path).convert('RGB')
                            
                            # Solo 1 augmentación simple: flip horizontal
                            flipped = ImageOps.mirror(image)
                            
                            # Guardar temporalmente y extraer características
                            temp_path = img_path.parent / f"temp_flip_{img_path.stem}.png"
                            flipped.save(temp_path)
                            
                            aug_features = feature_extractor.extract_all_features(temp_path)
                            X.append(aug_features)
                            y.append(class_idx)
                            
                            # Limpiar archivo temporal
                            if temp_path.exists():
                                temp_path.unlink()
                                
                        except Exception as aug_e:
                            print(f"     Error en augmentation para {img_path.name}: {aug_e}")
                
                except Exception as e:
                    print(f"     Error procesando {img_path.name}: {e}")
                    continue
    
    if len(X) == 0:
        raise ValueError("No se pudieron cargar imágenes del dataset")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Dataset cargado: {X.shape[0]} muestras con {X.shape[1]} características")
    print(f"   Distribución de clases: {np.bincount(y)}")
    
    return X, y
    
def load_test_dataset(data_dir, feature_extractor):
    """Cargar solo el dataset de test."""
    print("📊 Cargando dataset de test...")
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"   ⚠️  Directorio de clase {class_name} no existe: {class_dir}")
            continue
        
        # Buscar archivos de imagen
        image_files = (list(class_dir.glob('*.jpg')) + 
                      list(class_dir.glob('*.jpeg')) + 
                      list(class_dir.glob('*.png')) + 
                      list(class_dir.glob('*.JPG')) + 
                      list(class_dir.glob('*.PNG')))
        
        print(f"     {class_name}: {len(image_files)} imágenes encontradas")
        
        for img_path in image_files:
            try:
                features = feature_extractor.extract_all_features(img_path)
                X.append(features)
                y.append(class_idx)
            except Exception as e:
                print(f"     Error procesando {img_path.name}: {e}")
                continue
    
    if len(X) == 0:
        raise ValueError("No se pudieron cargar imágenes del dataset de test")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Dataset de test cargado: {X.shape[0]} muestras")
    print(f"   Distribución de clases: {np.bincount(y)}")
    
    return X, y

def create_advanced_ensemble_model():
    """Crear modelo ensemble avanzado pero eficiente."""
    print("🏗️  Creando modelo ensemble optimizado...")
    
    # Modelos base simplificados para mayor velocidad
    rf = RandomForestClassifier(
        n_estimators=100,  # Reducido para rapidez
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    svm = SVC(
        kernel='rbf',
        C=1.0,  # Reducido para rapidez
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # Reducido para rapidez
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,  # Reducido para rapidez
        random_state=42
    )
    
    # Ensemble simple con voting
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('svm', svm),
            ('mlp', mlp)
        ],
        voting='soft'
    )
    
    print(f"   Modelo ensemble creado con 3 algoritmos base optimizados")
    return ensemble

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Entrenar y evaluar el modelo."""
    print("\n🏋️  Entrenando modelo avanzado...")
    
    start_time = time.time()
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Crear y entrenar modelo
    model = create_advanced_ensemble_model()
    
    print("   Iniciando entrenamiento...")
    model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    print(f"   Entrenamiento completado en {training_time:.2f} segundos")
    
    # Evaluación
    print("\n📊 Evaluando modelo...")
    
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    inference_time = time.time() - start_time
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Reporte detallado
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    
    results = {
        'model_type': 'Advanced Sklearn Ensemble',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'training_time': training_time,
        'inference_time': inference_time,
        'samples_per_second': len(y_test) / inference_time,
        'total_samples': len(y_test),
        'feature_count': X_train.shape[1]
    }
    
    # Mostrar resultados
    print_results(results)
    
    return model, scaler, results

def print_results(results):
    """Imprimir resultados de evaluación."""
    print(f"\n🎯 RESULTADOS FINALES")
    print("="*50)
    print(f"   Exactitud: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precisión: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   F1-Score: {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    
    print(f"\n⚡ EFICIENCIA:")
    print(f"   Entrenamiento: {results['training_time']:.2f} segundos")
    print(f"   Inferencia: {results['inference_time']:.3f} segundos")
    print(f"   Velocidad: {results['samples_per_second']:.1f} muestras/segundo")
    print(f"   Características: {results['feature_count']} features")
    
    # Matriz de confusión
    cm = results['confusion_matrix']
    print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"{'':>15} {'Anthracnose':>12} {'healthy_guava':>12}")
    print("-" * 41)
    print(f"{'Anthracnose':>15} {cm[0][0]:>12} {cm[0][1]:>12}")
    print(f"{'healthy_guava':>15} {cm[1][0]:>12} {cm[1][1]:>12}")

def save_model_and_metadata(model, scaler, results, feature_extractor):
    """Guardar modelo y metadatos."""
    print(f"\n💾 Guardando modelo y metadatos...")
    
    # Guardar modelo
    model_path = MODELS_DIR / 'guayaba_advanced_sklearn_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Guardar scaler
    scaler_path = MODELS_DIR / 'guayaba_advanced_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Guardar feature extractor
    extractor_path = MODELS_DIR / 'guayaba_feature_extractor.pkl'
    with open(extractor_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    # Metadatos
    metadata = {
        'model_type': 'Advanced Sklearn Ensemble',
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'extractor_path': str(extractor_path),
        'training_date': datetime.now().isoformat(),
        'image_size': IMG_SIZE,
        'classes': CLASSES,
        'class_indices': {name: i for i, name in enumerate(CLASSES)},
        'results': results,
        'feature_count': results['feature_count']
    }
    
    # Guardar metadatos
    metadata_path = MODELS_DIR / 'advanced_sklearn_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Guardar índices de clases
    class_indices_path = MODELS_DIR / 'advanced_sklearn_class_indices.json'
    with open(class_indices_path, 'w') as f:
        json.dump(metadata['class_indices'], f, indent=2)
    
    print(f"   📄 Modelo: {model_path}")
    print(f"   📄 Scaler: {scaler_path}")
    print(f"   📄 Feature Extractor: {extractor_path}")
    print(f"   📄 Metadatos: {metadata_path}")
    print(f"   📄 Índices: {class_indices_path}")

def create_comparison_plot(results):
    """Crear gráficos de resultados."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Métricas
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [results[m] for m in metrics]
    
    bars = ax1.bar(metrics, values, color='#2E8B57', alpha=0.8)
    ax1.set_title('Métricas del Modelo Avanzado')
    ax1.set_ylabel('Valor')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Matriz de confusión
    cm = results['confusion_matrix']
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_title('Matriz de Confusión')
    
    # Etiquetas
    classes = ['Anthracnose', 'healthy_guava']
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    
    # Números en la matriz
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax2.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    ax2.set_ylabel('Clase Real')
    ax2.set_xlabel('Clase Predicha')
    
    # Comparación de tiempos
    times = ['Entrenamiento', 'Inferencia']
    time_values = [results['training_time'], results['inference_time']]
    
    ax3.bar(times, time_values, color=['#4169E1', '#32CD32'])
    ax3.set_title('Tiempos de Ejecución')
    ax3.set_ylabel('Tiempo (segundos)')
    ax3.grid(True, alpha=0.3)
    
    for i, v in enumerate(time_values):
        ax3.text(i, v + max(time_values)*0.01, f'{v:.3f}s', ha='center', va='bottom')
    
    # Información del modelo
    ax4.text(0.1, 0.8, f"Modelo: {results['model_type']}", fontsize=12, weight='bold')
    ax4.text(0.1, 0.7, f"Exactitud: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)", fontsize=11)
    ax4.text(0.1, 0.6, f"Características: {results['feature_count']:,}", fontsize=11)
    ax4.text(0.1, 0.5, f"Muestras de test: {results['total_samples']}", fontsize=11)
    ax4.text(0.1, 0.4, f"Velocidad: {results['samples_per_second']:.1f} fps", fontsize=11)
    ax4.text(0.1, 0.2, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Información del Modelo')
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_path = MODELS_DIR / 'advanced_sklearn_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Gráficos guardados en: {plot_path}")

def main():
    """Función principal."""
    print("🚀 ENTRENAMIENTO CON MODELO SKLEARN AVANZADO")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Crear extractor de características
        feature_extractor = AdvancedFeatureExtractor(IMG_SIZE)
        
        # Cargar dataset
        X, y = load_dataset_with_augmentation(DATA_DIR, feature_extractor, use_augmentation=True)
        
        # Dividir train/test (usar test existente)
        test_dir = DATA_DIR / "test"
        if test_dir.exists():
            print("📊 Cargando dataset de test separado...")
            X_test, y_test = load_test_dataset(test_dir, feature_extractor)
            
            # El resto es para entrenamiento (train + val)
            X_train, y_train = X, y
        else:
            # División manual si no hay test separado
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        print(f"   Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   Test: {X_test.shape[0]} muestras")
        
        # Entrenar y evaluar
        model, scaler, results = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        
        # Guardar modelo
        save_model_and_metadata(model, scaler, results, feature_extractor)
        
        # Crear gráficos
        create_comparison_plot(results)
        
        # Tiempo total
        total_time = time.time() - start_time
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"⏱️  Tiempo total: {total_time/60:.2f} minutos")
        print(f"🎯 Exactitud final: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"🚀 Velocidad: {results['samples_per_second']:.1f} muestras/segundo")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()