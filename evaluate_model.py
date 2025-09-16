#!/usr/bin/env python3
"""
Script para evaluar el modelo entrenado y mostrar métricas detalladas.
Incluye matriz de confusión, exactitud, precisión, recall y F1-Score.
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from PIL import Image

# Agregar el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent))

def load_model_and_metadata():
    """Cargar el modelo entrenado y sus metadatos."""
    
    # Rutas de los archivos
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / "guayaba_sklearn_model.pkl"
    metadata_path = models_dir / "sklearn_metadata.json"
    class_indices_path = models_dir / "class_indices.json"
    
    print(f"📁 Cargando modelo desde: {model_path}")
    print(f"📁 Cargando metadatos desde: {metadata_path}")
    print(f"📁 Cargando índices de clases desde: {class_indices_path}")
    
    # Verificar que los archivos existan
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadatos no encontrados: {metadata_path}")
    if not class_indices_path.exists():
        raise FileNotFoundError(f"Índices de clases no encontrados: {class_indices_path}")
    
    # Cargar modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Cargar metadatos
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Cargar índices de clases
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    return model, metadata, class_indices

def load_test_data():
    """Cargar datos de prueba del conjunto de test."""
    test_dir = Path(__file__).parent / "src" / "data" / "DataSetGuayabas" / "test"
    
    print(f"\n📂 Cargando datos de prueba desde: {test_dir}")
    
    X_test = []
    y_test = []
    class_names = []
    
    for class_dir in sorted(test_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_names.append(class_name)
            print(f"   Procesando clase: {class_name}")
            
            image_count = 0
            for img_path in class_dir.glob("*.png"):
                try:
                    # Cargar y procesar imagen
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((64, 64))
                    img_array = np.array(img).flatten() / 255.0
                    
                    X_test.append(img_array)
                    y_test.append(class_name)
                    image_count += 1
                    
                    if image_count % 50 == 0:
                        print(f"     Procesadas {image_count} imágenes...")
                        
                except Exception as e:
                    print(f"     ⚠️  Error procesando {img_path}: {e}")
            
            print(f"     ✅ Total procesadas: {image_count} imágenes")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\n📊 Datos de prueba cargados:")
    print(f"   Forma de X_test: {X_test.shape}")
    print(f"   Forma de y_test: {y_test.shape}")
    print(f"   Clases encontradas: {class_names}")
    
    return X_test, y_test, class_names

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluar el modelo y calcular todas las métricas."""
    print("\n🔍 Evaluando modelo...")
    
    # Convertir etiquetas de string a números si es necesario
    if isinstance(y_test[0], str):
        # Crear mapeo de clase a número
        label_to_num = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
        y_test_num = np.array([label_to_num[label] for label in y_test])
        print(f"   Mapeo de clases: {label_to_num}")
    else:
        y_test_num = y_test
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Convertir predicciones a las mismas etiquetas que y_test
    if isinstance(y_test[0], str):
        # Las predicciones ya son números, convertir a strings para comparación
        num_to_label = {idx: class_name for class_name, idx in label_to_num.items()}
        y_pred_str = np.array([num_to_label[pred] for pred in y_pred])
        
        # Usar las versiones string para métricas
        y_test_for_metrics = y_test
        y_pred_for_metrics = y_pred_str
        class_names_for_metrics = sorted(class_names)
    else:
        y_test_for_metrics = y_test_num
        y_pred_for_metrics = y_pred
        class_names_for_metrics = class_names
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_test_for_metrics, y_pred_for_metrics)
    precision_macro = precision_score(y_test_for_metrics, y_pred_for_metrics, average='macro')
    recall_macro = recall_score(y_test_for_metrics, y_pred_for_metrics, average='macro')
    f1_macro = f1_score(y_test_for_metrics, y_pred_for_metrics, average='macro')
    
    precision_weighted = precision_score(y_test_for_metrics, y_pred_for_metrics, average='weighted')
    recall_weighted = recall_score(y_test_for_metrics, y_pred_for_metrics, average='weighted')
    f1_weighted = f1_score(y_test_for_metrics, y_pred_for_metrics, average='weighted')
    
    # Matriz de confusión
    cm = confusion_matrix(y_test_for_metrics, y_pred_for_metrics, labels=class_names_for_metrics)
    
    # Reporte detallado
    report = classification_report(y_test_for_metrics, y_pred_for_metrics, labels=class_names_for_metrics, output_dict=True)
    
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
        'y_pred': y_pred_for_metrics,
        'y_pred_proba': y_pred_proba,
        'class_names_used': class_names_for_metrics
    }

def print_detailed_metrics(results, class_names):
    """Imprimir métricas detalladas de forma organizada."""
    print("\n" + "="*60)
    print("📊 REPORTE DETALLADO DE EVALUACIÓN DEL MODELO")
    print("="*60)
    
    # Métricas generales
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
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name:>15}:")
            print(f"      Precisión:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"      Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            print(f"      F1-Score:   {metrics['f1-score']:.4f} ({metrics['f1-score']*100:.2f}%)")
            print(f"      Soporte:    {metrics['support']} muestras")
            print()

def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """Crear y mostrar la matriz de confusión."""
    plt.figure(figsize=(10, 8))
    
    # Crear heatmap
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Número de muestras'}
    )
    
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Realidad', fontsize=12, fontweight='bold')
    
    # Añadir texto explicativo
    total_samples = confusion_matrix.sum()
    correct_predictions = np.trace(confusion_matrix)
    accuracy = correct_predictions / total_samples
    
    plt.figtext(0.02, 0.02, f'Exactitud: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                fontsize=10, ha='left')
    plt.figtext(0.02, 0.98, f'Total de muestras: {total_samples}', 
                fontsize=10, ha='left', va='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Matriz de confusión guardada en: {save_path}")
    
    plt.show()

def print_confusion_matrix_text(cm, class_names):
    """Imprimir la matriz de confusión en formato texto."""
    print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"{'':>15} " + " ".join(f"{name:>12}" for name in class_names))
    print("-" * (15 + 13 * len(class_names)))
    
    for i, actual_class in enumerate(class_names):
        row = f"{actual_class:>15} "
        row += " ".join(f"{cm[i][j]:>12}" for j in range(len(class_names)))
        print(row)
    
    print("\nInterpretación:")
    for i, class_name in enumerate(class_names):
        tp = cm[i][i]  # Verdaderos positivos
        total_actual = cm[i].sum()  # Total de casos reales de esta clase
        total_predicted = cm[:, i].sum()  # Total de predicciones de esta clase
        
        print(f"  {class_name}:")
        print(f"    Correctamente clasificadas: {tp} de {total_actual} ({tp/total_actual*100:.1f}%)")
        print(f"    Total predichas como {class_name}: {total_predicted}")

def main():
    """Función principal del script de evaluación."""
    try:
        print("🚀 Iniciando evaluación del modelo...")
        
        # Cargar modelo y metadatos
        model, metadata, class_indices = load_model_and_metadata()
        print("✅ Modelo cargado exitosamente")
        
        # Mostrar información del modelo
        print(f"\n📋 INFORMACIÓN DEL MODELO:")
        print(f"   Tipo: {metadata.get('model_type', 'No especificado')}")
        print(f"   Exactitud en entrenamiento: {metadata.get('training_accuracy', 'No disponible')}")
        print(f"   Fecha de entrenamiento: {metadata.get('training_date', 'No disponible')}")
        
        # Cargar datos de test
        X_test, y_test, class_names = load_test_data()
        
        if len(X_test) == 0:
            print("❌ No se encontraron datos de prueba")
            return
        
        # Evaluar modelo
        results = evaluate_model(model, X_test, y_test, class_names)
        
        # Usar los nombres de clases del resultado
        class_names_used = results.get('class_names_used', class_names)
        
        # Mostrar resultados
        print_detailed_metrics(results, class_names_used)
        print_confusion_matrix_text(results['confusion_matrix'], class_names_used)
        
        # Crear gráfico de matriz de confusión
        save_path = Path(__file__).parent / "confusion_matrix.png"
        plot_confusion_matrix(results['confusion_matrix'], class_names_used, save_path)
        
        print(f"\n🎉 Evaluación completada exitosamente!")
        print(f"📊 Exactitud final: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()