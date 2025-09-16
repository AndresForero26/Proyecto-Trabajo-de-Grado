#!/usr/bin/env python3
"""
Entrenamiento con ResNet50 para clasificación de guayabas.
Transfer Learning con fine-tuning para clasificación binaria: sana vs antracnosis.
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
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "DataSetGuayabas"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Parámetros del modelo
IMG_SIZE = (224, 224)  # ResNet50 requiere 224x224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
CLASSES = ['Anthracnose', 'healthy_guava']

# Configuración GPU (si está disponible)
def configure_gpu():
    """Configurar GPU si está disponible."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Permitir crecimiento de memoria gradual
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🎮 GPU configurada: {len(gpus)} dispositivo(s) encontrado(s)")
        except RuntimeError as e:
            print(f"⚠️  Error configurando GPU: {e}")
    else:
        print("💻 Usando CPU - No se detectó GPU")

def create_data_generators():
    """Crear generadores de datos con data augmentation."""
    print("\n📊 Creando generadores de datos...")
    
    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% para validación
    )
    
    # Solo rescaling para validación y test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generador de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Clasificación binaria
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Generador de validación
    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Generador de test
    test_generator = test_datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"   Entrenamiento: {train_generator.samples} muestras")
    print(f"   Validación: {validation_generator.samples} muestras")
    print(f"   Test: {test_generator.samples} muestras")
    print(f"   Clases detectadas: {list(train_generator.class_indices.keys())}")
    
    return train_generator, validation_generator, test_generator

def create_resnet50_model():
    """Crear modelo ResNet50 con transfer learning."""
    print("\n🏗️  Creando modelo ResNet50...")
    
    # Cargar ResNet50 pre-entrenado (sin las capas superiores)
    base_model = ResNet50(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar las capas base inicialmente
    base_model.trainable = False
    
    # Crear el modelo completo
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Clasificación binaria
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"   Modelo creado con {model.count_params():,} parámetros")
    print(f"   Parámetros entrenables: {sum([tf.size(p) for p in model.trainable_weights]):,}")
    
    return model

def create_callbacks(model_path):
    """Crear callbacks para el entrenamiento."""
    return [
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger(
            MODELS_DIR / 'training_history.csv',
            append=False
        )
    ]

def fine_tune_model(model, train_gen, val_gen, model_path):
    """Realizar fine-tuning del modelo."""
    print("\n🔧 Iniciando fine-tuning...")
    
    # Descongelar las últimas capas de ResNet50
    model.layers[0].trainable = True
    
    # Congelar las primeras capas (mantener características básicas)
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"   Parámetros entrenables después del fine-tuning: {sum([tf.size(p) for p in model.trainable_weights]):,}")
    
    # Entrenar con fine-tuning
    fine_tune_epochs = 20
    history_fine = model.fit(
        train_gen,
        epochs=fine_tune_epochs,
        validation_data=val_gen,
        callbacks=create_callbacks(model_path.replace('.h5', '_finetuned.h5')),
        verbose=1
    )
    
    return history_fine

def evaluate_model(model, test_generator):
    """Evaluar el modelo y mostrar métricas detalladas."""
    print("\n📊 Evaluando modelo...")
    
    # Evaluación básica
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        test_generator, 
        verbose=1
    )
    
    # Predicciones para métricas detalladas
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Obtener clases reales
    true_classes = test_generator.classes
    class_names = list(test_generator.class_indices.keys())
    
    # Reporte de clasificación
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_names,
        output_dict=True
    )
    
    # Matriz de confusión
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Mostrar resultados
    print(f"\n🎯 RESULTADOS FINALES:")
    print(f"   Pérdida de Test: {test_loss:.4f}")
    print(f"   Exactitud: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Precisión: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"   Recall: {test_recall:.4f} ({test_recall*100:.2f}%)")
    
    print(f"\n📋 MÉTRICAS POR CLASE:")
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name}:")
            print(f"      Precisión: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"      Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            print(f"      F1-Score: {metrics['f1-score']:.4f} ({metrics['f1-score']*100:.2f}%)")
    
    print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"{'':>15} " + " ".join(f"{name:>12}" for name in class_names))
    print("-" * (15 + 13 * len(class_names)))
    for i, actual_class in enumerate(class_names):
        row = f"{actual_class:>15} "
        row += " ".join(f"{cm[i][j]:>12}" for j in range(len(class_names)))
        print(row)
    
    return {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_loss': test_loss,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

def save_metadata(results, model_path, train_gen):
    """Guardar metadatos del modelo."""
    metadata = {
        'model_type': 'ResNet50',
        'model_path': str(model_path),
        'training_date': datetime.now().isoformat(),
        'image_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'classes': CLASSES,
        'class_indices': train_gen.class_indices,
        'training_samples': train_gen.samples,
        'test_accuracy': results['test_accuracy'],
        'test_precision': results['test_precision'],
        'test_recall': results['test_recall'],
        'test_loss': results['test_loss'],
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix']
    }
    
    # Guardar metadatos
    metadata_path = MODELS_DIR / 'resnet50_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Guardar índices de clases
    class_indices_path = MODELS_DIR / 'resnet50_class_indices.json'
    with open(class_indices_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    
    print(f"\n💾 Metadatos guardados:")
    print(f"   📄 {metadata_path}")
    print(f"   📄 {class_indices_path}")

def plot_training_history(history, save_path):
    """Crear gráficos del historial de entrenamiento."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Validación')
    ax1.set_title('Exactitud del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Exactitud')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True)
    
    # Precision
    ax3.plot(history.history['precision'], label='Entrenamiento')
    ax3.plot(history.history['val_precision'], label='Validación')
    ax3.set_title('Precisión del Modelo')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Precisión')
    ax3.legend()
    ax3.grid(True)
    
    # Recall
    ax4.plot(history.history['recall'], label='Entrenamiento')
    ax4.plot(history.history['val_recall'], label='Validación')
    ax4.set_title('Recall del Modelo')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Gráficos guardados en: {save_path}")

def main():
    """Función principal del entrenamiento."""
    print("🚀 ENTRENAMIENTO CON RESNET50")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # Configurar GPU
        configure_gpu()
        
        # Crear generadores de datos
        train_gen, val_gen, test_gen = create_data_generators()
        
        # Crear modelo
        model = create_resnet50_model()
        
        # Mostrar arquitectura
        model.summary()
        
        # Paths para guardar
        model_path = MODELS_DIR / 'guayaba_resnet50.h5'
        
        print(f"\n🏋️  Iniciando entrenamiento...")
        print(f"   Épocas: {EPOCHS}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Learning rate: {LEARNING_RATE}")
        
        # Entrenamiento inicial
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=create_callbacks(str(model_path)),
            verbose=1
        )
        
        # Fine-tuning
        history_fine = fine_tune_model(model, train_gen, val_gen, str(model_path))
        
        # Cargar el mejor modelo
        model = keras.models.load_model(str(model_path))
        
        # Evaluación final
        results = evaluate_model(model, test_gen)
        
        # Guardar metadatos
        save_metadata(results, model_path, train_gen)
        
        # Crear gráficos
        plot_path = MODELS_DIR / 'resnet50_training_history.png'
        plot_training_history(history, plot_path)
        
        # Tiempo total
        total_time = time.time() - start_time
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"⏱️  Tiempo total: {total_time/60:.2f} minutos")
        print(f"📂 Modelo guardado en: {model_path}")
        print(f"🎯 Exactitud final: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()