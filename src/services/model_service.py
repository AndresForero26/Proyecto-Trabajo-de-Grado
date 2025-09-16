import json
import os
import pickle
import numpy as np
from PIL import Image
from src.config.config import Config

# Variables globales para los modelos
_resnet50_model = None
_sklearn_model = None
_advanced_model = None
_advanced_scaler = None
_advanced_extractor = None
_class_indices = None
_metadata = None
_last_error = None

def last_error():
    return _last_error

def _set_error(msg):
    global _last_error
    _last_error = msg

def ensure_model_ready():
    """
    Carga perezosa del modelo. Prioridad: Modelo Avanzado -> ResNet50 -> scikit-learn.
    """
    global _resnet50_model, _sklearn_model, _advanced_model, _advanced_scaler, _advanced_extractor, _class_indices, _metadata
    
    # Si ya hay un modelo cargado, retornar True
    if (_resnet50_model is not None or _sklearn_model is not None or _advanced_model is not None) and _class_indices is not None:
        return True

    # 1. Intentar cargar modelo avanzado primero (máxima prioridad)
    advanced_model_path = os.path.join(Config.MODELS_DIR, 'guayaba_advanced_sklearn_model.pkl')
    advanced_scaler_path = os.path.join(Config.MODELS_DIR, 'guayaba_advanced_scaler.pkl')
    advanced_extractor_path = os.path.join(Config.MODELS_DIR, 'guayaba_feature_extractor.pkl')
    advanced_metadata_path = os.path.join(Config.MODELS_DIR, 'advanced_sklearn_metadata.json')
    advanced_indices_path = os.path.join(Config.MODELS_DIR, 'advanced_sklearn_class_indices.json')
    
    if os.path.exists(advanced_model_path) and os.path.exists(advanced_scaler_path) and os.path.exists(advanced_extractor_path):
        try:
            # Importar la clase AdvancedFeatureExtractor
            from src.services.feature_extractor import AdvancedFeatureExtractor
            
            # Cargar modelo, scaler y extractor
            with open(advanced_model_path, 'rb') as f:
                _advanced_model = pickle.load(f)
            
            with open(advanced_scaler_path, 'rb') as f:
                _advanced_scaler = pickle.load(f)
            
            with open(advanced_extractor_path, 'rb') as f:
                _advanced_extractor = pickle.load(f)
            
            print("✅ Modelo avanzado scikit-learn cargado exitosamente")
            
            # Cargar metadatos del modelo avanzado
            if os.path.exists(advanced_metadata_path):
                with open(advanced_metadata_path, 'r') as f:
                    _metadata = json.load(f)
            
            # Cargar índices de clases del modelo avanzado
            if os.path.exists(advanced_indices_path):
                with open(advanced_indices_path, 'r', encoding='utf-8') as f:
                    _class_indices = json.load(f)
            
        except Exception as e:
            print(f"⚠️  Error cargando modelo avanzado: {e}")
            _advanced_model = None
            _advanced_scaler = None
            _advanced_extractor = None

    # 2. Intentar cargar ResNet50 si el modelo avanzado no está disponible
    resnet50_path = Config.RESNET50_MODEL_PATH
    resnet50_indices_path = Config.RESNET50_CLASS_INDICES_PATH
    resnet50_metadata_path = Config.RESNET50_METADATA_PATH
    
    if _advanced_model is None and os.path.exists(resnet50_path):
        try:
            import tensorflow as tf
            _resnet50_model = tf.keras.models.load_model(resnet50_path)
            print("✅ Modelo ResNet50 cargado exitosamente")
            
            # Cargar metadatos de ResNet50
            if os.path.exists(resnet50_metadata_path):
                with open(resnet50_metadata_path, 'r') as f:
                    _metadata = json.load(f)
            
        except Exception as e:
            print(f"⚠️  Error cargando modelo ResNet50: {e}")
            _resnet50_model = None
    
    # 3. Si los modelos anteriores no funcionan, intentar scikit-learn básico como último fallback
    sklearn_model_path = Config.SKLEARN_MODEL_PATH
    sklearn_metadata_path = os.path.join(os.path.dirname(sklearn_model_path), 'sklearn_metadata.json')
    
    if _advanced_model is None and _resnet50_model is None and os.path.exists(sklearn_model_path):
        try:
            with open(sklearn_model_path, 'rb') as f:
                _sklearn_model = pickle.load(f)
            
            if os.path.exists(sklearn_metadata_path):
                with open(sklearn_metadata_path, 'r') as f:
                    _metadata = json.load(f)
            
            print("✅ Modelo scikit-learn básico cargado como último fallback")
        except Exception as e:
            print(f"⚠️  Error cargando modelo scikit-learn básico: {e}")
            _sklearn_model = None

    # 4. Cargar índices de clases (prioridad: avanzado -> ResNet50 -> básico)
    indices_paths = [advanced_indices_path, resnet50_indices_path, Config.CLASS_INDICES_PATH]
    for indices_path in indices_paths:
        if os.path.exists(indices_path):
            try:
                with open(indices_path, 'r', encoding='utf-8') as f:
                    _class_indices = json.load(f)
                break
            except Exception as e:
                continue
    
    # Fallback para índices de clases
    if _class_indices is None:
        _class_indices = {name: i for i, name in enumerate(Config.CLASSES)}

    # Verificar que al menos un modelo se cargó
    if _advanced_model is None and _resnet50_model is None and _sklearn_model is None:
        _set_error(
            f"No se encontró ningún modelo válido. "
            f"Avanzado: {advanced_model_path} | "
            f"ResNet50: {resnet50_path} | "
            f"Scikit-learn: {sklearn_model_path}"
        )
        return False

    return True

def _preprocess_advanced(image_path):
    """Preprocesamiento para modelo avanzado usando el feature extractor"""
    global _advanced_extractor
    if _advanced_extractor is None:
        raise ValueError("Feature extractor no está disponible")
    
    # Usar el extractor de características del modelo avanzado
    features = _advanced_extractor.extract_all_features(image_path)
    return np.expand_dims(features, axis=0)

def _preprocess_resnet50(image_path):
    """Preprocesamiento para modelo ResNet50 (224x224)"""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)

def _preprocess_sklearn(image_path):
    """Preprocesamiento para modelo scikit-learn"""
    target_size = (64, 64)  # Tamaño usado en entrenamiento sklearn
    if _metadata and 'target_size' in _metadata:
        target_size = tuple(_metadata['target_size'])
    
    img = Image.open(image_path).convert('RGB').resize(target_size)
    arr = np.array(img).flatten() / 255.0
    return np.expand_dims(arr, axis=0)

def model_predict(image_path):
    if not ensure_model_ready():
        return None

    try:
        # 1. Usar modelo avanzado si está disponible (máxima prioridad)
        if _advanced_model is not None and _advanced_scaler is not None:
            x = _preprocess_advanced(image_path)
            x_scaled = _advanced_scaler.transform(x)
            probs = _advanced_model.predict_proba(x_scaled)[0]
            model_type = 'Advanced Sklearn Ensemble'
            
        # 2. Usar modelo ResNet50 si está disponible
        elif _resnet50_model is not None:
            x = _preprocess_resnet50(image_path)
            preds = _resnet50_model.predict(x, verbose=0)[0]
            
            # Para clasificación binaria, ResNet50 devuelve una sola probabilidad
            if len(preds) == 1:
                # Binary classification: sigmoid output
                prob_anthracnose = float(preds[0])
                prob_healthy = 1.0 - prob_anthracnose
                probs = [prob_anthracnose, prob_healthy]
            else:
                # Multi-class: aplicar softmax si es necesario
                if preds.sum() <= 0 or not (preds.max() <= 1.0 and preds.min() >= 0.0):
                    import tensorflow as tf
                    preds = tf.nn.softmax(preds).numpy()
                probs = preds
            
            model_type = 'ResNet50'
        
        # 3. Usar modelo scikit-learn básico como último fallback
        elif _sklearn_model is not None:
            x = _preprocess_sklearn(image_path)
            probs = _sklearn_model.predict_proba(x)[0]
            model_type = 'scikit-learn (básico)'
        
        else:
            _set_error("No hay modelo disponible para predicción")
            return None

        # Construir mapa nombre->prob
        inv = sorted(_class_indices.items(), key=lambda kv: kv[1])
        names = [name for name, _ in inv]
        
        # Ajustar si hay diferencia en número de clases
        n = min(len(names), len(probs))
        names = names[:n]
        probs_adjusted = probs[:n]
        
        i_max = int(np.argmax(probs_adjusted))
        label = names[i_max]
        confidence = float(probs_adjusted[i_max])

        prob_map = {names[i]: float(probs_adjusted[i]) for i in range(n)}
        
        # Asegurar las 2 clases esperadas con nombres estándar
        standard_prob_map = {}
        standard_prob_map['Anthracnose'] = prob_map.get('Anthracnose', 0.0)
        standard_prob_map['healthy_guava'] = prob_map.get('healthy_guava', 0.0)

        return {
            'label': label,
            'confidence': confidence,
            'probs': standard_prob_map,
            'model_type': model_type,
            'prob_anthracnose': standard_prob_map['Anthracnose'],
            'prob_healthy': standard_prob_map['healthy_guava'],
            'pred_label': label
        }
        
    except Exception as e:
        _set_error(f"Error al predecir: {e}")
        return None