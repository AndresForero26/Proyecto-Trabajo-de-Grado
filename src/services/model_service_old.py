import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from src.config.config import Config

_model = None
_class_indices = None
_last_error = None

def last_error():
    return _last_error

def _set_error(msg):
    global _last_error
    _last_error = msg

def ensure_model_ready():
    """
    Carga perezosa del modelo. Devuelve True si está listo.
    Si falta el archivo, deja un error claro para la UI.
    """
    global _model, _class_indices
    if _model is not None and _class_indices is not None:
        return True

    model_path = Config.MODEL_PATH
    indices_path = Config.CLASS_INDICES_PATH

    if not os.path.exists(model_path):
        _set_error(
            f"No se encontró el modelo en: {model_path}. "
            f"Entrena con 'python training/train.py' para generarlo."
        )
        return False

    try:
        _model = tf.keras.models.load_model(model_path)
    except Exception as e:
        _set_error(f"Error cargando el modelo: {e}")
        return False

    # Cargar mapa de clases (si existe). Si no, inferir del modelo.
    if os.path.exists(indices_path):
        try:
            with open(indices_path, 'r', encoding='utf-8') as f:
                inv = json.load(f)  # {"Anthracnose": 0, ...}
            # lo queremos como lista ordenada por índice
            _class_indices = {k: int(v) for k, v in inv.items()}
        except Exception as e:
            _set_error(f"Error leyendo class_indices.json: {e}")
            return False
    else:
        # fallback: intentar inferir a partir de Config
        _class_indices = {name: i for i, name in enumerate(Config.CLASSES)}

    return True

def _preprocess(image_path):
    img = Image.open(image_path).convert('RGB').resize(Config.INPUT_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)

def model_predict(image_path):
    if not ensure_model_ready():
        return None

    try:
        x = _preprocess(image_path)
        preds = _model.predict(x, verbose=0)[0]  # vector de logits/softmax
        # softmax si el modelo no la trae
        if preds.ndim == 0 or preds.shape == ():
            preds = np.array([float(preds)])
        if preds.sum() <= 0 or preds.max() <= 1.0 and preds.min() >= 0.0:
            probs = preds
        else:
            probs = tf.nn.softmax(preds).numpy()

        # construir mapa nombre->prob
        inv = sorted(_class_indices.items(), key=lambda kv: kv[1])  # [(name, idx), ...] orden por idx
        names = [name for name, _ in inv]
        # si el modelo tiene diferente # clases, recortar/ajustar
        n = min(len(names), len(probs))
        names = names[:n]
        probs = probs[:n]
        i_max = int(np.argmax(probs))
        label = names[i_max]
        confidence = float(probs[i_max])

        prob_map = {names[i]: float(probs[i]) for i in range(n)}
        # asegurar las 2 clases esperadas en la UI
        for k in ['Anthracnose', 'healthy_guava']:
            prob_map.setdefault(k, 0.0)

        return {
            'label': label,
            'confidence': confidence,
            'probs': prob_map
        }
    except Exception as e:
        _set_error(f"Error al predecir: {e}")
        return None
