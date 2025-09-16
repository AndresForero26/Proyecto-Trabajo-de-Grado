# src/routes/index.py
import os
from datetime import datetime
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, current_app, url_for, render_template_string
from src.models.db import db, Detection
from src.services.model_service import model_predict, last_error
from src.services.metrics_service import calcular_metricas
from werkzeug.utils import secure_filename
from PIL import Image

bp = Blueprint("index", __name__)

# === Configuración ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> proyecto-flask-guayabas/
UPLOADS_DIR = PROJECT_ROOT / "static" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@bp.route('/fix_rutas', methods=['POST'])
def fix_rutas():
    """Corrige las rutas antiguas en la base de datos para que usen '/' en vez de '\\' en rel_path."""
    count = 0
    for d in Detection.query.all():
        if '\\' in d.rel_path:
            d.rel_path = d.rel_path.replace('\\', '/')
            count += 1
    db.session.commit()
    return f"Rutas corregidas: {count}", 200

@bp.route("/", methods=["GET", "POST"])
def home():
    view = request.args.get('view', 'detector')
    result = None
    
    # Cargar historial y métricas SIEMPRE para que el template siempre tenga datos
    all_detections = Detection.query.order_by(Detection.created_at.desc()).all()
    detections_db = [
        {
            "id": d.id,
            "rel_path": d.rel_path,
            "pred_label": d.pred_label,
            "real_label": d.real_label,
            "confidence": d.confidence,
            "prob_anthracnose": d.prob_anthracnose,
            "prob_healthy": d.prob_healthy,
            "created_at": d.created_at.strftime("%Y-%m-%d %H:%M")
        }
        for d in all_detections
    ]
    
    # Si hay detecciones y no se especifica vista, mostrar reportes
    if len(detections_db) > 0 and view == 'detector' and request.method == 'GET':
        view = 'reportes'
    
    # Solo considerar para métricas las filas donde el usuario ya marcó la realidad (ground truth)
    def normaliza_label(lbl):
        if lbl in ["Antracnosis", "Anthracnose"]:
            return "Anthracnose"
        elif lbl in ["Sana", "healthy_guava"]:
            return "healthy_guava"
        return lbl

    metricas = calcular_metricas([
        {"pred_label": normaliza_label(d.pred_label), "true_label": normaliza_label(d.real_label)}
        for d in all_detections if d.real_label is not None
    ]) if all_detections else None

    if request.method == "POST" and request.form.get("action") == "detector":
        file = request.files.get("file")
        if not file or file.filename == "":
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return render_template_string('<div class="alert alert-danger text-center mt-4">No seleccionaste archivo.</div>'), 400
            return render_template("index.html", view=view, result=None, error="No seleccionaste archivo."), 400
        
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return render_template_string('<div class="alert alert-danger text-center mt-4">Formato de archivo no permitido, sube una imagen en formato JPG o PNG.</div>'), 400
            return render_template("index.html", view=view, result=None, error="Formato de archivo no permitido, sube una imagen en formato JPG o PNG."), 400
        
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(file.filename)
        filepath = UPLOADS_DIR / filename
        file.save(filepath)
        
        try:
            img = Image.open(filepath).convert("RGB")
        except Exception:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return render_template_string('<div class="alert alert-danger text-center mt-4">No es una imagen válida.</div>'), 400
            return render_template("index.html", view=view, result=None, error="No es una imagen válida."), 400

        # Usar el servicio de modelo actualizado
        prediction = model_predict(str(filepath))
        
        if prediction is None:
            error_msg = last_error() or "Error desconocido en la predicción"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return render_template_string(f'<div class="alert alert-danger text-center mt-4">{error_msg}</div>'), 500
            return render_template("index.html", view=view, result=None, error=error_msg), 500

        label = prediction['label']
        confidence = prediction['confidence']
        probs = prediction['probs']
        
        # Normalizar etiquetas para mostrar en español
        if label == "Anthracnose":
            label_es = "Antracnosis"
            tiene_antracnosis = True
            mensaje = "La guayaba tiene Antracnosis."
        elif label == "healthy_guava":
            label_es = "Sana"
            tiene_antracnosis = False
            mensaje = "La guayaba está sana. No se detectaron signos de Antracnosis."
        else:
            label_es = label
            tiene_antracnosis = False
            mensaje = f"Se detectó: {label_es}."
        
        # Extraer probabilidades
        prob_anthracnose = probs.get('Anthracnose', 0.0)
        prob_healthy = probs.get('healthy_guava', 0.0)
        
        # Guardar en base de datos
        rel_path = f"uploads/{filename}".replace('\\', '/')
        detection = Detection(
            rel_path=rel_path,
            pred_label=label_es,
            confidence=round(confidence, 2),
            prob_anthracnose=round(prob_anthracnose, 2),
            prob_healthy=round(prob_healthy, 2),
            real_label=None,
            created_at=datetime.now()
        )
        db.session.add(detection)
        db.session.commit()

        result = {
            "nombre": filename,
            "ruta": f"/static/uploads/{filename}",
            "pred_label": label_es,
            "confianza": confidence,
            "prob_anthracnose": prob_anthracnose,
            "prob_healthy": prob_healthy,
            "mensaje": mensaje,
            "tiene_antracnosis": tiene_antracnosis,
            "detection_id": detection.id
        }
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render_template("_resultado.html", result=result)

    return render_template("index.html", 
                         view=view, 
                         result=result, 
                         detections=detections_db, 
                         metricas=metricas)

@bp.route('/update_label/<int:detection_id>', methods=['POST'])
def update_label(detection_id):
    """Actualizar la etiqueta real (ground truth) de una detección."""
    data = request.json
    new_label = data.get('label')
    
    detection = Detection.query.get(detection_id)
    if not detection:
        return jsonify({'error': 'Detección no encontrada'}), 404
    
    # Normalizar etiqueta
    if new_label in ["Antracnosis", "Anthracnose"]:
        detection.real_label = "Antracnosis"
    elif new_label in ["Sana", "healthy_guava"]:
        detection.real_label = "Sana"
    else:
        detection.real_label = new_label
    
    db.session.commit()
    return jsonify({'success': True})

@bp.route('/delete_detection/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    """Eliminar una detección."""
    detection = Detection.query.get(detection_id)
    if not detection:
        return jsonify({'error': 'Detección no encontrada'}), 404
    
    # Eliminar archivo de imagen si existe
    image_path = UPLOADS_DIR / os.path.basename(detection.rel_path)
    if image_path.exists():
        image_path.unlink()
    
    db.session.delete(detection)
    db.session.commit()
    return jsonify({'success': True})

@bp.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicción."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Formato de archivo no soportado'}), 400
    
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(file.filename)
    filepath = UPLOADS_DIR / filename
    file.save(filepath)
    
    # Realizar predicción
    prediction = model_predict(str(filepath))
    
    if prediction is None:
        return jsonify({'error': last_error() or 'Error en la predicción'}), 500
    
    label = prediction['label']
    confidence = prediction['confidence']
    probs = prediction['probs']
    
    # Normalizar para respuesta
    if label == "Anthracnose":
        label_es = "Antracnosis"
        message = "La guayaba tiene Antracnosis."
    elif label == "healthy_guava":
        label_es = "Sana"
        message = "La guayaba está sana. No se detectaron signos de Antracnosis."
    else:
        label_es = label
        message = f"Se detectó: {label_es}."
    
    return jsonify({
        'filename': filename,
        'label': label_es,
        'confidence': round(confidence, 4),
        'probabilities': {
            'Antracnosis': round(probs.get('Anthracnose', 0.0), 4),
            'Sana': round(probs.get('healthy_guava', 0.0), 4)
        },
        'message': message,
        'image_url': f"/static/uploads/{filename}"
    })