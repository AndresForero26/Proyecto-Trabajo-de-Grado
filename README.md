# Detección de Antracnosis en guayaba mediante redes convolucionales (CNN)

## Descripción general
Este repositorio contiene el código, datos y documentación del proyecto de grado orientado a detectar Antracnosis en hojas y frutos de guayaba utilizando modelos de Deep Learning (CNN). Se incluyen notebooks, scripts de entrenamiento y evaluación, una API para inferencia y una interfaz web mínima.

## Justificación académica
La detección temprana de enfermedades en cultivos mejora la capacidad de manejo y reduce pérdidas. Este trabajo explora arquitecturas CNN (MobileNetV2, VGG16, ResNet50) para la clasificación binaria: `Anthracnose` vs `healthy_guava`. El foco es metodológico: reproducibilidad, comparación de arquitecturas y análisis crítico de resultados.

## Estructura del repositorio
- `app.py` — Servidor / API para inferencia.
- `models/` — Checkpoints y metadata de modelos entrenados.
- `src/` — Código fuente: `training/`, `services/`, `routes/`, `data/`.
- `static/`, `templates/` — Frontend estático.
- `docs/` — Notebooks e informes (exploración del dataset, comparativos, API, frontend).
- `db/` — Base de datos local (si aplica).

Consultar el contenido de cada carpeta para mayor detalle.

---

## Requisitos
- Python 3.8+ (recomendado 3.9/3.10)
- pip
- GPU con CUDA (recomendado para entrenamiento; no obligatorio para inferencia)

Dependencias principales (detalladas en `requirements.txt`): `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `matplotlib`, `tensorflow` o `torch`, `flask`.

## Instalación rápida
1. Clonar el repositorio:
```bash
git clone <URL_DEL_REPO>
cd Proyecto-Trabajo-de-Grado
```
2. Crear y activar un entorno virtual:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
3. Instalar dependencias:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso básico
1. Verificar dataset:
```bash
python src/training/verify_dataset.py
```
2. Preprocesar (si aplica):
```bash
python src/training/preprocessing.py --input src/data/DataSetGuayabas/ --output data_processed/
```
3. Entrenar (ejemplo ResNet50):
```bash
python src/training/train_resnet50.py --data data_processed/ --epochs 30 --batch-size 32 --output models/resnet50_weights.h5
```
4. Evaluar:
```bash
python src/training/evaluate_resnet50.py --model models/resnet50_weights.h5 --data data_processed/test/
```
5. Ejecutar backend e interactuar con frontend:
```bash
python app.py
# Abrir http://127.0.0.1:5000/
```

## Ejemplo de petición al API (POST multipart/form-data)
Respuesta JSON ejemplo:
```json
{
  "prediction": "Anthracnose",
  "confidence": 0.92,
  "class_index": 0,
  "timestamp": "2025-09-01T12:34:56"
}
```

## Limitaciones
- Dataset potencialmente pequeño o sesgado; ver `docs/dataset_exploration.md`.
- Resultados dependientes de hardware, semilla aleatoria y versiones de librerías.
- No se garantiza rendimiento en dispositivos embebidos sin pruebas adicionales.

## Trabajo futuro
- Añadir Dockerfile y pipeline CI para validación automática.
- Exportar modelo a TFLite/ONNX y evaluar en dispositivos de borde.
- Implementar métodos de interpretabilidad (Grad-CAM) y evaluación robusta.

## Autores y contexto académico
- [Nombre del estudiante] — Proyecto de grado, [Universidad], tutor: [Nombre del tutor].

---

Para mayor detalle sobre sprints y commits históricos reconstruidos, revisar `docs/CHANGELOG_SPRINTS.md`.
