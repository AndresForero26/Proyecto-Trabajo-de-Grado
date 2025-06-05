# Sprint 1 — Exploración y limpieza del dataset

Resumen de tareas realizadas en el sprint:

- Conteo de imágenes por clase.
- Verificación de estructura `train/`, `val/`, `test/` con subcarpetas por clase.
- Detección de imágenes corruptas o con pesos 0.

Instrucciones reproducibles:

```bash
python src/training/verify_dataset.py --input src/data/DataSetGuayabas/ --report docs/dataset_report.csv
```

Observaciones: documentar balance de clases y decisiones sobre limpieza aquí.
