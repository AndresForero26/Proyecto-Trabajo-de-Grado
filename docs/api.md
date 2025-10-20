# Sprint 7 — API Backend

Endpoints principales propuestos:

- `POST /predict` — Subir imagen (multipart/form-data) y devolver predicción JSON.
- `GET /health` — Endpoint de salud.

Ejemplo de uso (curl):
```bash
curl -X POST -F "image=@ejemplo.jpg" http://127.0.0.1:5000/predict
```
