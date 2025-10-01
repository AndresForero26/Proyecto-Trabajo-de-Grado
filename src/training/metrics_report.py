"""
Sprint 6 — Métricas y reportes (stub)

Contendrá funciones para calcular precision, recall, F1, matriz de confusión
y generar gráficos guardables en `docs/results/`.
"""
import json
def generate_report(metrics: dict, out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    print('metrics_report.py: Stub para generación de métricas y reportes')
