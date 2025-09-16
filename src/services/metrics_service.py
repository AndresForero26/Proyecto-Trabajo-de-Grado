import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calcular_metricas(detecciones):
    """
    Calcula matriz de confusión, accuracy, precision, recall y F1-score a partir de una lista de detecciones.
    Cada detección debe ser un dict con 'pred_label' y 'true_label'.
    """
    if not detecciones:
        return None
    y_true = [d['true_label'] for d in detecciones]
    y_pred = [d['pred_label'] for d in detecciones]
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    return {
        'labels': labels,
        'confusion_matrix': cm.tolist(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
