"""
Sprint 2 — Pipeline de preprocesamiento (stub)

Este archivo contiene funciones de referencia para normalización, redimensionamiento
y augmentations reproducibles. Es un stub que debe completarse con las
transformaciones utilizadas en los experimentos.
"""
from typing import Tuple
import cv2
import numpy as np

def resize_and_normalize(image: np.ndarray, size: Tuple[int,int]=(224,224)) -> np.ndarray:
    """Redimensiona y normaliza a [0,1]."""
    img = cv2.resize(image, size)
    img = img.astype('float32') / 255.0
    return img

if __name__ == '__main__':
    print('preprocessing.py: Stub de pipeline de preprocesamiento')
