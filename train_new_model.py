#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar el nuevo modelo

Uso:
    python train_new_model.py

Este script entrena un nuevo modelo usando la misma arquitectura que el modelo viejo
pero potencialmente con diferentes hiperparámetros o datos.
"""

import sys
import os

# Agregar el directorio app-ia al path para los imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app-ia'))

if __name__ == "__main__":
    print("🤖 Entrenamiento del Nuevo Modelo de Censura de Twitter")
    print("=" * 60)
    
    try:
        from model.train_new import train_new_model
        train_new_model()
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        print("Asegúrate de estar en el directorio correcto y tener todas las dependencias instaladas")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("✅ Script completado") 