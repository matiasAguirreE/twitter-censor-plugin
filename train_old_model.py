#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar el modelo OLD con datos originales (sin augmentation)

Uso:
    python train_old_model.py

Este script entrena el modelo OLD usando:
- Datos originales: tweets_limpios.csv (1,000 tweets)
- Hiperparámetros optimizados para datos sin augmentation
- Soporte completo para MPS (Apple Silicon), CUDA y CPU
"""

import sys
import os

# Agregar el directorio app-ia al path para los imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app-ia'))

if __name__ == "__main__":
    print("🏛️  Entrenamiento del Modelo OLD (sin data augmentation)")
    print("=" * 70)
    print("📊 Datos: tweets_limpios.csv (1,000 tweets originales)")
    print("🎯 F1-Macro esperado: ~82.6%")
    print("=" * 70)
    
    try:
        from model.train_old import train_old_model
        train_old_model()
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        print("Asegúrate de estar en el directorio correcto y tener todas las dependencias instaladas")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 70)
        print("🏁 Script completado") 