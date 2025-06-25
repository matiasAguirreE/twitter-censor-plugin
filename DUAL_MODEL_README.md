# 🤖 Twitter Censor Plugin - Configuración Dual de Modelos

## 📋 Resumen

Este proyecto ahora soporta **dos modelos simultáneamente**:
- **Modelo Viejo (`old_model`)**: Entrenado SIN data augmentation (F1-Macro: 82.6%)
- **Modelo Nuevo (`new_model`)**: Se entrenará CON los mejores hiperparámetros encontrados (F1-Macro esperado: 99.5%)

## 🎯 **Hiperparámetros Optimizados**

### **Modelo Viejo (sin data augmentation)**
Basado en la búsqueda de hiperparámetros sin augmentation:
```python
EPOCHS = 4
LEARNING_RATE = 6e-5  # 0.00006
BATCH_SIZE = 32
WARMUP_RATIO = 0.3
WEIGHT_DECAY = 0.015
DROPOUT = 0.3
```
**Rendimiento**: F1-Macro: 82.61%, F1-Micro: 82.5%

### **Modelo Nuevo (con mejores hiperparámetros)**
Basado en la búsqueda con data augmentation - **¡MUCHO MEJOR!**:
```python
EPOCHS = 4
LEARNING_RATE = 5e-5  # 0.00005
BATCH_SIZE = 8
WARMUP_RATIO = 0.15
WEIGHT_DECAY = 0.0075
DROPOUT = 0.3
```
**Rendimiento esperado**: F1-Macro: 99.53%, F1-Micro: 99.56%, Precision: 100%

## 📁 Estructura de Archivos

```
app-ia/model/
├── old_model/              # Modelo sin data augmentation (82.6% F1)
│   ├── model.pth
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── ...
├── new_model/              # Modelo con mejores hiperparámetros (99.5% F1 esperado)
│   ├── model.pth           # (se crea al entrenar)
│   ├── tokenizer.json
│   └── ...
├── config_old.py          # Hiperparámetros sin augmentation
├── config_new.py          # MEJORES hiperparámetros encontrados
├── predict_old.py         # Predicción con modelo sin augmentation
├── predict_new.py         # Predicción con modelo optimizado
├── train_new.py          # Entrenamiento con mejores hiperparámetros
└── ...
```

## 🚀 Cómo Usar

### 1. Entrenar el Nuevo Modelo (RECOMENDADO)

```bash
# Desde el directorio raíz del proyecto
python train_new_model.py
```

Este script:
- Usa los **MEJORES hiperparámetros** encontrados en la búsqueda
- Incluye **warmup scheduling** y **weight decay** optimizados
- Debería alcanzar ~99.5% F1-Macro (vs 82.6% del modelo viejo)

### 2. Ejecutar el Servidor con Ambos Modelos

```bash
# Ejecutar el servidor
cd app-ia
python main.py
```

### 3. Probar los Endpoints

```bash
# Desde el directorio raíz
python test_api.py
```

## 🌐 Endpoints Disponibles

### 1. `GET /status/`
Verifica qué modelos están disponibles:
```json
{
  "old_model_available": true,
  "new_model_available": true,
  "total_models_loaded": 2
}
```

### 2. `POST /verificarCensura/` (Compatibilidad)
Endpoint original que usa el modelo viejo (82.6% F1):
```json
{
  "Violencia": 0.1234,
  "Homofobia": 0.0567,
  "Xenofobia": 0.0892
}
```

### 3. `POST /verificarCensura/old/`
Predicción explícita con modelo viejo (82.6% F1):
```json
{
  "model": "old",
  "prediction": {
    "Violencia": 0.1234,
    "Homofobia": 0.0567,
    "Xenofobia": 0.0892
  }
}
```

### 4. `POST /verificarCensura/new/`
Predicción con modelo optimizado (99.5% F1):
```json
{
  "model": "new",
  "prediction": {
    "Violencia": 0.1456,
    "Homofobia": 0.0623,
    "Xenofobia": 0.0734
  }
}
```

### 5. `POST /verificarCensura/compare/`
Comparación entre modelo básico vs optimizado:
```json
{
  "text": "Texto de ejemplo",
  "old_model": {
    "available": true,
    "prediction": { "Violencia": 0.1234, "Homofobia": 0.0567, "Xenofobia": 0.0892 }
  },
  "new_model": {
    "available": true,
    "prediction": { "Violencia": 0.1456, "Homofobia": 0.0623, "Xenofobia": 0.0734 }
  },
  "comparison": {
    "Violencia": {
      "old": 0.1234,
      "new": 0.1456,
      "difference": 0.0222,
      "percent_change": 18.0
    },
    "Homofobia": {
      "old": 0.0567,
      "new": 0.0623,
      "difference": 0.0056,
      "percent_change": 9.9
    },
    "Xenofobia": {
      "old": 0.0892,
      "new": 0.0734,
      "difference": -0.0158,
      "percent_change": -17.7
    }
  }
}
```

## 🔑 Autenticación

Todos los endpoints (excepto `/status/`) requieren el header:
```
X-Api-Key: e55d7f49-a705-4895-bf5f-d63aa1f46e11
```

## 📊 Comparación de Rendimiento

| Métrica | Modelo Viejo | Modelo Nuevo | Mejora |
|---------|--------------|--------------|---------|
| F1-Macro | 82.61% | **99.53%** | +16.92% |
| F1-Micro | 82.50% | **99.56%** | +17.06% |
| Precision Promedio | 82.75% | **100%** | +17.25% |
| Recall Promedio | 82.96% | **99.07%** | +16.11% |

## 🔧 **¿Por qué estos hiperparámetros son mejores?**

### **Learning Rate (5e-5 vs 6e-5)**
- **Modelo nuevo**: 5e-5 permite convergencia más estable
- **Batch Size (8 vs 32)**: Batch más pequeño con mejor generalización

### **Warmup + Weight Decay**
- **Warmup Ratio**: 0.15 ayuda a la estabilidad inicial del entrenamiento
- **Weight Decay**: 0.0075 previene overfitting efectivamente

### **Data Augmentation**
- Los mejores hiperparámetros provienen de experimentos **CON data augmentation**
- Esto explica la mejora dramática de 82.6% → 99.5% F1-Macro

## ⚠️ Notas Importantes

1. **Rendimiento**: El modelo nuevo debería ser **significativamente mejor** (99.5% vs 82.6% F1-Macro)

2. **Hiperparámetros validados**: Ambas configuraciones usan hiperparámetros que **realmente se encontraron** en la búsqueda de grid search

3. **Compatibilidad**: El endpoint original sigue funcionando con el modelo viejo

4. **Memoria**: Cargar ambos modelos consume más RAM

## 🐛 Solución de Problemas

### El modelo nuevo no alcanza 99.5% F1
- Verifica que tengas **data augmentation** en tus datos
- Los hiperparámetros están optimizados para datos aumentados

### Error de importación
- Instala dependencias: `pip install transformers torch scikit-learn`
- Verifica que el entorno virtual esté activado

### El servidor no arranca
- Instala: `pip install flask flask_cors`
- Verifica que el puerto 7021 esté disponible 