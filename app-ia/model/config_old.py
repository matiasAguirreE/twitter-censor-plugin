import os

# Base directory of the current file
BASE_DIR = os.path.dirname(__file__)

# Paths for data and model artifacts (OLD MODEL - SIN DATA AUGMENTATION)
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'labeled', 'tweets_limpios.csv')  # 1,000 tweets originales
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'old_model')
MODEL_PATH = os.path.join(BASE_DIR, 'old_model', 'model.pth')

# Model configuration
MODEL_NAME = "dccuchile/tulio-chilean-spanish-bert"
NUM_LABELS = 3
LABELS = ["Violencia", "Homofobia", "Xenofobia"]
MAX_LEN = 128

# HIPERPARÁMETROS DEL MODELO VIEJO (sin data augmentation)
# Datos: tweets_limpios.csv (1,000 tweets originales)
# Estos fueron los mejores encontrados en la búsqueda sin augmentation
# Mejor F1-Macro: (4, 6e-05, 32, 0.3, 0.015) con F1-Macro = 82.61%
EPOCHS = 4
LEARNING_RATE = 6e-5  # 0.00006
BATCH_SIZE = 32       # el mejor para F1-Macro sin augmentation
WARMUP_RATIO = 0.3
WEIGHT_DECAY = 0.015
DROPOUT = 0.3

# Nota: Estos son los hiperparámetros que dieron mejores resultados SIN data augmentation
# usando el dataset original de 1,000 tweets 