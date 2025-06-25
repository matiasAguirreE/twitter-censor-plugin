import os

# Base directory of the current file
BASE_DIR = os.path.dirname(__file__)

# Paths for data and model artifacts (NEW MODEL - CON DATA AUGMENTATION)
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'labeled', 'tweets_limpios_2.csv')  # 4,449 tweets con augmentation
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'new_model')
MODEL_PATH = os.path.join(BASE_DIR, 'new_model', 'model.pth')

# Model configuration
MODEL_NAME = "dccuchile/tulio-chilean-spanish-bert"
NUM_LABELS = 3
LABELS = ["Violencia", "Homofobia", "Xenofobia"]
MAX_LEN = 128

# MEJORES HIPERPARÁMETROS ENCONTRADOS EN LA BÚSQUEDA CON DATA AUGMENTATION
# Datos: tweets_limpios_2.csv (4,449 tweets con augmentation - 4.4x más datos)
# Resultados: F1-Macro = 99.5%, F1-Micro = 99.6%
# Mejor combinación: (4, 5e-05, 8, 0.15, 0.0075)
EPOCHS = 4
LEARNING_RATE = 5e-5  # 0.00005 - el mejor encontrado CON augmentation
BATCH_SIZE = 8        # el mejor encontrado CON augmentation 
WARMUP_RATIO = 0.15   # el mejor encontrado CON augmentation
WEIGHT_DECAY = 0.0075 # el mejor encontrado CON augmentation
DROPOUT = 0.3

# Nota: Estos hiperparámetros dieron los mejores resultados con data augmentation
# El dataset aumentado permite usar batch size más pequeño y learning rate más alto
# resultando en un rendimiento significativamente superior (99.5% vs 82.6%) 