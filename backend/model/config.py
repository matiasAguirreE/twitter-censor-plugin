import os

# Base directory of the current file
BASE_DIR = os.path.dirname(__file__)

# Paths for data and model artifacts
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'labeled', 'labeled_tweets.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'model.pth')

# Model configuration
MODEL_NAME = "dccuchile/tulio-chilean-spanish-bert"
NUM_LABELS = 3
LABELS = ["Violencia", "Homofobia", "Xenofobia"]
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5