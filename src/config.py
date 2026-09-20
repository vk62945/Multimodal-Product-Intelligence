from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

FINAL_MODEL_PATH = MODEL_DIR / "multimodal_final.pth"
MODEL_CONFIG_PATH = MODEL_DIR / "multimodal_final_config.json"

NUM_CLASSES = 20

IMAGE_SIZE = 224

MAX_TEXT_LENGTH = 32

IMAGE_EMBEDDING_DIM = 2048
IMAGE_PROJECTION_DIM = 512

TEXT_EMBEDDING_DIM = 768
TEXT_PROJECTION_DIM = 512

FUSION_INPUT_DIM = 1024
FUSION_HIDDEN_DIM = 256

DROPOUT = 0.3

TEXT_MODEL_NAME = "distilbert-base-uncased"

IMAGE_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGE_STD = [
    0.229,
    0.224,
    0.225
]
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")

else:
    DEVICE = torch.device("cpu")