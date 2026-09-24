import torch
from huggingface_hub import hf_hub_download

from src.config import (
    DEVICE,
    MODEL_REPO_ID,
    MODEL_FILENAME,
    NUM_CLASSES,
)

from src.model import (
    create_image_encoder,
    DistilBERTTextEncoder,
    MultimodalClassifier,
)


def load_model():
    image_encoder = create_image_encoder()
    text_encoder = DistilBERTTextEncoder()

    model = MultimodalClassifier(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        num_classes=NUM_CLASSES,
    )

    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model