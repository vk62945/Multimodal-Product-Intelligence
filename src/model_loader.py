import torch

from src.config import (
    DEVICE,
    FINAL_MODEL_PATH,
    NUM_CLASSES
)

from src.model import (
    create_image_encoder,
    DistilBERTTextEncoder,
    MultimodalClassifier
)


def load_model():
    """
    Create the production multimodal model,
    load the trained checkpoint, and prepare
    it for inference.
    """

    image_encoder = create_image_encoder()

    text_encoder = DistilBERTTextEncoder()

    model = MultimodalClassifier(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        FINAL_MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model