import torch
import torch.nn as nn

from torchvision import models
from transformers import AutoModel

from src.config import (
    NUM_CLASSES,
    IMAGE_EMBEDDING_DIM,
    IMAGE_PROJECTION_DIM,
    TEXT_EMBEDDING_DIM,
    TEXT_PROJECTION_DIM,
    FUSION_INPUT_DIM,
    FUSION_HIDDEN_DIM,
    DROPOUT,
    TEXT_MODEL_NAME
)

def masked_mean_pooling(
    last_hidden_state,
    attention_mask
):
    """
    Mean-pool DistilBERT token embeddings while
    ignoring padding tokens.
    """

    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed_embeddings = torch.sum(
        last_hidden_state * mask,
        dim=1
    )

    summed_mask = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed_embeddings / summed_mask

class DistilBERTTextEncoder(nn.Module):

    def __init__(
        self,
        model_name=TEXT_MODEL_NAME
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(
            model_name
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_embedding = masked_mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        return text_embedding

def create_image_encoder():
    """
    Create the ResNet-50 feature extractor used
    by the multimodal model.
    """

    resnet = models.resnet50(
        weights=None
    )

    image_encoder = nn.Sequential(
        *list(resnet.children())[:-1]
    )

    return image_encoder

class MultimodalClassifier(nn.Module):

    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_classes=NUM_CLASSES
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_projection = nn.Sequential(
            nn.Linear(
                IMAGE_EMBEDDING_DIM,
                IMAGE_PROJECTION_DIM
            ),
            nn.ReLU()
        )

        self.text_projection = nn.Sequential(
            nn.Linear(
                TEXT_EMBEDDING_DIM,
                TEXT_PROJECTION_DIM
            ),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                FUSION_INPUT_DIM,
                FUSION_HIDDEN_DIM
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                FUSION_HIDDEN_DIM,
                num_classes
            )
        )

    def forward(
        self,
        images,
        input_ids,
        attention_mask
    ):

        image_features = self.image_encoder(
            images
        )

        image_features = torch.flatten(
            image_features,
            start_dim=1
        )

        image_features = self.image_projection(
            image_features
        )

        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = self.text_projection(
            text_features
        )

        combined_features = torch.cat(
            [
                image_features,
                text_features
            ],
            dim=1
        )

        logits = self.fusion(
            combined_features
        )

        return logits