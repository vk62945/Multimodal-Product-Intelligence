import json
import torch

from src.config import DEVICE, MODEL_CONFIG_PATH
from src.model_loader import load_model
from src.preprocessing import preprocess_image, preprocess_text

MODEL = load_model()


def load_label_mapping():
    """Load class index to product category mapping."""

    mapping_path = MODEL_CONFIG_PATH.parent.parent / "data" / "processed" / "label_mapping.json"

    with open(mapping_path, "r", encoding="utf-8") as file:
        category_to_index = json.load(file)

    index_to_category = {
        index: category
        for category, index in category_to_index.items()
    }

    return index_to_category


LABEL_MAPPING = load_label_mapping()


def predict(image_path, product_text, top_k=5):
    """
    Run multimodal product classification.

    Parameters
    ----------
    image_path : str or Path
        Path to the product image.

    product_text : str
        Product name or description.

    top_k : int
        Number of top predictions to return.

    Returns
    -------
    dict
        Prediction results containing category, confidence,
        and top-k predictions.
    """

    if not product_text or not str(product_text).strip():
        raise ValueError("Product text cannot be empty.")

    if top_k < 1 or top_k > len(LABEL_MAPPING):
        raise ValueError(
            f"top_k must be between 1 and {len(LABEL_MAPPING)}."
        )

    image = preprocess_image(image_path).to(DEVICE)

    text_inputs = preprocess_text(product_text)

    input_ids = text_inputs["input_ids"].to(DEVICE)
    attention_mask = text_inputs["attention_mask"].to(DEVICE)

    with torch.inference_mode():
        logits = MODEL(
            images=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(logits, dim=1)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1,
    )

    top_probabilities = top_probabilities[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    predicted_index = top_indices[0]
    confidence = top_probabilities[0]

    predicted_category = LABEL_MAPPING[predicted_index]

    top_predictions = []

    for class_index, probability in zip(
        top_indices,
        top_probabilities,
    ):
        top_predictions.append(
            {
                "class_index": class_index,
                "category": LABEL_MAPPING[class_index],
                "probability": probability,
            }
        )

    return {
        "predicted_class_index": predicted_index,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "top_k": top_predictions,
    }