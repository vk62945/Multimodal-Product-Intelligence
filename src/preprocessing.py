from PIL import Image
from torchvision import transforms

from src.config import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD
)

image_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGE_MEAN,
        std=IMAGE_STD
    )
])

def preprocess_image(image_path):
    """
    Load and preprocess a product image.

    Parameters
    ----------
    image_path : str or Path
        Path to the product image.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor with shape:
        [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = image_transform(
        image
    )
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

from transformers import AutoTokenizer

from src.config import (
    TEXT_MODEL_NAME,
    MAX_TEXT_LENGTH
)
# Text Tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_NAME
)

def preprocess_text(text):
    """
    Tokenize a product name for DistilBERT.

    Parameters
    ----------
    text : str
        Product name or product description.

    Returns
    -------
    dict
        input_ids and attention_mask with batch dimension.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if not text:
        raise ValueError(
            "Product text cannot be empty."
        )

    encoded = tokenizer(
        text,
        max_length=MAX_TEXT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }