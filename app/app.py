import sys
from pathlib import Path

import streamlit as st


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference import predict

st.set_page_config(
    page_title="Multimodal Product Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛍️ Multimodal Product Intelligence")

st.markdown(
    """
    Use **product imagery + product text** to classify products
    into one of 20 product categories using a multimodal deep
    learning model.
    """
)

st.divider()

with st.sidebar:

    st.header("About the Project")

    st.markdown(
        """
        ### Model Architecture

        **Image Branch**
        - ResNet-50
        - Transfer Learning
        - Image Embeddings

        **Text Branch**
        - DistilBERT
        - Transformer Embeddings

        **Fusion**
        - Feature Projection
        - Multimodal Feature Fusion
        - 20-Class Classification
        """
    )

    st.divider()

    st.markdown(
        """
        ### Technologies

        - PyTorch
        - Torchvision
        - Hugging Face Transformers
        - Streamlit
        - Python
        """
    )

    st.divider()

    st.caption(
        "Multimodal Product Intelligence System"
    )

st.subheader("Product Information")

input_col1, input_col2 = st.columns(
    [1.2, 1],
    gap="large",
)

with input_col1:

    st.markdown("#### 🖼️ Product Image")

    uploaded_image = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG.",
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Product Image",
            width=400,
        )

with input_col2:

    st.markdown("#### 📝 Product Description")

    product_text = st.text_area(
        "Product name or description",
        placeholder=(
            "Example:\n"
            "Peter England Men Party Blue Jeans"
        ),
        height=150,
        help="Enter the product name or a short product description.",
    )

    if product_text.strip():

        st.success(
            "Product information entered.",
            icon="✅",
        )

st.divider()

st.subheader("Product Classification")

predict_button = st.button(
    "🔍 Predict Product Category",
    type="primary",
    use_container_width=True,
)


if predict_button:

    if uploaded_image is None:

        st.warning(
            "Please upload a product image before predicting."
        )

    elif not product_text.strip():

        st.warning(
            "Please enter a product name or description "
            "before predicting."
        )

    elif uploaded_image.type not in [
        "image/jpeg",
        "image/png",
    ]:

        st.error(
            "Unsupported image format. "
            "Please upload a JPG, JPEG, or PNG image."
        )

    else:

        with st.spinner(
            "sAnalyzing product image and text..."
        ):

            try:

                result = predict(
                    image_path=uploaded_image,
                    product_text=product_text,
                    top_k=5,
                )

                st.session_state[
                    "prediction_result"
                ] = result

                st.success(
                    "Prediction completed successfully."
                )

            except ValueError as error:

                st.warning(
                    f"Input validation error: {error}"
                )

            except Exception:

                st.error(
                    "Something went wrong while processing "
                    "the product. Please try again with a "
                    "valid image and product description."
                )

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.subheader("Prediction Result")

    predicted_category = result[
        "predicted_category"
    ]

    confidence = result[
        "confidence"
    ]

    confidence_level = result[
        "confidence_level"
    ]

    result_col1, result_col2 = st.columns(
        [1.5, 1],
        gap="large",
    )

    with result_col1:

        st.markdown(
            "### 🏷️ Predicted Category"
        )

        st.markdown(
            f"# {predicted_category}"
        )

    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence:.2%}",
        )

        st.caption(
            f"Confidence Level: {confidence_level}"
        )

        st.progress(
            min(confidence, 1.0)
        )

    st.markdown("### Top Predictions")

    top_predictions = result["top_k"]

    top_k_table = []

    for rank, prediction in enumerate(
        top_predictions,
        start=1,
    ):

        top_k_table.append(
            {
                "Rank": rank,
                "Category": prediction[
                    "category"
                ],
                "Probability": (
                    f"{prediction['probability']:.4%}"
                ),
            }
        )

    st.table(top_k_table)

st.divider()

st.caption(
    "Built with PyTorch, ResNet-50, DistilBERT "
    "and Streamlit."
)