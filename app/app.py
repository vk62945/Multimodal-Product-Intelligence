import sys
from pathlib import Path

import streamlit as st

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
    ### AI-powered product category classification

    This application uses both **product images** and **product text**
    to predict the product category using a multimodal deep learning model.
    """
)

st.subheader("1. Upload Product Image")

uploaded_image = st.file_uploader(
    "Choose a product image",
    type=["jpg", "jpeg", "png"],
    help="Upload a product image for classification.",
)

if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Uploaded Product Image",
        width=400,
    )


st.subheader("2. Enter Product Information")

product_text = st.text_input(
    "Product name or description",
    placeholder="e.g. Peter England Men Party Blue Jeans",
    help="Enter the product name or a short product description.",
)

if product_text.strip():
    st.success("Product information entered.")

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.subheader("3. Product Classification")

predict_button = st.button(
    "🔍 Predict Product Category",
    type="primary",
    use_container_width=True,
)

if predict_button:

    # Validate image
    if uploaded_image is None:
        st.warning(
            "Please upload a product image before predicting."
        )

    # Validate text
    elif not product_text.strip():
        st.warning(
            "Please enter a product name or description before predicting."
        )

    # Validate image type
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
            "Analyzing product image and text..."
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
                    "Something went wrong while "
                    "processing the product. "
                    "Please try again with a valid image "
                    "and product description."
                )

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.subheader("Prediction Result")

    predicted_category = result["predicted_category"]
    confidence = result["confidence"]
    confidence_level = result["confidence_level"]

    st.markdown(
        f"### 🏷️ {predicted_category}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Confidence",
            f"{confidence:.2%}",
        )

    with col2:
        st.metric(
            "Confidence Level",
            confidence_level,
        )

    st.markdown("#### Top Predictions")

    top_predictions = result["top_k"]

    top_k_table = []

    for rank, prediction in enumerate(
        top_predictions,
        start=1,
    ):
        top_k_table.append(
            {
                "Rank": rank,
                "Category": prediction["category"],
                "Probability": f"{prediction['probability']:.4%}",
            }
        )

    st.table(top_k_table)

with st.sidebar:

    st.header("About the Project")

    st.markdown(
        """
        **Multimodal Product Intelligence System**

        This system combines:

        - 🖼️ ResNet-50 for image understanding
        - 📝 DistilBERT for text understanding
        - 🔗 Multimodal feature fusion
        - 🧠 PyTorch-based inference

        **Model Output**

        - Predicted category
        - Confidence score
        - Top-K predictions
        """
    )

    st.divider()

    st.caption(
        "Multimodal Product Intelligence"
    )