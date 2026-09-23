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

st.subheader("3. Product Classification")

predict_button = st.button(
    "🔍 Predict Product Category",
    type="primary",
    use_container_width=True,
)

if predict_button:

    if uploaded_image is None:
        st.warning("Please upload a product image first.")

    elif not product_text.strip():
        st.warning("Please enter a product name or description.")

    else:
        with st.spinner("Analyzing product image and text..."):
            try:
                result = predict(
                    image_path=uploaded_image,
                    product_text=product_text,
                    top_k=5,
                )
                st.success("Prediction completed successfully.")
                st.session_state["prediction_result"] = result
            except Exception as error:
                st.error(
                    "An error occurred while processing the product."
                )
                st.exception(error)

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