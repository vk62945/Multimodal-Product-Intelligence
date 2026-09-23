import streamlit as st

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