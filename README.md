# Multimodal Product Intelligence System

A production-oriented multimodal deep learning system that combines **product images and product text** to classify products into **20 product categories**.

The system uses a pretrained **ResNet-50** image encoder and **DistilBERT** text encoder. The two modalities are converted into embeddings, projected into a shared feature space, fused, and passed through a classification head.

The final model is deployed as an interactive **Streamlit application**, with the production model hosted on **Hugging Face Hub**.

## 🚀 Live Demo

**[Open the Live Streamlit Application](https://multimodal-prduct-intelligence-ai.streamlit.app)**

## 🎯 Problem Statement

Modern product catalogs contain multiple sources of information, particularly product images and textual descriptions.

Images provide visual information such as:

- Product type
- Shape
- Color
- Visual characteristics

Product text provides complementary semantic information such as:

- Product name
- Brand
- Gender
- Product description

This project combines both modalities to build a multimodal product classification system.

## 🧠 Solution Overview

The system processes the image and text independently and combines their learned representations before classification.

```text
                         Product
                       /         \
                      /           \
             Product Image     Product Text
                   |                |
                   v                v
               ResNet-50        DistilBERT
              Image Encoder    Text Encoder
                   |                |
                   v                v
              2048-D Image      768-D Text
               Embedding         Embedding
                   |                |
                   v                v
             Image Projection   Text Projection
                  512-D              512-D
                   |                |
                   └────────┬────────┘
                            |
                            v
                    Feature Fusion
                         1024-D
                            |
                            v
                      Fusion Layer
                          256-D
                            |
                            v
                    20-Class Classifier
                            |
                            v
                    Product Category
       
## 🏗️ Model Architecture

### Image Branch — ResNet-50

The image branch uses a pretrained ResNet-50 model to extract visual features from product images.

Key points:

- ResNet-50 pretrained on ImageNet
- Input image size: 224 × 224
- 2048-dimensional image embedding
- Transfer learning
- Selective fine-tuning of ResNet Layer 4 during final optimization

### Text Branch — DistilBERT

The text branch uses DistilBERT to extract semantic information from product names and descriptions.

Key points:

- `distilbert-base-uncased`
- Maximum sequence length: 32 tokens
- 768-dimensional text embedding
- Masked mean pooling
- Transformer encoder frozen during final multimodal optimization

### Multimodal Fusion

The two embeddings are projected into the same dimensional space:

```text
Image Embedding: 2048 → 512
Text Embedding:   768 → 512

📊 Model Performance
Final Optimized Multimodal Model

The final model selectively fine-tunes ResNet-50 Layer 4 while keeping the DistilBERT encoder frozen.

Metric	Result
Test Accuracy	99.58%
Macro F1	98.95%
Weighted F1	99.58%
Test Samples	6,459
Test Errors	27

🔍 Explainability

The project includes explainability techniques for both image and text modalities.

Image Explainability — Grad-CAM

Grad-CAM is used to visualize image regions contributing to the model's prediction.

The technique was applied to the final ResNet-50 feature layers to inspect which visual regions influenced the classification.

Text Explainability — Gradient × Embedding

Gradient × Embedding attribution is used to estimate the contribution of individual text tokens to the multimodal prediction.

These attribution values are interpreted as local sensitivity signals rather than causal explanations.

📂 Dataset

The project uses the Fashion Product Images (Small) dataset.

The final modeling dataset contains:

42,859 records
20 product categories
Leakage-safe train/validation/test splits
Product-name grouping during splitting to reduce data leakage
All 20 categories represented across the splits

Product Categories
The model predicts:
Bags
Belts
Bottomwear
Dress
Eyewear
Flip Flops
Fragrance
Innerwear
Jewellery
Lips
Loungewear and Nightwear
Makeup
Nails
Sandal
Saree
Shoes
Socks
Topwear
Wallets
Watches

🛠️ Technology Stack
Python
PyTorch
Torchvision
Hugging Face Transformers
ResNet-50
DistilBERT
Streamlit
Hugging Face Hub
Pillow
NumPy
Pandas
Scikit-learn

📁 Project Structure
Multimodal-Product-Intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   │   ├── label_mapping.json
│   │   ├── model_dataset.csv
│   │   ├── styles_clean.csv
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
│   └── raw/
│       └── fashion-product-images-small/
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Data_Preparation.ipynb
│   ├── 03_Image_Model.ipynb
│   ├── 04_Text_Model.ipynb
│   └── 05_Multimodal_Model.ipynb
│
├── src/
│   ├── clean_styles.py
│   ├── config.py
│   ├── inference.py
│   ├── model.py
│   ├── model_loader.py
│   └── preprocessing.py
│
├── .gitignore
├── requirements.txt
└── README.md

🤗 Hugging Face Model Hosting

The production model is hosted separately on Hugging Face Hub.

Model repository:

https://huggingface.co/vk62945/multimodal-product-intelligence

Production checkpoint:

multimodal_final.pth

The application downloads the model using hf_hub_download() at runtime.

This keeps the large model checkpoint out of the GitHub source repository.

⚙️ Local Setup
1. Clone the Repository
git clone <https://github.com/vk62945/Multimodal-Product-Intelligence>
cd Multimodal-Product-Intelligence
2. Create a Virtual Environment
python -m venv .venv

On macOS/Linux:

source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Application
streamlit run app/app.py

The application will be available at:

http://localhost:8501

The production model is downloaded automatically from Hugging Face Hub when the application starts.

deployment
👨‍💻 Author

Vivek Kumar

End-to-end multimodal AI engineering project demonstrating deep learning, computer vision, NLP/Transformers, multimodal fusion, explainability, production inference, model hosting, and cloud deployment.
