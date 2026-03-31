import streamlit as st
import torch
import torchvision
from PIL import Image
import os
import sys
from typing import List

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))
from src.inference import load_model, predict

# Hardcoded Class Names (Matching ImageFolder's alphabetical order)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Configuration
MODEL_PATH = "models/Plant disease classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# App UI
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")

st.title("🌿 Plant Disease Classifier")
st.write("Upload an image of a plant leaf to identify potential diseases.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure your trained model is in the 'models' directory.")
    st.info("The app requires the trained .pth file to perform classification.")
else:
    # Load model (cached)
    @st.cache_resource
    def cached_load_model():
        return load_model(MODEL_PATH, len(CLASS_NAMES), DEVICE)

    model = cached_load_model()
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Classify"):
            with st.spinner("Analyzing image..."):
                # Save uploaded file temporarily
                temp_path = "temp_image.jpg"
                image.save(temp_path)
                
                # Perform prediction
                pred_class, prob = predict(model, temp_path, CLASS_NAMES, DEVICE)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Display results
                display_name = pred_class.replace('___', ' ').replace('_', ' ')
                st.subheader(f"Prediction: {display_name}")
                st.write(f"Confidence: {prob:.2%}")
                
                # Visual feedback
                if "healthy" in pred_class.lower():
                    st.success(f"The {pred_class.split('___')[0]} plant appears to be healthy!")
                else:
                    st.warning(f"Potential disease detected: {display_name}")

st.sidebar.title("App Info")
st.sidebar.write("Built with PyTorch & EfficientNet-B0")
st.sidebar.write(f"Model recognizes {len(CLASS_NAMES)} plant disease categories.")
