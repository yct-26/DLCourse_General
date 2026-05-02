import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Setup Page Configuration
st.set_page_config(page_title="Age Classifier", page_icon="🎂")

# 2. Load the model with caching
# This ensures the model only loads once, making the app much faster.
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

age_classifier = load_classifier()

# 3. Streamlit UI Elements
st.title("🎂 Age Classification using ViT")
st.write("Upload a photo of a face to estimate the age range using a Vision Transformer (ViT).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert and display the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner('Analyzing age...'):
            # Run classification
            age_predictions = age_classifier(image)
            
            # Sort predictions by score
            age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

            # Display the top result
            top_label = age_predictions[0]['label']
            top_score = age_predictions[0]['score']

            st.subheader("Prediction Results")
            st.success(f"**Predicted Age Range: {top_label}**")
            st.info(f"Confidence: {top_score:.2%}")

            # Optional: Show other possibilities
            with st.expander("See all probability scores"):
                for pred in age_predictions:
                    st.write(f"{pred['label']}: {pred['score']:.4f}")
