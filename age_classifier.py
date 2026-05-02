import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Setup Page Configuration
st.set_page_config(page_title="Age & Gender Classifier", page_icon="👤")

# 2. Load the models with caching
# We load both models here so they stay in memory and don't reload on every click.
@st.cache_resource
def load_models():
    age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")
    gender_pipe = pipeline("image-classification", model="rizvandwiki/gender-classification-2")
    return age_pipe, gender_pipe

age_classifier, gender_classifier = load_models()

# 3. Streamlit UI Elements
st.title("👤 Age & Gender Classification")
st.write("Upload a photo to estimate both the age range and gender using Vision Transformers.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert and display the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns: Left for Image, Right for Results
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        
        with st.spinner('Analyzing image details...'):
            # --- Age Prediction Logic ---
            age_preds = age_classifier(image)
            age_preds = sorted(age_preds, key=lambda x: x['score'], reverse=True)
            top_age = age_preds[0]

            # --- Gender Prediction Logic ---
            gender_preds = gender_classifier(image)
            gender_preds = sorted(gender_preds, key=lambda x: x['score'], reverse=True)
            top_gender = gender_preds[0]

            # Display Age Result
            st.markdown("### **Age Group**")
            st.success(f"**{top_age['label']}**")
            st.caption(f"Confidence: {top_age['score']:.2%}")

            st.divider()

            # Display Gender Result
            st.markdown("### **Gender**")
            st.info(f"**{top_gender['label'].capitalize()}**")
            st.caption(f"Confidence: {top_gender['score']:.2%}")

            # Optional: Detailed Probabilities
            with st.expander("View detailed breakdown"):
                st.write("**Age Scores:**")
                st.json(age_preds)
                st.write("**Gender Scores:**")
                st.json(gender_preds)
