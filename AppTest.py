import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Set app title
st.set_page_config(page_title="Image Caption Generator")
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image, and the AI will describe it.")

# Load model once (caching helps with performance)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("Generating caption..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.markdown("### 📝 Generated Caption:")
    st.success(caption)
