import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the pre-trained VGG16 model
model = VGG16()

# Streamlit UI
st.title("Image Classification with VGG16")
st.write("Upload an image to classify using the VGG16 model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_array)
    top_prediction = decode_predictions(predictions, top=1)[0][0]  # Get the top prediction
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Display top prediction
    imagenet_id, label, score = top_prediction
    st.write(f"### Prediction: **{label}** ({score:.4f})")
