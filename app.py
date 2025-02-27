import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("potato_leaf_model.h5")
class_names = ["Early Blight", "Healthy", "Late Blight"]

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🍃 Potato Leaf Disease Detection")

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        result = class_names[class_idx]

        st.subheader(f"🟢 Prediction: {result}")

        # Display confidence scores
        for i in range(len(class_names)):
            st.write(f"🔹 {class_names[i]}: {predictions[0][i]*100:.2f}%")