import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

# Function to load the model and predict the class of the uploaded image
def load_and_predict(model_path, image_obj, class_names=None):
    """
    Load a trained model and predict the class of a given image object.

    Parameters:
    - model_path: Path to the saved .keras model file
    - image_obj: PIL Image object
    - class_names: Optional list of class names (if not provided, will use indices)

    Returns:
    - Predicted class name or index
    - Confidence score
    """
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    img_array = image.img_to_array(image_obj.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class_index]
    
    # Convert to class name if class names are provided
    if class_names:
        predicted_class = class_names[predicted_class_index]
    else:
        predicted_class = predicted_class_index
    
    return predicted_class, confidence_score


def get_class_names(train_dir):
    """
    Retrieve class names from the training directory.

    Parameters:
    - train_dir: Path to the training directory

    Returns:
    - List of class names
    """
    # return sorted(os.listdir(train_dir))
    return ["fake", "real"]


# Streamlit app
st.title("Image Class Prediction")
st.write("Upload an image to predict its class.")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image_obj = Image.open(uploaded_image)
    st.image(image_obj, caption="Uploaded Image", use_column_width=True)
    
    # Paths (modify these as needed)
    model_path = "final_densenet_model.h5"  # Path to your saved model
    train_dir = 'dataset3'  # Path to the training directory used during model creation

    # Get class names
    class_names = get_class_names(train_dir)

    # Predict
    st.write("Predicting...")
    predicted_class, confidence = load_and_predict(
        model_path,
        image_obj,
        class_names
    )

    # Display results
    if predicted_class == "real":
        st.success(f"**Predicted Class:** {predicted_class}")
    else:
        st.error(f"**Predicted Class:** {predicted_class}")
        
    st.write(f"**Confidence Score:** {confidence * 100:.2f}%")
