import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("./flower_model.h5") 
    return model

model = load_model()
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip'] 

def preprocess_image(img):
    img = img.resize((224, 224))  
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0
    return img_array

st.title("Flower Classification App")
st.write("Upload a flower image and get its prediction!")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image",  use_container_width=True)
    
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}%")
