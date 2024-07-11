import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model('model_mobilenet.hdf5')
  return model
model = load_model()

file=st.file_uploader("Upload an image", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size= (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape= img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
  st.text("Please upload an image file")
else:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  predictions=import_and_predict(image,model)
  output=""
  if (predictions>0.5):
    output="Dog"
  else:
    output="Cat"
  st.write("Your image is a "+output)
