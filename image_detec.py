import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np
from model import teachable_machine_classification
import time
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im
from io import StringIO
from tensorflow.keras.preprocessing import image as image_utils
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def object_detection_image(uploaded_file):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
    holistic=mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete)

    label = teachable_machine_classification(image)

    image2 = np.array(image)
    
        #st.write("Your face shape is ",label)

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        results = holistic.process(image2)
    mp_drawing.draw_landmarks(image2, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
 
    return label,image,image2
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def object_detection_video(img):
    model = keras.models.load_model('model.h5')
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    alphabet = ('heart','oblong','oval','round','square')
    dictionary = {}
    for i in range(5):
        dictionary[i] = alphabet[i]
    # Create the array of the right shape to feed into the keras model
    size = (224,224)
    img = Image.fromarray(img)
    image = ImageOps.fit(img, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
    image_array = np.asarray(image)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = image_array
    prediction = model.predict(data)
    
    predicted_shape = dictionary[np.argmax(prediction)]
    # run the inference
    return predicted_shape # return position of the highest probability
    
    
    