#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing import image as image_utils
import streamlit as st
from io import StringIO
import pickle

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def load_model():
    return keras.models.load_model('model.h5')

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def teachable_machine_classification(img):
    # Load the model
    model = load_model()
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    alphabet = ('heart','oblong','oval','round','square')
    dictionary = {}
    for i in range(5):
        dictionary[i] = alphabet[i]
    # Create the array of the right shape to feed into the keras model
    size = (224,224)
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

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def class_model(gender,typee,styls,util,mats,shape):
    alphabet = ('Ligne des sourcils', 'Oeil de chat', 'Papillon', 'Wayfarer', 'Rectangulaire', 'Ovale' ,'Verre Unique', 'Rectangulaire ovale', 'Masque', 'Aviator' ,'Carré')
    dictionary = {}
    for i in range(11):
        dictionary[i] = alphabet[i]
    #gender
    gender=gender.replace('enfant','0')
    gender=gender.replace('femme','1')
    gender=gender.replace('homme','2')
    #shape

    #typee
    typee=typee.replace('de vue','1')
    typee=typee.replace('contre soleil','0')
    #styles
    styls=styls.replace('classique','0')
    styls=styls.replace('fashion','1')
    styls=styls.replace('luxe','2')
    styls=styls.replace('professionnel','3')
    styls=styls.replace('sport','4')
    styls=styls.replace('vintage','5')
    #matiere
    mats=mats.replace('acétate','0')
    mats=mats.replace('bois','1')
    mats=mats.replace('fibres de carbonne','2')
    mats=mats.replace('metal','3')
    mats=mats.replace('plastique','4')
    mats=mats.replace('titane','5')
    #util
    util=util.replace('lecture','0')
    util=util.replace('quotidienne','1')
    util=util.replace('randonnée','2')
    util=util.replace('sortie en mer','3')
    util=util.replace('vélo','4')
    util=util.replace('voiture','5')
    
    new_input=np.array([[int(gender), int(shape),int(typee),int(styls),int(util),int(mats)]])
    pickled_model = pickle.load(open('best_model_Voting.h5', 'rb'))
    #pickled_model.predict(X_test)
    result=pickled_model.predict(new_input)
    return dictionary[result[0]]
