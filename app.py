import streamlit as st
import requests
import numpy as np
import threading
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
st.set_page_config(page_title="KatYos2",page_icon=":tada:",layout="wide")
from typing import Union
import av
from model import teachable_machine_classification
from model import class_model
import mediapipe as mp
from image_detec import object_detection_image
from image_detec import object_detection_video

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as image_utils
from PIL import Image
import os 
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

       
        
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
def load_lottie1(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()
glasses=load_lottie1("https://assets4.lottiefiles.com/packages/lf20_i0ZzAQ.json")
person=load_lottie1("https://assets4.lottiefiles.com/packages/lf20_2cbmucbb.json")
logo=Image.open("images/logo.png")


genre = ['homme' ,'femme', 'enfant']
formes_de_visage = ['rond' ,'Long' ,'carré' ,'Ovale', 'cœur']
types = ['de vue' ,'contre soleil']
styles = ['fashion', 'professionnel', 'luxe', 'classique' ,'sport' ,'vintage']
utilisations = ['sortie en mer', 'voiture', 'quotidienne', 'lecture', 'randonnée', 'vélo']
matiere = ['acétate' ,'plastique', 'fibres de carbonne', 'bois','titane', 'metal']
formes_de_montures = ['ligne de sourcil', 'œil de chat', 'papillon', 'wayfarer', 'rectangulaire', 'ovale' ,'verre unique', 'rectangulaire ovale', 'masque', 'aviator' ,'carré']

glasses=load_lottie1("https://assets4.lottiefiles.com/packages/lf20_i0ZzAQ.json")
person=load_lottie1("https://assets4.lottiefiles.com/packages/lf20_2cbmucbb.json")
logo=Image.open("images/logo.png")

#navigation bar
with st.sidebar:
    selected=option_menu(menu_title=None,options=["Image","Video"],icons=["house-door-fill","camera-fill","camera-reels-fill"],
                    )
    
#using css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



local_css("style/style.css")
flag=0
#nheader

        
#body

if selected=="Image":
    
    with st.container():
        image_column,test_column=st.columns((1,2))
    with image_column:
        st_lottie(person,height=200,key="person")
    with test_column:
        st.title("Welcome to KatYos Virtual Assistant :wave:")
        st.subheader("I will help you choose your perfect Eye-Frames")
    st.write("---")

    with st.container():
        if flag==0:
            left,right=st.columns(2)
            with left:
                st.title("Please Upload a selfie ")
                holder = st.empty()
                uploaded_file = holder.file_uploader('', type="jpg")

            with right:
                st.empty()
        left,right=st.columns(2)
        with left:
            if uploaded_file is not None:
                label,image,image2=object_detection_image(uploaded_file)
                st.image(image2,width =400)
                st.subheader("Your face shape is "+label.title())
                flag=1
                holder.empty()
        if flag==1:
            with right:
                st.title("Please select some additional options")
                gender=st.selectbox('Stereotype',genre)
                typee=st.selectbox('Type',types)
                styls=st.selectbox('Style',styles)
                util=st.selectbox('Utilisations',utilisations)
                mats=st.selectbox('Matiere',matiere)
                if st.button('Next'):
                    lunette=class_model(gender,typee,styls,util,mats,label)
                    st.subheader("Seems like "+lunette.title()+" fits you well")
                    st.subheader("Please click below to redirect you to our shop")
                    lunette=lunette.title()
                    lunette=lunette.replace(" ","+")
                    st.subheader("https://katyos2.katyos.com/3-lunettes-de-vue?q=Formes-"+lunette.title())

                    #st.write('Predict') #displayed when the button is clicked



            
if selected=="Video":
 
    with st.container():
        image_column,test_column=st.columns((1,2))


    with st.container():
        if flag==0:
            left,right=st.columns(2)
            with left: 
                st_lottie(glasses,height=200,key="glasses")
            with right:
                st.title("Coming Soon")
        


            
            
            
            
            
            
            
            
            
            
            
            
            
