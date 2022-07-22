import threading
from typing import Union
import av
from streamlit import caching

import requests
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import os 
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import streamlit as st
import requests
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
st.set_page_config(page_title="KatYos",page_icon="images\logo.png",layout="wide")
from io import StringIO
from model import teachable_machine_classification
from model import class_model

from image_detec import object_detection_image
from image_detec import object_detection_video

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as image_utils
from PIL import Image

label="1"
def main():
    global label
    genre = ['homme' ,'femme', 'enfant']
    formes_de_visage = ['rond' ,'Long' ,'carré' ,'Ovale', 'cœur']
    types = ['de vue' ,'contre soleil']
    styles = ['fashion', 'professionnel', 'luxe', 'classique' ,'sport' ,'vintage']
    utilisations = ['sortie en mer', 'voiture', 'quotidienne', 'lecture', 'randonnée', 'vélo']
    matiere = ['acétate' ,'plastique', 'fibres de carbonne', 'bois','titane', 'metal']
    formes_de_montures = ['ligne de sourcil', 'œil de chat', 'papillon', 'wayfarer', 'rectangulaire', 'ovale' ,'verre unique','rectangulaire ovale', 'masque', 'aviator' ,'carré']
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

    holistic=mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  
        out_image: Union[np.ndarray, None]
        
        def __init__(self) -> None:
            self.frame_lock = threading.Lock()

            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            out_image = frame.to_ndarray(format="bgr24")
            out_image = process(out_image)
            with self.frame_lock:

                self.out_image = out_image
            return out_image

    def process(image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
        return cv2.flip(image, 1)

    flag=1
    def load_lottie1(url):
        r=requests.get(url)
        if r.status_code!=200:
            return None
        return r.json()

    person=load_lottie1("https://assets4.lottiefiles.com/packages/lf20_2cbmucbb.json")
    
    with st.container():
        image_column,test_column=st.columns((1,2))
    with image_column:
        st_lottie(person,height=200,key="person")
    with test_column:
        st.title("Welcome to KatYos Virtual Assistant :wave:")
        st.subheader("I will help you choose your perfect Eye-Frames")
    st.write("---")

    
    with st.container():
        
        left,right=st.columns(2)
        with left:
            st.title("↓ Let's start")
            ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer,rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            if ctx.video_transformer:
                snap = st.button("Snapshot")
                if snap:
                    
                    with ctx.video_transformer.frame_lock:
                        out_image = ctx.video_transformer.out_image
                        
                    if out_image is not None:
                        my_path = os.path.abspath(os.path.dirname(__file__))              
                        label=object_detection_video(out_image)
                        st.image(out_image, channels="BGR")
                        st.subheader("Your face shape is "+label.title())         
                    else:
                        st.warning("No frames available yet.")

        with right:
            st.title("Please select some additional options")
            gender=st.selectbox('Stereotype',genre)
            typee=st.selectbox('Type',types)
            styls=st.selectbox('Style',styles)
            util=st.selectbox('Utilisations',utilisations)
            mats=st.selectbox('Matiere',matiere)
            label=label.replace('heart','1')
            label=label.replace('oblong','2')
            label=label.replace('oval','3')
            label=label.replace('round','4')
            label=label.replace('square','0')
            x=int(label)
            if st.button('Next'):
                lunette=class_model(gender,typee,styls,util,mats,x)
                st.subheader("Seems like "+lunette.title()+" fits you well")
                st.subheader("Please click below to redirect you to our shop")
                if lunette=="Ligne des sourcils":
                    lunette=" Ligne des sourcils"
                lunette=lunette.replace(" ","+")
                st.subheader("https://katyos2.katyos.com/3-lunettes-de-vue?q=Formes-"+lunette)
                class_model.clear()
                object_detection_video.clear()
                del ctx
                
if __name__ == "__main__":
    
    main()
#caching.clear_cache()
