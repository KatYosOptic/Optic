import streamlit as st
import requests
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
st.set_page_config(page_title="KatYos2",page_icon=":tada:",layout="wide")

from model import teachable_machine_classification
from model import class_model

from image_detec import object_detection_image
from image_detec import object_detection_video

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as image_utils
from PIL import Image

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


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
def load_lottie1(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

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

    
#using css



local_css("style/style.css")
flag=0


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
            

