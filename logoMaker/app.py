import os
import tempfile
import streamlit as st

from utils import makeLogo

st.set_page_config(page_title="Logo Manipulator", 
                   page_icon=":sparkles:",
                   menu_items={
                       "Report a bug": "mailto:jamesriri03@gmail.com",
                       "About": "#### This application utilizes CV2's image processing techniques to manipulate logos with combination of other images.\
                       \n\nDeveloper Portfolio: :blue[https://ririnjaramba.onrender.com]"
                   })


st.title("Logo Manipulator: Manipulate logos with just a click. :sunglasses:")


manipulator = makeLogo()
manipulator.read_images()

if st.button('Manipulate logo'):
    manipulator.manipulate_logo()

sidebar = st.sidebar
                
