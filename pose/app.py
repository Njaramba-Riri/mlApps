import os
from PIL import Image
import numpy as np
import cv2 as cv
import streamlit as st

from utils import detect_points, display_points

cd = os.getcwd()
image_path = os.path.join(cd, 'data/images/Tiger_Woods_crop.png')

st.set_page_config(page_title="Pose Estimator")
st.title("Pose Estimator: Draw Skeleton Masks on Image Frames.")
st.page_link("https://ririnjaramba.onrender.com", label=":blue-background[Developer Portfolio]", icon=":material/globe:")

st.text("For better results, upload images with only one person.")

file = st.file_uploader("Select an image file", type=['jpg', 'jpeg', 'png'])

if file is not None:
    try:
        image = Image.open(file)
        image = np.array(image) 
        points, detImage = detect_points(image)
        fig = display_points(detImage, points)
        st.pyplot(fig)
        st.image(image, caption="Uploaded Image", channels="RGB")
    except Exception as e:
        st.error(f"An error occured while processing the image: {e}")
else:
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    points, detImage = detect_points(image)
    fig = display_points(detImage, points)
    st.caption("A sample output.", )
    st.pyplot(fig)
    
    
    
