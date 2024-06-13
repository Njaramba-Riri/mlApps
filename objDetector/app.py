import os
from PIL import Image
import numpy as np
import streamlit as st
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from utils import detect_objects, display_objects, detect_video, VideoTransformer

cd = os.getcwd()
default = os.path.join(cd, 'data/videos/faces.mp4')
file = os.path.join(cd, 'data/images/street.jpg')

st.set_page_config(page_title="Object Detector", page_icon=":smiley:")

st.title("Object Detector: Identify objects in image or video frames.")
st.page_link("https://ririnjaramba.onrender.com", label=":blue-background[Developer Portfolio]", icon=":material/globe:")
st.text("This model is capable of detecting 92 different classes.")

option = st.selectbox("Select a file type to upload", ("--------------------", "Image", "Video", "Camera"))

if option == "--------------------":    
    image = Image.open(file)
    image = np.array(image)
    objects = detect_objects(image)
    drawn = display_objects(image, objects, threshold=0.25, show=True)
    st.image(drawn, "Detected Objects")
    
    if st.button('Show video sample'):
        detect_video(default)
        # st.rerun()

if option == "Image":
    file = st.file_uploader("Select a file", type=['jpg', 'jpeg', 'png'])
    if file is not None:
        try:
            image = Image.open(file)
            image = np.array(image)
            objects = detect_objects(image)
            drawn = display_objects(image, objects, show=True)
            st.image(drawn, "Detected Objects")
        except Exception as e:
            st.error(f"An error occured: {e}")

elif option == "Video":
    file = st.file_uploader("Select a video file", type=['mp4', 'mpeg', 'avi', 'webm'])
    if file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        tfile.close()
        try:
            detect_video(tfile.name)
        except Exception as e:
            st.error("Error while reading the video: {}".format(e))
            
        if st.button('Re-run'):
            st.rerun()
        
elif option == "Camera":
    st.write("Click 'Start' to open the webcam and perform live face detection.")
    webrtc_ctx = webrtc_streamer(
        key="live-face-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    # detect_video(flip=True)
