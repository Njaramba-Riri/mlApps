import time
import cv2 as cv
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from utils import VideoProcessor, callback

st.set_page_config("Picture Perfect", layout="wide",
                   menu_items={
                       'Report a bug': 'mailto:jamesriri03@gmail.com',
                       'About': "This app shows various and possible transformations that can be done on image frames to\
                           enhance model generazability or be applied in image filters.\n\n- Made by :red[Riri].\n\n- :blue[https://ririnjaramba.onrender.com]"
                   })

st.title("Image Transformations: Cool :blue[OpenCV's] :camera: Filters")
st.page_link("https://ririnjaramba.onrender.com", label=":blue-background[Developer Portfolio]", icon=":material/globe:")
st.text("To enhance your computer vision models some transformations become a necessity...")
st.text("This app demonstrates some of these.")

sidebar = st.sidebar

sidebar.header("How do you want to trasform your frames?.")
options = sidebar.radio("Select Image Filter", ("`Default`", "`Canny`", "`Blur`", "`Features`", "`Flip`", "`GrayScale`", "`Crop`"), horizontal=True)

processor = VideoProcessor()

ctx = webrtc_streamer(
        key="Image Filters",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio":False},
        video_processor_factory=lambda: processor,
        rtc_configuration={
            'iceServers': [{
                'urls': ['stun:stun.1.google.com:19302']
            }]
        },
        async_processing=True
    )

if options == "`Default`":
    st.toast("You are now in `Default` mode :camera:")
    if ctx.video_processor:
        ctx.video_processor.mode = "default"

elif options == "`Canny`":
    st.toast("You are now in `Canny` mode :camera:")
    if ctx.video_processor:
        thresh1 = sidebar.slider("Threshold 1", min_value=0, max_value=255, step=10, value=30)
        thresh2 = sidebar.slider("Threshold 2", min_value=0, max_value=255, step=10, value=100)
        ctx.video_processor.mode = "canny"
        ctx.video_processor.threshold1 = thresh1
        ctx.video_processor.threshold2 = thresh2

elif options == "`Blur`":
    st.toast("You are now in `Blur` mode :camera:")
    if ctx.video_processor:
        blur_size = sidebar.slider("Blur size", 1, 25, (13, 13))        
        ctx.video_processor.mode = "blur"
        ctx.video_processor.blur = (blur_size, blur_size)

elif options == "`Features`":
    st.toast("You are in `Features` mode :camera:")
    if ctx.video_processor:
        maxCorners = sidebar.slider("Max Corners", min_value=100, max_value=1000, step=20, value=500)
        qltLevel = sidebar.slider('Quality Level', min_value=0.0, max_value=1.0, step=0.1, value=0.2)
        minDistance = sidebar.slider('Min Distance', min_value=0.0, max_value=100.0, step=5.0, value=15.0)
        block = sidebar.slider('Block size', min_value=1, max_value=20, step=1, value=10)
        ctx.video_processor.mode = "features"
        ctx.video_processor.maxCorners = maxCorners
        ctx.video_processor.qualityLevel = qltLevel
        ctx.video_processor.minDistance = minDistance
        ctx.video_processor.blockSize = block
        
elif options == "`Flip`":
    st.toast("You are in `Flip` mode :camera:")
    if ctx.video_processor:
        mode = sidebar.radio("Select flip direction", options=("Vertical", "Horizontal"), horizontal=True)
        ctx.video_processor.mode = "flip"
        if mode == "Vertical":
            ctx.video_processor.flip = 0
        elif mode == "Horizontal":
            ctx.video_processor.flip = 1

elif options == "`GrayScale`":
    st.toast("You are `Grayscale` mode :camera:")
    if ctx.video_processor:
        thresh = sidebar.slider("Threshold", min_value=1, max_value=255, step=5, value=10)
        ctx.video_processor.mode = "gray"
        ctx.video_processor.threshold = thresh

elif options == "`Crop`":
    st.toast("You are in `Crop` mode :camera:")
    if ctx.video_processor:
        height = sidebar.slider("Image height", min_value=0, max_value=1000, step=100, value=300)
        width = sidebar.slider("Image width", min_value=0, max_value=1000, step=100, value=300)
        ctx.video_processor.mode = "crop"
        ctx.video_processor.height = height
        ctx.video_processor.width = width
    
