from PIL import Image
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


from utils import enhanceImage

st.set_page_config(page_title="Image Enhancement", 
                   page_icon=":shark:",
                   menu_items={
                       "Report a bug": "mailto:jamesriri03@gmail.com",
                       "About": "This app enhances images taken in low/high resolution frames.\n\nDeveloper Portfolio: :blue[https://ririnjaramba.onrender.com]"
                   })
st.title("Image Enhancement:")
st.page_link("https://ririnjaramba.onrender.com", label=":blue-background[Developer Portfolio]", icon=":material/globe:")


sidebar = st.sidebar
sidebar.title("How do you want to enhance the image?.")

option = sidebar.radio(label="Adjust...", options=["Brightness", "Contrast", "Thresholding"])
# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
# c = (alt.Chart(chart_data)
#      .mark_circle()
#      .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"]))
# st.altair_chart(c, use_container_width=True)

# image = enhanceImage
# st.image(image, caption="Sample of Enhanced Images", channels="BGR")   
file = st.file_uploader("Upload a photo", type=["jpeg", "png", "jpg"])
if file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tfile:
        tfile.write(file.read())
        tfile.close()     
         
    if option == "Brightness":    
            factor = sidebar.slider("Factor", min_value=0, max_value=255, step=5, value=50)
            enhance = enhanceImage(factor=factor)
            image = enhance.process(tfile.name)
            st.header("Adjusting Brightness", divider="rainbow")
            st.image(image, caption="Enhanced Images", channels="BGR", use_column_width=True)
            st.image(Image.open(tfile.name), caption="Original Image")

    elif option == "Contrast":
        dark = sidebar.slider("Low Contrast", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
        bright = sidebar.slider("High Contrast", min_value=1.0, max_value=3.0, step=0.2, value=1.3)
        enhance = enhanceImage(mode="contrast", low=dark, high=bright)
        image = enhance.process(tfile.name)
        st.header("Adjusting Contrast", divider="rainbow")
        st.image(image, "Processed images", use_column_width=True)
        st.image(Image.open(tfile.name), caption="Original Image")
        
        
    elif option == "Thresholding":
        thresh = sidebar.slider("Threshold", min_value=0, max_value=255, value=100, step=5)
        block = [i for i in range(100) if i % 2 == 1 and i > 0]
        block = sidebar.slider("Block size", max_value=15, min_value=3, step=2, value=11)
        c = sidebar.slider("C", max_value=20, value=7, step=1, min_value=1)
        enhance = enhanceImage(mode="thresholding", thresh=thresh, blockSize=block, c=c)
        image = enhance.process(tfile.name)
        st.header("Thresholding", divider="rainbow")
        st.image(image, caption="Processed images", use_column_width=True)
        # st.image(Image.open(tfile.name), caption="Original Image")
