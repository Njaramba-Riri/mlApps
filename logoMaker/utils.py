import os
import numpy as np
import cv2 as cv
import streamlit as st

class makeLogo:
    def __init__(self, logo=None, manip=None, aspectRatio:float=None, threshold:int=100):
        self.logo = logo
        self.manip = manip
        self.aspect = aspectRatio
        self.threshold = threshold
        
    def read_images(self):
        image1 = st.file_uploader("Choose a logo to manipulate...", type=['jpeg', 'jpg', 'png'], key="image1")
        if image1 is not None:
            self.logo = cv.imdecode(np.frombuffer(image1.read(), np.uint8), 1)
            st.image(self.logo, caption="Uploaded logo to manipulate", channels="BGR", use_column_width=True)
            
        image2 = st.file_uploader("Choose an image to manipulate logo with...", type=['jpeg', 'jpg', 'png'], key='image2')
        if image2 is not None:
            self.manip = cv.imdecode(np.frombuffer(image2.read(), np.uint8), 1)
            st.image(self.manip, caption="Uploaded logo manipulator", channels="BGR", use_column_width=True)
            
    def manipulate_logo(self):
        if self.logo is not None and self.manip is not None:
            gray_logo = cv.cvtColor(self.logo, cv.COLOR_RGB2GRAY)
            gray_manip = cv.cvtColor(self.manip, cv.COLOR_BGR2GRAY)
            
            # logo_h, logo_w = self.logo.shape[0], self.logo.shape[1]
            # aspect_ratio = logo_h / self.manip.shape[1]
            # dim = (logo_w, int(self.manip.shape[0] * aspect_ratio))
            # gray_manip = cv.resize(gray_manip, (self.logo.shape[1], self.logo.shape[0]), interpolation=cv.INTER_AREA)
            # both = np.hstack((gray_logo, gray_manip))
            # st.image(both, caption="Grayscale image to manipulate", use_column_width=True)
            
            
            # height, width = gray_logo.shape
            # logo_resized = cv.resize(self.image2, (width, height), interpolation=cv.INTER_AREA)
            
            # logo_gray = cv.cvtColor(gray_logo, cv.COLOR_BGR2GRAY)
            
            _, mask = cv.threshold(gray_logo, 180, 255, cv.THRESH_BINARY_INV)
            mask_inv = cv.bitwise_not(mask)
            #mask_inv = np.invert(mask)
            
            
            height, width, channels = self.logo.shape
            area = self.manip[0 :height, 0 :width]
                        
            img1_bg = cv.bitwise_and(area, area, mask=mask_inv)
            img2_fg = cv.bitwise_and(self.logo, self.logo , mask=mask)
            
            self.result = cv.add(img1_bg, img2_fg)
            # self.manip[0: height, 0 : width] = self.result
            
            st.image(self.result, caption="Resulting manipulated logo", channels="BGR", use_column_width=True)
            st.toast("Manipulation Complete", icon=':material/thumb_up:')
