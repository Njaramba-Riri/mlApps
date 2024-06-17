import av
import numpy as np
import cv2 as cv
import streamlit as st
from streamlit_webrtc import VideoProcessorBase

class VideoProcessor(VideoProcessorBase):
    def __init__(self, mode="default", threshold=5, flip=0, threhold1=100, threshold2=200, 
                 blur=(13, 13), blocksize=9, maxCorners=500, qualityLevel=0.2, minDistance=15.0,
                 height=300, width=300):
        self.mode  = mode
        self.threshold = threshold
        self.flip =  flip
        self.threshold1 = threhold1
        self.threshold2 =  threshold2
        self.blur =  blur 
        self.blockSize =  blocksize
        self.maxCorners =  maxCorners
        self.qualityLevel =  qualityLevel
        self.minDistance =  minDistance
        self.height =  height
        self.width = width 
        
    def recv(self, img_frame: av.VideoFrame) -> av.VideoFrame:
        frame = img_frame.to_ndarray(format="bgr24")
        result = frame
        
        if self.mode == "default":
            result = frame
        elif self.mode == "canny":
            result = cv.cvtColor(
                cv.Canny(frame, threhold1=self.threshold1, threshold2=self.threshold2), cv.COLOR_BGR2GRAY
            )
        elif self.mode == "blur":
            result = cv.blur(frame, ksize=self.blur)
        elif self.mode == "flip":
            result = cv.flip(frame, flipCode=self.flip)
        elif self.mode == "gray":
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, frame = cv.threshold(gray, thresh=self.threshold, maxval=255, type=cv.THRESH_BINARY)
            result = frame
        elif self.mode == "features":
            result = frame
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners = cv.goodFeaturesToTrack(gray, maxCorners=self.maxCorners, qualityLevel=self.qualityLevel,
                                             minDistance=self.minDistance, blockSize=self.blockSize)
            if corners is not None:
                for x, y in np.float32(corners).reshape(-1, 2):
                    cv.circle(result, center=(int(x), int(y)), radius=10, color=(0, 255, 0), thickness=1)
        elif self.mode == "crop":
            result = cv.resize(frame, dsize=(self.height, self.width))
        
        return av.VideoFrame.from_ndarray(result, format="bgr24")


def callback(frame):
    img = frame.to_ndarray("bgr24")
    img = cv.cvtColor(cv.Canny(img, threshold1=130, threshold2=200), cv.COLOR_GRAY2BGR)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        
