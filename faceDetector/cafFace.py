import os
import sys
import numpy as np
import cv2 as cv
import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase

cd = os.getcwd()
model = os.path.join(cd, "models/res10_300x300_ssd_iter_140000.caffemodel")
proto = os.path.join(cd, "models/configs/deploy.prototxt")

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]


model = cv.dnn.readNetFromCaffe(prototxt=proto,
                                caffeModel=model)

dim = 300
mean = (104.0, 177.0, 123.0)
conf_thresh = 0.2

class VideoTransformer(VideoProcessorBase):
    def __init__(self, thresh=0.5):
        self.conf_thresh = thresh
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        frame_height, frame_width = img.shape[:2]

        blob = cv.dnn.blobFromImage(img, 0.5, (dim, dim), mean, swapRB=False, crop=False)
        model.setInput(blob)
        detections = model.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_thresh:
                x_bottom_left = int(detections[0, 0, i, 3] * frame_width)
                y_bottom_left = int(detections[0, 0, i, 4] * frame_height)
                x_top_right = int(detections[0, 0, i, 5] * frame_width)
                y_top_right = int(detections[0, 0, i, 6] * frame_height)

                cv.rectangle(img, (x_bottom_left, y_bottom_left), (x_top_right, y_top_right), (255, 117, 234), 2)

                label = "Confidence: {:.2f}%".format(confidence * 100)
                label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .5, 2)
                cv.rectangle(img, (x_bottom_left, y_bottom_left - label_size[1]),
                             (x_bottom_left + label_size[0], y_bottom_left + baseline), (255, 255, 243), cv.FILLED)
                cv.putText(img, label, (x_bottom_left, y_bottom_left), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        
        t, _ = model.getPerfProfile()
        label = "Inference time: %.2f ms" % (t * 1000 / cv.getTickFrequency())
        cv.putText(img, label, (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def detect_faces(img):
    image = np.array(img)
    h, w = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (dim, dim)), 1.0, (dim, dim), mean, swapRB=False, crop=False)
    
    model.setInput(blob)
    detections = model.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, endX, startY, endY) = box.astype("int")
            cv.rectangle(image, (startX - 10, endX), (startY, endY + 20), (0, 255, 0), 2)
            
            label = "Confidence: {:.2f}%".format( confidence * 100)
            label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv.rectangle(image, (startX - 10, endX - label_size[1]),
                        (startX + label_size[0], endX + baseline), (255, 255, 243), cv.FILLED)
            cv.putText(image, label, (startX, endX), cv.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 1)
    return image

def process_frame(video_frame, model=model, threshold=0.5):
    frame = cv.flip(video_frame, 1)
    frame_height, frame_width = frame.shape[:2]
    
    blob = cv.dnn.blobFromImage(frame, 0.5, (dim, dim), mean, swapRB=False, crop=False)
    
    model.setInput(blob=blob)
    detections = model.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            x_bottom_left = int(detections[0, 0, i, 3] * frame_width)
            y_bottom_left = int(detections[0, 0, i, 4] * frame_height)
            x_top_right = int(detections[0, 0, i, 5] * frame_width)
            y_top_right = int(detections[0, 0, i, 6] * frame_height)
            
            cv.rectangle(frame, (x_bottom_left, y_bottom_left), (x_top_right, y_top_right), (255, 117, 234), 2)
            
            label = "Confidence: {:.2f}%".format(confidence * 100)
            label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv.rectangle(frame, (x_bottom_left, y_bottom_left - label_size[1]),
                            (x_bottom_left + label_size[0], y_bottom_left + baseline), (255, 255, 243), cv.FILLED)
            cv.putText(frame, label, (x_bottom_left, y_bottom_left), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    
    t, _ = model.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000 / cv.getTickFrequency())
    cv.putText(frame, label, (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255))
    
    return frame

def real_timeDetection(source=s, conf_thresh=0.5):
    cap = cv.VideoCapture(source)
    stframe = st.empty()
    
    if not cap.isOpened():
        st.error("There is a problem while trying to access the video file.")   
        
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = process_frame(frame, threshold=conf_thresh)
        stframe.image(processed, channels="BGR", use_column_width=True)

    cap.release()
