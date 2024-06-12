import os
import sys
import requests
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st
from streamlit_webrtc import VideoProcessorBase


cd = os.getcwd()

modelFile = os.path.join(cd, "models/frozen_inference_graph.pb")
configFile = os.path.join(cd, 'models/configs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
labelFile = os.path.join(cd, 'models/labels/coco_class_labels.txt')


model = cv.dnn.readNetFromTensorflow(modelFile, configFile)
with open(labelFile) as cl:
    labels = cl.read().split('\n')

DIM = 300
MEAN = (0, 0, 0)
FONTFACE = cv.FONT_HERSHEY_SIMPLEX
FONTSCALE = 0.7
THICKNESS = 2

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
    
def detect_objects(img, model=model):
    # image = cv.imread(img, cv.COLOR_RGB2BGR)
    blob = cv.dnn.blobFromImage(img, 1.0, size=(DIM, DIM), mean=MEAN, swapRB=True, crop=False)
    
    model.setInput(blob)
    
    detected_objs = model.forward() 
    
    return detected_objs

def display_text(img, text, x, y):
    # dim, baseline = cv.getTextSize(text, FONTFACE, FONTFACE, THICKNESS)
    textSize = cv.getTextSize(text, FONTFACE, FONTSCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
    cv.rectangle(img, (x, y-dim[1] - baseline), (x + dim[0], y  + baseline), (255, 255, 255), cv.FILLED)
    cv.putText(img, text, (x, y-5), FONTFACE, FONTSCALE, (0, 0, 0), 1, cv.LINE_AA)
    
def display_objects(img, objects, threshold=0.3, show=False):
    height = img.shape[0]
    width = img.shape[1]
    
    total = []
    total_dic = {}
    
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score   = float(objects[0, 0, i, 2])
        
        x = int(objects[0, 0, i, 3] * width)
        y = int(objects[0, 0, i, 4] * height)
        w = int(objects[0, 0, i, 5] * width - x)
        h = int(objects[0, 0, i, 6] * height - y)
        
        if score > threshold:
            display_text(img, "{}".format(labels[classId]).capitalize(), x, y)
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), THICKNESS) 
            
            total.append(labels[classId])
    
    for cls in total:
        if cls in total_dic:
            total_dic[cls] += 1
        else:
            total_dic[cls] = 1
        
    formatted_counts = ', '.join([f"{cls}: {count}" for cls, count in total_dic.items()])
    
    if show:
        st.text(formatted_counts)
    
    return img


def detect_video(source=s, flip=False):
    cap = cv.VideoCapture(source)
    stframe = st.empty()
    if not cap.isOpened():
        st.error("There was an error trying to open your webcam.")
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv.flip(frame, 1)
        objects = detect_objects(frame)
        identified = display_objects(frame, objects, threshold=0.55)
        
        stframe.image(identified, "Detected Objects", channels="BGR")
            
    cap.release()
    
        
