import os
import sys
import requests
import zipfile
import av
import cv2 as cv
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import VideoProcessorBase

cd = os.getcwd()
modelFile = os.path.join(cd, "models/frozen_inference_graph.pb")
configFile = os.path.join(cd, 'models/configs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
labelFile = os.path.join(cd, 'models/configs/labels/coco_class_labels.txt')
model_url = 'https://github.com/ChiekoN/OpenCV_SSD_MobileNet/raw/master/model/frozen_inference_graph.pb'

def download_and_unzip(url, extract_to="."):
    # Download the zip file
    response = requests.get(url)
    zip_path = os.path.join(extract_to, "model.zip")
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(zip_path)


def load_model_from_url(url):
    response = requests.get(url)
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(response.content)
    
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

url = 'https://github.com/yourusername/yourrepository/raw/master/model.zip'
# download_and_unzip(url, extract_to="models")

# modelFile = load_model_from_url(model_url)

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
    

class VideoTransformer(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        frame_height, frame_width = img.shape[:2]

        blob = cv.dnn.blobFromImage(img, 0.5, (DIM, DIM), MEAN, swapRB=False, crop=False)
        model.setInput(blob)
        detections = model.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                classId = int(detections[0, 0, i, 1])
                x_bottom_left = int(detections[0, 0, i, 3] * frame_width)
                y_bottom_left = int(detections[0, 0, i, 4] * frame_height)
                x_top_right = int(detections[0, 0, i, 5] * frame_width)
                y_top_right = int(detections[0, 0, i, 6] * frame_height)

                cv.rectangle(img, (x_bottom_left, y_bottom_left), (x_top_right, y_top_right), (255, 117, 234), 2)

                label = "{}: {:.2f}%".format(labels[classId].capitalize(), confidence * 100)
                label_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .5, 2)
                cv.rectangle(img, (x_bottom_left, y_bottom_left - label_size[1]),
                             (x_bottom_left + label_size[0], y_bottom_left + baseline), (255, 255, 243), cv.FILLED)
                cv.putText(img, label, (x_bottom_left, y_bottom_left), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)

        t, _ = model.getPerfProfile()
        label = "Inference time: %.2f ms" % (t * 1000 / cv.getTickFrequency())
        cv.putText(img, label, (5, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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
        
        stframe.image(identified, caption="Detected Objects", channels="BGR")
            
    cap.release()
    
        
