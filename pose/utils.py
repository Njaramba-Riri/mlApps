import os
import requests
import tempfile
import matplotlib.pyplot as plt
import cv2 as cv

cd = os.getcwd()
proto = os.path.join(cd, 'models/configs/pose_deploy_linevec_faster_4_stages.prototxt')
caffe = os.path.join(cd, 'models/pose_iter_160000.caffemodel')


tPoints = 15
dim = 368
mean = (0, 0, 0)
POSE_PAIRS = [[0, 1], [1, 2], [2, 3],
              [3, 4], [1, 5], [5, 6],
              [6, 7], [1, 14], [14, 9],
              [9, 10],[14, 11], [11, 12],
              [12, 13]]

def load_model(modelFile=caffe, protoFile=proto):
    model = cv.dnn.readNetFromCaffe(prototxt=protoFile,
                                    caffeModel=modelFile)    
    return model

def detect_points(image, thresh=0.2):
    img_width = image.shape[1]
    img_height = image.shape[0]
    
    model = load_model()    
    
    blob = cv.dnn.blobFromImage(image, 1/255, (dim, dim), mean=mean, swapRB=True, crop=False)
 
    model.setInput(blob)
    output = model.forward()
    
    scaleY = img_height / output.shape[2]
    scaleX = img_width / output.shape[3]
    
    points = []
    
    for i in range(tPoints):
        probMap = output[0, i, :, :]
        minval, prob, minLoc, point = cv.minMaxLoc(probMap)
        x = scaleX * point[0]
        y = scaleY * point[1]
        
        if prob > thresh:
            points.append((int(x), int(y)))
        else:
            points.append(None)
        
    return points, image
    
def display_points(img, points: list):
    imPoints = img.copy()
    imSkeleton = img.copy()
    
    for i, p in enumerate(points):
        cv.circle(imPoints, p, 8, (255, 255, 0), -1, cv.FILLED)
        cv.putText(imPoints, "{}".format(i), p, cv.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 1, cv.LINE_AA)
        
    for pair in POSE_PAIRS:
        A = pair[0]
        B = pair[1]
        
        if points[A] and points[B]:
            cv.line(imSkeleton, points[A], points[B], (255, 255, 0), 2)
            cv.circle(imSkeleton, points[A], 8, color=(255, 0, 0), thickness=-1, lineType=cv.FILLED)
        
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(121)
        ax.set_yticks([]);ax.set_xticks([]);
        ax.imshow(imPoints);ax.set_title("Detected Points.", size=15, weight='bold')
        ax = fig.add_subplot(122)
        ax.set_yticks([]);ax.set_xticks([]);
        ax.imshow(imSkeleton);ax.set_title("Skeleton Mask.", size=15, weight='bold')       
        
    return fig
