import os
import numpy as np
import cv2 as cv
from PIL import Image

default = os.path.join(os.getcwd(), 'data', 'images', 'cat.jpeg')

class enhanceImage:
    def __init__(self, mode="brightness", factor=50, low=0.8, high=1.2, thresh=100,
                 blockSize=11, c=7):
        self.mode = mode
        self.factor = factor
        self.lowContrast = low
        self.highContrast = high
        self.thresh = thresh
        self.blockSize = blockSize
        self.C = c
        
    def process(self, image=default):
        bgr = cv.imread(image, cv.IMREAD_COLOR)
        image_resize = cv.resize(bgr, (bgr.shape[1], bgr.shape[0]))
        rgb = cv.cvtColor(image_resize, cv.COLOR_BGR2RGB)
        
        if self.mode == "brightness":
            matrix = np.ones(rgb.shape, dtype='uint8') * self.factor    
            high = cv.add(image_resize, matrix)
            low = cv.subtract(image_resize, matrix)
            both = np.hstack((low, high))
            both_labeled = self.label_images(both, ['Darkened', 'Brightened'])
                
            return both_labeled
        
        elif self.mode == "contrast":
            matrix1 = np.ones(rgb.shape) * self.lowContrast
            matrix2 = np.ones(rgb.shape) * self.highContrast
            
            darker = np.uint8(cv.multiply(np.float64(rgb), matrix1))
            brighter = np.uint8(np.clip(cv.multiply(np.float64(rgb), matrix2), a_min=0, a_max=255))
            both = np.hstack((darker, brighter))
            both_labeled = self.label_images(both, ['Low Contrast', 'High Contrast'])
            
            return both_labeled           
        
        elif self.mode == "thresholding":
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            # img = img.reshape(3)
            ret, img_thresh = cv.threshold(img, thresh=self.thresh, maxval=255, type=cv.THRESH_BINARY)
            ada_thresh = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                                              thresholdType=cv.THRESH_BINARY, blockSize=self.blockSize, C=self.C)
            gauss_thresh = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                thresholdType=cv.THRESH_BINARY, blockSize=self.blockSize, C=self.C)
            num = np.hstack((img, img_thresh))
            den = np.hstack((gauss_thresh, ada_thresh))
            whole = np.vstack((num, den))
            # both_labeled = self.label_images(whole, ["Original Grayscale", f"Threshold: {self.thresh}"])
            
            return whole
                      
    def label_images(self, image, labels):
        _, width, _ = image.shape
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 1
        
        section_width = width//len(labels)
        labeled_image = image.copy()
        for i, label in enumerate(labels):
            position = (i * section_width + 10, 30)
            start = i*section_width + 10
            label_size, baseline = cv.getTextSize(label, font, font_scale, thickness)
            cv.rectangle(labeled_image, 
                         (start, 30 - label_size[1]), 
                         (start + label_size[0], 30 + baseline), 
                         (0, 0, 0), cv.FILLED)
            cv.putText(labeled_image, label, position, font, font_scale, font_color, thickness, cv.LINE_AA)
            
        return labeled_image
