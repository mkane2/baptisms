#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:24:27 2023

@author: mk192787
"""

import cv2
# import numpy as np
# import tempfile
# from pdf2image import convert_from_path

# read image. the image should be sized up to whatever dpi is being used to
# to convert pdfs to images in the ocr process, otherwise the coords and 
# bounding boxes will be wrong
img = cv2.imread("deerpark1.png")

# create a grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# lightly blur the image to reduce noise. Mess with the values in the tuple to change results
blur = cv2.GaussianBlur(gray, (7, 7), 1)

# run an adaptive threshold. Messing with the last number can change the results a lot
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 14) # even number for 2nd number for adaptive threshholding for sampling area

## many small bounding boxes from small kernel sizes will result in too
## many column boundaries being found. too many dilation iterations will result
## in only large, full-page boxes being found and so no columns detected
# set kernel size - fudge numbers to capture bigger or smaller regions
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
# dilate image - create large blobs of text to detect clusters
dilation = cv2.dilate(thresh, rect_kernel, iterations=2)
# standard contouring method
contours = cv2.findContours(dilation, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_NONE)[0]

# get contours
result = img.copy()
# contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x,y,w,h = cv2.boundingRect(cntr)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    print("x,y,w,h:",x,y,w,h)
 
# save resulting image
cv2.imwrite('result.jpg',result)      
