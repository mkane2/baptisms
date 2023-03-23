#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:50:26 2023

@author: mk192787
"""

# from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import tempfile
import numpy as np
import cv2
# import pandas as pd
import scipy.ndimage
# from collections import Counter
# import string
import os

# psm 6 for type 3 with hanging indent/hanging ditto marks
# psm 4 for neatly separated, whitespaced columns
# psm 1 for 2 separated columns
# blacklist out garbage characters that don't occur
# whitelist expected characters; include a whitespace in the list as well as
# preserve_interword_spaces so words don't run together
blacklist = "Äòûúùîò¬ß¶§√°¢"
whitelist = "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,. ()"
tesseract_config = '--psm 1 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="{}" -c tessedit_char_blacklist="{}"'.format(whitelist, blacklist)

# this_dir can contain numbered subfolders but should not contain string-named
# folders. this_dir doesn't need to contain subfolders
files = []
this_dir = "../../pdfs/type_1/2cols"

# expects pdfs to be structed in numbered subfolders, numbered by expected_cols
for subdir, dirs, items in os.walk(this_dir):
    for file in items:
        # print(subdir, dirs, file)
        if file.endswith(".pdf"):
            item = {"filename": file}
            files.append(item)
            # print(item)

# convert the pdf to a collection of images with the specified dpi            
def convert_pdf(pdf_doc, dpi):
    images = []
    images.extend(
                    list(
                            map(
                                lambda image: cv2.cvtColor(
                                    np.asarray(image), code=cv2.COLOR_RGB2BGR
                                    ),
                                convert_from_path(pdf_doc, dpi=dpi),
                                )
                            )
                    )
    return images

# detects best rotation and returns a de-skewed image
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = scipy.ndimage.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
          borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

for f in files:
    print(f['filename'])
    file_path = this_dir + "/" + f['filename']
    slug = f['filename'].replace(".pdf","")
    
    out_lst = []

    # get temporary directory of Image objects
    with tempfile.TemporaryDirectory() as path:
        print("Converting pages . . . ")
        images = convert_pdf(file_path, 300)
        
        n = 1
        for i in images:
            print(f'Processing page {n}...')
    
            try:
                ## correct the image
                # correct the image skew
                im = correct_skew(i)[1]
                
                # create a grayscale image
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                
                # lightly blur the image to reduce noise. Mess with the values 
                # in the tuple to change results
                blur = cv2.GaussianBlur(gray, (7, 7), 1)
                
                # run an adaptive threshold. Messing with the last number can 
                # change the results a lot
                # even number for 2nd number for adaptive threshholding for 
                # sampling area
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 14)
    
                # ocr the corrected image
                # nld for Dutch, deu for German
                txt = pytesseract.image_to_string(thresh, config=tesseract_config, lang="deu+nld+eng")
                out_lst.append(txt)
                
            except ValueError:
                print(f"Page {n} can't be processed.")
                newrow = ["Page {n} can't be processed."]
                out_lst.append(newrow)

            n += 1
        
        # target subdir should exist if specified, it won't be created if it 
        # doesn't exist
        outfile = "type_1_output/" + slug + ".txt"
        with open(outfile, 'a') as file:
            file.writelines(out_lst)