#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:52:04 2023

@author: mk192787
"""

import camelot
# from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import tempfile
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage
# from collections import Counter
# import string
import os

# slug = "ny-schenectady-drc-3"
# file_path = 'type_2/trimmed_pdfs/' + slug + '.pdf'

# pdfs to be processed should be run through remove_blanks to remove blank pagess,
# plus hand check for blank pages. pdfs should be structured into type folders:
# tye_1 for weird, bad, and irregular
# type_2 for columnized with whitespace -> further separated into subfolders by
# expected number of columns
# type_3 for prose block formatting. type_3_ditto for type_3 with hanging dittos
# or hanging indent; run type_3 through type3ocr with psm 6.  pages with prose 
# blocks in two columns can be run through type3ocr at psm 1.

# --user-words file seems like it should be able to live anywhere but pytesseract
# docs have it in /tessdata as eng.user-words
# preserve_interword_whitespace should preserve spaces between words when a 
# whitelist is passed but doesn't actually seem to do that; whitespace needs to
# be included in the whitelist.
# blacklist shouldn't technically be necessary but whitelist alone was still
# outputting diacritics and special characters
# "/Users/mk192787/opt/miniconda3/envs/spyder-env/share/tessdata/eng.user-patterns"
# "/Users/mk192787/opt/miniconda3/envs/spyder-env/share/tessdata/eng.user-words"
# --user-words eng.user-words 
blacklist = "Äòûúùîò¬ß¶§√°¢"
whitelist = "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,. "
tesseract_config = '--psm 4 bazaar -c preserve_interword_spaces=1 -c tessedit_char_whitelist="{}" -c tessedit_char_blacklist="{}"'.format(whitelist, blacklist)
# char_margin defaults to 2 but when not specified will still order words incorrectly; 
# without specifying char_margin, words will appear out of order within a cell
# larger char_margin will group words from a row together in a single cell
layout_dict = {'char_margin': 2}

files = []
this_dir = "../../pdfs/type_2/trimmed_pdfs/"

# expects pdfs to be structed in numbered subfolders, numbered by expected_cols
for subdir, dirs, items in os.walk(this_dir):
    for file in items:
        # print(subdir, dirs, file)
        if file.endswith(".pdf"):
            item = {"filename": file, "expected_cols": subdir[-1]}
            files.append(item)

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

# sort a list of x coordinates from smallest to largest
def sort_coords(value):
    return sorted(value.split(","), key=float)

# group the input list so that each element has no more than maxgap
# between each element; there is no control over the number of groups returned
# data is expected to be a list of ints; small maxgap will produce more smaller
# groups, large maxgap will produce fewer bigger groups
def cluster(data, maxgap):
    data = sorted(data)
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    groups = sorted(groups, key=len)
    # print(data)
    # print(groups)
    return groups

# get the n largest subgroups in a clustered list; clustered is produced by cluster()
def find_cols(clustered, expected_cols):
    cols = sorted(clustered, key=len)[-expected_cols:]
    minlist = sorted([item[0] for item in cols])
    # remove first item because we only want locations of separators between
    # columns, not the location of the start of the first column
    minlist.pop(0)
    return minlist

# detects best rotation and returns a de-skewed image
# https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
# limit is the maximum degree of skew to be corrected; most pdfs in the set have
# been 2-3 degrees skewed. delta is the steps to iterate to find the best angle 
# and will iterate up to the limit
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = scipy.ndimage.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    # https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
    # cv.threshold(src, thresh, maxval, type)
    # any pixel greater than thresh is set to zero and any pixel that is less 
    # than thresh is set to maxval then inverted; results in a pure black/white 
    # output with more or less captured as white against a black ground; drops 
    # gray background noise from image. otsu thresholding takes a 0 for thresh
    # because otsu will automatically detect an optimal thresh and replace 0
    # use [1] to actually get the processed image itself; [0] is the threshold
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
    print(f['filename'], f['expected_cols'])
    file_path = this_dir + "/" + str(f['expected_cols']) + "/" + f['filename']
    slug = f['filename'].replace(".pdf","")
    expected_cols = int(f['expected_cols'])
    
    out_df = pd.DataFrame()

    # get temporary directory of Image objects
    with tempfile.TemporaryDirectory() as path:
        print("Converting pages . . . ")
        images = convert_pdf(file_path, 300)
        # run process on each image
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
                # in the tuple to change results, smaller numbers blurs across 
                # a smaller area, larger numbers across a larger area
                blur = cv2.GaussianBlur(gray, (7, 7), 1)
                
                # https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
                # run an adaptive threshold. Messing with the last number can 
                # change the results a lot; use an even number for 2nd number 
                # for adaptive threshholding for sampling area
                # cv2.adaptiveThreshold(src, output thresh value, adaptive threshold method, 
                # threshold method, int for square neighborhood size, constant for tuning)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 14)
                
                ## get the bounding boxes of each cluster in order to get their
                ## coordinates for producing columns
                # set kernel size - fudge numbers to capture bigger or smaller regions
                rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
                
                # dilate image - create large blobs of text to detect clusters
                dilation = cv2.dilate(thresh, rect_kernel, iterations=2)
                
                # standard contouring method
                contours = cv2.findContours(dilation, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)[0]
                # get bboxes for all contours
                bbox = [cv2.boundingRect(c) for c in contours]
                
                # extract x coordinates only
                x_coords = [c[0] for c in bbox]
                
                # cluster together the x coords so that the first and last of 
                # each group have only n between them.  Small n for many smaller
                # groups, big n for fewer bigger groups
                clustered = cluster(x_coords,50)
                sorted_keys = find_cols(clustered, int(expected_cols))
    
                # ocr the corrected image
                # nld for Dutch, deu for German
                # hocr puts the text on the page so that camelot can use x coords
                # to find and segement it into columns
                pdf = pytesseract.image_to_pdf_or_hocr(thresh, extension='pdf', config=tesseract_config, lang="nld+eng")
                
                with open('temp.pdf', 'w+b') as f:
                    f.write(pdf)
                
                try:
                    # print("Extracting tables...")
                    
                    # print(sorted_keys, type(sorted_keys))
                    # print(','.join(str(x) for x in sorted_keys))
                    sorted_keys = ','.join(str(x) for x in sorted_keys)
                    
                    # use the n most frequent coordinates to identify tables
                    # with the expected number of columns
                    # split_text=True in read_pdf will break text at column
                    # boundaries, but will produce bad results if the column
                    # boundaries aren't perfect
                    newtables = camelot.read_pdf('temp.pdf', flavor="stream", edge_tol=1800, row_tol=10, pages="all", strip_text="\n\r", columns=[sorted_keys], layout_kwargs=layout_dict)
                    # print("Total tables extracted:", newtables.n)

                    for nt in newtables:
                        # print(nt.df)
                        nt.df['pdf_pg'] = n
                        out_df = pd.concat([out_df, nt.df])
                except:
                    print(f"Tables for page {n} can't be processed.")
                    insert_row = {1: "table error", "pdf_pg": n}
                    out_df = pd.concat([out_df, pd.DataFrame([insert_row])])
                    pass
            except ValueError:
                print(f"Page {n} can't be processed.")
                insert_row = {1: "page error", "pdf_pg": n}
                out_df = pd.concat([out_df, pd.DataFrame([insert_row])])
            n += 1

        # print(df)
        
        # out_df = out_df.replace(r'\n',' ', regex=True) 
        out_df.to_csv("camelot_rerun/charmarg-" + slug + ".csv")