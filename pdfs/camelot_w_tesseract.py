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

# when there's white space between the month and the day, 
# add an extra expected column
# expected_cols = 4

# --user-words file seems like it should be able to live anywhere but pytesseract
# docs have it in /tessdata as eng.user-words
# preserve_interword_whitespace should preserve spaces between words when a 
# whitelist is passed but doesn't actually seem to do that; whitespace needs to
# be included in the whitelist.
# blacklist shouldn't technically be necessary but whitelist only was still
# outputting diacritics and special characters
# "/Users/mk192787/opt/miniconda3/envs/spyder-env/share/tessdata/eng.user-patterns"
# "/Users/mk192787/opt/miniconda3/envs/spyder-env/share/tessdata/eng.user-words"
# --user-words eng.user-words 
blacklist = "Äòûúùîò¬ß¶§√°¢"
whitelist = "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,. "
tesseract_config = '--psm 4 bazaar -c preserve_interword_spaces=1 -c tessedit_char_whitelist="{}" -c tessedit_char_blacklist="{}"'.format(whitelist, blacklist)
layout_dict = {'boxes_flow': 1.0}

files = []
this_dir = "../../pdfs/type_2/trimmed_pdfs/"

# expects pdfs to be structed in numbered subfolders, numbered by expected_cols
for subdir, dirs, items in os.walk(this_dir):
    for file in items:
        # print(subdir, dirs, file)
        if file.endswith(".pdf"):
            item = {"filename": file, "expected_cols": 4}
            files.append(item)

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

def sort_coords(value):
    return sorted(value.split(","), key=float)

# group the input list so that each element has no more than maxgap
# between each element
def cluster(data, maxgap):
    data = sorted(data)
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    groups = sorted(groups, key=len)
    print(data)
    print(groups)
    return groups

# get the n largest subgroups in a clustered list
def find_cols(clustered, expected_cols):
    cols = sorted(clustered, key=len)[-expected_cols:]
    minlist = sorted([item[0] for item in cols])
    # remove first item because we only want locations of separators between
    # columns, not the location of the start of the first column
    minlist.pop(0)
    return minlist

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

for f in files[:10]:
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
        for i in images[:10]:
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
                
                ## get the bounding boxes of each cluster
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
                pdf = pytesseract.image_to_pdf_or_hocr(thresh, extension='pdf', config=tesseract_config, lang="nld+eng")
                
                with open('temp.pdf', 'w+b') as f:
                    f.write(pdf)
                
                try:
                    # print("Extracting tables...")
                    
                    # print(sorted_keys, type(sorted_keys))
                    # print(','.join(str(x) for x in sorted_keys))
                    sorted_keys = ','.join(str(x) for x in sorted_keys)
                    # use the n most frequent coordinates to re-identify tables
                    # with the expected number of columns
                    # split_text=True, 
                    newtables = camelot.read_pdf('temp.pdf', flavor="stream", edge_tol=900, row_tol=10, pages="1-end", columns=[sorted_keys], layout_kwargs=layout_dict)
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
        out_df.to_csv("camelot/boxesflowvert-" + slug + ".csv")