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
from collections import Counter
# import os

slug = "ny-schenectady-drc-3"

file_path = 'type_2/trimmed_pdfs/' + slug + '.pdf'

# when there's white space between the month and the day, 
# add an extra expected column
expected_cols = 5

# files = []
# this_dir = "type_2/trimmed_pdfs"
# for x in os.listdir(this_dir):
#     if x.endswith(".pdf"):
#         files.append(x)

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

# for f in files:
#     print(f)
#     file_path = this_dir + "/" + f
#     slug = f.replace(".pdf","")
    
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
            # correct the image skew
            im = correct_skew(i)[1]
            # create a grayscale image
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # lightly blur the image to reduce noise. Mess with the values in the tuple to change results
            blur = cv2.GaussianBlur(gray, (7, 7), 1)
            # run an adaptive threshold. Messing with the last number can change the results a lot
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 14)

            # ocr the corrected image
            pdf = pytesseract.image_to_pdf_or_hocr(thresh, extension='pdf', config="--psm 4")
            
            with open('temp.pdf', 'w+b') as f:
                f.write(pdf)
            
            try:
                tables = camelot.read_pdf('temp.pdf', flavor="stream", edge_tol=900, row_tol=10, pages="1-end")
                print("Total tables extracted:", tables.n)
                
                for table in tables:
                    
                    coords = []
                    cells = table.cells
                    
                    # just get the coords for the first row, all rows have the
                    # same coords
                    row = cells[0]
                    for item in row:
                        cell = str(item).replace('<Cell ',"").replace('>',"").replace('x1=','').replace('x2=','').replace('y1=','').replace('y2=','')
                        splitcell = cell.split()
                        splitcell.append(n)
                        print("cell", splitcell)
                        coords.append(splitcell)
                    
                    # get just the bottom left x coordinates from each cell in
                    # the first row, check that they're unique, and put back in
                    # a list to use as a header row for table.df
                    coordx = [item[0] for item in coords]
                    coordl = set(coordx)
                    coord_list = list(coordl)

                    # make a dict of the contents of the page table and give
                    # it a header with the x coordinates of each column
                    table.df = table.df.replace('', np.nan)
                    coord_list = [str(a) for a in coord_list]
                    table.df.columns = coord_list
                    coord_dict = table.df.count().to_dict()
                    
                    # soort the dict to find the most n frequent x coordinates
                    # where n is the number of expected columns
                    k = Counter(coord_dict)
                    keep_coords = k.most_common(expected_cols)
                    keep = {tup[0]: tup[1:] for tup in keep_coords}
                    
                    # keep only the n most frequent x coodinates
                    sorted_keys = sorted(list(keep.keys()), key=float)
                    
                    # use the n most frequent coordinates to re-identify tables
                    # with the expected number of columns
                    newtables = camelot.read_pdf('temp.pdf', flavor='stream', columns=[','.join(sorted_keys)])
                    for nt in newtables:
                        nt.df['pdf_pg'] = n
                        out_df = pd.concat([out_df, nt.df])
            except:
                pass
        except ValueError:
            print(f"Page {n} can't be processed.")
        n += 1
        
    # print(df)
    
    out_df = out_df.replace(r'\n',' ', regex=True) 
    out_df.to_csv("camelot/trial-" + slug + "-camelot-imgcorrect.csv")