#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:47:04 2023

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
import os

# slug = "ny-schenectady-drc-3"

# file_path = 'type_2/trimmed_pdfs/' + slug + '.pdf'

# when there's white space between the month and the day, 
# add an extra expected column
expected_cols = 5

files = []
this_dir = "type_2/trimmed_pdfs"
for x in os.listdir(this_dir):
    if x.endswith(".pdf"):
        files.append(x)

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

for f in files:
    print(f)
    file_path = this_dir + "/" + f
    slug = f.replace(".pdf","")
    
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
                im = correct_skew(i)[1]
    
                pdf = pytesseract.image_to_pdf_or_hocr(im, extension='pdf', config="--psm 4")
                
                with open('temp.pdf', 'w+b') as f:
                    f.write(pdf)
                
                try:
                    tables = camelot.read_pdf('temp.pdf', flavor="stream", edge_tol=500, row_tol=10, pages="1-end")
            
                    # print("Total tables extracted:", tables.n)
            
                    for t in tables:
                        coords = []
                        cells = t.cells
                        for c in cells: 
                            for item in c:
                                cell = str(item).replace('<Cell ',"").replace('>',"").replace('x1=','').replace('x2=','').replace('y1=','').replace('y2=','')
                                splitcell = cell.split()
                                splitcell.append(n)
                                coords.append(splitcell)
                                # print(splitcell)
                        
                        coord_df = pd.DataFrame(coords, columns=['bottom_left_x1', 'bottom_left_y1', 'top_right_x2', 'top_right_y2', 'pg'])
                        coord_list = list(coord_df['bottom_left_x1'].unique())
                        # print("Columns:", list(t.df.columns))
                        # print("Coord list:", coord_list)
                        # t.df.columns = t.df.columns.toList() + coord_list
                        # print("Columns:", list(t.df.columns))
                        t.df = t.df.replace('', np.nan)
                        coord_list = [str(a) for a in coord_list]
                        t.df.columns = coord_list
                        # print("Count:", t.df.count())
                        coord_dict = t.df.count().to_dict()
                        # print("Coord dict:", coord_dict)
                        k = Counter(coord_dict)
                        # print(k)
                        keep_coords = k.most_common(expected_cols)
                        keep = {tup[0]: tup[1:] for tup in keep_coords}
                        # print("Keep these:", keep)
                        sorted_keys = sorted(list(keep.keys()), key=float)
                        # print("Keep keys:", sorted_keys)
                        
                        # coord_df.to_csv("coord" + str(n) + ".csv")
                        
                        # this is so ugly. Get the n of expected columns on the page 
                        # and count the n most frequent x1 coordinates on the page.
                        # pgs_df = (coord_df.groupby('pg')['bottom_left_x1']
                        #           .value_counts().groupby(level=0).cumcount()
                        #           .loc[lambda x: x < expected_cols]
                        #           .reset_index(name='c')
                        #           .pivot(index='pg',columns='c',values='bottom_left_x1')
                        #           .rename_axis(None, axis=1)
                        #           .reset_index())
                        
                        # .rename(columns={0:'first_', 1:'second_', 2:'third_', 3:'fourth_'})
                        # .add_suffix('x')
                        
                        # pgs_df = pgs_df.astype(str)
                        
                        # end_col = expected_cols + 1
                        
                        # pgs_df['col_coords'] = pgs_df.iloc[:,1:end_col].agg(','.join, axis=1)
                        
                        # pgs_df['sorted_coords'] = pgs_df['col_coords'].map(sort_coords)
                        # # print(pgs_df)
                        
                        # coords = ','.join(pgs_df.iloc[-1].sorted_coords)
                        # print(coords)
                        # print(','.join(sorted_keys))
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
        out_df.to_csv("camelot/trial-" + slug + "-camelot-frequent_coords.csv")