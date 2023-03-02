#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:13:37 2023

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
# import os

slug = "ny-oysterbay-drc"

file_path = 'type_2/trimmed_pdfs/' + slug + '.pdf'

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
    for i in images:
        print(f'Processing page {n}...')

        try:
            # Get a searchable PDF
            pdf = pytesseract.image_to_pdf_or_hocr(i, extension='pdf', config="--psm 4")
            
            with open('temp.pdf', 'w+b') as f:
                f.write(pdf)
            
            try:
                tables = camelot.read_pdf('temp.pdf', flavor="stream", row_tol=10, pages="1-end")
        
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
                    
                    coord_df = pd.DataFrame(coords, columns=['bottom_left_x1', 'bottom_left_y1', 'top_right_x2', 'top_right_y2', 'pg'])
                    
                    # this is so ugly. Get the n of columns on the page and count the n most
                    # frequent x1 coordinates on the page. Rewrite this later to not rename
                    # the columns and just work with column indices so it's flexible for 
                    # different n of columns
                    pgs_df = (coord_df.groupby('pg')['bottom_left_x1']
                              .value_counts().groupby(level=0).cumcount()
                              .loc[lambda x: x < 4]
                              .reset_index(name='c')
                              .pivot(index='pg',columns='c',values='bottom_left_x1')
                              .rename(columns={0:'first_', 1:'second_', 2:'third_', 3:'fourth_'})
                              .add_suffix('x')
                              .rename_axis(None, axis=1)
                              .reset_index())
                    
                    pgs_df = pgs_df.astype(str)
                    
                    pgs_df['col_coords'] = pgs_df[['first_x','second_x','third_x','fourth_x']].agg(','.join, axis=1)
                    
                    pgs_df['sorted_coords'] = pgs_df['col_coords'].map(sort_coords)
                    # print(pgs_df)
                    
                    coords = ','.join(pgs_df.iloc[-1].sorted_coords)
                    newtables = camelot.read_pdf('temp.pdf', flavor='stream', columns=[coords])
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
    out_df.to_csv("camelot/" + slug + "-camelot-coords.csv")