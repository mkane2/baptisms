#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:41:32 2023

@author: mk192787
"""

import camelot
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import tempfile
import numpy as np
import cv2
import pandas as pd

slug = "ny-schenectady-drc-3"

file_path = 'type_2/trimmed_pdfs/' + slug + '.pdf'

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

df = pd.DataFrame()

# get temporary directory of Image objects
with tempfile.TemporaryDirectory() as path:
    print("Converting pages . . . ")
    images = convert_pdf(file_path, 300)
    # run process on each image
    n = 1
    for i in images:
        print(f'Processing page {n}...')

        # Get a searchable PDF
        pdf = pytesseract.image_to_pdf_or_hocr(i, extension='pdf')
        
        with open('temp.pdf', 'w+b') as f:
            f.write(pdf) # pdf type is bytes by default
        
        tables = camelot.read_pdf('temp.pdf', flavor="stream", row_tol=10, pages="1-end")

        # print("Total tables extracted:", tables.n)

        for t in tables:
            t.df['pdf_pg'] = n
            print(t.df)
            df = df.append(t.df)
        n += 1
    
    print(df)
    df.to_csv(slug + "-output.csv")