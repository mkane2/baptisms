#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:28:20 2023

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
import os

files = []
this_dir = "type_2/trimmed_pdfs"
for x in os.listdir(this_dir):
    if x.endswith(".pdf"):
        files.append(x)

# slug = "ny-oysterbay-drc"

# file_path = 'type_2/trimmed_pdfs/' + slug + '.pdf'

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

for f in files:
    print(f)
    file_path = this_dir + "/" + f
    slug = f.replace(".pdf","")
    with tempfile.TemporaryDirectory() as path:
        print("Converting pages . . . ")
        images = convert_pdf(file_path, 300)
        # run process on each image
        n = 1
        for i in images:
            print(f'Processing page {n}...')
    
            try:
                text = pytesseract.image_to_string(i, config="--psm 4")
                print(text)
                with open(slug + "-tesseract-output.txt", "a+") as file:
                    file.write("Page " + str(n))
                    file.write(text)
            except:
                pass
            n += 1