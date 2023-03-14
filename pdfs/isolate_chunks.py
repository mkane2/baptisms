#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:08:30 2023

@author: dnelson
"""

from pdf2image import convert_from_path
import tempfile
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
# from PIL import Image
import pytesseract
import csv


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

slug = "ny-newtown-drc"

file_path = 'type_2/trimmed_pdfs/5/' + slug + '.pdf'

# list for holding output
output = []

# get temporary directory of Image objects
with tempfile.TemporaryDirectory() as path:
    print("Converting pages . . . ")
    images = convert_pdf(file_path, 300)
    # run process on each image
    n = 1
    for i in images:
        print(f'Processing page {n}...')
        # print("Page size ", str(images[n].shape[0]))
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        # threshing to improve contouring - need binary image
        pg_width = 500 #images[n].shape[0]
        gray = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        # set kernel size - fudge numbers to capture bigger or smaller regions
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
        # dilate image - create large blobs of text to detect clusters
        dilation = cv2.dilate(gray, rect_kernel, iterations=3)
        # standard contouring method
        contours = cv2.findContours(dilation, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_NONE)[0]

        # initialize empty list to sort clusters
        sorted_clusters = []
        # get bboxes for all contours
        bbox = [cv2.boundingRect(c) for c in contours]
        # extract y coordinates only
        y_coords = [(0, c[1]) for c in bbox]

        # apply clustering algorithm
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="manhattan",
                linkage="complete",
                distance_threshold=25.0)
            clustering.fit(y_coords)

            # loop over all clusters
            for q in np.unique(clustering.labels_):
                idxs = np.where(clustering.labels_ == q)[0]
                avg = np.average([bbox[i][1] for i in idxs])
                sorted_clusters.append((q, avg))
                # sort clusters by average y coordinate
                sorted_clusters.sort(key=lambda x: x[1])

            # empty list to hold tesseract output
            tesseract_output = []
            for (q, _) in sorted_clusters:
                # extract indices for the coordinates for each cluster
                idxs = np.where(clustering.labels_ == q)[0]
                # extract x coorindates from items in each cluster, then sort
                # left to right
                x_coords = [bbox[x][0] for x in idxs]
                sorted_idxs = idxs[np.argsort(x_coords)]
                row = []
                for foo in sorted_idxs:
                    x, y, w, h = bbox[foo]
                    cropped = i[y:y + h, x:x + w]
                    text = pytesseract.image_to_string(cropped)
                    print(x, y, text.strip())
                    row.append(text.strip())
                with open(slug + '-output.csv', 'a', newline='',
                          encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
        except ValueError:
            print(f"Page {n} can't be processed.")
        n = n + 1

# with open('output.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(output))