#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 21:44:54 2023

@author: mk192787
"""

from PyPDF2 import PdfReader

reader = PdfReader("pdfs/type_3/ca-prarieduchien-catholic-1.pdf")
page = reader.pages[0]
print(page.extract_text())