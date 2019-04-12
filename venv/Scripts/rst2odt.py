#!C:\Users\dioda\odrive\RIT - Google Drive\Spring 2019\SWEN 780 Capstone Research Project\commit-multi-label-classification\venv\Scripts\python.exe

# $Id: rst2odt.py 5839 2009-01-07 19:09:28Z dkuhlman $
# Author: Dave Kuhlman <dkuhlman@rexx.com>
# Copyright: This module has been placed in the public domain.

"""
A front end to the Docutils Publisher, producing OpenOffice documents.
"""

import sys
try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

from docutils.core import publish_cmdline_to_binary, default_description
from docutils.writers.odf_odt import Writer, Reader


description = ('Generates OpenDocument/OpenOffice/ODF documents from '
               'standalone reStructuredText sources.  ' + default_description)


writer = Writer()
reader = Reader()
output = publish_cmdline_to_binary(reader=reader, writer=writer,
    description=description)

