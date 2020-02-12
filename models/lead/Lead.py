#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
import logging
import sys
sys.path.append("/idiap/temp/jbello/preprocess/")
from textcleaner import clean_text_by_sentences as _clean_text_by_sentences

logger = logging.getLogger(__name__)
def summarize(text, nb_sentences = 3):
    """
    Returns the first n sentences of the text, as its summarized version.
    """
    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text)
    
    # If only one, two or three sentences are present, the function return the input text (Avoids ZeroDivisionError). 
    if len(sentences) < nb_sentences:
        logger.warning("Summarization not performed since the document has already few sentences.")
        return text
    else:
        sentences = sentences[:nb_sentences]
        return "\n".join([sentence.text for sentence in sentences])
