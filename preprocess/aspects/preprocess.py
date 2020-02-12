import os
import sys
import codecs
import re
sys.path.append("/idiap/temp/jbello/preprocess/aspects/")
from textcleaner import clean_text_by_sentences as _clean_text_by_sentences
import nltk
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
import re

data_directory = "/idiap/temp/jbello/data/cleaned/aspects/"

relative_save_path_3 = "/idiap/temp/jbello/data/preprocessed/sent3/aspects/"


def filter_numb_sent(directory, name_text, min_nb_sent):
    '''Return texts and id of texts that have more than a specific nb of sentences'''
    names, docs = read_directory(directory, name_text)
    filtered_text = []
    filtered_id = []
    for i in range(0,len(docs)):
        sentences = _clean_text_by_sentences(docs[i])
        if len(sentences) > min_nb_sent:
            filtered_text.append(docs[i])
            filtered_id.append(re.findall(r'\d+',names[i])[0])
    if (i%1000 == 0):
        print(str(i), 'out of', str(len(docs)), ' texts processed')
    return filtered_id,filtered_text

def filter_sum_with_docs(directory,filtered_id_docs):
    '''Let just summaries that have associated documents'''
    name_sum, summaries = read_directory(directory,name_text = 'summary.')
    filter_sum = []
    filter_id = []
    for i in range(0, len(name_sum)):
        id_sum = re.findall(r'\d+',name_sum[i])[0]
        if id_sum in id_docs:
            filter_sum.append(summaries[i])
            temp_sum = name_sum[i].replace('summary.','')
            temp_sum = temp_sum.replace('.txt','')
            filter_id.append(temp_sum)
    if (i%1000 == 0):
        print(str(i), 'out of', str(len(name_sum)), ' texts processed')
    return filter_id, filter_sum

print('start preprocessing')
print('filtering documents with more than 3 sentences:')
id_docs,filter_docs = filter_numb_sent(data_directory,"document.",3)
save_texts(relative_save_path_3,"document.",filter_docs,id_docs)
print('Saved documents: ',len(id_docs))
filter_id, filter_sum = filter_sum_with_docs(data_directory,id_docs)
save_texts(relative_save_path_3,"summary.",filter_sum,filter_id)
print('Saved summaries: ',len(filter_id))

