import sys
sys.path.append('/idiap/temp/jbello/others/')
from utils import *
sys.path.append('/idiap/temp/jbello/preprocess/')
from textcleaner import clean_text_by_sentences
import numpy as np
import seaborn as sns
import json

data_directory = '/idiap/temp/jbello/data/preprocessed/sent3/'
partition_file = '/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json'
save_path = '/idiap/temp/jbello/data/preprocessed/sent3/length.sentences.partition.json'


print('begin exploration!')
parts = ['training', 'validation','test']
type_doc = ['document.','summary.']
for part in parts:
    for td in type_doc:
        filename, docs = select_partition(data_directory, partition_file, partition = part, name_text = td)
        length_docs_sent = []
        for i in range (0, len(docs)):
            sentences = clean_text_by_sentences(docs[i])
            length_docs_sent.append(len(sentences))
        print('The median number of sentences per ', str(td), ' in ',str(part),' is ',np.quantile(length_docs_sent,0.5))
print('End of exploration')

'''
print('begin exploration')
length_docs = {}
parts = ['training','validation','test']
for part in parts:
    print('begin: ', part)
    filename, docs = select_partition(data_directory, partition_file, partition = part, name_text = 'document.')
    ids = extract_id(filename)
    length_docs[part] = {}
    for i in range (0,len(docs)):
        length_docs[part][ids[i]] = len(clean_text_by_sentences(docs[i]))

with open(save_path, 'w') as fp:
    json.dump(length_docs,fp)
print('Saved documents!')
'''
'''
print('begin exploration')
length_docs = []
parts = ['training','validation','test']
for part in parts:
    print('begin ', part)
    #the partition file contains already the filtering for language
    _,docs = select_partition(data_directory, partition_file, partition = part, name_text = 'document.')
    for i in range (0, len(docs)):
        sentences = clean_text_by_sentences(docs[i])
        length_docs.append(len(sentences))
    
ax = sns.distplot(length_docs)
ax.set_yscale("log")
ax.set(xlabel='nb. of sentences per document', ylabel='nb. of documents')
ax.figure.savefig("doc_length_log_dist.png")
print("75% of the documents have a length below: ",np.quantile(length_docs,0.75), " sentences.")
print("90% of the documents have a length below: ", np.quantile(length_docs,0.9), " sentences.")
print("95% of the documents have a length below: ", np.quantile(length_docs, 0.95), " sentences.")
'''
