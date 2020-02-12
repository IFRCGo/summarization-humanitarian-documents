import sys
sys.path.append('/idiap/temp/jbello/others/')
from utils import *
sys.path.append('/idiap/temp/jbello/preprocess/')
from nltk.tokenize import RegexpTokenizer
import numpy as np


data_directory = '/idiap/temp/jbello/data/preprocessed/sent3/'
partition_file = '/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json'

tokenizer = RegexpTokenizer(r'\w+')

parts = ['training', 'validation','test']
type_doc = ['document.','summary.']

print('begin word exploration!')
for td in type_doc:
	for part in parts:
		filename, docs = select_partition(data_directory, partition_file, partition = part, name_text = td)
		length_docs_words = []
		for i in range (0, len(docs)):
			tokens = tokenizer.tokenize(docs[i])
			length_docs_words.append(len(tokens))
		print('The average number of words per ', str(type_doc), ' in ',str(part),' is ',np.quantile(length_docs_words, 0.5))
print('end word exploration!')
