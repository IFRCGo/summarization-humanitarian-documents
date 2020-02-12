import sys
sys.path.append('/idiap/temp/jbello/others/')
from utils import *
import numpy as np
import json

data_directory = '/idiap/temp/jbello/data/preprocessed/sent3/aspects/'
data_file = 'aspects.partition.json'

sectors = ['Agriculture','Cross','Education','Food','Health','Livelihood','Logistic','NFI','Nutrition','Protection','Shelter','WASH']

fn, _ = read_directory(data_directory, 'summary.')
mask = []
for sector in sectors:
	if len(mask) != 0:
		mask = [sector in obs for obs in fn]
	else:
		temp = [sector in obs for obs in fn]
		mask = mask or temp
doc_filter_asp = np.array(fn)[mask]
doc_filter_asp = [doc.replace('summary.','') for doc in doc_filter_asp]
doc_filter_asp = [doc.replace('.txt','') for doc in doc_filter_asp]
mask_n = np.bitwise_not(mask)
doc_not_filter = np.array(fn)[mask_n]
doc_not_filter = [doc.replace('summary.','') for doc in doc_not_filter]
doc_not_filter = [doc.replace('.txt','') for doc in doc_not_filter]
doc_filter = [doc_filter_asp,doc_not_filter]
print('Saving partition')
with open(os.path.join(data_directory,data_file),'w',encoding= 'utf8') as outfile:
	json.dump(doc_filter,outfile)
print('Saved partition')


