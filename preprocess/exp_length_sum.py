import sys
sys.path.append('/idiap/temp/jbello/others/')
from utils import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

data_train_src = '/idiap/temp/jbello/data/training/neusum/train.src.txt'
data_val_src = '/idiap/temp/jbello/data/validation/neusum/val.src.txt'
data_test_src = '/idiap/temp/jbello/data/results/neusum/test.src.txt'

data_train_tgt = '/idiap/temp/jbello/data/training/neusum/train.tgt.txt'
data_val_tgt = '/idiap/temp/jbello/data/validation/neusum/val.tgt.txt'
data_test_tgt = '/idiap/temp/jbello/data/results/neusum/test.tgt.txt'

relative_save_path = '/idiap/temp/jbello/preprocess/'

data_src = [data_train_src, data_val_src, data_test_src]
data_tgt = [data_train_tgt,data_val_tgt,data_test_tgt]

print('Begin to read documents!')
src = []
for path in data_src:
	with open(path) as txt_file:
		file = txt_file.read()
		file = file.split('\n')
		for line in file:
			src.append(line.count('###SENT###')+1)
src = np.array(src)

print('Begin to read summaries!')
tgt = []
for path in data_tgt:
	with open(path) as txt_file:
		file = txt_file.read()
		file = file.split('\n')
		for line in file:
			tgt.append(line.count('###SENT###')+1)
tgt = np.array(tgt)

mask_short = [length < 20 for length in src]
mask_short = np.array(mask_short)

tgt_short = tgt[mask_short]
tgt_long = tgt[~mask_short]

print('Number of short documents: ', len(tgt_short))
print('Number of long documents: ', len(tgt_long))

#if we focus our attention in short summaries
#tgt_short = tgt_short[tgt_short < 10]
#tgt_long = tgt_long[tgt_long < 10]
#ax_short = sns.distplot(tgt_short, color = 'red')
#ax_short.set(xlabel = 'nb. of sentences per summary', ylabel = 'proportion of documents')
#save_path = relative_save_path + "short_doc_sum_len_10s.png"
#plt.savefig(save_path)
print('50\% of the short documents, has summaries with a length below: ', np.quantile(tgt_short,0.5), ' sentences.')
#plt.close()

#ax_long = sns.distplot(tgt_long, color = 'red')
#ax_long.set(xlabel = 'nb. of sentences per summary', ylabel = 'proportion of documents')
#save_path = relative_save_path + "long_doc_sum_len_10s.png"
#plt.savefig(save_path)
print('50\% of the long documents, has summaries with a length below: ', np.quantile(tgt_long,0.55), ' sentences.')
