import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from itertools import compress

data_train_src = '/idiap/temp/jbello/data/training/neusum/train.src.txt'
data_val_src = '/idiap/temp/jbello/data/validation/neusum/val.src.txt'

data_train_tgt = '/idiap/temp/jbello/data/training/neusum/train.tgt.txt'
data_val_tgt = '/idiap/temp/jbello/data/validation/neusum/val.tgt.txt'

data_rouge_oracle = '/idiap/temp/jbello/data/training/neusum/train.rouge_bigram_F1.oracle.regGain'

src_train_short_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/short_docs/train.src.txt'
src_train_long_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/long_docs/train.src.txt'
src_val_short_save_path = '/idiap/temp/jbello/data/validation/neusum_aspect/short_docs/val.src.txt'
src_val_long_save_path = '/idiap/temp/jbello/data/validation/neusum_aspect/long_docs/val.src.txt'

tgt_train_short_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/short_docs/train.tgt.txt'
tgt_train_long_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/long_docs/train.tgt.txt'
tgt_val_short_save_path = '/idiap/temp/jbello/data/validation/neusum_aspect/short_docs/val.tgt.txt'
tgt_val_long_save_path = '/idiap/temp/jbello/data/validation/neusum_aspect/long_docs/val.tgt.txt'

oracle_short_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/short_docs/train.rouge_bigram_F1.oracle.regGain'
oracle_long_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/long_docs/train.rouge_bigram_F1.oracle.regGain'
print('Begin to read documents!')

with open(data_train_src) as txt_file:
    file = txt_file.read()
    src_train = file.split('\n')
        
with open(data_train_tgt) as txt_file:
    file = txt_file.read()
    tgt_train = file.split('\n')

with open(data_val_src) as txt_file:
    file = txt_file.read()
    src_val = file.split('\n')

with open(data_val_tgt) as txt_file:
    file = txt_file.read()
    tgt_val = file.split('\n')

with open(data_rouge_oracle) as txt_file:
    file = txt_file.read()
    rouge_oracle = file.split('\n')

print('Documents read!')

len_train = []
for line in src_train:
    len_train.append(line.count('###SENT###')+1)
len_train = np.array(len_train)

len_val = []
for line in src_val:
    len_val.append(line.count('###SENT###')+1)
len_val = np.array(len_val)
print('dataset created')

mask_short_train = [length < 20 for length in len_train]
mask_short_train = np.array(mask_short_train)

mask_short_val = [length < 20 for length in len_val]
mask_short_val = np.array(mask_short_val)

src_train_short = list(compress(src_train,mask_short_train))
src_train_long = list(compress(src_train,~mask_short_train))
src_val_short = list(compress(src_val,mask_short_val))
src_val_long = list(compress(src_val,~mask_short_val))

rouge_oracle_short = list(compress(rouge_oracle,mask_short_train))
rouge_oracle_long = list(compress(rouge_oracle,~mask_short_train))

tgt_train_short = list(compress(tgt_train,mask_short_train))
tgt_train_long = list(compress(tgt_train,~mask_short_train))
tgt_val_short = list(compress(tgt_train,mask_short_val))
tgt_val_long = list(compress(tgt_train,~mask_short_val))

print('Number of short documents in training set: ', len(src_train_short))
print('Number of long documents in training set: ', len(src_train_long))
print('Number of short documents in validation set: ', len(src_val_short))
print('Number of long documents in validation set: ', len(src_val_long))

save_paths = [src_train_short_save_path, src_train_long_save_path, src_val_short_save_path, src_val_long_save_path,tgt_train_short_save_path,tgt_train_long_save_path, tgt_val_short_save_path,tgt_val_long_save_path,oracle_short_save_path,oracle_long_save_path]
datasets = [src_train_short,src_train_long,src_val_short,src_val_long,tgt_train_short,tgt_train_long,tgt_val_short,tgt_val_long,rouge_oracle_short,rouge_oracle_long]
for i in range(0,len(save_paths)):
    with open(save_paths[i],'w', encoding = 'utf-8') as f:
        for row in datasets[i]:
            f.write(str(row) + '\n')
print('Saved documents!')

