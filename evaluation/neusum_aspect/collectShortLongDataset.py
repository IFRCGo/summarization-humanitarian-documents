import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from itertools import compress

data_test_src = '/idiap/temp/jbello/data/results/neusum/test.src.txt'
data_test_tgt = '/idiap/temp/jbello/data/results/neusum/test.tgt.txt'
#data_train_src = '/idiap/temp/jbello/data/training/neusum/train.src.txt'
#data_val_src = '/idiap/temp/jbello/data/validation/neusum/val.src.txt'

src_test_short_save_path = '/idiap/temp/jbello/data/results/neusum_aspect/short_docs/test.src.txt'
src_test_long_save_path = '/idiap/temp/jbello/data/results/neusum_aspect/long_docs/test.src.txt'
tgt_test_short_save_path = '/idiap/temp/jbello/data/results/neusum_aspect/short_docs/test.tgt.txt'
tgt_test_long_save_path = '/idiap/temp/jbello/data/results/neusum_aspect/long_docs/test.tgt.txt'

part_test_short_save_path = '/idiap/temp/jbello/data/results/neusum_aspect/short_docs/test.part.txt'
part_test_long_save_path = '/idiap/temp/jbello/data/results/neusum_aspect/long_docs/test.part.txt'

partition = '/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json'

'''
src_train_short_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/short_docs/train.tgt.txt'
_train_long_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/long_docs/train.tgt.txt'
tgt_val_short_save_path = '/idiap/temp/jbello/data/validation/neusum_aspect/short_docs/val.tgt.txt'
tgt_val_long_save_path = '/idiap/temp/jbello/data/validation/neusum_aspect/long_docs/val.tgt.txt'

oracle_short_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/short_docs/train.rouge_bigram_F1.oracle.regGain'
oracle_long_save_path = '/idiap/temp/jbello/data/training/neusum_aspect/long_docs/train.rouge_bigram_F1.oracle.regGain'
'''
print('Begin to read documents!')

with open(data_test_src) as txt_file:
    file = txt_file.read()
    src_test = file.split('\n')

with open(data_test_tgt) as txt_file:
    file = txt_file.read()
    tgt_test = file.split('\n')

with open(partition) as f:
    file = json.load(f)
    idx_test = file[2]

print('Documents read!')

len_test = []
for line in src_test:
    len_test.append(line.count('###SENT###')+1)
len_test = np.array(len_test)


mask_short_test = [length < 20 for length in len_test]
mask_short_test = np.array(mask_short_test)

src_test_short = list(compress(src_test,mask_short_test))
src_test_long = list(compress(src_test,~mask_short_test))
part_short = list(compress(idx_test,mask_short_test))
part_long = list(compress(idx_test,~mask_short_test))

#src_val_short = list(compress(src_val,mask_short_val))
#src_val_long = list(compress(src_val,~mask_short_val))

#rouge_oracle_short = list(compress(rouge_oracle,mask_short_train))
#rouge_oracle_long = list(compress(rouge_oracle,~mask_short_train))

tgt_test_short = list(compress(tgt_test,mask_short_test))
tgt_test_long = list(compress(tgt_test,~mask_short_test))
#tgt_val_short = list(compress(tgt_train,mask_short_val))
#tgt_val_long = list(compress(tgt_train,~mask_short_val))

'''
print('Number of short documents in training set: ', len(src_train_short))
print('Number of long documents in training set: ', len(src_train_long))
print('Number of short documents in validation set: ', len(src_val_short))
print('Number of long documents in validation set: ', len(src_val_long))


save_paths = [src_train_short_save_path, src_train_long_save_path, src_val_short_save_path, src_val_long_save_path,tgt_train_short_save_path,tgt_train_long_save_path, tgt_val_short_save_path,tgt_val_long_save_path,oracle_short_save_path,oracle_long_save_path]
datasets = [src_train_short,src_train_long,src_val_short,src_val_long,tgt_train_short,tgt_train_long,tgt_val_short,tgt_val_long,rouge_oracle_short,rouge_oracle_long]
'''
save_paths = [src_test_short_save_path,src_test_long_save_path,tgt_test_short_save_path,tgt_test_long_save_path,part_test_short_save_path, part_test_long_save_path]
datasets = [src_test_short,src_test_long,tgt_test_short,tgt_test_long,part_short,part_long]
for i in range(0,len(save_paths)):
    with open(save_paths[i],'w', encoding = 'utf-8') as f:
        for row in datasets[i]:
            f.write(str(row) + '\n')
print('Saved documents!')

