import sys
sys.path.append('/idiap/temp/jbello/models/neusum/NeuSum/neusum_pt/neusum/')
from xinit import *
import torch
import codecs
import os
import numpy as np

vocab_size = 100000
dim = 100
vocab_path = '/idiap/temp/jbello/data/training/neusum/neusum_random/sent3_100k_100d_80s/vocab.txt.100k'
save_path= '/idiap/temp/jbello/data/training/neusum/neusum_random/sent3_100k_100d_80s/'

res = torch.FloatTensor(vocab_size,dim) 
embed = torch.nn.init.xavier_uniform(res, math.sqrt(3))
parameters = embed.numpy()

with codecs.open(vocab_path, 'r') as f:
    data = f.readlines()
vocab = []
for line in data:
    vocab.append(line.split(" ")[0])

print(len(parameters))
print(len(parameters[0]))

count = 0
for item in parameters:
    count+=1
    print (count)
    item = np.array2string(item, separator= ' ')

f_emb = ''
for i in range(0,len(parameters)):
    print (i)
    f_emb = f_emb + str(vocab[i]) + str(np.round(parameters[i],decimals = 4)) + '\n'


'''
f_emb = ''
for i in range(0,len(vocab)):
    print (i)
    f_emb = f_emb + vocab[i] + ' '
    for d in range(dim):
        if (d == dim):
            f_emb = f_emb + str(embed[i][d]) + '\n'
        else:
            f_emb = f_emb + str(embed[i][d]) + ' '
'''

def save_texts(relative_path, name_text, texts, id_texts):
    for i in range(0,len(texts)):
        complete_text_name = os.path.join(relative_path,name_text+ str(id_texts[i]) +'.txt')
        with open(complete_text_name,'w', encoding = 'utf-8') as f:
            f.write(texts[i])

save_texts(save_path, 'random.100d',[f_emb],[''])

