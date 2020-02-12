import sys
import torch
import torch.nn as nn
sys.path.append('/idiap/temp/jbello/models/neusum/NeuSum/neusum_pt/')

model = '/idiap/temp/jbello/training/neusum/neusum_out_domain/model_devRouge_0.1952_e13.pt'

checkpoint = torch.load(model)
model_opt = checkpoint['opt']
src_dict = checkpoint['dicts']['src']
tgt_dict = checkpoint['dicts']['tgt']
enc_rnn_size = model_opt.doc_enc_size
dec_rnn_size = model_opt.dec_rnn_size
print('dec rnn size: ', dec_rnn_size)

