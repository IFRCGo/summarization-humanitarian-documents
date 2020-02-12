import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
sys.path.append('/idiap/temp/jbello/others/')
from utils import save_texts

class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)
                                                            
        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
                                                                                                    
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
                                                                                                                                                
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x

def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.cuda()

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss).cuda()


model_path = '/idiap/temp/jbello/training/neusum/neusum_in_domain/sent4_10k_100d_80s/gloveDEEP.pt'

save_path = '/idiap/temp/jbello/data/training/neusum/neusum_in_domain/sent4_10k_100d_80s/'

model = GloveModel(272449,100)


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
print('loaded model')
emb_i = model.wi.weight.cpu().data.numpy()
emb_j = model.wj.weight.cpu().data.numpy()
emb = emb_i + emb_j
print(len(emb))

class GloveDataset:
        
    def __init__(self, text, n_words=200000, window_size=5):
        self._window_size = window_size
        self._tokens = text.split(" ")
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
                                                                            
        self._id_tokens = [self._word2id[w] for w in self._tokens]
                                                                                            
        self._create_coocurrence_matrix()
                                                                                                            
        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)

        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))

            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx).cuda()
        self._j_idx = torch.LongTensor(self._j_idx).cuda()
        self._xij = torch.FloatTensor(self._xij).cuda()

    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]

dataset = GloveDataset(open("/idiap/temp/jbello/data/training/neusum/train.src.txt").read())

print(len(dataset._id2word))

f_emb = ''
for i in range(0, len(dataset._id2word)):
    print(i)
    f_emb = f_emb + str(dataset._id2word[i]) + str(np.round(emb[i], decimals = 4)) + '\n'

save_texts(save_path,'glove_emb.100d',[f_emb],[''])

top_k = 300
tsne = TSNE(metric = 'cosine', random_state = 123)
embed_tsne = tsne.fit_transform(emb[:top_k,:])
fig,ax = plt.subplot(figsize = (14,14))
for idx in range(top_k):
    plt.scatter(*embed_tsne[idx,:],color = red)
    plt.annotate(dataset._id2word[idx],(embed_tsne[idx,0],embed_tsne[idx,1]),alpha = 0.7)
plt.savefig('/idiap/temp/jbello/training/neusum/neusum_in_domain/sent4_10k_100d_80s/glove_emb100d.png')
print('saved embedding plot!')
