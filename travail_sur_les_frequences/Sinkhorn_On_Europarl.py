import io
import numpy as np
import torch
import pickle
import SinkhornAutoDiff.sinkhorn_pointcloud as spc


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    f.close()
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


src_path = 'C:\POLYTECHNIQUE\\3A\EANLP\embeddings\multilingual_muse\\vector-en.txt'
tgt_path = 'C:\POLYTECHNIQUE\\3A\EANLP\embeddings\multilingual_muse\\vector-fr.txt'
nmax = 50000  # maximum number of word embeddings to load

src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)



# avec EUROPARL DATASET
# ca marche pas pcq c trop grand, on va faire des batchs, voir plus loin

import pickle

frequences_fr = pickle.load(open("frequences_fr.pkl",'rb'))
frequences_en = pickle.load(open("frequences_en.pkl",'rb'))

europarl_id_list_fr=[tgt_word2id[w] for w in frequences_fr.keys()&tgt_word2id.keys()]
europarl_id_list_en=[src_word2id[w] for w in frequences_en.keys()&src_word2id.keys()]

europarl_word_list_fr = [w for w in frequences_fr.keys()&tgt_word2id.keys()]
europarl_word_list_en = [w for w in frequences_en.keys()&src_word2id.keys()]

europarl_emb_list_fr = [tgt_embeddings[id] for id in europarl_id_list_fr]
europarl_emb_list_en = [src_embeddings[id] for id in europarl_id_list_en]

europarl_id_to_word_fr={i:x for i,x in enumerate(europarl_word_list_fr)}
europarl_id_to_word_en={i:x for i,x in enumerate(europarl_word_list_en)}


## faisons sinkhorn maintenant
distrib1=europarl_emb_list_en
distrib2=europarl_emb_list_fr

freq_lang1=[frequences_en[x] for x in europarl_word_list_en]
freq_lang2=[frequences_fr[x] for x in europarl_word_list_fr]

epsilon = 0.01
niter = 100

# Wrap with torch tensors
X = torch.FloatTensor(distrib1)
Y = torch.FloatTensor(distrib2)

l1 = spc.FREQ_sinkhorn_loss(X,Y,epsilon,freq_lang1, freq_lang2,niter)

print("Sinkhorn loss : ", l1[0].data.item())

#l1[1][:,0].sum()   # on retrouve bien les fr�quences du d�but

MATRIX_FREQ=np.array(l1[1])
for n in range(MATRIX_FREQ.shape[0]):
    for m in range(MATRIX_FREQ.shape[1]):
        MATRIX_FREQ[n][m]=100*MATRIX_FREQ[n][m]


seuil=80
for n in range(MATRIX_FREQ.shape[0]):
    for m in range(MATRIX_FREQ.shape[1]):
        if MATRIX_FREQ[n][m]>seuil:
            en=src_id2word[n]
            es=tgt_id2word[m]
            print("traduction probable : {}  --->  {}".format(es,en))



