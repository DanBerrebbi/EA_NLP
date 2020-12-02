import numpy as np
from numpy import dot
from numpy.linalg import norm

def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def list_emb_to_muse(list_emb): # list_emb est du format retourné par la fonction read
    embeddings, id2word, word2id = [], {}, {}
    taille_voc = len(list_emb[0])
    for k in range(taille_voc):
        embeddings.append(normalize(list_emb[1][k]))
        id2word[k]=list_emb[0][k]
        word2id[list_emb[0][k]]=k
    return np.array(embeddings), id2word, word2id



def read_dico(file):
    lines = file.readlines()
    matching = []
    for line in lines:
        s = line.split(' ')
        matching.append([s[0], s[1][:-1]])
    return matching




f=open(r"C:\POLYTECHNIQUE\3A\EANLP\vecmap-master\en.emb.txt",errors='surrogateescape',encoding='utf-8')
emb_en=read(f)
src_embeddings, src_id2word, src_word2id = list_emb_to_muse(emb_en)
f.close()

f=open(r"C:\POLYTECHNIQUE\3A\EANLP\vecmap-master\es.emb.txt",errors='surrogateescape',encoding='utf-8')
emb_es=read(f)
tgt_embeddings, tgt_id2word, tgt_word2id = list_emb_to_muse(emb_es)
f.close()

del emb_en
del emb_es

f=open(r"C:\POLYTECHNIQUE\3A\EANLP\vecmap-master\dictionaries\en-es.train.txt",errors='surrogateescape',encoding='utf-8')
matching = read_dico(f)
f.close()


#######################################################
##      on ne garde que les mots du dictionnaire
########################################################

# TTI car là on met les bons id

def keep_only_dictionnary(src_embeddings, src_id2word, src_word2id, tgt_embeddings, tgt_id2word, tgt_word2id, dico):
    src_embeddings2, src_id2word2, src_word2id2=[], {}, {}
    tgt_embeddings2, tgt_id2word2, tgt_word2id2=[], {}, {}
    curr_id=0
    for couple in dico:
        src, tgt = couple
        id_src = src_word2id[src]
        id_tgt = tgt_word2id[tgt]
        src_embeddings2.append(src_embeddings[id_src])
        tgt_embeddings2.append(tgt_embeddings[id_tgt])
        src_id2word2[curr_id]=src
        tgt_id2word2[curr_id]=tgt
        src_word2id2[src]=curr_id
        tgt_word2id2[tgt]=curr_id
        curr_id+=1
        print(curr_id)
    return src_embeddings2, src_id2word2, src_word2id2, tgt_embeddings2, tgt_id2word2, tgt_word2id2


results=keep_only_dictionnary(src_embeddings, src_id2word, src_word2id, tgt_embeddings, tgt_id2word, tgt_word2id, matching)

src_embeddings_5000, src_id2word_5000, src_word2id_5000, tgt_embeddings_5000, tgt_id2word_5000, tgt_word2id_5000 = results




###############################################################
##     CCA
###############################################################


import numpy as np
from sklearn.cross_decomposition.cca_ import CCA
import time

t1=time.time()

cca=CCA(n_components=10)
cca.fit(X=src_embeddings_5000[300:400], Y=tgt_embeddings_5000[300:400])

print(time.time()-t1)

X_c, Y_c = cca.transform(X=src_embeddings_5000, Y=tgt_embeddings_5000)



import matplotlib.pyplot as plt

X=[]
Y_orig, Y_cca = [], []

for k in range(1000,2000, 10):
    X.append(k)
    Y_orig.append(cos(src_embeddings_5000[k],tgt_embeddings_5000[k]))
    Y_cca.append(cos(X_c[k],Y_c[k]))

Y_orig, Y_cca= np.array(Y_orig), np.array(Y_cca)
print("Initial mean : {}    Initial std : {}".format(Y_orig.mean(),Y_orig.std()))
print("After cca mean : {}      After cca std : {}".format(Y_cca.mean(),Y_cca.std()))

plt.plot(X,Y_orig, color="r")
plt.plot(X,Y_cca, color="b")
#plot the means
plt.plot(X,[Y_orig.mean() for _ in X], color="r")
plt.plot(X,[Y_cca.mean() for _ in X], color="b")
plt.savefig("cosinus_avant_et_apres_cca_avec_10_composantes.png")


DIM, MOY, STD, TEMPS = [], [], [], []

for dim in range(5,100,5):
    # perform cca
    t1 = time.time()

    cca = CCA(n_components=dim)
    cca.fit(X=src_embeddings_5000[1000:1200], Y=tgt_embeddings_5000[1000:1200])
    X_c, Y_c = cca.transform(X=src_embeddings_5000, Y=tgt_embeddings_5000)

    print("dim ",dim, "    temps : ",time.time() - t1)
    TEMPS.append(time.time() - t1)
    Y_cca=[]
    for k in range(2000, 5000):
        Y_cca.append(cos(X_c[k], Y_c[k]))
    Y_cca=np.array(Y_cca)
    DIM.append(dim)
    MOY.append(Y_cca.mean())
    STD.append(Y_cca.std())

plt.plot(DIM, MOY, color="r")
plt.plot(DIM, STD, color="b")
plt.plot(DIM, TEMPS, color = "g")
