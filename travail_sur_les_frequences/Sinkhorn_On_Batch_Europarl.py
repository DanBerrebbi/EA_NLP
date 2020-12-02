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



################################################
##      ON FAIT AVEC DES BATCHS
################################################

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

europarl_word_to_id_fr={i:x for x,i in europarl_id_to_word_fr.items()}
europarl_word_to_id_en={i:x for x,i in europarl_id_to_word_en.items()}


## faisons sinkhorn maintenant
# on ne garde que certains batchs !!!

from nltk.tokenize import word_tokenize

######################################
##  USEFUL FONCTIONS
######################################

def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')

##############################
## for batch work
##############################

def batch_words(debut, fin, filename):
	doc=load_doc(filename)
	sentences = to_sentences(doc)
	LISTE=[]
	for phrase in sentences[debut:fin]:
		LISTE+= word_tokenize(phrase)
	return set(LISTE)

#from work_on_europarl import batch_words

PATH = "C:\POLYTECHNIQUE\\3A\EANLP\Europarl_fr-en"
filename_fr= PATH+ '\europarl-v7.fr-en.fr'
filename_en = PATH+ '\europarl-v7.fr-en.en'

debut, fin = 0, 100
BATCH_WORDS_FR= batch_words(debut,fin,filename_fr)   # 1mn to run on single cpu (mon ordi)
BATCH_WORDS_EN=batch_words(debut, fin, filename_en)

print(len(BATCH_WORDS_FR))
print(len(BATCH_WORDS_EN))

batch_fr_id_list=[europarl_word_to_id_fr[mot.lower()] for mot in BATCH_WORDS_FR & europarl_word_to_id_fr.keys()]
print(len(batch_fr_id_list))    # on perd beaucoup , c'est bizarre --> à analyser

batch_fr_embeddings=[europarl_emb_list_fr[id] for id in batch_fr_id_list]

batch_fr_id2word={i:x for i,x in enumerate(BATCH_WORDS_FR& europarl_word_to_id_fr.keys())}

batch_fr_word2id={v:k for k,v in batch_fr_id2word.items()}


batch_en_id_list=[europarl_word_to_id_en[mot.lower()] for mot in BATCH_WORDS_EN & europarl_word_to_id_en.keys()]
print(len(batch_en_id_list))

batch_en_embeddings=[europarl_emb_list_en[id] for id in batch_en_id_list]

batch_en_id2word={i:x for i,x in enumerate(BATCH_WORDS_EN& europarl_word_to_id_en.keys())}

batch_en_word2id={v:k for k,v in batch_en_id2word.items()}


distrib1=batch_en_embeddings
distrib2=batch_fr_embeddings

freq_lang1=[frequences_en[word] for word in BATCH_WORDS_EN & europarl_word_to_id_en.keys()]
freq_lang2=[frequences_fr[x] for x in BATCH_WORDS_FR & europarl_word_to_id_fr.keys()]

epsilon = 0.01
niter = 150

# Wrap with torch tensors
X = torch.FloatTensor(distrib1)
Y = torch.FloatTensor(distrib2)

l1 = spc.FREQ_sinkhorn_loss(X,Y,epsilon,freq_lang1, freq_lang2,niter)

print("Sinkhorn loss : ", l1[0].data.item())


MATRIX_FREQ=np.array(l1[1])
for n in range(MATRIX_FREQ.shape[0]):
    coeff_ligne=sum([ x for x in MATRIX_FREQ[n]])
    for m in range(MATRIX_FREQ.shape[1]):
        if coeff_ligne!=0:
            MATRIX_FREQ[n][m]=100*MATRIX_FREQ[n][m]/coeff_ligne


############################################
##    Première analyse des résultats
############################################


liste_score=[MATRIX_FREQ[n].max() for n in range(len(MATRIX_FREQ))]
liste_score=np.array(liste_score)

print(batch_fr_id2word[130], batch_en_id2word[1])

seuil=95
for n in range(MATRIX_FREQ.shape[0]):
    for m in range(MATRIX_FREQ.shape[1]):
        if MATRIX_FREQ[n][m]>seuil:
            print(batch_fr_id2word[m], batch_en_id2word[n])
