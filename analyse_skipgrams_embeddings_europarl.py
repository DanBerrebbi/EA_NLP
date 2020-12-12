from gensim.models import Word2Vec
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt


"""def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


def to_sentences(doc):
	return doc.strip().split('\n')



def split_for_skipgram(txt_file):
	doc=load_doc(txt_file)
	sentences= to_sentences(doc)
	LISTE=[]
	for phrase in sentences:
		LISTE.append(phrase.split())
	return LISTE


PATH = "C:\POLYTECHNIQUE\\3A\EANLP\Europarl_fr-en"
filename_fr = PATH+'\europarl-v7.fr-en.fr'

text=split_for_skipgram(filename_fr)

print("skipgram begining")
t1=time.time()

model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)

model.save("word2vec_europarl_fr.model")

print("skipgram over !")


print("temps : ", time.time()-t1)

PATH = "C:\POLYTECHNIQUE\\3A\EANLP\Europarl_fr-en"
filename_en = PATH+'\europarl-v7.fr-en.en'

text=split_for_skipgram(filename_en)

print("skipgram begining")
t1=time.time()

model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)

model.save("word2vec_europarl_en.model")

print("skipgram over !")


print("temps : ", time.time()-t1)"""

import numpy as np
import pickle
from gensim.models import Word2Vec


print("modules loaded")
print("loading model ...")

PATH="C:\POLYTECHNIQUE\\3A\EANLP\skipgram_models"

model_fr = Word2Vec.load(PATH+"\\word2vec_europarl_fr.model")
model_en = Word2Vec.load(PATH+"\\word2vec_europarl_en.model")



model_fr.corpus_total_words
model_fr.wv.most_similar("chat")

y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
n = [58, 651, 393, 203, 'papa']

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))

mots = ["chat", "chien", "bonjour", "fenêtre", "porte"]
words = ["cat", "dog", "hello", "window", 'door']


from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)


def plot_embeddings_monolingue(mots,  model, titre):
	embeddings = np.array([list(model.wv.get_vector(mot)) for mot in mots])
	low_dim_embs = pca.fit_transform(embeddings)
	X, Y = [a[0] for a in low_dim_embs], [a[1] for a in low_dim_embs]
	print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())
	fig, ax = plt.subplots()
	ax.scatter(X,Y)
	for i, mot in enumerate(mots):
		ax.annotate(mot, (X[i], Y[i]))
	plt.title(titre)
	plt.savefig(titre+".png")

plot_embeddings_monolingue(mots, model_fr, "skipgrams' embeddings of french words in Europarl")
plot_embeddings_monolingue(words, model_en, "skipgrams' embeddings of english words in Europarl")


def plot_embeddings_multilingue(mots1,  model1, mots2, model2, titre):
	embeddings1 = [list(model1.wv.get_vector(mot)) for mot in mots1]
	embeddings2 = [list(model2.wv.get_vector(mot)) for mot in mots2]
	mots=mots1+mots2
	embeddings = np.array(embeddings1 + embeddings2)
	low_dim_embs = pca.fit_transform(embeddings)
	X, Y = [a[0] for a in low_dim_embs], [a[1] for a in low_dim_embs]
	print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())
	fig, ax = plt.subplots()
	ax.scatter(X,Y)
	for i, mot in enumerate(mots):
		ax.annotate(mot, (X[i], Y[i]))
	plt.title(titre)
	plt.savefig(titre+".png")

plot_embeddings_multilingue(mots, model_fr, words, model_en, "skipgrams' embeddings of french and english words in Europarl")


########################################
##     on va maintenant comparer nos embeddings avec ceux de FastText
#########################################

embeddings = np.array([list(model_fr.wv.get_vector(mot)) for mot in mots])
low_dim_embs2 = pca.fit_transform(embeddings)
XX, YY = [a[0] for a in low_dim_embs2], [a[1] for a in low_dim_embs2]


def get_word_emb(word,src_emb, src_id2word):
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    return word_emb

low_dim_embs=pca.fit_transform(np.array(EMBEDDINGS))
#EMBEDDINGS sont es vecteurs de FastText pour les mots de la liste
X, Y = [a[0] for a in low_dim_embs], [a[1] for a in low_dim_embs]
print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.scatter(X, Y, color='r', label='Embeddings monolingue de FastText')
ax.scatter(XX, YY, color='b', label = 'Embeddings de skipgram entrainé sur Europarl')
ax.legend()
for i, mot in enumerate(mots):
	ax.annotate(mot, (X[i], Y[i]), color='r')
	ax.annotate(mot, (XX[i], YY[i]), color='b')

plt.title("Comparaison des embeddings de skipgram et de ceux de FastText")
plt.savefig("Comparaison des embeddings de skipgram et de ceux de FastText.png")


