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

mots = ["chat", "chien", "bonjour", "fenêtre"]
words = ["cat", "dog", "hello", "window"]


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
