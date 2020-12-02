from gensim.models import Word2Vec
import pickle
import time
import numpy as np


def load_doc(filename):
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
    doc = load_doc(txt_file)
    sentences = to_sentences(doc)
    LISTE=[]
    for phrase in sentences:
        LISTE.append(phrase.split())
    return LISTE



PATH = "/data/projet_citations/EANLP/Europarl_fr-en"
filename_fr = PATH+'/europarl-v7.fr-en.fr'

text=split_for_skipgram(filename_fr)

print("skipgram begining")
t1=time.time()

model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)

model.save("word2vec_europarl_fr.model")

print("skipgram over !")


print("temps : ", time.time()-t1)


filename_en =PATH+ '/europarl-v7.fr-en.en'

text=split_for_skipgram(filename_en)

print("skipgram begining")
t1=time.time()

model = Word2Vec(sentences=text, window=5, min_count=1, workers=4)

model.save("word2vec_europarl_en.model")

print("skipgram over !")


print("temps : ", time.time()-t1)


