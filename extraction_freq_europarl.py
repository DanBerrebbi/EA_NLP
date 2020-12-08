import numpy as np
from nltk.tokenize import word_tokenize


# load doc into memory
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

# shortest and longest sentence lengths
def sentence_lengths(sentences):
	lengths = [len(s.split()) for s in sentences]
	return min(lengths), max(lengths)


PATH = "C:\POLYTECHNIQUE\\3A\EANLP\Europarl_fr-en"

"""# load English data
filename = PATH+ '\europarl-v7.fr-en.en'
doc = load_doc(filename)
sentences = to_sentences(doc)
minlen, maxlen = sentence_lengths(sentences)
print('English data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))

# load French data
filename = PATH+'\europarl-v7.fr-en.fr'
doc = load_doc(filename)
sentences = to_sentences(doc)
minlen, maxlen = sentence_lengths(sentences)
print('French data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))"""



from collections import Counter

def occurences(filename):
	doc=load_doc(filename)
	sentences = to_sentences(doc)

	LISTE= []
	for phrase in sentences :
		LISTE+= word_tokenize(phrase)
	dico = Counter(LISTE)
	return dico

filename_fr = PATH+'\europarl-v7.fr-en.fr'
occurences_fr = occurences(filename_fr)

print("aaa")


filename_en = PATH+ '\europarl-v7.fr-en.en'
occurences_en = occurences(filename_en)


# apres ca c'est facile

import pickle

with open("occurences_en.pkl","wb") as f:
	pickle.dump(occurences_en,f)


####################################
# transforming occurence counts into frequences
#####################################

import pickle

occurences_fr=pickle.load(open("occurences_fr.pkl", "rb"))
occurences_en=pickle.load(open("occurences_en.pkl", "rb"))

fr_count=sum([x for x in occurences_fr.values()])
en_count=sum([x for x in occurences_en.values()])

frequences_fr={k:v/fr_count for k,v in occurences_fr.items()}
frequences_en={k:v/en_count for k,v in occurences_en.items()}


"""with open("frequences_fr.pkl","wb") as f:
	pickle.dump(frequences_fr,f)"""


##############################
## for batch work
##############################

def batch_words(debut, fin, filename):
	doc=load_doc(filename)
	sentences = to_sentences(doc)
	LISTE=[]
	for phrase in sentences[debut:fin]:
		LISTE+= phrase.split()
	return set(LISTE)
