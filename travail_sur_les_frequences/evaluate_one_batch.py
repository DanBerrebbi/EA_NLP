import json

seuil=95
LISTE_TRAD_FREQ=json.load(open("dumped_trad_freq_batch_2_seuil{}.json".format(seuil),"r"))
LISTE_TRAD_UNIF=json.load(open("dumped_trad_unif_batch_2_seuil{}.json".format(seuil),"r"))


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

def to_trad(liste):
    return [x.split() for x in liste]

def to_french_dico(liste):
    dico={}
    for couple in liste:
        a,b = couple
        if b not in dico.keys():
            dico[b]=[a]
        else:
            dico[b].append(a)
    return dico


DICO = load_doc("C:\POLYTECHNIQUE\\3A\EANLP\MUSE-master\data\crosslingual\dictionaries\\en-fr.txt")
DICO = to_sentences(DICO)

aux=to_trad(DICO)
fr2en=to_french_dico(aux)


######################################################
##          evaluation
######################################################

def accuracy(dico_true, dico_pred):
    keys= list( dico_true.keys() & dico_pred.keys())
    ok=0.
    for fr in keys:
        if dico_pred[fr] in dico_true[fr]:
            ok+=1.
    print(len(keys))
    print(ok)
    print(round(ok/len(keys),2))

accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_FREQ)
accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_UNIF)



seuil=80
LISTE_TRAD_FREQ=json.load(open("dumped_trad_freq_batch_2_seuil{}.json".format(seuil),"r"))
LISTE_TRAD_UNIF=json.load(open("dumped_trad_unif_batch_2_seuil{}.json".format(seuil),"r"))

accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_FREQ)
accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_UNIF)



seuil=50
LISTE_TRAD_FREQ=json.load(open("dumped_trad_freq_batch_2_seuil{}.json".format(seuil),"r"))
LISTE_TRAD_UNIF=json.load(open("dumped_trad_unif_batch_2_seuil{}.json".format(seuil),"r"))

accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_FREQ)
accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_UNIF)

