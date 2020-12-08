import json

seuil=50
LISTE_TRAD_FREQ=json.load(open("dumped_trad_freq_seuil{}.json".format(seuil),"r"))
LISTE_TRAD_UNIF=json.load(open("dumped_trad_unif_seuil{}.json".format(seuil),"r"))


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


import pickle

frequences_fr = pickle.load(open("frequences_fr.pkl",'rb'))
frequences_en = pickle.load(open("frequences_en.pkl",'rb'))



def evaluate_freq_diff(freq1, freq2, matching):
    lang1, lang2 = [x[0] for x in matching] , [x[1] for x in matching]
    DIFF=0
    for i in range(len(lang1)):
        assert lang1[i] in freq1.keys()
        assert lang2[i] in freq2.keys()
        f1, f2 = freq1[lang1[i]], freq2[lang2[i]]
        DIFF += abs(f2-f1)
    print(DIFF/len(matching)*10000)

def dict_to_matching(dict):
    return [[a,b] for a,b in dict.items()]

evaluate_freq_diff(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_FREQ))
evaluate_freq_diff(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_UNIF))




###########################################################
###      bizarre , on devrait obtenir le contraire
###########################################################

seuil=95
LISTE_TRAD_FREQ_100 = json.load(open("dumped_trad_freq_seuil{}_top100.json".format(seuil),"r"))

LISTE_TRAD_UNIF_100 = json.load(open("dumped_trad_unif_seuil{}_top100.json".format(seuil),"r"))



accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_FREQ_100)
accuracy(dico_true=fr2en, dico_pred=LISTE_TRAD_UNIF_100)

evaluate_freq_diff(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_FREQ_100))
evaluate_freq_diff(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_UNIF_100))


##################################################################
##          Je tente un nouveau truc
##################################################################


def evaluate_freq_diff_2(freq1, freq2, matching):
    lang1, lang2 = [x[0] for x in matching] , [x[1] for x in matching]
    DIFF=0
    for i in range(len(lang1)):
        assert lang1[i] in freq1.keys()
        assert lang2[i] in freq2.keys()
        f1, f2 = freq1[lang1[i]], freq2[lang2[i]]
        DIFF += abs(f2-f1)/(f1+f2)
    print(DIFF/len(matching)*10000)

def dict_to_matching(dict):
    return [[a,b] for a,b in dict.items()]

evaluate_freq_diff_2(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_FREQ))
evaluate_freq_diff_2(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_UNIF))



######################################################################
###             encore un autre autre
######################################################################

def evaluate_freq_diff_3(freq1, freq2, matching):
    lang1, lang2 = [x[0] for x in matching] , [x[1] for x in matching]
    DIFF=0
    for i in range(len(lang1)):
        assert lang1[i] in freq1.keys()
        assert lang2[i] in freq2.keys()
        f1, f2 = freq1[lang1[i]], freq2[lang2[i]]
        DIFF += f2/f1
    print(DIFF/len(matching))

def dict_to_matching(dict):
    return [[a,b] for a,b in dict.items()]

evaluate_freq_diff_3(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_FREQ))
evaluate_freq_diff_3(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_UNIF))

