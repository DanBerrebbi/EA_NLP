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


#################################################################
##          en fixant une population maintenant
#################################################################

population_fr = LISTE_TRAD_UNIF.keys() & LISTE_TRAD_FREQ.keys()
print(len(population_fr))

def evaluate_freq_diff_population(freq1, freq2, matching, population):
    lang1, lang2 = [x[0] for x in matching] , [x[1] for x in matching]
    DIFF=0
    for i in range(len(lang1)):
        assert lang1[i] in freq1.keys()
        assert lang2[i] in freq2.keys()
        if lang1[i] in population:
            f1, f2 = freq1[lang1[i]], freq2[lang2[i]]
            DIFF += abs(f2-f1)
    print(DIFF/len(population)*10000)

def dict_to_matching(dict):
    return [[a,b] for a,b in dict.items()]

evaluate_freq_diff_population(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_FREQ), population=population_fr)
evaluate_freq_diff_population(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_UNIF), population=population_fr)


def evaluate_freq_diff_population_3(freq1, freq2, matching, population):
    lang1, lang2 = [x[0] for x in matching] , [x[1] for x in matching]
    DIFF=0
    for i in range(len(lang1)):
        assert lang1[i] in freq1.keys()
        assert lang2[i] in freq2.keys()
        if lang1[i] in population:
            f1, f2 = freq1[lang1[i]], freq2[lang2[i]]
            DIFF += f2/f1
    print(DIFF/len(population_fr))

evaluate_freq_diff_population_3(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_FREQ), population=population_fr)
evaluate_freq_diff_population_3(freq1=frequences_fr, freq2=frequences_en, matching=dict_to_matching(LISTE_TRAD_UNIF), population=population_fr)
