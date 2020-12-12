# EA_NLP
Projet d'EA sous la supervision de Mr. Francois Yvon

# Liens utiles : 

1) github de [Facebook MUSE](https://github.com/facebookresearch/MUSE)
2) embeddings monolingues de [FastText (Facebook)](https://fasttext.cc/docs/en/pretrained-vectors.html)
3) Dataset [Europarl](http://www.statmt.org/europarl/)
4) [Dictionnaires et Embeddings](https://github.com/artetxem/vecmap/blob/master/get_data.sh) produits par Mikel Artetxe 
5) Une implémentation possible de l'[algorithme de Sinkhorn](https://github.com/gpeyre/SinkhornAutoDiff)


# Premières implémentations : 

## SkipGram : 

Nous avons utlisé l'imentation Gensim de Word2Vec (SkipGram) pour obtenir les vecteurs des mots présents dans le corpus parallèle d'Europarl. Nous avons fait cela dans le fichier [skipgram.py](https://github.com/DanBerrebbi/EA_NLP/blob/main/skipgram.py), puis comparons les vecteurs obtenus en français avec ceux de FastText. Cette comparaison est faite dans le fichier [analyse_skipgrams_embeddings_europarl.py](https://github.com/DanBerrebbi/EA_NLP/blob/main/analyse_skipgrams_embeddings_europarl.py). 

![alt text](https://github.com/DanBerrebbi/EA_NLP/blob/main/Comparaison%20des%20embeddings%20de%20skipgram%20et%20de%20ceux%20de%20FastText.png) 

## CCA : 

Nous avons repris l'idée de Faruqui et Dyer et avons utilisé une CCA pour projeter les mots et leur traduction dans un espace commun. L'implémentation de cette méthode est dans le fichier [CCA.py](https://github.com/DanBerrebbi/EA_NLP/blob/main/CCA.py).

![alt text](https://github.com/DanBerrebbi/EA_NLP/blob/main/cosinus_avant_et_apres_cca_avec_10_composantes.png) 


# Travail sur les fréquences 


