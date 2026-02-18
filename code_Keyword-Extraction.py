#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:44:10 2023

@author: adalgisaperrelli e giorgialandonio
"""

#Fonte dataset: Xiaotao Gu*, Zihan Wang*, Zhenyu Bi, Yu Meng, Liyuan Liu, Jiawei Han, Jingbo Shang, 
##"UCPhrase: Unsupervised Context-aware Quality Phrase Tagging", in Proc. of 2021 ACM SIGKDD Int. 
##Conf. on Knowledge Discovery and Data Mining (KDD'21), Aug. 2021

#Il dataset kp5k raccoglie metadati da diverse librerie online digitali come: ACM, WebofScience e Wiley; i metadati sono relativi a 5.000 paper

#Il dataset contiene 5.000 righe e ciascuna riga presenta una lista di stringhe, ovvero frasi estratte dal documento di riferimento

# Importiamo il file
import jsonlines
import pandas as pd

def read_jsonl_file(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader.iter():
            data.append(line)
    return data

file_path = 'C:/Users/utente/OneDrive/Desktop/DATA PROCESSING AND ANALYSIS/LAVORO ESAME/kp5k.jsonl'
jsonl_data = read_jsonl_file(file_path)

# Stampiamo il file letto
for item in jsonl_data:
    print(item)

# Convertiamo il file in formato jsonl in un dataset
def convert_jsonl_to_dataset(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader.iter():
            data.append(line)
    
    df = pd.DataFrame(data)
    return df


dataset = convert_jsonl_to_dataset(file_path)

print(dataset)



############################################ FASE 1: PREPROCESSING     
  

from tqdm import tqdm
tqdm.pandas()
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import re
from nltk.stem import PorterStemmer




#1. Convertiamo tutti i caratteri in minuscolo e sostituiamo i caratteri presenti nella lista [^\w\s\d] con lo spazio vuoto qualora fossero presenti
def lower_function(element):
    
    return element.lower()

dataset["sents_lower"] = dataset["sents"].apply(lambda row: [lower_function(element) for element in row])
print(dataset)


def remove_characters(string):
    pattern = r'[^\w\s\d]'
    cleaned_string = re.sub(pattern, '', string)
    return cleaned_string

dataset["sents_removechar"] = dataset["sents_lower"].apply(lambda row: [remove_characters(element) for element in row])
print(dataset)


#2. Stemming
def split_function(element):

    return element.split()

dataset["sents_split"] = dataset["sents_removechar"].apply(lambda row: [split_function(element) for element in row])
print(dataset)


stemmer = PorterStemmer()

dataset_stemmed = []
for riga in dataset["sents_split"]:
    riga_stemmed = [tuple(stemmer.stem(parola) for parola in tupla) for tupla in riga]
    dataset_stemmed.append(riga_stemmed)


for riga_stemmed in dataset_stemmed:
    print(riga_stemmed)
 

#3. Rimozione delle stop word
def import_stop_words(file_path):
    with open(file_path, 'r') as file:
        stop_words = file.read().splitlines()
    return stop_words

elenco_stopwords = import_stop_words('C:/Users/utente/OneDrive/Desktop/DATA PROCESSING AND ANALYSIS/LAVORO ESAME/stopwords.txt')


def remove_stop_words(tuple_words, el_stopwords):
    filtered_words = [word for word in tuple_words if word not in el_stopwords]
    return tuple(filtered_words)

dataset_filtered = []
for riga in dataset_stemmed:
    riga_filtered = [remove_stop_words(tupla, elenco_stopwords) for tupla in riga]
    dataset_filtered.append(riga_filtered)
    




############################################ FASE 2: VALUTAZIONE DELLE FRASI CANDIDATE


## Metodo I: Calcolo della term-frequency (tf)
from collections import Counter

# Prepariamo il dataset *dataset_filtered* per il calcolo del tf unendo le stringhe contenute nelle tuple all'interno delle liste che costituiscono le righe del dataset    
ds_per_tfidf = []
for riga in dataset_filtered:
    riga_tfidf = [' '.join(tupla) for tupla in riga]
    ds_per_tfidf.append(riga_tfidf)

for riga_tfidf in ds_per_tfidf:
    print(riga_tfidf)

def concatena_stringhe(dataset):
    risultati = []
    for riga in dataset:
        stringa_concatenata = ' '.join(riga)
        risultati.append(stringa_concatenata)
    return risultati
liste_complete = concatena_stringhe(ds_per_tfidf) # La lista *liste_complete* contiene le stringhe ottenute concatenando le unità testuali di ogni documento


# Assumiamo che ogni stringa di *liste_complete* rappresenti un documento


def crea_dizionario_conteggio(dataset):
    tf_matrix = []

    for riga in dataset:
        conteggio_parole = Counter(riga.split()) # La funzione "Counter" conta il numero di volte in cui ogni parola occorre nella riga del dataset in cui è contenuta
        parole_filtrate = {parola: conteggio for parola, conteggio in conteggio_parole.items() if conteggio > 2}
        parole_ordinate = dict(sorted(parole_filtrate.items(), key=lambda item: item[1], reverse=True))
        tf_matrix.append(parole_ordinate)

    return tf_matrix


tf_matrix = crea_dizionario_conteggio(liste_complete) # Si tratta di una lista di dizionari; all'interno di ogni dizionario le chiavi rappresentano le parole più frequenti nel documento e i valori il numero di occorrenze, ovvero il valore del tf

print(tf_matrix)

##################################################################

# Metodo II: Attribuzione del punteggio d'importanza alle frasi candidate sulla base della presenza di parole chiave

from nltk.tokenize import sent_tokenize, word_tokenize


# Leggiamo il file json contenete le keyword
import json

file_keyw = "C:/Users/utente/OneDrive/Desktop/DATA PROCESSING AND ANALYSIS/LAVORO ESAME/test.json"
dati_json = []

with open(file_keyw, 'r') as file:
    for riga in file:
        try:
            oggetto_json = json.loads(riga)
            dati_json.append(oggetto_json)
        except json.JSONDecodeError as e:
            print(f"Errore: {e}")

print(dati_json)

# Convertiamo il file json letto in un dataset
dataset_keyw = pd.DataFrame(dati_json)
print(dataset_keyw)


dataset_keyw["kw_split"] = dataset_keyw["keywords"].apply(lambda x: [word.strip() for phrase in x.split(';') for word in phrase.split()])
print(dataset_keyw.head(5))



def calculate_importance_score(sentence, dataset_keyw):
    tokens = word_tokenize(sentence)
    importance_score = 0

    for keywords in dataset_keyw["kw_split"]:
        for keyword in keywords:
            if keyword in [token.lower() for token in tokens]:
                importance_score += 1
    
    return importance_score

diz_punteggi_keyw = {}
for idx, document in enumerate(dataset["sents_removechar"]):
    if isinstance(document, list):
        diz_punteggi_keyw[idx] = {}
        for sentence in document:
            if isinstance(sentence, str):
                sentences = sent_tokenize(sentence)
                for sentence_token in sentences:
                    importance_score = calculate_importance_score(sentence_token, dataset_keyw)
                    diz_punteggi_keyw[idx][sentence_token] = importance_score 
    
# Otteniamo un dizionario di dizionari, in cui per ogni documento è associata una chiave 
## e il rispettivo valore è un dizionario in cui la chiave è 
## la frase e il valore associato è il suo punteggio di importanza.

print(diz_punteggi_keyw)

##################################################################
    
# Metodo III: Attribuzione del punteggio d'importanza alle frasi candidate sulla base del POS tagging


'''
CC coordinating conjunction 
CD cardinal digit 
DT determiner 
EX existential there (like: “there is” … think of it like “there exists”) 
FW foreign word 
IN preposition/subordinating conjunction 
JJ adjective – ‘big’ 
JJR adjective, comparative – ‘bigger’ 
JJS adjective, superlative – ‘biggest’ 
LS list marker 1) 
MD modal – could, will 
NN noun, singular ‘- desk’ 
NNS noun plural – ‘desks’ 
NNP proper noun, singular – ‘Harrison’ 
NNPS proper noun, plural – ‘Americans’ 
PDT predeterminer – ‘all the kids’ 
POS possessive ending parent’s 
PRP personal pronoun –  I, he, she 
PRP$ possessive pronoun – my, his, hers 
RB adverb – very, silently, 
RBR adverb, comparative – better 
RBS adverb, superlative – best 
RP particle – give up 
TO – to go ‘to’ the store. 
UH interjection – errrrrrrrm 
VB verb, base form – take 
VBD verb, past tense – took 
VBG verb, gerund/present participle – taking 
VBN verb, past participle – taken 
VBP verb, sing. present, non-3d – take 
VBZ verb, 3rd person sing. present – takes 
WDT wh-determiner – which 
WP wh-pronoun – who, what 
WP$ possessive wh-pronoun, eg- whose 
WRB wh-adverb, eg- where, when
'''
lista_POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 
             'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH',
             'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


combinazioni_POS = [
    [("NNP",), ("VB",)],
    [("JJ", "NN"), ("VB",)],
    [("NN",), ("VBD",)],
    [("NNP",), ("VBD",)],
    [("NNP",), ("VBG",)],
    [("NN",), ("VBG",)],
    [("NNP",), ("VBN",)],
    [("NN",), ("VBN",)],
    [("NNP",), ("VBP",)],
    [("NN",), ("VBP",)],
    [("NNP",), ("VBZ",)],
    [("NN",), ("VBZ",)]
]


def calculate_importance_score1(sentence):
    from nltk import pos_tag
    insieme_tokens = word_tokenize(sentence)
    tokens_tagged = pos_tag(insieme_tokens)
    importance_score = 0
   
    
    for combinazione in combinazioni_POS:
        found_combination = False
        tag_positions = [0] * len(combinazione)
        
        for i, tag in enumerate(combinazione):
            for j, (_, pos_tag1) in enumerate(tokens_tagged[tag_positions[i]:]):
                if pos_tag1 == tag[0]:
                    tag_positions[i] = tag_positions[i] + j
                    break
            else:
                break
        else:
            found_combination = True

        if found_combination:
            importance_score += 1
    
    return importance_score

diz_punteggi_POS = {}
for idx, document in enumerate(dataset["sents_removechar"]):
    if isinstance(document, list):
        diz_punteggi_POS[idx] = {}
        for sentence in document:
            if isinstance(sentence, str):
                sentences = sent_tokenize(sentence)
                for sentence_token in sentences:
                    importance_score = calculate_importance_score1(sentence_token)
                    diz_punteggi_POS[idx][sentence_token] = importance_score 
    

print(diz_punteggi_POS)



############################################ FASE 3: CLASSIFICA ED ESTRAZIONE DELLE FRASI CHIAVE

###### 1. Classifica delle parole contenute in ciascun documento, ovvero ciascuna riga del dataset, in base al valore del tf
def crea_dizionario_conteggio(dataset):
    tf_matrix = []

    for riga in dataset:
        conteggio_parole = Counter(riga.split())
        parole_filtrate = {parola: conteggio for parola, conteggio 
                           in conteggio_parole.items() if conteggio > 2}
        parole_ordinate = dict(sorted(parole_filtrate.items(), key=lambda item: item[1], reverse=True))
        tf_matrix.append(parole_ordinate)

    return tf_matrix

tf_matrix = crea_dizionario_conteggio(liste_complete)

print(tf_matrix)



###### 2. Classifica delle frasi candidate rispetto al valore di importanza ottenuto in base alla presenza di parole chiave
for chiave in diz_punteggi_keyw:
    diz_punteggi_keyw[chiave] = dict(sorted(diz_punteggi_keyw[chiave].items(), 
                                            key=lambda item: item[1], reverse=True))
print(diz_punteggi_keyw)

for key, sub_dict in diz_punteggi_keyw.items():
    top_3_keys = sorted(sub_dict, key=sub_dict.get, reverse=True)[:3]
    print(f"Top 3 chiavi per il dizionario {key}: {top_3_keys}")



###### 3. Classifica delle frasi candidate rispetto al valore di importanza ottenuto in base alla presenza di combinazioni di POS tag
for chiave in diz_punteggi_POS:
    diz_punteggi_POS[chiave] = dict(sorted(diz_punteggi_POS[chiave].items(), 
                                            key=lambda item: item[1], reverse=True))
print(diz_punteggi_POS)

for key, sub_dict in diz_punteggi_POS.items():
    top_3_keys = sorted(sub_dict, key=sub_dict.get, reverse=True)[:3]
    print(f"Top 3 chiavi per il dizionario {key}: {top_3_keys}")








