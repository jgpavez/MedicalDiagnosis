'''
    Data augmentation strategies for facts lists
'''

from bs4 import BeautifulSoup
import urllib
import pdb
import json
import os
import pandas as pd
import re
import numpy as np
import pickle

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from pattern.en import singularize,pluralize
import random

from utils import get_stories

random.seed(1234)
np.random.seed(1234)

def check_repeated(name,repeated_list):
    name = name.lower().strip()
    return name if not (name in repeated_list) else repeated_list[name]

def synonyms(data):
    augment_n = 10
    data_dict = dict((key,[val]) for val,key,_ in data)

    is_plural = lambda word: singularize(word) <> word
    stops = set(stopwords.words('english') + ['l'])

    for disease in data:
        for _ in range(augment_n):
            new_facts_list = []
            for fact in disease[0]:
                new_fact = fact[:]
                for k,word in enumerate(fact):
                    if word not in stops:
                        syn = wordnet.synsets(word)
                        if syn:
                            random_syn = syn[0]              
                            random_lemma = random.choice(random_syn.lemma_names())
                            random_lemma = pluralize(random_lemma) if is_plural(word)\
                                                else random_lemma
                            random_lemma = random_lemma.lower()
                            random_lemma = random_lemma.replace('_',' ')
                            random_lemma = random_lemma.replace('-',' ')
                            if ' ' in random_lemma:
                                continue
                            new_fact[k] = random_lemma
                new_facts_list.append(new_fact)
            #print new_facts_list
            data_dict[disease[1]].append(new_facts_list[:])
    return data_dict

    # TODO: this is adding the name of the disease by synonym, check it!

def remove(data,data_dict):

    num_delete = (3,15) # Number of facts to delete
    min_delete = 5 # delete only if you have more than 8 facts
    n_augment = 20

    for (values,name,_) in data:
        facts = data_dict[name]
        new_facts = []
        for k in range(n_augment):
            fact = random.choice(facts)
            n_facts = len(fact)
            if n_facts > min_delete:
                max_facts = num_delete[1] if n_facts > 15 else n_facts - 1
                min_facts = num_delete[0]
                n_choice = np.random.randint(min_facts, max_facts)
                choice = np.random.choice(n_facts, n_choice, replace=False)
                new_fact = [f for k,f in enumerate(fact) if k not in choice]
                data_dict[name].append(new_fact)
                new_facts.append(new_fact)
    return data_dict


def permute(data,data_dict):
    n_augment = 10

    for (values, name,_) in data:
        facts = data_dict[name]
        new_facts = []
        for k in range(n_augment):
            fact = random.choice(facts)
            n_facts = len(fact)
            permutations = np.random.permutation(n_facts)
            new_fact = np.array(fact[:])
            new_fact = new_fact[permutations]
            new_facts.append(new_fact)
            data_dict[name].append(list(new_fact))

    return data_dict


file_names = ['facts_list_all.txt']

training_set = []
test_set = []
data = []

for file_name in file_names:
    print 'Reading {0} .....'.format(file_name)
    read_file = open('data/{0}'.format(file_name), 'r')
    # Data in format:
    # [([[fact1],[fact2],..][answer])...]
    # where each fact is a list of words
    
    data += get_stories(read_file)

    read_file.close()


# Data augmenting strategies
#1. Changing randomly nouns by synonyms
print('Data augmentation: synonyms')
data_dict = synonyms(data)
print('Number of diseases: {0}'.format(len(data_dict)))
#2. Removing facts randomly
print('Data augmentation: removing')
data_dict = remove(data,data_dict)
#3. Changing facts order
print('Data augmentation: permutation')
data_dict = permute(data,data_dict)


for (values, name,_) in data:
    # Save training and test data
    data_len = len(data_dict[name])
    training_size = int(0.7 * data_len)
    test_size = int(0.3 * data_len)
    facts = np.array(data_dict[name])
    indexes = np.random.permutation(len(facts))
    training_facts = facts[indexes[:training_size]]
    test_facts = facts[indexes[training_size:]]
    training_set += zip(list(training_facts),[name]*len(training_facts))
    test_set += zip(list(test_facts),[name]*len(test_facts))   


print(len(training_set))
print(len(test_set))
pickle.dump(training_set,open('data/training_set.dat','w'))
pickle.dump(test_set,open('data/test_set.dat','w'))
print 'Saved'