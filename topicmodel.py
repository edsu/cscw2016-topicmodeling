#!/usr/bin/env python

import os
import re
import json
import codecs
import string
import argparse

from glob import glob
from gensim import corpora, models, similarities

punctuation = set(string.punctuation)

def papers():
    for filename in glob("data/*[0-9].txt"):
        yield words(filename)

def abstracts():
    for filename in glob("data/*-abstract.txt"):
        yield words(filename)

def words(filename):
    stop_words = get_stop_words('data/stop_words.txt')
    text = codecs.open(filename, 'r', 'utf8').read()
    words = text.lower().split(' ')
    new_words = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if word in stop_words:
            continue
        if re.match(r'^(@|\#|http)', word):
            continue
        if len(word) < 4:
            continue

        word = ''.join(ch for ch in word if ch not in punctuation)
        new_words.append(word)
    return new_words

def get_stop_words(path):
    stops = {}
    for word in open('data/stop_words.txt'):
        word = word.strip().lower()
        stops[word] = True
    return stops

def get_dictionary(path):
    if not os.path.isfile(path):
        dictionary = corpora.Dictionary(abstracts())
        dictionary.save(path)
    else:
        dictionary = corpora.Dictionary.load(path)
    return dictionary

def get_corpus(path, dictionary):
    def ids():
        for doc in abstracts():
            yield dictionary.doc2bow(doc)
    if not os.path.isfile(path):
        corpus = corpora.MmCorpus.serialize(path, ids())
    else:
        corpus = corpora.MmCorpus(path)
    return corpus

def topics(corpus=papers, num_words=5, num_topics=10):
    dictionary_file = "data/%s.dict" % corpus.__name__
    dictionary = get_dictionary(dictionary_file)

    model_file = "data/%s.mm" % corpus.__name__
    corpus = get_corpus(model_file, dictionary)

    lda = models.ldamodel.LdaModel(
        corpus, 
        id2word=dictionary,
        num_topics=num_topics
    )

    count = 0
    for topic in lda.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
        count += 1 
        print("topic #%i: %s" % (count, ', '.join(t[0] for t in topic[1])))
        print()
