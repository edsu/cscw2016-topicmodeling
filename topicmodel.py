#!/usr/bin/env python

import os
import re
import json
import codecs
import string
import argparse

from glob import glob
from gensim import corpora, models

def papers():
    for filename in glob("data/*[0-9].txt"):
        yield words(filename)

def abstracts():
    for filename in glob("data/*-abstract.txt"):
        yield words(filename)

def words(filename):
    text = codecs.open(filename, 'r', 'utf8').read()
    return [w for w in re.split(r'\W+', text) if w and len(w) >= 4]

def get_dictionary(sources):
    return corpora.Dictionary(sources())
    if refresh or not os.path.isfile(path):
        dictionary = corpora.Dictionary(sources())
        dictionary.save(path)
    else:
        dictionary = corpora.Dictionary.load(path)
    return dictionary

def remove_stopwords(sources, stopwords):
    def f():
        for doc in sources():
            new_doc = []
            for word in doc:
                if word.lower() not in stopwords:
                    new_doc.append(word)
            yield new_doc
    return f

def get_corpus(dictionary):
    def ids():
        for doc in abstracts():
            yield dictionary.doc2bow(doc)
    path = "data/corpus.mm"
    corpora.MmCorpus.serialize(path, ids())
    corpus = corpora.MmCorpus(path)
    return corpus

def topics(sources=papers, num_words=5, num_topics=5, passes=10, iterations=50, ignore=None):

    if ignore is not None:
        sources = remove_stopwords(sources, ignore)

    dictionary = get_dictionary(sources)
    corpus = get_corpus(dictionary)

    lda = models.ldamodel.LdaModel(
        corpus, 
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations
    )

    topics = lda.top_topics(corpus, num_words=num_words)

    num = 0
    for topic in topics:
        num += 1
        print("%s. %s" % (num, ', '.join([t[1] for t in topic[0]])))

stopwords = set([w.strip() for w in open("data/stopwords.txt")])
