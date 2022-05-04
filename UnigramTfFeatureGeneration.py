import random
from collections import Counter

import numpy as np

from nltk.tokenize import word_tokenize
import codecs
#from dbn_outside.dbn.tensorflow import SupervisedDBNClassification
stopwords = codecs.open("hindi_stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')


# Creating a set of lexicons which is a kind of dictionary of words.
def create_lexicon(pos, neg):
    lexicon = []
    for file_name in [pos, neg]:
        with codecs.open(file_name, 'r',encoding='utf-8',errors='ignore') as f:
            contents = f.read()
            for line in contents.split('$'):
                data = line.strip('\n')
                if data:
                    all_words = word_tokenize(data)
                    lexicon += list(all_words)
    lexicons = []
    for word in lexicon:
        if not word in stopwords:
            lexicons.append(word)
    word_counts = Counter(lexicons)  # it will return kind of dictionary
    l2 = []
    for word in word_counts:
        if 60 > word_counts[word]:
            l2.append(word)
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []
    with codecs.open(sample, 'r', encoding="utf8",errors='ignore') as f:
        contents = f.read()
        for line in contents.split('$'):
            data = line.strip('\n')
            if data:
                all_words = word_tokenize(data)
                all_words_new = []
                for word in all_words:
                    if not word in stopwords:
                        all_words_new.append(word)
                features = np.zeros(len(lexicon))
                for word in all_words_new:
                    if word in lexicon:
                        idx = lexicon.index(word)
                        features[idx] = 1
                features = list(features)
                featureset.append([features, classification])
    return featureset


def create_feature_set_and_labels(pos, neg, test_size=0.2):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, 1)
    features += sample_handling(neg, lexicon, 0)
    #print(features)
    random.shuffle(features)
    features = np.array(features)
    # print(features)
    # print(len(features))
    training_size = int((1 - test_size) * len(features))

    x_train = list(features[:, 0][:training_size])  # taking features array upto training_size
    y_train = list(features[:, 1][:training_size])  # taking labels upto training_size

    x_test = list(features[:, 0][training_size:])
    y_test = list(features[:, 1][training_size:])
    return x_train, y_train, x_test, y_test



