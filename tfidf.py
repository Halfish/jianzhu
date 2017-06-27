import re
import jieba
jieba.dt.tmp_dir = './'
jieba.initialize()

import json
data_pair = [json.loads(data) for data in open('static/dataset/data_pair.json', 'r')]
print len(data_pair)
print data_pair[0].keys()

class Dictionary(object):
    def __init__(self):
        self.index2word = []
        self.word2index = {}
        self.wordcount = {}
        self.stopwords = [word.strip().decode('utf-8') for word in open('static/dataset/stopwords.txt', 'r')]

    def update(self, words):
        words = re.sub(u'[\r\t\n ]', u' ', words)
        words = list(jieba.cut_for_search(words))
        words = filter(lambda x: x not in self.stopwords, words)
        for word in words:
            if self.word2index.has_key(word):
                self.wordcount[word] += 1
            else:
                self.wordcount[word] = 1
                self.index2word.append(word)
                self.word2index[word] = len(self.index2word) - 1

    def shrink(self, min_count=2):
        index2word, word2index, wordcount = [], {}, {}
        for word in self.index2word:
            if self.wordcount[word] >= min_count:
                index2word.append(word)
                word2index[word] = len(index2word) - 1
                wordcount[word] = self.wordcount[word]
        self.index2word = index2word
        self.word2index = word2index
        self.wordcount = wordcount

dic = Dictionary()
for data in data_pair:
    dic.update(data['title'])
print 'have %d words before shrink' %(len(dic.index2word))
dic.shrink(min_count=5)
print 'have %d words after shrink' %(len(dic.index2word))

import numpy as np

class Tfidf(object):
    def __init__(self, dic):
        self.dic = dic
        self.term_freq = []
        self.inverse_doc_freq = {key:0 for key, value in dic.wordcount.iteritems()}
        self.n_docs = 0

    def _clean(self, words):
        words = re.sub('[\r\t\n ]', ' ', words)
        words = list(jieba.cut_for_search(words))
        words = filter(lambda x: (x not in self.dic.stopwords) and (x in self.dic.wordcount), words)
        return words

    def update(self, words):
        words = self._clean(words)
        #assert len(words) > 0, 'detect useless doc'
        doc_tf = {word:0 for word in words}
        for word in words:
            doc_tf[word] += 1
        self.term_freq.append(doc_tf)
        for term in doc_tf:
            self.inverse_doc_freq[term] += 1
        self.n_docs += 1

    def parse(self, words):
        words = self._clean(words)
        doc_tf = {word:0 for word in words}
        for word in words:
            doc_tf[word] += 1
        return self._get_tfidf(doc_tf)

    def __getitem__(self, index):
        assert index < self.n_docs, 'out of range'
        return self._get_tfidf(self.term_freq[index])

    def __iter__(self):
        for i in range(self.n_docs):
            yield self[i]

    def _get_tfidf(self, tf):
        tfidf = {}
        for word, freq in tf.iteritems():
            tfidf[word] = freq * np.log(self.n_docs / (1e-6 + self.inverse_doc_freq[word]))
        tfidf = sorted(tfidf.iteritems(), key=lambda x: x[1], reverse=True)
        tfidf = {k:v for k, v in tfidf} # list to dict
        return tfidf

    def to_numpy(self, tfidf):
        doc = np.zeros(len(self.dic.wordcount))
        for word, score in tfidf.iteritems():
            doc[self.dic.word2index[word]] = score
        return doc

    def numpy(self):
        return np.array([self.to_numpy(tfidf) for tfidf in self])

tfidf = Tfidf(dic)
for data in data_pair[0:100]:
    tfidf.update(data['title'])

docs = tfidf.numpy()
print docs.shape
