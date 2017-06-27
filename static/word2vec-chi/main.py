import jieba
from gensim.models import word2vec

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname) s %(message)s',
                    datefmt = '%a, %d %d %Y %H: %M:%S')
#                    datefmt = '%a, %d %d %Y %H: %M:%S', filename = 'myapp.log', filemode='w')

class MySentences(object):
    def __init__(self, corpusFilename):
        self.corpusFilename = corpusFilename

    def __iter__(self):
        for document in open(self.corpusFilename, 'r'):
            words = list(jieba.cut(document[9:-11], cut_all=False))
            yield words


def trainW2V():
    print 'start to train word2vec'
    sentences = MySentences('full_corpus.txt')
    model = word2vec.Word2Vec(sentences, size=300, window=5, min_count=20, workers=10)
    model.init_sims(replace=True)
    print 'done training word2vec'
    print model

    model.save('word2vec_news.model')

if __name__ == '__main__':
    trainW2V()
