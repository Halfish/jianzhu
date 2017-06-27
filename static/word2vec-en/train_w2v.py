from gensim.models import word2vec
import argparse

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname) s %(message)s',
                    datefmt = '%a, %d %d %Y %H: %M:%S')
#                    datefmt = '%a, %d %d %Y %H: %M:%S', filename = 'myapp.log', filemode='w')


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str, default='wiki_corpus.txt', help="corpus filename")
parser.add_argument("-n", "--savename", type=str, default='word2vec_en.model', help="model save name")
parser.add_argument("-s" , "--size", type=int, default=300, help="word embedding dimension")
parser.add_argument("-w" , "--workers", type=int, default=4, help="workers for word2vec training")
args = parser.parse_args()
print args

class MySentences(object):
    def __init__(self, corpusFilename):
        self.corpusFilename = corpusFilename

    def __iter__(self):
        for document in open(self.corpusFilename, 'r'):
            words = document.split(' ')
            words = [word.decode('utf-8') for word in words]
            yield words


def trainW2V():
    print 'start to train word2vec'
    sentences = MySentences(args.filename)
    model = word2vec.Word2Vec(sentences, size=args.size, window=5, min_count=20, workers=args.workers)
    model.init_sims(replace=True)
    print 'done training word2vec'
    print model

    model.save(args.savename)

if __name__ == '__main__':
    trainW2V()
