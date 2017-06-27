import jieba
import h5py
import logging
import numpy as np
from gensim.models.word2vec import Word2Vec
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

jieba.dt.tmp_dir = './'
jieba.initialize()

class JiebaCut(object):
    def __call__(self, sentence):
        return list(jieba.cut(sentence, cut_all=False))

class W2VEmbedding(object):
    def __init__(self, wvModel):
        self.wvModel = wvModel

    def __call__(self, sentence):
        words_emb = []
        for word in sentence:
            try:
                emb = self.wvModel[word]
                words_emb.append(emb)
            except KeyError:
                logging.debug("No Such Key as", word)
        words_emb = np.array(words_emb)
        return words_emb


class GlobalVariable(object):
    def __init__(self):
        resnet = models.resnet18(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet = resnet.eval().cuda()

        self.wvModel = Word2Vec.load('static/word2vec-chi/word2vec_news.model')

        self.img_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        self.sentence_transform = transforms.Compose([
            JiebaCut(),
            W2VEmbedding(self.wvModel),
        ])

        self.sentences = np.load('sentences.npy')

        try:
            with h5py.File('arch_feats.h5', 'r') as hf:
                self.arch_feats = hf['feats'][:]
                self.sentence_ids = hf['sentence_ids'][:]
        except:
            print 'failed to open arch_feats.h5'


