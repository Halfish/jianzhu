#coding=utf-8

from torch.autograd import Variable
from linecache_light import LineCache
from configure import GlobalVariable

import numpy as np
import json
import torch
import urllib

gvar = GlobalVariable()
gvar.resnet = gvar.resnet.cuda()

model = torch.load('static/models/multimodal.new.pt')
sentence_static = np.load('static/models/sentence_static.new.npy')
image_static = np.load('static/models/image_static.new.npy')[0]
data_pair = LineCache('static/dataset/data_pair.json')


def find_sentences(query_vector):
    dists = np.square(sentence_static - query_vector).sum(1)
    index = dists.argsort()
    dists = dists[index]
    similar_sentences = [json.loads(t)['title'] for t in data_pair[index[0:5]]]
    return similar_sentences


def find_images(query_vector):
    dists = np.square(image_static['images'] - query_vector).sum(1)
    index = dists.argsort()
    dists = dists[index]
    temp_sentence_ids = image_static['sentence_ids'][index]
    topk = map(json.loads, data_pair[temp_sentence_ids[0:5]])

    similar_images = []
    for i in range(len(topk)):
        filename = filter(lambda x: x['image_id'] == index[i], topk[i]['images'])[0]['image_name']
        filename = 'static/dataset/' + topk[i]['poster'] + '/' + filename
        filename = urllib.quote(filename.encode('utf-8'))
        similar_images.append(filename)
    return similar_images

def query_from_sentence(query_sen):
    sentence_emb = torch.from_numpy(gvar.sentence_transform(query_sen))
    sentence_emb = sentence_emb.unsqueeze_(0)
    sentence_emb = Variable(sentence_emb.cuda(), requires_grad=False)
    output, (hidden_state, cell_state) = model.rnn.forward(sentence_emb)
    output = output.data.cpu().mean(1)[0][0].numpy()

    return find_sentences(output), find_images(output)

def query_from_image(query_image):
    image_emb = gvar.img_transform(query_image)
    image_emb = Variable(image_emb.unsqueeze_(0).cuda(), requires_grad=False)
    image_features = gvar.resnet.forward(image_emb)
    output = model.mlp(image_features.view(-1, 512))
    output = output.data[0].cpu().numpy()

    return find_sentences(output), find_images(output)


if __name__  == '__main__':
    query_sen = u"亭子"
    for sentence in query_from_sentence(query_sen):
        print sentence
