#coding=utf-8

from torch.autograd import Variable
from configure import GlobalVariable
from PIL import Image

import numpy as np
import json
import urllib

gvar = GlobalVariable()
gvar.resnet = gvar.resnet.cuda()
data_pair = [json.loads(line.strip()) for line in open('static/dataset/data_pair.json')]
query_num = 10


def handle_sentence(query_sen):
    dists = np.square(gvar.sentences - query_sen).sum(1)
    index = dists.argsort()
    dists = dists[index]
    topk = [data_pair[index[i]] for i in range(query_num)]
    print(topk)
    images = ['static/dataset/'+topk[i]['imgpath'] for i in range(query_num)]
    for i in range(query_num):
        for key in topk[i]:
            topk[i][key] = str(topk[i][key])
    return topk, images

    '''
    # get images
    result_images = []
    for i in range(len(topk)):
        sentence_images = []
        for img in topk[i]['images']:
            filename = 'static/dataset/' + topk[i]['poster'] + '/' + img['image_name']
            filename = urllib.quote(filename.encode('utf-8'))
            sentence_images.append((filename, str(img['image_id'])))
        result_images.append(sentence_images)
        del topk[i]['images']

    for t in topk:
        for k in t:
            if type(t[k]) == int:
                t[k] = str(t[k])

    return topk, result_images
    '''


def handle_image(query_img):
    dists = np.square(gvar.arch_feats - query_img).sum(1)
    index = dists.argsort()
    dists = dists[index]
    topk = [data_pair[index[i]] for i in range(query_num)]
    images = ['static/dataset/'+topk[i]['imgpath'] for i in range(query_num)]
    for i in range(query_num):
        for key in topk[i]:
            topk[i][key] = str(topk[i][key])
    return topk, images
    '''
    #temp_sentence_ids = gvar.sentence_ids[index]
    #topk = [data for data in data_pair[temp_sentence_ids[0:query_num]]]

    result_images = []
    for i in range(len(topk)):
        images = topk[i]['images']
        # put the target image in the front
        similar_image = filter(lambda x: x['image_id'] == index[i], images)[0]
        images.remove(similar_image)
        images.insert(0, similar_image)
        sentence_images = []
        for img in images:
            filename = 'static/dataset/' + topk[i]['poster'] + '/' + img['image_name']
            filename = urllib.quote(filename.encode('utf-8'))
            sentence_images.append((filename, str(img['image_id'])))
        result_images.append(sentence_images)
        del topk[i]['images']
    for t in topk:
        for k in t:
            if type(t[k]) == int:
                t[k] = str(t[k])
    return topk, result_images
    '''


def query_from_sentence(query_sen):
    sentence_emb = gvar.sentence_transform(query_sen).mean(0)
    return handle_sentence(sentence_emb)

def query_from_image(query_image):
    image_emb = gvar.img_transform(query_image)
    image_emb = Variable(image_emb.unsqueeze_(0), requires_grad=False).cuda()
    image_features = gvar.resnet.forward(image_emb)
    output = image_features.cpu().data.squeeze().numpy()

    return handle_image(output)


if __name__  == '__main__':
    query_sen = u"亭子"
    for sentence in query_from_sentence(query_sen):
        print(sentence)

    image = Image.open('tingzi.jpg').convert('RGB')
    for image in query_from_image(image):
        print(image)
