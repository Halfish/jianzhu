import random
import os.path
import json
import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from PIL import Image, ImageFile
from zutil.config import Config

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ArchDataset(torch.utils.data.Dataset):
    def __init__(self, config, global_variable, max_len=100000):

        self.config = config
        self.gvar = global_variable
        self.max_len = max_len

        assert config.which_text in {'title', 'detail'}

        with open(os.path.join(self.config.root_dir, self.config.filename), 'r') as f:
            # data_pair is list of dict
            data_pair = [json.loads(line.strip()) for line in f] 
            cut_point = int(len(data_pair) * self.config.split_rate)
            self.data_pair = data_pair[:cut_point] \
                        if self.config.train else data_pair[cut_point:]
            random.shuffle(self.data_pair)

    def __len__(self):
        return min(self.max_len, len(self.data_pair))

    def __getitem__(self, index):
        # get sentence
        sentence = self.data_pair[index][self.config.which_text]
        sentence = self.gvar.sentence_transform(sentence)
        if sentence.shape[0] == 0: # fix crash bug, in case no words left after jieba
            sentence = np.random.randn(1, self.gvar.wvModel.vector_size)
        # load image tensor from gvar
        image = self.gvar.arch_feats[index]
        return image, sentence

    def __iter__(self):
        while True:
            indices = random.sample(range(self.__len__()), self.config.neg_sample_num)
            yield [self.__getitem__(index) for index in indices]

    def next_batch(self):
        batch_data = []
        for t, data in enumerate(self):
            batch_data.append(data)
            if (t + 1) % self.config.batch_size == 0:
                yield self.get_padded_tensor_batch(batch_data)
                batch_data = []

    def get_padded_tensor_batch(self, batch_data):
        bag_sentences, bag_images, bag_lengths = [], [], []
        for i in range(self.config.neg_sample_num):
            images = np.array([data[i][0] for data in batch_data])
            # list of numpy arrays with variable lengths
            sentences = [data[i][1] for data in batch_data]  
            lengths = np.array([sentence.shape[0] for sentence in sentences])

            # padding words embeddings
            actual_batch_size, max_seq_len = len(lengths), max(lengths)
            if self.config.model_type == 'cnn':
                # at least 9 words, considering convolution operation
                max_seq_len = max(lengths[0], 9) 
            padded_sentences = np.zeros(
                    (actual_batch_size, max_seq_len, self.config.word_emb_dim))
            for j in range(actual_batch_size):
                padded_sentences[i, 0:lengths[j], :] = sentences[j]

            padded_sentences = Variable(torch.from_numpy(padded_sentences)).float()
            images = Variable(torch.from_numpy(images)).float()
            if self.config.cuda:
                padded_sentences = padded_sentences.cuda()
                images = images.cuda()

            bag_sentences.append(padded_sentences)
            bag_images.append(images)
            bag_lengths.append(lengths)

        return bag_sentences, bag_images, bag_lengths


class RawArchDataset(torch.utils.data.Dataset):
    '''
        return raw image or sentence dataset
    '''
    def __init__(self, config, global_variable, max_len=1000000):
        super(RawArchDataset, self).__init__()

        self.max_len = max_len
        self.gvar = global_variable
        self.config = config
        assert self.config.mode in {'image', 'title', 'detail'}, 'wrong mode!'

        with open(os.path.join(self.config.root_dir, self.config.filename), 'r') as f:
            data_pair = map(json.loads, f.readlines())
            if self.config.mode == 'image':
                self.images = [data['imgpath'] for data in data_pair]
            else:
                self.sentences = [data[self.config.mode] for data in data_pair]

    def __len__(self):
        if self.config.mode == 'image':
            return min(self.max_len, len(self.images))
        else:
            return min(self.max_len, len(self.sentences))

    def __getitem__(self, index):
        if self.config.mode == 'image':
            image_name = self.images[index]
            image_path = os.path.join(self.config.root_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.gvar.img_transform(image)
            return image
        else:
            sentence = self.sentences[index]
            sentence = self.gvar.sentence_transform(sentence)
            # fix crash bug, in case no words left after jieba
            if sentence.shape[0] == 0:
                sentence = np.random.randn(1, self.gvar.wvModel.vector_size)
            return sentence

    def __iter__(self):
        index = 0
        while index < self.__len__():
            yield self.__getitem__(index)
            index += 1

    def next_batch(self):
        batch_data = []
        for t, data in enumerate(self):
            batch_data.append(data)
            if (t + 1) % self.config.batch_size == 0:
                yield self.split_batch(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield self.split_batch(batch_data)

    def split_batch(self, batch_data):
        if self.config.mode == 'image':
            images = Variable(torch.stack(batch_data))
            if self.config.cuda:
                images = images.cuda()
            return images
        else:
            sentences = batch_data
            lengths = np.array([sentence.shape[0] for sentence in sentences])

            # padding words embeddings
            actual_batch_size, max_seq_len = len(lengths), max(lengths)
            if self.config.model_type == 'cnn':
                max_seq_len = max(lengths[0], 9) # at least 9 words, considering convolution operation
            padded_sentences = torch.zeros(actual_batch_size, max_seq_len, self.config.word_emb_dim)
            for i in range(actual_batch_size):
                padded_sentences[i][0:lengths[i], :] = torch.from_numpy(sentences[i])
            padded_sentences = Variable(padded_sentences)
            if self.config.cuda:
                padded_sentences = padded_sentences.cuda()
            return padded_sentences, lengths


if __name__ == '__main__':
    import configure
    gvar = configure.GlobalVariable()
    dataset = ArchDataset(Config(source='parameters.json', train=True), gvar)
    for bag_sentences, bag_images, bag_lengths in dataset.next_batch():
        print(len(bag_sentences), len(bag_images), len(bag_lengths))
        print(bag_sentences[0].data)
        print(bag_images[0].data)
        print(bag_lengths[0])
        break

    '''
    print 'testing images'
    rawarch = RawArchDataset(Config('parameters.json', mode='image'), gvar, max_len = 10)
    for sentence_id, image_id, image in rawarch:
        print sentence_id, image_id, image.size(), image.mean()

    print 'testing keywords'
    rawarch = RawArchDataset(Config('parameters.json', mode='keywords'), gvar, max_len = 10)
    for sentence_id, sentence in rawarch:
        print sentence_id, sentence.size(), sentence.mean()

    print 'testing batch images'
    rawdataset = RawArchDataset(Config('parameters.json', mode='image'), gvar, max_len = 10)
    for sentence_id, image_id, image in rawdataset.next_batch():
        print sentence_id, image_id, image.size(), image.mean()

    print 'testing batch titles'
    rawdataset = RawArchDataset(Config('parameters.json', mode='title'), gvar, max_len = 10)
    for sentence_id, sentence in rawdataset.next_batch():
        print sentence_id, sentence
    '''
