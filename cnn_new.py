# coding: utf-8

from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
from zutil.config import Config
from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import os.path
import random
import json


class ArchDataset(Dataset):
    def __init__(self, config):
        self.config = config
        fn_labels = 'static/dataset/handcraft_labels.txt'
        fn_datapair = 'static/dataset/data_pair.json'
        all_labels = [label.strip().split(' ') for label in open(fn_labels, 'r')]
        self.all_labels = [label.decode('utf8') for label, weight in all_labels] # to unicode
        self.n_classes = len(self.all_labels) + 1 # extra class
        self.data_pair = [json.loads(data) for data in open(fn_datapair, 'r')]
        self.img_transform = transforms.Compose([
                        transforms.Scale(100),
                        transforms.CenterCrop(100),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):
        item = self.data_pair[index]
        # get image
        img_name = random.choice(item['images'])['image_name']
        img_path = 'static/dataset/' + os.path.join(item['poster'], img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)
        # get target
        target = []
        for i, label in enumerate(self.all_labels):
            if label in item['title']:
                target.append(i)
        if len(target) == 0:
            target.append(len(self.all_labels))     # extra class
        for i in range(self.n_classes - len(target)):
            target.append(-1)
        return image, target


    def __iter__(self):
        for i in range(len(self.data_pair)):
            yield self.__getitem__(i)

    def next_batch(self, mode='train'):
        cut_point = int(self.__len__() * self.config.split_rate)
        while True:
            batch_data = []
            for i in range(self.config.batch_size):
                if mode == 'train':
                    index = random.randint(0, cut_point-1)
                elif mode == 'valid':
                    index = random.randint(cut_point, self.__len__() - 1)
                else:
                    raise Exception('no such mode!')
                batch_data.append(self.__getitem__(index))
            yield self._split_batch(batch_data)

    def _split_batch(self, batch_data, requires_grad=True):
        images = torch.cat([image.unsqueeze(0) for image, _ in batch_data], 0)
        targets = torch.LongTensor([target for _, target in batch_data])
        images, targets = Variable(images), Variable(targets)
        if self.config.gpu:
            images, targets = images.cuda(0), targets.cuda(0)
        return images, targets

    def raw_image(self):
        for data in self.data_pair:
            for image in data['images']:
                img_path = 'static/dataset/' + os.path.join(data['poster'], image['image_name'])
                img_pil = Image.open(img_path).convert('RGB')
                yield self.img_transform(img_pil)

    def raw_batch(self):
        batch_data = []
        for image in self.raw_image():
            batch_data.append(image)
            if len(batch_data) == self.config.batch_size:
                yield self.split_image(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield self.split_image(batch_data)

    def split_image(self, batch_data):
        images = torch.cat([image.unsqueeze(0) for image in batch_data], 0)
        images = Variable(images, requires_grad=False)
        if self.config.gpu:
            images = images.cuda(0)
        return images


config = Config(batch_size=1000, gpu=True, split_rate=0.8, learning_rate=1e-4, save_freq=10, max_epoch=20)
dataset = ArchDataset(config)
print config
print dataset


from zutil.convblock import ConvBlockModule
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.conv = ConvBlockModule(dims=[3, 16, 32, 64, 64])
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(512, 150)

    def forward(self, images):
        output = self.conv(images)
        output = output.view(self.config.batch_size, -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output

    def train(self):
        self.training = True
        self.conv.train()
        self.fc.train()

    def eval(self):
        self.training = False
        self.conv.eval()
        self.fc.eval()

'''
model = CNN(config)
if config.gpu:
    model = model.cuda(0)
print model
criterion = nn.MultiLabelMarginLoss() # or MultiLabelSoftMarginLoss, or KLDivLoss
print criterion

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def train(epoch):
    epoch_train_loss = []
    for batchid, (images, targets) in enumerate(dataset.next_batch(mode='train')):
        outputs = model(images)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.data.mean()
        epoch_train_loss.append(loss)
        #print 'epoch = %d, batch_id = %d, train loss = %.3f' %(epoch, batchid, loss)

        if batchid > 10:
            return np.array(epoch_train_loss).mean()

def validate(epoch):
    epoch_valid_loss = []
    for batchid, (images, targets) in enumerate(dataset.next_batch(mode='valid')):
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss = loss.data.mean()
        epoch_valid_loss.append(loss)
        #print 'epoch = %d, batch_id = %d, valid loss = %.3f' %(epoch, batchid, loss)

        if batchid > 3:
            return np.array(epoch_valid_loss).mean()

def do_loop():
    for epoch in range(config.max_epoch):
        model.train()
        train_loss = train(epoch)
        model.eval()
        valid_loss = validate(epoch)
        print 'epoch = %d, train_loss = %.3f, valid_loss = %.3f' % (epoch, train_loss, valid_loss)

        if (epoch+1) % config.save_freq == 0:
            torch.save(model, 'cnn.pt')

do_loop()
'''

model = torch.load('resnet.pt')
print model.conv.training
print model.fc.training

model = nn.Sequential(*list(model.children())[:-1])
#model = model.cuda()
print model
print 'training', model.training

outputs = []
for batchid, images in enumerate(dataset.raw_batch()):
    output = model[1](model[0](images).view(len(images), -1))
    output = output.data.cpu()
    outputs.append(output)
    print batchid, output.size(), output.mean()
outputs = torch.cat(outputs, 0)
print outputs.size()
np.save('images.npy', outputs.numpy())
