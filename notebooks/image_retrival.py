
# coding: utf-8

# In[1]:


from configure import GlobalVariable
from linecache_light import LineCache

gvar = GlobalVariable()
data_pair = LineCache('static/dataset/data_pair.json')

print gvar.arch_feats.shape
print gvar.sentence_ids.shape, gvar.sentence_ids
print data_pair.num_lines


# In[2]:


import numpy as np
arch_feats = np.load('images.npy')
print arch_feats.shape

from torchvision import transforms
img_transform = transforms.Compose([
                transforms.Scale(100),
                transforms.CenterCrop(100),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])


# In[3]:


import torch
import torch.nn as nn
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

model = torch.load('cnn.pt')
model = nn.Sequential(*list(model.children())[:-1])
print model


# In[23]:


import json
import os
from PIL import Image

def findImgUrlByIndex(index):
    sentence_id = gvar.sentence_ids[index]
    data = json.loads(data_pair[sentence_id])
    filename = [image for image in data['images'] if image['image_id'] == index][0]['image_name'] 
    filename = os.path.join('static/dataset/' + data['poster'],filename)
    return filename

imgpath = findImgUrlByIndex(61644) # tingzi: 61644 youeryuan: 30361
img = Image.open(imgpath).convert('RGB')
img


# In[24]:


from torch.autograd import Variable
img_tensor = Variable(gvar.img_transform(img).unsqueeze_(0))
output = gvar.resnet(img_tensor).squeeze().cpu().data.numpy()
print output.shape, output[0:10]

dists = np.square(gvar.arch_feats - output).sum(1)
index = dists.argsort()
dists = dists[index]

print index[0:5]


# In[25]:


from torch.autograd import Variable
img_tensor = Variable(img_transform(img).unsqueeze_(0)).cuda(0)
output = model[1](model[0](img_tensor).view(len(img_tensor), -1))
output = output.squeeze().cpu().data.numpy()
print output.shape, output[0:10]

dists = np.square(arch_feats - output).sum(1)
index = dists.argsort()
dists = dists[index]

print index[0:5]


# In[26]:


count = 0


# In[42]:


print index[count]
imgpath = findImgUrlByIndex(index[count])
img = Image.open(imgpath)
count += 1
img

