
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (6.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[11]:


opt = {'word_emb_dim': 300,
       'sentence_emb_dim': 400,
       'filter_num': 32,
       'dropout_rate': 0.5,
       'hidden_size': 256,
       'model_type': 'cnn',
       'num_layers': 1,
       'dataset_type': 'flickr8k',
       'cuda': False,
       'max_epoch': 1000,
       'savefreq': 10,
       'batch_size': 100,
       'learning_rate': 1e-2,
       'max_clip_norm': 5,
       }


# In[16]:


from datautil import BatchDataWrapper
opt['train'] = True
dataset = BatchDataWrapper(opt)
print dataset

from torch.nn.utils.rnn import pad_packed_sequence

from gensim.models.word2vec import Word2Vec
model = Word2Vec.load('static/word2vec-chi/word2vec_news.model')
print model


# In[7]:


for word, score in model.most_similar(u'学校'):
    print word, score
print ''
for word, score in model.most_similar(u'建筑'):
    print word, score
print ''
for word, score in model.most_similar(u'商业'):
    print word, score


# In[18]:


import numpy as np
count = 0
for sentences, images, flags in dataset.next_batch():
    words = sentences[0].data.numpy()
    print 'count = ', count
    count += 1
    if count > 20:
        break
    print 'flag is ', flags[0].data[0] == 1
    print ''.join([model.most_similar([w], topn=1)[0][0] for w in words])

    img = images[0].data.numpy()
    img2 = np.random.randn(224, 224, 3)
    img2[:, :, 0] = img[0, :, :] * 0.229 + 0.485
    img2[:, :, 1] = img[1, :, :] * 0.224 + 0.456
    img2[:, :, 2] = img[2, :, :] * 0.225 + 0.406
    plt.figure()
    plt.imshow(img2)
    plt.show()
    print '---------------------------------------------------------'


# In[33]:


dataset.dataset.filename


# In[12]:


count = 0
for sentences, images, flags in dataset.next_batch():
    print 'count = %d, batch size = %d' %(count, flags.size()[0])
    count += 1


# In[30]:


import numpy as np
print np.random.choice(range(10), 10, replace=True)

import random
index = range(10)
print random.shuffle(index)
print index

