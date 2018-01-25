
# coding: utf-8

# In[10]:


import json
data_pair = [json.loads(data) for data in open('static/dataset/data_pair.json', 'r')]
print len(data_pair)
print data_pair[0].keys()


# In[16]:


print data_pair[0]['title']


# In[88]:


import jieba

title_words_lengths = [len(jieba.lcut(data['detail'])) for data in data_pair]
title_chars_lengths = [len(data['detail']) for data in data_pair]
print title_words_lengths[0:10]
print title_chars_lengths[0:10]


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

plt.plot(title_chars_lengths)
#plt.show()

plt.plot(title_words_lengths)
plt.show()


# In[17]:


from configure import GlobalVariable

gvar = GlobalVariable()
print gvar.


# In[101]:


import random
import numpy as np

def getitem(index):
    title = data_pair[index]['keywords']
    sentence = gvar.sentence_transform(title)
    if sentence.shape[0] == 0:
        sentence = np.zeros((1, 300))
    return sentence.mean(0)

index = random.randint(0, len(data_pair) - 1)
sentence = getitem(index)
print sentence.shape

dataset = np.array([getitem(i) for i in range(len(data_pair))])
print dataset.shape


# In[2]:


import numpy as np
dataset = np.load('docs_tfidf.npy')


# In[6]:


# PCA, t-SNE 看一下

import numpy as np
from sklearn.manifold import TSNE

model = TSNE(n_components=2, init='pca', random_state=0)
result = model.fit_transform(dataset[np.random.choice(len(dataset), 5000)])
print result.shape

x_data = np.array([x for x, y in result])
y_data = np.array([y for x, y in result])
print x_data.shape, y_data.shape

plt.scatter(x_data, y_data)
plt.show()


# In[7]:


from sklearn.cluster import KMeans

model = KMeans(n_clusters=50, n_jobs=10, verbose=False)
print model

results = model.fit(dataset)
print results.labels_


# In[63]:


count = -1


# In[89]:


count += 1
for i in range(len(data_pair)):
    if results.labels_[i] == count:
        print count, data_pair[i]['title'].replace('\n', ' ')


# In[ ]:


from sklearn.cluster import DBSCAN
model = DBSCAN(n_jobs=10)
print model

result = model.fit(dataset)
print result


# In[28]:


result.labels_


# In[48]:


for i in range(len(data_pair)):
    if result.labels_[i] == 17:
        print data_pair[i]['title'].replace('\n', ' ')

