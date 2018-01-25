
# coding: utf-8

# In[15]:


from configure import GlobalVariable

gvar = GlobalVariable()
print gvar
gvar.arch_feats.shape

from linecache_light import LineCache
data_pair = LineCache('static/dataset/data_pair.json')
print data_pair.num_lines


# In[18]:


import json
import os
from PIL import Image

def findImgByIndex(index):
    sentence_id = gvar.sentence_ids[index]
    data = json.loads(data_pair[sentence_id])
    filename = [image for image in data['images'] if image['image_id'] == index][0]['image_name'] 
    filename = os.path.join('static/dataset/' + data['poster'],filename)
    img = Image.open(filename).convert('RGB')
    return img


# In[6]:


get_ipython().run_cell_magic('time', '', 'from sklearn.cluster import KMeans\n\nmodel = KMeans(n_clusters=50, n_jobs=10, verbose=False)\nprint model\n\nresults = model.fit(gvar.arch_feats[0:50000])\nprint results.labels_')


# In[26]:


cluster_id = -1


# In[59]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

count = 0
cluster_id += 1

plt.figure()
for i in range(50000):
    if results.labels_[i] == cluster_id:
        print i, cluster_id
        count += 1
        if count > 30:
            break
        img = findImgByIndex(i)
        plt.imshow(img)
        plt.show()

