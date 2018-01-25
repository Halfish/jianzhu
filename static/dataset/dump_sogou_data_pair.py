#encoding=utf8
import re
import os
import sys
import json
import random
#import jieba.analyse
#jieba.analyse.set_stop_words('stopwords.txt')

from PIL import Image

data_pair = []
for i, line in enumerate(open('formalCompetition4/trainMatching.txt', 'r')):
    if (i+1) % 1000 == 0:
        print(i)
    imgpath, textpath = line.split()
    imgpath = os.path.join('formalCompetition4/News_pic_info_train', imgpath)
    textpath = os.path.join('formalCompetition4/News_info_train', textpath)
    with open(textpath) as ft:
        text = ft.readline()    # this file only contains one line
    try:
        url, title, detail = text.split('\t')
        pair = {'url':url, 'title':title, 'detail':detail.strip(), 'imgpath':imgpath}
        img = Image.open(imgpath) # may failed to open this image
        data_pair.append(pair)
    except OSError as e:
        print('OSError:', e)
        print(i, line)
    except ValueError as e:
        print('ValueError', e)
        print(i, line)
    except:
        print("unexpected error:", sys.exc_info()[0])
        raise

random.shuffle(data_pair) # shuffle is important!
print('got', len(data_pair), 'valid data pairs')

# add data_id
for i in range(len(data_pair)):
    data_pair[i]['data_id'] = i


# save the result to json file
print('saving to data_pair.json')
with open('data_pair.json', 'w') as f:
    for pair in data_pair:
        f.write(json.dumps(pair) + '\n')

#print(json.dumps(data_pair[0:3]))
