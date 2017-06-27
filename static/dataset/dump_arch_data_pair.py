#encoding=utf8
import re
import os
import json
import random
import jieba.analyse
jieba.analyse.set_stop_words('stopwords.txt')

data_pair = []
for i, line in enumerate(open('jianzhu_tag.json', 'r')):
    if (i+1) % 1000 == 0:
        print i
    line = json.loads(line)
    detail = line['detail'].strip()
    title, tag = line['title'].strip(), line['tag'].strip()
    title = title != '' and title or tag
    url = line['other'].has_key('url') and line['other']['url'] or ''
    poster = re.sub('^/data/crawler/', '', line['poster']) # for poster of archgo
    if len(detail) > 1 and os.path.exists(poster):
        website = poster.split('/')[1]
        detail = re.sub('[\+\-][0-9]+|[\r\n\t]+', '', detail.strip()) # remove redundant lines
        if website == 'ARCHGO':
            detail = re.sub(u'网盘|下载链接', '', detail)
            detail = re.sub(u'登录，付费 ￥\d.\d元 下载；节约您的宝贵时间', '', detail)
        if website == 'IKUKU':
            detail = re.sub(u'添加到采风集提出问题分享给朋友', '', detail)
        keywords = jieba.analyse.extract_tags(detail, topK=20, withWeight=False, allowPOS=())
        keywords = ' '.join(keywords)
        img_names = os.listdir(poster)
        images = []
        for img_name in img_names:
            images.append({'image_name':img_name})
        img_names = [os.path.join(poster, img_name) for img_name in img_names]
        pair = {'title':title, 'detail':detail, 'keywords':keywords,
                'poster':poster, 'images': images, 'url':url, 'website':website}
        data_pair.append(pair)

random.shuffle(data_pair) # shuffle is important!
print 'got', len(data_pair), 'valid data pairs'

# add sentence_id
imgcount = 0
for i in range(len(data_pair)):
    data_pair[i]['sentence_id'] = i
    for j in range(len(data_pair[i]['images'])):
        data_pair[i]['images'][j]['image_id'] = imgcount
        imgcount += 1

# save the result to json file
print 'saving to data_pair.json'
with open('data_pair.json', 'w') as f:
    for pair in data_pair:
        f.write(json.dumps(pair) + '\n')

print json.dumps(data_pair[0:1])
