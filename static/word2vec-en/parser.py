#!/usr/bin/python
# -*- coding: utf-8 -*-

import xml.sax
import re

'''
parse english wiki xml file, get clean corpus for word2vec
'''

def filter(text):
    text = re.sub('[^a-zA-Z ]', ' ', text)
    text = re.sub('[ ]+', ' ', text)
    return text

class WikiHandler(xml.sax.ContentHandler):
    def __init__(self, filename):
        self.flag = False
        self.file = open(filename, 'w')
        self.count = 0

    def startElement(self, tag, attributes):
        if tag == "text":
            self.flag = True

    def endElement(self, tag):
        if tag == "text":
            self.flag = False

    def characters(self, content):
        if self.flag:
            line = filter(content)
            if len(line) > 100:
                self.file.write(line.encode('utf-8') + u'\n')
            elif len(line) > 3:
                self.file.write(line.encode('utf-8'))
            self.count += 1
            if self.count % 100000 == 0:
                print 'count = ', self.count


if __name__ == '__main__':
    parser = xml.sax.make_parser()
    # turn off namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    Handler = WikiHandler('corpus.txt')
    parser.setContentHandler(Handler)

    try:
        #parser.parse("wiki.small.xml")
        parser.parse("enwik9")
    except Exception, e:
        print e
        print 'bad end of xml'
