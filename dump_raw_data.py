from data import RawArchDataset
from zutil.config import Config
from configure import GlobalVariable
import numpy as np
import time
import h5py

config = Config('parameters.json')
gvar = GlobalVariable()

def dump_feats():
    dataset = RawArchDataset(config.copy(mode='image'), gvar)
    print 'got', len(dataset), 'images'
    static_sentence_ids = []
    static_image_ids = []
    static_feats = []

    for sentence_ids, image_ids, images in dataset.next_batch():
        print 'sentence_ids\n', sentence_ids
        print 'image_ids\n', image_ids
        print sentence_ids.shape
        static_sentence_ids.append(sentence_ids)
        static_image_ids.append(image_ids)
        output = gvar.resnet(images).cpu().data
        output = output.view(-1, 512)   # output = output.view(-1, 512, 49)
        static_feats.append(output.numpy())

    static_sentence_ids = np.hstack(static_sentence_ids)
    static_image_ids = np.hstack(static_image_ids)
    static_feats = np.concatenate(static_feats)

    h5file = h5py.File(config.featsname, 'w')
    h5file.create_dataset('sentence_ids', data=static_sentence_ids)
    h5file.create_dataset('image_ids', data=static_image_ids)
    h5file.create_dataset('feats', data=static_feats)
    h5file.close()

def dump_sentences():
    dataset = RawArchDataset(config.copy(mode='title'), gvar)
    sentences = []
    for sentence_id, sentence in dataset:
        print sentence_id
        sentences.append(sentence.mean(0))
    sentences = np.array(sentences)
    np.save('sentences.npy', sentences)

if __name__ == '__main__':
    start = time.time()
    dump_feats()
    dump_sentences()
    end = time.time()
    print 'cost', end - start, 'seconds'
