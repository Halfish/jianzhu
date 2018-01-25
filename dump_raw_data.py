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
    print('got %d images' % len(dataset))
    static_feats = []

    for batch_id, images in enumerate(dataset.next_batch()):
        print('processing batch_id = ', batch_id)
        output = gvar.resnet(images).cpu().data
        output = output.view(-1, 512)   # output = output.view(-1, 512, 49)
        static_feats.append(output.numpy())

    print('concat static feats...')
    static_feats = np.concatenate(static_feats)

    print('saving as arch_feats.h5')
    h5file = h5py.File('arch_feats.h5', 'w')
    h5file.create_dataset('feats', data=static_feats)
    h5file.close()

def dump_sentences():
    dataset = RawArchDataset(config.copy(mode='title'), gvar)
    sentences = []
    for sentence in dataset:
        sentences.append(sentence.mean(0))
    sentences = np.array(sentences)
    np.save('sentences.npy', sentences)

if __name__ == '__main__':
    start = time.time()
    #dump_feats()
    dump_sentences()
    end = time.time()
    print('cost %.3f seconds' % (end - start))
