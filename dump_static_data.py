import torch
from data import RawArchDataset
from configure import GlobalVariable
from torch.autograd import Variable
import numpy as np
import argparse
from zutil.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--modeltype', type=str, default='rnn')
args = parser.parse_args()

gvar = GlobalVariable()
config = Config('parameters.json')
if args.modeltype == 'rnn':
    model = torch.load('static/models/multimodal.new.pt')
model.eval()
print config
print model

def dump_sentences():
    dataset = RawArchDataset(config.update(mode='title'), gvar)
    sentence_static_data = np.zeros((len(dataset), config.lstm_hidden_size))
    for sentence_ids, sentences, lengths in dataset.next_batch():
        print sentence_ids
        if args.modeltype == 'rnn':
            output, (hidden_state, cell_state) = model.rnn.forward(sentences)
            output = torch.cat([output[i, 0:lengths[i], :].mean(0) for i in range(len(output))])
            sentences_emb = output.data.cpu().numpy() # detach data
        if args.modeltype == 'cnn':
            sentences_emb = model.cnn.forward(sentences)
            sentences_emb = sentences_emb.data.cpu().numpy() # detach data
        sentence_static_data[sentence_ids] = sentences_emb
    print 'saving sentence_static.npy'
    np.save('sentence_static.npy', sentence_static_data)

def dump_images():
    gvar.arch_feats = torch.from_numpy(gvar.arch_feats).cuda()
    static_images = np.zeros((len(gvar.arch_feats), config.lstm_hidden_size))
    for index in range(len(gvar.arch_feats) / config.batch_size + 1):
        start = index * config.batch_size
        end = min((index + 1) * config.batch_size, len(gvar.arch_feats))
        print 'start -> end', start, end
        image_features = Variable(gvar.arch_feats[start:end])
        images_emb = model.mlp(image_features)
        static_images[range(start, end)] = images_emb.data.cpu().numpy()
    image_static_data = np.array([{'images':static_images,
                                   'sentence_ids':gvar.sentence_ids}])
    print 'saving image_static.npy'
    np.save('image_static.npy', image_static_data)


if __name__ == '__main__':
    print 'dumping sentences'
    dump_sentences()
    print 'dumping images'
    dump_images()
