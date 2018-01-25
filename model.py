import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
from zutil.convblock import ConvBlockModule
from zutil.config import Config


class AttentionModule(nn.Module):
    def __init__(self, config):
        super(AttentionModule, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(config.word_emb_dim + config.resnet_features,
        #self.rnn = nn.LSTM(config.word_emb_dim,
                           config.lstm_hidden_size, config.recurrent_layers,
                           batch_first=False, bidirectional=False, dropout=config.recurrent_dropout)
        self.weight_feature2attn = Parameter(torch.Tensor(config.resnet_features))
        self.weight_hidden2attn = Parameter(torch.Tensor(config.lstm_hidden_size, config.resnet_fmap_size))
        self.bias_attention = Parameter(torch.Tensor(config.resnet_fmap_size))
        self.mlp = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, config.mlp_hidden_size),
            #nn.Linear(config.lstm_hidden_size+config.resnet_features, config.mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden_size, 1),
            nn.Sigmoid()
        )
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 2.0 / math.sqrt(weight.size(-1))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, rawwords_emb, image_features):
        words_emb, lengths = pad_packed_sequence(rawwords_emb, batch_first=False)
        seq_len, batch_size = words_emb.size()[0:2]
        hidden_state = Variable(torch.zeros(self.config.recurrent_layers, batch_size, self.config.lstm_hidden_size))
        cell_state = Variable(torch.zeros(self.config.recurrent_layers, batch_size, self.config.lstm_hidden_size))
        if self.config.cuda:
            hidden_state, cell_state = hidden_state.cuda(), cell_state.cuda()
        weight = self.weight_feature2attn.expand(batch_size, self.config.resnet_features).unsqueeze(2)
        feature2a = torch.bmm(image_features.transpose(1, 2), weight).squeeze(2)
        outputs = []
        for i in range(seq_len):
            # feature2a, attn_weight are both (batch_size, 512)
            attn_weight = F.softmax(feature2a + torch.mm(hidden_state[-1], self.weight_hidden2attn) +
                    self.bias_attention.expand(batch_size, self.config.resnet_fmap_size))
            attn_out = torch.bmm(image_features, attn_weight.unsqueeze(2)).squeeze(2) # (bz, 512)
            inputs = torch.cat([words_emb[i], attn_out], 1).unsqueeze(0)
            output, (hidden_state, cell_state) = self.rnn(inputs, (hidden_state, cell_state))
            outputs.append(output.squeeze(0))
        outputs = torch.cat([outputs[lengths[i]-1][i].unsqueeze(0) for i in range(len(lengths))], 0)
        scores = self.mlp(outputs).squeeze(1)
        '''
        outputs, (hidden_state, cell_state) = self.rnn(rawwords_emb)
        inputs = torch.cat([hidden_state[-1], image_features.mean(2).squeeze(2)], 1)
        scores = self.mlp(inputs).squeeze(1)
        '''
        return scores

    def eval(self):
        self.rnn.eval()
        self.mlp.eval()

    def train(self):
        self.rnn.train()
        self.mlp.train()


class MultimodalModule(nn.Module):
    def __init__(self, config):
        super(MultimodalModule, self).__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type == 'rnn':
            self.rnn = nn.LSTM(config.word_emb_dim, config.lstm_hidden_size,
                               config.recurrent_layers, batch_first=True)
        elif self.model_type == 'cnn':
            self.cnn = CNNTextModule(config)

        if self.model_type == 'rnn':
            sentence_emb_dim = config.lstm_hidden_size
        elif self.model_type == 'cnn':
            sentence_emb_dim = config.sentence_emb_dim
        elif self.model_type == 'simple':
            sentence_emb_dim = config.word_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(config.resnet_features, config.mlp_hidden_size),
            nn.BatchNorm1d(config.mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden_size, sentence_emb_dim),
        )

    def forward(self, bag_batch_data):
        bag_sentences, bag_images, bag_lengths = bag_batch_data
        bag_semantic, bag_visual = [], []
        for bagid in range(len(bag_sentences)):
            if self.model_type == 'rnn':
                output, (hidden_state, cell_state)= self.rnn(bag_sentences[bagid])
                sentences_emb = torch.stack([output[i, 0:bag_lengths[bagid][i], :].mean(0)
                                    for i in range(len(output))])
            elif self.model_type == 'cnn':
                sentences_emb = self.cnn(words_emb)
            elif self.model_type == 'simple':
                sentences_emb = words_emb.mean(1).squeeze(1)
            bag_semantic.append(sentences_emb)

            # embedding image feature to multi-modal vector space
            images_emb = self.mlp(bag_images[bagid])
            bag_visual.append(images_emb)

        return bag_semantic, bag_visual

    def eval(self):
        self.training = False
        self.mlp.eval()
        if self.model_type == 'cnn':
            self.cnn.eval()
        elif self.model_type == 'rnn':
            self.rnn.eval()

    def train(self):
        self.training = True
        self.mlp.train()
        if self.model_type == 'cnn':
            self.cnn.train()
        elif self.model_type == 'rnn':
            self.rnn.train()


class CNNTextModule(nn.Module):
    def __init__(self, config):
        super(CNNTextModule, self).__init__()
        self.config = config
        for i, ks in enumerate(self.config.kernel_sizes):
            block = ConvBlockModule(dims=config.kernel_nums[0:2],
                                    kernel_size=(ks, self.config.word_emb_dim),
                                    basic_layers='conv, batchnorm, relu')
            setattr(self, 'conv'+str(i), block)
        self.deep_conv = ConvBlockModule(dims=config.kernel_nums[1:],
                                        kernel_size=(3, 1),
                                        pooling_size=(2, 1),
                                        prefix='deep',
                                        basic_layers='conv, batchnorm, relu, pooling')
        self.fc = nn.Linear(config.kernel_nums[-1] * len(self.config.kernel_sizes), config.sentence_emb_dim)

    def conv_util(self, i, x):
        out = getattr(self, 'conv'+str(i))(x)
        out = self.deep_conv(out)
        out = out.squeeze(3)
        out, _ = out.max(2) # out is (batch_size, last_channel, 1)
        return out

    def forward(self, x):
        '''
        input: x, Variable, (batch_size, seq_len, word_emb_dim)
        output: out, Variable, (batch, sentence_emb_dim)
        '''
        out = x.unsqueeze(1)
        out = [self.conv_util(i, out) for i in range(len(self.config.kernel_sizes))]
        out = torch.cat(out, 2)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    config = Config('parameters.json', model_type='rnn', which_text='title')
    print(config)
    multimodal = MultimodalModule(config)
    print(multimodal)
    words_emb = torch.autograd.Variable(torch.randn(10, 56, 300))
    img_features = torch.autograd.Variable(torch.randn(10, 512))
    sentence, image_features = multimodal(words_emb, img_features)
    print('output of multimodal', sentence.shape)

    cnn = CNNTextModule(config)
    input = torch.autograd.Variable(torch.randn(10, 56, 300))
    output = cnn(input)
    print('output of cnn')
    print(output.size())
