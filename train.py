from data import ArchDataset
from model import MultimodalModule, AttentionModule
from configure import GlobalVariable
import numpy as np
import math
import torch
import time
from hyperboard import Agent
agent = Agent(address='127.0.0.1', port=5001)

from zutil.config import Config
config = Config(source='parameters.json')

print('building model')
assert config.model_type in {'rnn', 'cnn', 'simple', 'attention'}
if config.model_type == 'attention':
    model = AttentionModule(config)
    criterion = torch.nn.BCELoss()
else:
    model = MultimodalModule(config)
    criterion = torch.nn.CosineEmbeddingLoss()

print(model)
print(criterion)

if config.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

params = model.parameters()
#optimizer = torch.optim.Adam(
#    params, lr=config.learning_rate, weight_decay=config.weight_decay)
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

def dist(x1, x2, p=2):
    result = torch.abs(x1 - x2) ** p
    result = result.sum(1).squeeze()
    result = result ** (1 / (p + 1e-6))
    return result

def constractive_loss(bag_semantic, bag_visual, margin=1):
    assert(len(bag_semantic) >= 2)
    loss = 0
    for i in range(len(bag_semantic) - 1):
        x, v_neg = bag_semantic[0], bag_visual[i+1]
        v, x_neg = bag_visual[0], bag_semantic[i+1]
        loss += torch.nn.ReLU()(dist(x, v) - dist(x, v_neg) + margin).mean()
        loss += torch.nn.ReLU()(dist(x, v) - dist(x_neg, v) + margin).mean()
    loss = loss / (len(bag_semantic) - 1)
    return loss

def calc_norm(weights, ptype='weight'):
    total_norm = 0
    for w in weights:
        if ptype == 'weight':
            total_norm += w.data.norm() ** 2
        elif ptype == 'grad':
            total_norm += w.grad.data.norm() ** 2
    return math.sqrt(total_norm)

def train(epoch, dataset):
    epoch_train_loss = []
    batchid = 0
    for bag_batch_data in dataset.next_batch():
        if config.model_type == 'attention':
            #scores = model.forward(words_emb, images_features)
            #loss = criterion(scores, flags)
            loss = 0
        else:
            bag_semantic, bag_visual = model.forward(bag_batch_data)
            loss = constractive_loss(bag_semantic, bag_visual)
        optimizer.zero_grad()
        loss.backward()
        weight_norm = calc_norm(model.parameters(), ptype='weight')
        grad_norm = calc_norm(model.parameters(), ptype='grad')
        print('\ttotal weight norm = %.5f, total grad norm = %.5f' 
                %(weight_norm, grad_norm))
        if config.model_type in {'rnn', 'attention'}:
            rnn_weight_norm = calc_norm(model.rnn.parameters(), ptype='weight')
            rnn_grad_norm_before = calc_norm(model.rnn.parameters(), ptype='grad')
            torch.nn.utils.clip_grad_norm(model.rnn.parameters(), config.max_clip_norm)
            rnn_grad_norm_after = calc_norm(model.rnn.parameters(), ptype='grad')
            print('\trnn weight norm = %.5f, rnn norm before = %.5f, after = %.5f' 
                    % (rnn_weight_norm, rnn_grad_norm_before, rnn_grad_norm_after))
        optimizer.step()
        loss = loss.data.mean()
        epoch_train_loss.append(loss)
        batchid += 1
        print('epoch = %d, batch id = %d, train loss = %.3f' % (epoch, batchid, loss))
        if batchid > 100:
            break
    epoch_train_loss = np.array(epoch_train_loss).mean()
    return epoch_train_loss


def validation(epoch, dataset):
    epoch_val_loss = []
    batchid = 0
    for bag_batch_data in dataset.next_batch():
        if config.model_type == 'attention':
            #scores = model.forward(words_emb, images_features)
            #loss = criterion(scores, flags)
            loss = 0
        else:
            bag_semantic, bag_visual = model.forward(bag_batch_data)
            loss = constractive_loss(bag_semantic, bag_visual)
        loss = loss.data.mean()
        epoch_val_loss.append(loss)
        batchid += 1
        print('epoch = %d, batch id = %d, evaluation loss = %.3f' % (epoch, batchid, loss))
        if batchid > 30:
            break
    epoch_val_loss = np.array(epoch_val_loss).mean()
    return epoch_val_loss


def epoch_loop():
    train_parameters = config.select('model_type, learning_rate, which_text') + {'param':'train_loss', 'time':time.ctime()}
    val_parameters = config.select('model_type, learning_rate, which_text') + {'param':'val_loss', 'time':time.ctime()}
    # register parameters for hyperboard
    train_loss_agent = agent.register(train_parameters.getdict(), 'loss', overwrite = True)
    val_loss_agent = agent.register(val_parameters.getdict(), 'loss', overwrite = True)

    gvar = GlobalVariable()
    trainDataset = ArchDataset(config.copy(train=True), gvar)
    validDataset = ArchDataset(config.copy(train=False), gvar)

    checkpoints = config.select('model_type, learning_rate, which_text') + {'train_loss':[], 'valid_loss':[]}
    checkpoints.update(start_time = time.asctime())

    for epoch in range(config.max_epoch):
        model.train()
        train_loss = train(epoch, trainDataset)
        agent.append(train_loss_agent, epoch, train_loss)
        checkpoints.train_loss.append(train_loss)

        model.eval()
        val_loss = validation(epoch, validDataset)
        agent.append(val_loss_agent, epoch, val_loss)
        checkpoints.valid_loss.append(val_loss)

        if (epoch+1) % config.savefreq == 0:
            print('saving model')
            #checkpoints.update(model = model)
            checkpoints.update(end_time = time.asctime())
            torch.save(model, config.checkpoint)


if __name__ == '__main__':
    epoch_loop()
