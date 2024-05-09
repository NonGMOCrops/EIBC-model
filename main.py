# coding=utf-8
import torch
import torch.nn as nn
import time
import json
from torch.autograd import Variable
from config import Config
from model import BERT_LSTM_CRF, ERNIE_CRF
from model import ERNIE_IDCNN_GRU_CRF
from model import ERNIE_IDCNN_CRF
import torch.optim as optim
from utils2 import load_vocab, read_corpus, load_model, save_model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
from seqeval.metrics import classification_report
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from torchsummary import summary
torch.cuda.current_device()
torch.cuda._initialized = True
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

mode = 'idcnn'


def train(**kwargs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = Config()
    config.update(**kwargs)
    print('当前设置为:\n', config)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)
    print('loading corpus')
    vocab = load_vocab(config.vocab)
    with open(config.label_file) as file:
        label_dic = json.load(file)
    index2label = dict([(index, key) for key, index in label_dic.items()])
    tagset_size = len(label_dic)
    train_data = read_corpus(config.train_file, max_length=config.max_length,
                             label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length,
                           label_dic=label_dic, vocab=vocab)
    print(dev_data)
    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=False,# num_samples should be a positive integer value, but got num_samples=0,,,shuffle改false
                              batch_size=config.batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])

    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=False,
                            batch_size=config.batch_size)

    # model = Multi_Intention(config.bert_path, tagset_size,
    #                       config.bert_embedding, config.rnn_hidden,
    #                       config.rnn_layer, dropout_ratio=config.dropout_ratio,
    #                       dropout1=config.dropout1, use_cuda=config.use_cuda)
    # model = BERT_LSTM_CRF(config.bert_path, tagset_size,
    #                       config.bert_embedding, config.rnn_hidden,
    #                       config.rnn_layer, dropout_ratio=config.dropout_ratio,
    #                       dropout1=config.dropout1, use_cuda=config.use_cuda)
    # model = ERNIE_IDCNN_CRF(config.bert_path, tagset_size,
    #                       config.bert_embedding, config.rnn_hidden,
    #                       config.rnn_layer, dropout_ratio=config.dropout_ratio,
    #                       dropout1=config.dropout1, use_cuda=config.use_cuda)

    model = ERNIE_IDCNN_GRU_CRF(config.bert_path, tagset_size,
                          config.bert_embedding, config.rnn_hidden,
                          config.rnn_layer, dropout_ratio=config.dropout_ratio,
                          dropout1=config.dropout1, use_cuda=config.use_cuda)
    # model = ERNIE_CRF(config.bert_path, tagset_size,
    #                             config.bert_embedding, config.rnn_hidden,
    #                             config.rnn_layer, dropout_ratio=config.dropout_ratio,
    #                             dropout1=config.dropout1, use_cuda=config.use_cuda)

    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)

    model.to(device)
    model.train()

    embeds_params = [cell[1] for cell in list(filter(lambda kv: 'embeds' in kv[0], model.named_parameters()))]
    others_params = [cell[1] for cell in
                     list(filter(lambda kv: 'embeds' not in kv[0], model.named_parameters()))]
    optimizer = optim.Adam([{"params": embeds_params, "lr": 0.00001}, {"params": others_params, "lr": 0.0003},
                            ], weight_decay=0.00005)


    eval_loss = 10000
    for epoch in range(config.base_epoch):
        step = 0
        for i, batch in enumerate(train_loader):
            step += 1
            model.zero_grad()
            inputs, masks, tags = batch
            inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            if config.use_cuda:
                inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()

            feats = model(inputs, masks, mode)
            loss = model.loss(feats, masks, tags)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))

        loss_temp = dev(model, dev_loader, epoch, config, index2label)
        # if loss_temp < eval_loss:
        #     save_model(model, epoch)


def dev(model, dev_loader, epoch, config, index2label):
    global device
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0#ZeroDivisionError: division by zero  0—>1
    for i, batch in enumerate(dev_loader):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()


        feats = model(inputs, masks, mode)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        eval_trans(pred, true, best_path.cpu().numpy().tolist(),
                   tags.cpu().numpy().tolist(), index2label)
        # pred.extend([t for t in best_path])
        # true.extend([t for t in tags])
    print('mode : {}| eval  epoch: {}|  loss: {}'.format(mode, epoch, eval_loss / length))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(classification_report(true, pred, digits=4))
    model.train()
    return eval_loss


def eval_trans(pred, true, best_path, tags, index2label):
    for pred_path, true_path in zip(best_path, tags):

        pred_path_string = [normalize_string(index2label.get(cell)) for cell in pred_path]
        true_path_string = [normalize_string(index2label.get(cell)) for cell in true_path]
        pad_len = true_path.count(2)
        pred.append(pred_path_string[:-pad_len][1:-1])
        true.append(true_path_string[:-pad_len][1:-1])


def normalize_string(string):
    return string.replace('B_', 'B-'). \
        replace('E_', 'I-').replace('M_', 'I-').replace('o', 'O')
# def normalize_string(string):
#     return string.replace('B-', 'B-'). \
#         replace('E-', 'I-').replace('M-', 'I-').replace('S', 'O').\
#         replace('E', 'I').replace('M', 'I')


if __name__ == '__main__':
    train()
    # test()
