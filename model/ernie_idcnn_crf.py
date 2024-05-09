# coding=utf-8
# coding=utf-8
import torch.nn as nn
import torch
from transformers import BertModel,AutoModel,AlbertModel
from model import CRF
from model.idcnn import IDCNN
from torch.autograd import Variable
import torch

import warnings
warnings.filterwarnings("ignore")

class ERNIE_IDCNN_CRF(nn.Module):
    """
    bert_lstm_crf model
    """
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
        super(ERNIE_IDCNN_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = BertModel.from_pretrained(bert_config)
        self.idcnn = IDCNN(input_size=embedding_dim, filters=64)
        self.linear_matchDim=nn.Linear(64,768)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio, batch_first=True)
        self.gru=nn.GRU(input_size=embedding_dim,hidden_size=hidden_dim,num_layers=rnn_layers,bias=True,
                        batch_first=True,dropout=dropout_ratio,bidirectional=True)

        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.linear = nn.Linear(hidden_dim*2, tagset_size+2)
        self.tagset_size = tagset_size
        self.linear2=nn.Linear(768,14)#换成MRSA的时候，14改成15（tagsize=dic-label的长度）
    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        # return Variable(
        #     torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
        #     torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))
        return Variable(torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim))

    def forward(self, sentence, attention_mask=None,mode="gru"):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''

        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds, _ = self.word_embeds(sentence, attention_mask=attention_mask)
        hidden = self.rand_init_hidden(batch_size)
        device = "cuda:0"
        # if embeds.is_cuda:
            # hidden = (i.cuda() for i in hidden)
            # hidden1 = torch.tensor(hidden[0].cuda())
            # hidden2 = torch.tensor(hidden[1].cuda())
            # hidden=(hidden1,hidden2)
        # hidden=hidden.cuda()

        if mode=='idcnn':
            out = self.idcnn(embeds, seq_length)
            idcnn_out = embeds + self.dropout1(self.linear_matchDim(out))
        out = idcnn_out.contiguous().view(-1, 768)
        d_out = self.dropout1(out)
        feat = self.linear2(d_out)
        feat = feat.contiguous().view(batch_size, seq_length, -1)

        return feat

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value



