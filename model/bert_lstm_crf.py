# coding=utf-8
# coding=utf-8
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel,BertModel
from model import CRF
from model.idcnn import IDCNN
from torch.autograd import Variable
import torch

import warnings
warnings.filterwarnings("ignore")

class BERT_LSTM_CRF(nn.Module):
    """
    bert_lstm_crf model
    """
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
        super(BERT_LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = AutoModel.from_pretrained(bert_config)
        self.idcnn = IDCNN(input_size=embedding_dim, filters=64)
        self.linear_matchDim=nn.Linear(64,768)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio, batch_first=True)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.liner = nn.Linear(hidden_dim*2, tagset_size+2)
        self.attn=nn.Linear(hidden_dim*2,1)
        self.tagset_size = tagset_size

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))

    def forward(self, sentence, attention_mask=None):
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
        if embeds.is_cuda:
            hidden = (i.cuda() for i in hidden)
        hidden1 = hidden[0].cuda()
        hidden2 = hidden[1].cuda()
        hidden = (hidden1, hidden2)
        # if mode=='idcnn':
        out = self.idcnn(embeds, seq_length)
        idcnn_out = embeds + self.dropout1(self.linear_matchDim(out))
        lstm_out, hidden = self.lstm(idcnn_out, hidden)
        # else:
        #     lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        # 注意力attn_weights=self.attn(lstm_out).squeeze(2)
        # attn_weights = F.softmax(attn_weights, dim=1)
        # attn_applied = torch.matmul(attn_weights.unsqueeze(1), lstm_out)
        d_lstm_out = self.dropout1(lstm_out)
        l_out = self.liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

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



