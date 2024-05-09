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

class ERNIE_CRF(nn.Module):
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
        super(ERNIE_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = AutoModel.from_pretrained(bert_config)
        self.linear=nn.Linear(768,15)
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)

        self.tagset_size = tagset_size

    def forward(self, sentence, attention_mask=None,mode="gru"):
        out, _ = self.word_embeds(sentence, attention_mask=attention_mask)
        d_out = self.dropout1(out)
        # l_out = self.linear(d_out)
        # gru_feat = l_out.contiguous().view(batch_size, seq_length, -1)
        feat=self.linear(d_out)
        return feat

    def loss(self, feats, mask, tags):
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        # loss_value= torch.argmax(feats,mask,tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value



