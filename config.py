# coding=utf-8
class Config(object):
    def __init__(self):
        # self.label_file = 'data/MSRA_label_dict3.json'
        # self.train_file = './data/weibo/weiboNER.conll.train'
        # self.dev_file = './data/weibo/weiboNER.conll.dev'
        # self.test_file = './data/weibo/weiboNER.conll.test'
        self.label_file = './data/MSRA_label_dict3.json'
        self.train_file = './data/MSRA/train.txt'
        self.dev_file = './data/MSRA/dev.txt'
        self.test_file = './data/MSRA/test.txt'
        self.vocab = './ERNIE_Pretrain_Gram/vocab.txt'#/home/id_pos_mapping/Bert-Chinese-Text-Classification-Pytorch/bert_pretrain/vocab.txt'
        self.bert_path = './ERNIE_Pretrain_Gram'  # '/home/id_pos_mapping/Bert-Chinese-Text-Classification-Pytorch/bert_pretrain'
        self.max_length = 130
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 1
        self.rnn_hidden = 500
        self.bert_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 0.00001
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.base_epoch = 10

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':

    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
