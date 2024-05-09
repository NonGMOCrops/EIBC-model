# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class IDCNN(nn.Module):
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [#定义Dilation Width
            {"dilation": 1},
            {"dilation": 2},
            {"dilation": 4},
            {"dilation": 8}
            ]#再加一层
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(130) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(130) for _ in range(num_block)])
        for i in range(len(self.layers)):#定义DilatedCNNBlock，每个Block中有DilationWidth分别为1，1，2的三层卷积层
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, single_block)#卷积层
            net.add_module("relu", nn.ReLU())#激活层
            net.add_module("layernorm", norms_1[i])#归一化

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()


        for i in range(num_block):#定义IDCNN，由四个相同的DilatedCNNBlock拼接而成
            self.idcnn.add_module("block%i" % i, net)#DilatedCNN层
            self.idcnn.add_module("relu", nn.ReLU())#激活层
            self.idcnn.add_module("layernorm", norms_2[i])#归一化

    def forward(self, embeddings):
        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2