import torch
import torch.nn as nn
from .common import *
import torch.nn.functional as F


class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x


'''
假设输入张量的维度为 [batch_size, n_chan, height, width]，其中 n_chan 表示输入张量的通道数，height 和 width 表示输入张量的高度和宽度。
经过第一层卷积操作 self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1) 后，输出张量的维度为 [batch_size, chan_embed, height, width]，其中 chan_embed 是你在网络初始化时指定的通道数。
接着，经过第二层卷积操作 self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1) 后，输出张量的维度仍然为 [batch_size, chan_embed, height, width]。
最后，经过第三层卷积操作 self.conv3 = nn.Conv2d(chan_embed, n_chan, 1) 后，输出张量的维度变为 [batch_size, n_chan, height, width]，与输入张量的维度相同。
因此，这个2层CNN的输入和输出维度是一致的，都为 [batch_size, n_chan, height, width]。在网络的前向传播过程中，输入经过卷积操作后的输出张量维度保持不变。
'''


def pair_downsampler(img): # 输入一张【带噪】图像
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2