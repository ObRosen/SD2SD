

from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import *
import cv2
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/lai/uniform_ycbcr/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/lai/uniform_test_2", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
opt = parser.parse_args()
#print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")


files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('kernel_01') != -1:
        opt.kernel_size = [31, 31]
    if imgname.find('kernel_02') != -1:
        opt.kernel_size = [51, 51]
    if imgname.find('kernel_03') != -1:
        opt.kernel_size = [55, 55]
    if imgname.find('kernel_04') != -1:
        opt.kernel_size = [75, 75]

    _, imgs = get_image(path_to_image, -1) # load image and convert to np.
    y = np_to_torch(imgs).type(dtype) # 1x1x680x1024, 4dim

    img_size = imgs.shape
    print(imgname)
    # ######################################################################
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

    '''
    x_net:
    '''
    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype) # 1x8x710x1054=1 x input_depth x spatial_size[0] x spatial_size[1]

    net = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU') # 这一步返回的是模型

    net = net.type(dtype)

    
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_() # 200, 1dim

    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
    net_kernel = net_kernel.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    #
    net_input_saved = net_input.detach().clone() # 1x8x710x1054, 4dim
    net_input_kernel_saved = net_input_kernel.detach().clone() # 这行是不是没用

    ### start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_() # 1x8x710x1054, 4dim
        # net_input_kernel = net_input_kernel_saved + reg_noise_std*torch.zeros(net_input_kernel_saved.shape).type_as(net_input_kernel_saved.data).normal_()
        '''
        .normal_()对全零张量进行正态分布采样，生成具有随机噪声的张量。
        reg_noise_std是一个标量，表示噪声的标准差。
        reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()生成了一个具有随机噪声的张量，其形状与net_input_saved相同，并且噪声服从均值为0、标准差为reg_noise_std的正态分布。
        因此，最终的net_input是原始输入net_input_saved与具有随机噪声的张量相加而得，其形状与net_input_saved相同。  
        '''
        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input) # 这一步输入的参数是模型的参数，而不是skip的参数(参考DIP网络：输入是一个噪声图像或待优化的图像，输出是一个经过优化的图像。)
        out_k = net_kernel(net_input_kernel)
    
        out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1]) # 1x1x31x31, 4dim
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None) # 1x1x680x1024, 4dim
        # out_height = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        # out_width = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

        if step < 500:
            total_loss = mse(out_y, y)
        else:
            total_loss = 1 - ssim(out_y, y)

        total_loss.backward()
        optimizer.step()

        if (step+1) % opt.save_frequency == 0:
            #print('Iteration %05d' %(step+1))

            save_path = os.path.join(opt.save_path, '%s_x.png'%imgname)
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            #out_x_np = out_x_np.astype(np.uint8)
            out_x_np = np.uint8(out_x_np*255)
            imsave(save_path, out_x_np)

            save_path = os.path.join(opt.save_path, '%s_k.png'%imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            #out_k_np = out_k_np.astype(np.uint8)
            out_k_np = np.uint8(out_k_np*255)
            imsave(save_path, out_k_np)

            torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))
