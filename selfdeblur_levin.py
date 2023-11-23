
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
from networks.cnn import network, pair_downsampler
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
parser.add_argument('--num_iter', type=int, default=5000,
                    help='number of epochs of training')
parser.add_argument('--img_size', type=int,
                    default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int,
                    default=[21, 21], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str,
                    default="datasets/levin/", help='path to blurry image')
parser.add_argument('--save_path', type=str,
                    default="results/levin/", help='path to save results')
parser.add_argument('--save_frequency', type=int,
                    default=100, help='lfrequency to save results')
opt = parser.parse_args()
# print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# Losses
mse = torch.nn.MSELoss().type(dtype)
ssim = SSIM().type(dtype)

# loss函数由mse+lambda*TV改成mse+l_res+l_cons
def criterion(output, target, net_input_image, step, residual_on=False): # TODO: 这里的net_input_image应该是一张图像，但后面实际输入的net_input恐怕并不是图
    loss_mse = mse(output, target)
    loss_ssim = 1-ssim(output, target)
    blurry1, blurry2 = pair_downsampler(net_input_image)

    if residual_on:
        pred1 = blurry1-net(blurry1)  # 直接学图片，先不学残差
        pred2 = blurry2-net(blurry2)
    else:
        pred1 = net(blurry1)
        pred2 = net(blurry2)

    loss_res = 1/2*(mse(blurry1, pred2)+mse(blurry2, pred1))

    if residual_on:
        blurry_deblurred = net_input_image - net(net_input_image)
    else:
        blurry_deblurred = net(net_input_image)  # 直接学图片，先不学残差
    deblurred1, deblurred2 = pair_downsampler(blurry_deblurred)

    loss_cons = 1/2*(mse(pred1, deblurred1) + mse(pred2, deblurred2))

    if step < 1000:
        loss = loss_mse + loss_res + loss_cons
    else:
        loss = loss_ssim+loss_res + loss_cons

    return loss

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

    if imgname.find('kernel1') != -1:
        opt.kernel_size = [17, 17]
    if imgname.find('kernel2') != -1:
        opt.kernel_size = [15, 15]
    if imgname.find('kernel3') != -1:
        opt.kernel_size = [13, 13]
    if imgname.find('kernel4') != -1:
        opt.kernel_size = [27, 27]
    if imgname.find('kernel5') != -1:
        opt.kernel_size = [11, 11]
    if imgname.find('kernel6') != -1:
        opt.kernel_size = [19, 19]
    if imgname.find('kernel7') != -1:
        opt.kernel_size = [21, 21]
    if imgname.find('kernel8') != -1:
        opt.kernel_size = [21, 21]

    _, imgs = get_image(path_to_image, -1)  # load image and convert to np.
    y = np_to_torch(imgs).type(dtype)

    img_size = imgs.shape
    print(imgname)
    # ######################################################################
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

    '''
    x_net:

    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

    net = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)
    '''

    '''
    new x_net:
    '''
    input_depth = 8
    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)
    net = network()  # TODO: 输入用什么比较好呢？本来的输入是什么作用？
    net = net.type(dtype)  # 什么作用？

    '''
    k_net:
    '''
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()

    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
    net_kernel = net_kernel.type(dtype)


    # optimizer
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {
                                 'params': net_kernel.parameters(), 'lr': 1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[
                            2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone() # TODO:后面怎么没有用到这个

    # start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std * \
            torch.zeros(net_input_saved.shape).type_as(
                net_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)

        out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])
        # print(out_k_m)
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

        '''
        if step < 1000:
            total_loss = mse(out_y, y)
        else:
            total_loss = 1-ssim(out_y, y)
        '''
        total_loss = criterion(out_y, y, net_input, step) # TODO：就是这里的net_input输入，不太对，net_input到底是什么要看get_noise()函数

        total_loss.backward()
        optimizer.step()

        if (step+1) % opt.save_frequency == 0:
            # print('Iteration %05d' %(step+1))

            save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[padh//2:padh//2 +
                                img_size[1], padw//2:padw//2+img_size[2]]

            out_x_np = np.uint8(out_x_np*255)
            imsave(save_path, out_x_np)

            save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            out_k_np = np.uint8(out_k_np*255)
            imsave(save_path, out_k_np)

            torch.save(net, os.path.join(
                opt.save_path, "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(
                opt.save_path, "%s_knet.pth" % imgname))
