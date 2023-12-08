
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
import time

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
                    default="results/levin_cnn_originloss/", help='path to save results')
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
def criterion(output, target, net_input_image, net_input_kernel, step, residual_on=False):
    loss_mse = mse(output, target)
    loss_ssim = 1-ssim(output, target)
    blurry1, blurry2 = pair_downsampler(net_input_image) # D_1(y), D_2(y) # torch.Size([1, 1, 135, 135]) 

    
    if residual_on:
        pred_1 = blurry1-net(blurry1) # 这里的f应该不是单独的cnn，而是整个去模糊网络? NO.
        pred_2 = blurry2-net(blurry2)
    else:
        pred_1 = net(blurry1)  # 直接学图片，先不学残差
        pred_2 = net(blurry2)
    
    '''
    # 问题出在blurry1与pred_1的大小不同，无法计算残差。cnn的输入输出尺度相同，但fcn x cnn并非如此， 需要resize图片（经计算不可能满足）
    
    out_x_1 = net(blurry1) # torch.Size([1, 1, 135, 135])
    out_x_2 = net(blurry2)
    out_k = net_kernel(net_input_kernel) # torch.Size([289])
    out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1]) # torch.Size([1, 1, 17, 17])
    pred_1 = nn.functional.conv2d(out_x_1, out_k_m, padding=0, bias=None)  # f_\theta(D_1(y)) # torch.Size([1, 1, 119, 119])
    pred_2 = nn.functional.conv2d(out_x_2, out_k_m, padding=0, bias=None)  # f_\theta(D_2(y))

    if residual_on: # 学残差
        pred_1=blurry1-pred_1 # D_1(y)-f_\theta(D_1(y))
        pred_2=blurry2-pred_2 # D_2(y)-f_\theta(D_2(y))
    '''

    loss_res = 1/2*(mse(blurry1, pred_2)+mse(blurry2, pred_1))

    
    if residual_on:
        blurry_deblurred = net_input_image - net(net_input_image)
    else:
        blurry_deblurred = net(net_input_image)  # 直接学图片，先不学残差
    deblurred_1, deblurred_2 = pair_downsampler(blurry_deblurred)
    
    '''
    deblur_x=net(net_input_image) # torch.Size([1, 1, 271, 271])
    deblurred=nn.functional.conv2d(deblur_x, out_k_m, padding=0, bias=None) # f_\theta(y) # torch.Size([1, 1, 255, 255])
    deblurred_1, deblurred_2= pair_downsampler(deblurred) # D_1(f_\theta(y)), D_2(f_\theta(y)) # torch.Size([1, 1, 127, 127])

    if residual_on: # 学残差
        deblurred=net_input_image-nn.functional.conv2d(deblur_x, out_k_m, padding=0, bias=None) # y-f_\theta(y) 
        deblur_1,deblur_2=pair_downsampler(deblurred) # D_1(y-f_\theta(y)), D_2(y-f_\theta(y)) 
        deblurred_1=blurry1-deblur_1
        deblurred_2=blurry2-deblur_2
    '''
    loss_cons = 1/2*(mse(pred_1, deblurred_1) + mse(pred_2, deblurred_2))

    if step < 1000:
        loss = loss_mse + loss_res + loss_cons
    else:
        loss = loss_ssim + loss_res + loss_cons

    return loss


def draw_loss(loss,step):
    epoch=np.arange(step)
    loss=np.array(loss)
    plt.plot(epoch, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    plt.savefig('loss_cnn.png',dpi=600)

# start #image
for f in files_source:
    time1=time.time()
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f # 'datasets/levin/im1_kernel1_img.png'
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
    y = np_to_torch(imgs).type(dtype)  # imgs为图像，torch.Size([1, 1, 255, 255])

    img_size = imgs.shape
    print(imgname)
    # ######################################################################
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1 # [17-1, 17-1]
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw # [271, 271]

    '''
    new x_net:
    '''
    input_depth = 1  # 实测在skip网络中这个变量设置为1也可以跑通，所以这个变量的作用是什么？
    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype) # size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) = torch.Size([1, 1, 271, 271])

    net = network(input_depth) # 在cnn定义中，此处n_chan=input_depth，即输入的net_input的第二维的大小
    net = net.type(dtype) # 将net模型对象的数据类型转换为dtype所指定的数据类型，确保模型的输入和输出张量的数据类型与dtype一致。

    '''
    k_net:
    '''
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()

    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
    net_kernel = net_kernel.type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': net_kernel.parameters(), 'lr': 1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()  # 没用

    losses=[]

    # start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_() # torch.Size([1, 1, 271, 271])

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)  # 输入和输出大小不变，网络要满足这个要求 # torch.Size([1, 1, 271, 271])
        out_k = net_kernel(net_input_kernel) # torch.Size([289])

        out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1]) # torch.Size([1, 1, 17, 17])
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)  # 这一行决定out_X的第二维=1 # torch.Size([1, 1, 255, 255])

        
        if step < 1000:
            total_loss = mse(out_y, y)
        else:
            total_loss = 1-ssim(out_y, y)

        # total_loss = criterion(out_y, y, net_input, net_input_kernel, step)
        losses.append(total_loss.item())
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
    time2=time.time()
    print(f'训练5000代耗时{time2-time1}s')
    draw_loss(losses, num_iter)
    break
