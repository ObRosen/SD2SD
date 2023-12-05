
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
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
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/levin/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/levin_small/", help='path to save results')
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

def draw_loss(loss,step,index):
    epoch=np.arange(step)
    loss=np.array(loss)
    print(loss[-1])
    plt.plot(epoch, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    plt.savefig(f'loss_levin_small_{index}.png',dpi=600)
    return loss[-1]

# params_grid = {'num_channels_down': [[128,128,128,128,128],[128,128,128,128,64],[128,128,128,64,64],[128,128,64,64,64],[128,64,64,64,64],[64,64,64,64,64]], 'num_channels_skip': [[16,16,16,16,16],[16,16,16,16,8],[16,16,16,8,8]]}
# params_grid = {'num_channels_down': [[128,64,64,32,32],[64,64,64,32,32],[64,32,32,32,32],[32,32,32,32,32],[64,32,32,16,16]], 'num_channels_skip': [[16,16,16,8,8],[16,8,8,8,8],[8,8,8,8,8],[16,16,8,8,4],[16,16,8,4,4],[8,8,8,4,4],[4,4,4,4,4]]}
# params_grid= {'num_channels_down': [[128,128,128,128],[128,128,64,64],[128,64,64,64],[64,64,64,64],[128,128,64,32],[128,64,64,32],[64,64,32,32],[32,32,32,32]], 'num_channels_skip': [[16,16,16,16],[16,16,16,8],[16,16,8,8],[16,8,8,8],[8,8,8,8],[8,8,4,4],[4,4,4,4]]}
#params_grid = {'num_channels_down': [[128,128,128],[128,128,64],[128,64,64],[64,64,64],[64,64,32],[64,32,32],[32,32,32],[32,16,16],[16,16,16],[16,8,8],[8,8,8],[4,4,4]], 'num_channels_skip': [[16,16,16],[16,16,8],[16,8,8],[8,8,8],[8,4,4],[4,4,4],[2,2,2]]}
# params_grid={'num_channels_down': [[128,128],[128,64],[64,64],[64,32],[32,32],[32,16],[16,16],[16,8]], 'num_channels_skip': [[16,16],[16,8],[8,8],[8,4],[4,4],[4,2],[2,2]]}
params_grid={'num_channels_down': [[128,128,128,128,128]],'num_channels_skip': [[16,16,16,16,16]]}
index=300
time_record=[]
loss_record=[]
for num_channels_down in params_grid['num_channels_down']:
    for num_channels_skip in params_grid['num_channels_skip']:
        time1=time.time()
        index+=1
        print(num_channels_down, num_channels_skip)
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

            _, imgs = get_image(path_to_image, -1) # load image and convert to np.
            y = np_to_torch(imgs).type(dtype)

            img_size = imgs.shape
            print(imgname)
            # ######################################################################
            padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
            opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

            '''
            x_net:
            '''
            input_depth = 8

            net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

            net = skip( input_depth, 1,
                        num_channels_down = num_channels_down,
                        num_channels_up   = num_channels_down,
                        num_channels_skip = num_channels_skip,
                        upsample_mode='bilinear',
                        need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

            net = net.type(dtype)

            '''
            k_net:
            '''
            n_k = 200
            net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
            net_input_kernel.squeeze_()

            net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
            net_kernel = net_kernel.type(dtype)

            # Losses
            mse = torch.nn.MSELoss().type(dtype)
            ssim = SSIM().type(dtype)

            # optimizer
            optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
            scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

            # initilization inputs
            net_input_saved = net_input.detach().clone()
            net_input_kernel_saved = net_input_kernel.detach().clone()

            losses=[]

            ### start SelfDeblur
            sys.stderr = open('/dev/null', 'w')
            for step in tqdm(range(num_iter)):

                # input regularization
                net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

                # change the learning rate
                scheduler.step(step)
                optimizer.zero_grad()

                # get the network output
                out_x = net(net_input)
                out_k = net_kernel(net_input_kernel)
            
                out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
                # print(out_k_m)
                out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

                if step < 1000:
                    total_loss = mse(out_y,y) 
                else:
                    total_loss = 1-ssim(out_y, y) 
                losses.append(total_loss.item())
                total_loss.backward()
                optimizer.step()

                if (step+1) % opt.save_frequency == 0:
                    #print('Iteration %05d' %(step+1))

                    save_path = os.path.join(opt.save_path, f'{imgname}_x-{index}.png')
                    out_x_np = torch_to_np(out_x)
                    out_x_np = out_x_np.squeeze()
                    out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
                    out_x_np = np.uint8(out_x_np*255)
                    imsave(save_path, out_x_np)

                    save_path = os.path.join(opt.save_path, f'{imgname}_k-{index}.png')
                    out_k_np = torch_to_np(out_k_m)
                    out_k_np = out_k_np.squeeze()
                    out_k_np /= np.max(out_k_np)
                    out_k_np = np.uint8(out_k_np*255)
                    imsave(save_path, out_k_np)

                    torch.save(net, os.path.join(opt.save_path, f"{imgname}_xnet-{index}.pth"))
                    torch.save(net_kernel, os.path.join(opt.save_path, f"{imgname}_knet-{index}.pth"))
            sys.stderr = sys.__stderr__
            time2=time.time()
            print(f'参数组合{index}耗时{time2-time1}s。')
            time_record.append(time2-time1)
            last_loss=draw_loss(losses, num_iter,index)
            loss_record.append(last_loss)
            break

i=-1
with open('out_4.log', 'w') as f:
    # 遍历参数网格的键值对
    for num_channels_down in params_grid['num_channels_down']:
        for num_channels_skip in params_grid['num_channels_skip']:
            i+=1
            f.write(f'num_channels_down: {str(num_channels_down)}, num_channels_skip: {str(num_channels_skip)}, time cost: {str(time_record[i])}, last_loss: {str(loss_record[i])}\n')
