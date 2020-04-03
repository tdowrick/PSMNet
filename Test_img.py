from __future__ import print_function
import argparse
import os
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *
import cv2

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= None,
                    help='load model')
parser.add_argument('--rightimg', default= None,
                    help='load model')
parser.add_argument('--outfile', default=None,
                    help='Location to save output disparity map')
parser.add_argument('--isgray', default= False,
                    help='load model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--indir', type=str, default="",
                    help='Process all files in directory')
parser.add_argument('--outdir', type=str, default="",
                    help="Write output to this dir")

def load_model(model='stackhourglass', loadmodel='./trained/pretrained_model_KITTI2015.tar', maxdisp=192):

    if model == 'stackhourglass':
        model = stackhourglass(maxdisp)
    elif model == 'basic':
        model = basic(args.maxdisp)
    else:
        print('no model')

    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    if loadmodel is not None:
        print('load PSMNet')
        state_dict = torch.load(loadmodel)
        model.load_state_dict(state_dict['state_dict'])

    return model

def scale_dowm_images(left_img:np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Reduce image size to fit in CUDA memory. """

    new_width = 640
    new_height = 360
    left_resized = skimage.transform.resize(left_img, (new_height, new_width))
    right_resized = skimage.transform.resize(right_img, (new_height, new_width))

    return left_resized, right_resized

def process_image(left_img: str, right_img: str, model=model) -> np.ndarray:
       start_time = time.time()

       processed = preprocess.get_transform(augment=False)
       imgL_o = (skimage.io.imread(left_img))
       imgR_o = (skimage.io.imread(right_img))
       
       original_height, original_width = imgL_o.shape[:2]
       
       imgL, imgR = scale_dowm_images(imgL_o, imgR_o)

       imgL = processed(imgL).numpy()
       imgR = processed(imgR).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       # pad to width and hight to 16 times
       if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
       else:
            top_pad = 0 
       if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            left_pad = (times+1)*16-imgL.shape[3]
       else:
            left_pad = 0     
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       if torch.cuda.is_available():
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

       imgL, imgR= Variable(imgL), Variable(imgR)

       with torch.no_grad():
            disp = model(imgL,imgR)

       disp = torch.squeeze(disp)
       pred_disp = disp.data.cpu().numpy()

       print('time = %.2f' %(time.time() - start_time))
       if top_pad !=0 or left_pad != 0:
            img = pred_disp[top_pad:,left_pad:]
       else:
            img = pred_disp

       print(img.shape)

       #Resize disparity to size of original image. Also need to scale 
       disparity_scale_factor = original_width / img.shape[1]
       img = skimage.transform.resize(img, (original_height, original_width))
       img = img * disparity_scale_factor

       img = (img*256).astype('uint16')
       return img

def process_directory(dir_path: str, output_path: str):
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('png') or f.endswith('jpg')]
    left_files = [f for f in files if '-l' in f]
    right_files = [f for f in files if '-r' in f]

    for l, r in zip(left_files, right_files):
        print(f"Processsing {l}, {r}")
        frame_num = l.split('-')[0]

        l_file = os.path.join(dir_path, l)
        r_file = os.path.join(dir_path, r)

        output_file = f'disparity-{frame_num}.png'
        output_full_path = os.path.join(output_path, output_file)
        print(f"Writing {output_full_path}")
        img = process_image(l_file, r_file)
        cv2.imwrite(output_full_path,img)


def main():
    if args.indir:
        process_directory(args.indir, args.outdir)
    
    else:
        img = process_image(args.leftimg, args.rightimg)
        cv2.imwrite(args.outfile ,img)


if __name__ == '__main__':

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = load_model(args.model, args.loadmodel, args.maxdisp)
    model.eval()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    main()






