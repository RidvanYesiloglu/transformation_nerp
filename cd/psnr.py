# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 02:45:03 2022

@author: ridva
"""
import runset_train.parameters as parameters
import torch
import torch.optim as optim
import torch.distributions.bernoulli as Bernoulli
import numpy as np
import math
import os
import errno
from tqdm import tqdm #(for time viewing)
import time
import models.nerp.write_actions_nerp as wr_acts
from pathlib import Path

import torch.backends.cudnn as cudnn
from utils import mri_fourier_transform_3d, save_image_3d, PSNR, check_gpu

import glob

from torchnufftexample import create_radial_mask, project_radial, backproject_radial
from networks import Positional_Encoder, FFN, SIREN
from skimage.metrics import structural_similarity as ssim
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
def update_runset_summary(args, runset_folder):
    reads = ""
    for i in range(args.totalInds):
        if Path(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(i+1))).is_file():
            run_i_file = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(i+1)), "r+")
            reads += run_i_file.read() + "\n"
            run_i_file.close()
    summary_file = open(os.path.join(runset_folder, 'runset_{}.txt'.format(args.runsetName)), "w")
    summary_file.write(reads)
    summary_file.close()

def main(args=None, im_ind=None):
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=7)
    args = Namespace(net = {"network_input_size":512,'network_width':350,'network_depth':8,'network_output_size':1}, \
                     encoder = {'embedding':'gauss','scale':3, 'embedding_size':256, 'coordinates_size':3}, \
                         img_path='../data/patient19/volume_1739.npy', gpu_id=int(input('Enter GPU ID: ')))
    des_n_proj = int(input('Desired nproj or -1 if prior: '))
    start_ind = 1739
    end_ind = 1801
    psnrs = np.zeros((end_ind-start_ind+1))
    ssims = np.zeros((end_ind-start_ind+1))
    args.img_path = '../data/patient19/volume_{}.npy'.format(1731)
    image_pr = torch.from_numpy(np.expand_dims(np.load(args.img_path),(0,-1))).cuda(args.gpu_id)
    for im_ind in range(start_ind,end_ind+1):
        args.img_path = '../data/patient19/volume_{}.npy'.format(im_ind)
        # Setup input encoder:
        encoder = Positional_Encoder(args)
        # Setup model as SIREN
        model = SIREN(args.net)
        # Setup loss function
        mse_loss_fn = torch.nn.MSELoss()
        if des_n_proj != -1:
            a_dir = '../detailed_results/vol_{}'.format(im_ind)
            for i in range(2):
                kk = [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
                a_dir += '/'+kk[0]
            model_list = [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
            for name in model_list:
                npr_ind = name.find('nproj_')
                found_nproj = int(name[npr_ind+6:name.find('&', npr_ind)])
                if found_nproj == des_n_proj:
                    a_dir += '/'+name
            model_path = a_dir + '/' + ([name for name in os.listdir(a_dir) if name.endswith('.pt')][0])
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
            model.load_state_dict(state_dict['net'])
            encoder.B = state_dict['enc'].cuda(args.gpu_id)
            model = model.cuda(args.gpu_id)
        #print('Load pretrain model: {}'.format(model_path))
        
        # Setup data loader
        #print('Load image: {}'.format(args.img_path))
        #data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])
        
        # Input coordinates (x, y, z) grid and target image
        #grid = grid.cuda()  # [bs, c, h, w, 3], [0, 1]
        #image = image.cuda()  # [bs, c, h, w, 1], [0, 1]
        
        image = torch.from_numpy(np.expand_dims(np.load(args.img_path),(0,-1))).cuda(args.gpu_id)
        grid_np = np.asarray([(x,y,z) for x in range(128) for y in range(128) for z in range(64)]).reshape((1,128,128,64,3))
        grid_np = (grid_np/np.array([128,128,64])) + np.array([1/256.0,1/256.0,1/128.0])
        grid = torch.from_numpy(grid_np.astype('float32')).cuda(args.gpu_id)
        # print('Image min: {} and max: {}'.format(image.min(), image.max()))
        # print('Image and grid created. Image shape: {}, grid shape: {}'.format(image.shape, grid.shape))
        # print('Image element size: {} and neleements: {}, size in megabytes: {}'.format(image.element_size(), image.nelement(), (image.element_size()*image.nelement())/1000000.0))
        # print('Grid element size: {} and neleements: {}, size in megabytes: {}'.format(grid.element_size(), grid.nelement(), (grid.element_size()*grid.nelement())/1000000.0))

        # Data loading
        test_data = (grid, image)
        test_embedding = encoder.embedding(test_data[0])
       
        with torch.no_grad():
            test_output = model(test_embedding) if des_n_proj != -1 else image_pr
    
            test_loss = 0.5 * mse_loss_fn(test_output, image)
            #print(test_output.shape, test_data[1].shape)
            test_psnr = - 10 * torch.log10(2 * test_loss).item()
            #print('PRETRAIN MODEL PSNR:')
            #print('Test psnr: {:.5f}'.format(test_psnr))
            
            test_loss = test_loss.item()
            test_ssim = ssim(test_output.cpu().numpy().squeeze(), image.cpu().numpy().squeeze(), data_range=1)
        #np.save(os.path.join(inps_dict['save_folder'], 'pretrainmodel_out'), test_output.detach().cpu().numpy())
        psnrs[im_ind-start_ind] = test_psnr
        ssims[im_ind-start_ind] = test_ssim
    print('ALL PSNRS: ')
    print(psnrs)
    print('Average: {:.5f}, min: {:.5f} (at {}), max: {:.5f} (at {})'.format(psnrs.mean(), psnrs.min(), psnrs.argmin()+start_ind, psnrs.max(), psnrs.argmax()+start_ind))
    print('ALL SSIMS: ')
    print(ssims)
    print('Average: {:.5f}, min: {:.5f} (at {}), max: {:.5f} (at {})'.format(ssims.mean(), ssims.min(), ssims.argmin()+start_ind, ssims.max(), ssims.argmax()+start_ind))
    if des_n_proj == -1:
        np.save('psnr_prior', psnrs)
        np.save('ssim_prior', ssims)
    else:
        np.save('psnr_nproj_{}'.format(des_n_proj), psnrs)
        np.save('ssim_nproj_{}'.format(des_n_proj), ssims)
if __name__ == "__main__":
    main() 
