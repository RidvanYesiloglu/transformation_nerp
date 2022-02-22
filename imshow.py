# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:53:41 2022

@author: ridva
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchnufftexample import create_radial_mask, project_radial, backproject_radial

def calcpsnr(reconstructed, original):
    mse = torch.nn.MSELoss()(reconstructed, original)
    psnr = - 20 * torch.log10(torch.sqrt(mse)) #assuming max=1
    return psnr
ref_ind = 41
rec = np.load('savedrec_run0_ep5000_33.47dB.npy')[0,...,0]#np.load('savedrec_run0_ep250_35.68dB.npy')[0,...,0]
ref = np.load('data73/ims_tog.npy')[ref_ind]
pri = np.load('data73/ims_tog.npy')[0]
psnr = calcpsnr(torch.from_numpy(rec),torch.from_numpy(ref))


image = torch.from_numpy(np.expand_dims(np.load('data73/ims_tog.npy')[ref_ind], (0)))#.permute((2,0,1)).unsqueeze(1)
image = image+0j*image
nproj=20
ktraj, im_size, grid_size = create_radial_mask(nproj, (64,1,128,128), -1, plot=False)
kdata = project_radial(image, ktraj, im_size, grid_size)
image_blurry_numpy, image_sharp_numpy = backproject_radial(kdata, ktraj, image, im_size, grid_size, plot=False) 
image_blurry_numpy = np.abs(np.transpose(image_blurry_numpy, (1,2,0)))
image_sharp_numpy = np.abs(np.transpose(image_sharp_numpy, (1,2,0)))
image_sharp_numpy = image_sharp_numpy/(image_sharp_numpy.max())
image_blurry_numpy = image_blurry_numpy/(image_blurry_numpy.max())
psnr2 = calcpsnr(torch.from_numpy(image_sharp_numpy),torch.from_numpy(ref))
print('rec,ref', psnr)
print('bpr,ref', psnr2)


min_psnr = 100000
min_ind = -1
for i in range(64):
    psnr = calcpsnr(torch.from_numpy(rec[...,i]),torch.from_numpy(ref[...,i]))
    if psnr < min_psnr:
        min_psnr = psnr
        min_ind = i
print('a', min_psnr)

fig,ax = plt.subplots(1,5)
ax[0].imshow(ref[...,min_ind], cmap='gray')#, vmin=0, vmax=1)
ax[1].imshow(rec[...,min_ind], cmap='gray')#, vmin=0, vmax=1)
ax[2].imshow(pri[...,min_ind], cmap='gray')#, vmin=0, vmax=1)
im3 = ax[3].imshow(ref[...,min_ind]-rec[...,min_ind], )
divider = make_axes_locatable(ax[3])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')
ax[4].imshow(image_sharp_numpy[...,min_ind], cmap='gray')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[4].axis('off')
ax[0].set(title='Ground Truth')
ax[1].set(title='Reconstructed Image with 2 Projections')
ax[2].set(title='Prior')
ax[3].set(title='Difference Image')
ax[4].set(title='Backprojection')
plt.suptitle('The Lowest-PSNR Slice with 2 Radial Projections (PSNR: {:.5f})'.format(min_psnr))
        # plt.sca(axs[i,0])
        # plt.title('Axial Image at Slice {}'.format(iminds[i]))
        # plt.sca(axs[i,1])
        # plt.title('Reconstruction with {} Radial Projections'.format(nproj))