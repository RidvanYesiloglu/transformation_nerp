# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:23:34 2022

@author: ridva
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio
#from utilsRadialMRI import SampleRadial, ReconRadial

from pynufft import NUFFT
import math
import torch
from torchnufftexample import create_radial_mask, project_radial, backproject_radial

def SampleRadial_torch(volume, nProj): 
    print('girdi su anda')
    #volume = np.moveaxis(volume, 0, -1) #[128 128 64]
    imgSize = volume.shape
    
    projections = torch.zeros((imgSize[0], imgSize[2], nProj))
    
    radius = imgSize[0]
    #om = np.empty((radius,2), dtype = np.float32)
    
    NufftObj = NUFFT()
    Nd = (radius,radius)
    Kd = (radius*2,radius*2)
    Jd = (6,6)
    
    kspace = torch.zeros((imgSize[0], imgSize[2]), dtype = torch.complex64)
    for aid in range(nProj):
        
        radian = aid*111.246118/180*math.pi 
#om[0 : imgSize[0] ,0]
        mt1 = math.cos(radian) * (torch.arange(0, radius) - radius/2 )*math.pi/ radius*2 # normalized between -1 and 1
        mt2 = math.sin(radian) * (torch.arange(0, radius) - radius/2 )*math.pi/ radius*2 # normalized between -1 and 1
        mt = torch.stack((mt1,mt2),-1)
        #mt.retain_grad()
        NufftObj.plan(mt, Nd, Kd, Jd)
        
        
        for sid in range(imgSize[2]):
            kspace[:,sid] = NufftObj.forward(volume[:,:,sid])
            
        recon = torch.fft.ifftshift(kspace,0)	
        recon = torch.fft.ifft(recon,radius, 0)
        recon = torch.fft.fftshift(recon,0)
        recon = torch.abs(recon)
        
        projections[:,:,aid] = recon
        
    print(projections.shape, 'bas')
    #projections = np.moveaxis(projections, -1, 0)
    #print(projections.shape)
    #projections = np.moveaxis(projections, 1, -1) 
    #print(projections.shape)
    projections = projections/60
    return projections
def radialrecon(img, nproj):
    # projections = SampleRadial(img, nproj)
    # solved = ReconRadial(projections, img.shape, nproj)

    #fig, ax = plt.subplots(1,2,figsize=(5,10))
   
    # IMPORTANT ANIMATION CODE HERE
    # Used to keep the limits constant
    #ax.set_ylim(0, y_max)

    ktraj, grid_size = create_radial_mask(nproj, img.shape, -1, plot=False)
    kdata = project_radial(img, ktraj)
    image_blurry_numpy, image_sharp_numpy = backproject_radial(kdata, ktraj, img, grid_size, plot=False)   
    
    iminds = [10,54]
    fig, axs = plt.subplots(len(iminds), 2, figsize=(5*len(iminds),10))
    for i in range(len(iminds)):
        axs[i,0].imshow(np.abs(img[iminds[i],0]), cmap='gray')
        axs[i,0].axis('off')
        axs[i,1].imshow(np.abs(image_sharp_numpy[iminds[i]]), cmap='gray')
        axs[i,1].set(title='Number of projections: {}'.format(nproj))
        axs[i,1].axis('off')
        
        plt.sca(axs[i,0])
        plt.title('Axial Image at Slice {}'.format(iminds[i]))
        plt.sca(axs[i,1])
        plt.title('Reconstruction with {} Radial Projections'.format(nproj))
        #plt.suptitle()
        #plt.figtext(0.5,1-i/float(len(iminds)), 'Axial Image at Slice {}'.format(iminds[i]), ha='center', va='center')
        # Adjust vertical_spacing = 0.5 * axes_height
        #plt.subplots_adjust(hspace=0.5)
    
    # Add text in figure coordinates
        # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

#kwargs_write = {'fps':1.0, 'quantizer':'nq'}
image = torch.from_numpy(np.load('../data/patient19/volume_1732.npy')).permute((2,0,1)).unsqueeze(1)
image = image+0j*image
#projections = SampleRadial_torch(torch.from_numpy(img), 2)

imageio.mimsave('radialsamplings_torch_1.gif', [radialrecon(image, nproj) for nproj in (list(range(2,12))+list(range(15,101,5)))], fps=0.7)






    

    
        
 
def ReconRadial_torch(projections,imgSize,nProj):
    kspace_restore = np.empty([imgSize[0],imgSize[2],nProj],dtype = 'complex') # 128*64*12
    for aid in range(nProj):
        kspace_restore[:,:,aid] = fftc(projections[:,:,aid],0)
        
    kspace_restore = np.moveaxis(kspace_restore,1,2)
    kspace_restore = np.reshape(kspace_restore,(imgSize[0]*nProj,imgSize[2]))
    
    radius = imgSize[0]
    spoke_range = (np.arange(0, radius) - radius/2 )*np.pi/ radius*2
    om_total = np.empty((radius,2,nProj), dtype = np.float32)
    for aid in range(nProj):       
        radian = aid*111.246118/180*np.pi 
        spoke_x =  spoke_range * np.cos(radian)
        spoke_y =  spoke_range * np.sin(radian)
        om_total[0 : imgSize[0],0,aid] = spoke_x
        om_total[0 : imgSize[0],1,aid] = spoke_y

    om_total = np.moveaxis(om_total,1,2)
    om_total = np.reshape(om_total,[128*nProj,2])

    NufftObj = NUFFT()
    Nd = (radius,radius)
    Kd = (radius*2,radius*2)
    Jd = (6,6)	
    NufftObj.plan(om_total, Nd, Kd, Jd)

    nSamples =radius;
    deltaRho = 1/nSamples;
    RHOxy = np.sqrt(np.sum(np.square(om_total),1))
    upperRHOxy = (RHOxy + deltaRho/2);
    lowerRHOxy = (RHOxy - deltaRho/2);
    densityCompensation = np.pi*np.abs(np.square(upperRHOxy) - np.sign(lowerRHOxy*upperRHOxy)*np.square(lowerRHOxy))/2;
    rhoxy0 = 0.8
    if nProj<3:
        rhoxy0 = 0.5
        
    Gfilter = np.exp(-1*np.square((RHOxy/rhoxy0)));
    densityCompensation2 = densityCompensation*Gfilter; 
    
    solved = np.empty(imgSize,dtype = np.float32)
    for sid in range(imgSize[2]):
        oneslice = kspace_restore[:,sid]*densityCompensation2
        solved[:,:,sid] = np.abs(NufftObj.adjoint(oneslice))   
    
    solved = solved/25*(10**6)
    if nProj<3:
        solved = solved*5   
    solved = solved*1.5
	
    #plt.imshow(solved[:,:,32], aspect='equal', cmap=cm.gray, vmax=1, vmin=0)
    #plt.show()
    return solved