# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:06:20 2022

@author: ridva
"""
from warnings import filterwarnings

#import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
#from skimage.data import shepp_logan_phantom


def project_radial(image, ktraj, im_size, grid_size):
    #print('image shape was ', image.shape)
    image  = image.permute((3,0,1,2,4)).squeeze(-1)
    #image.retain_grad()
    image = torch.stack((image, image*0),-1)
    #image.retain_grad()
    #print('now image shape', image.shape)
    # create NUFFT objects, use 'ortho' for orthogonal FFTs
    nufft_ob = tkbn.KbNufft(
        im_size=im_size, grid_size=grid_size).to(image)
    
    #print(nufft_ob)
    # calculate k-space data
    kdata = nufft_ob(image, ktraj)
    return kdata

def create_radial_mask(nspokes, image_shape, gpu_id, plot=False):
    # create a k-space trajectory and plot it
    im_size = (image_shape[-2], image_shape[-1])
    spokelength = image_shape[-1] * 2
    grid_size = (2*image_shape[-2],2*image_shape[-1])
    
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes), dtype=np.float32)
    ky = np.zeros(shape=(spokelength, nspokes), dtype=np.float32)
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)
    
    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    if plot:
        # plot the first 40 spokes
        plt.plot(kx[:, :].transpose(), ky[:, :].transpose())
        plt.axis('equal')
        plt.title('k-space trajectory (first 40 spokes)')
        plt.show()
    
    
    # convert k-space trajectory to a tensor
    ktraj = torch.tensor(ktraj)#.cuda(gpu_id)
    if gpu_id is not -1:
         ktraj = ktraj.cuda(gpu_id)
    print('ktraj shape: {}'.format(ktraj.shape))
    return ktraj, im_size, grid_size

def backproject_radial(kdata, ktraj, image, grid_size, plot=False):
    adjnufft_ob = tkbn.KbNufftAdjoint(
        im_size=(image.shape[-2],image.shape[-1]), grid_size=(2*image.shape[-2],2*image.shape[-1])).to(image)
    #print(adjnufft_ob)
    
    # adjnufft back
    # method 1: no density compensation (blurry image)
    image_blurry = adjnufft_ob(kdata, ktraj)
    # method 2: use density compensation
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=image.shape)
    image_sharp = adjnufft_ob(kdata * dcomp, ktraj)   
    
    # show the images
    image_blurry_numpy = np.squeeze(image_blurry.cpu().numpy())
    
    image_sharp_numpy = np.squeeze(image_sharp.cpu().numpy())
    if plot:
        plt.figure(0)
        plt.imshow(np.absolute(image_blurry_numpy))
        plt.gray()
        plt.title('blurry image')
        
        plt.figure(1)
        plt.imshow(np.absolute(image_sharp_numpy))
        plt.gray()
        plt.title('sharp image (with Pipe dcomp)')
        
        plt.show()
    return image_blurry_numpy, image_sharp_numpy

def main():
    filterwarnings("ignore") # ignore floor divide warnings
    gpu_id=0
    # create a simple shepp logan phantom and plot it
    image = shepp_logan_phantom().astype(complex)
    #im_size = image.shape
    # plt.imshow(np.absolute(image))
    # plt.gray()
    # plt.title('Shepp-Logan Phantom')
    # plt.show()
    
    # convert the phantom to a tensor and unsqueeze coil and batch dimension
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    print('image shape: {}'.format(image.shape))
    
    nspokes = 20
    ktraj, grid_size = create_radial_mask(nspokes, image.shape, gpu_id, plot=True)
    kdata = project_radial(image, ktraj, grid_size, nspokes)
    image_blurry_numpy, image_sharp_numpy = backproject_radial(kdata, ktraj, image, grid_size, plot=False)

