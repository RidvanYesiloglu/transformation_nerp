# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:09:39 2022

@author: ridva
"""
from runset_train import train
def main(args=None):
    for i in range(1739,1760): #(1739,1760) (1760,1781), (1781,1802)
        args.img_path = '../data/patient19/volume_{}.npy'.format(i)
        train.main(args, i)
    
if __name__ == "__main__":
    main() 
    
    