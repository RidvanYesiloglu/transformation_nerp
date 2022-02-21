import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

class SigmBernNes(nn.Module):
    def __init__(self, inps_dict, device, dtype):
        super(SigmBernNes, self).__init__()
        args = inps_dict['args']
        self.sg_slope = args.sgSlp
        self.device = device
        self.dtype = dtype
        if args.ini == 'cns':
            y = torch.from_numpy(1-2*np.ones((args.K,args.N))*np.random.randint(2, size=(args.K,1))).to(device)
            x = torch.randn((args.K, args.N), device=device, dtype=dtype)
            self.w = nn.Parameter((y==1)*((x<0)*(-x)+(x>=0)*x) + (y==-1)*((x<0)*x+(x>=0)*(-x)), requires_grad=True)
        elif args.ini == 'rnd':
            self.w = nn.Parameter(torch.randn((args.K, args.N), device=device, dtype=dtype), requires_grad=True)
        elif (args.ini == np.asarray(['gold','weil','gw_c'])).sum() > 0:
            raise ValueError('Gold weil and gw_c are not implemented yet.')
            #self.w = nn.Parameter(torch.from_numpy(np.load('{}_param_for_{}_{}.npy')).to(device),  requires_grad=True)
        elif (args.ini == np.asarray(['sp1','sp2'])).sum() > 0:
            if args.N == 8:
                y = torch.from_numpy(1-2*np.ones((args.K,args.N))*np.random.randint(2, size=(args.K,1))).to(device)
                x = torch.randn((args.K, args.N), device=device, dtype=dtype)
                self.w = nn.Parameter((y==1)*((x<0)*(-x)+(x>=0)*x) + (y==-1)*((x<0)*x+(x>=0)*(-x)), requires_grad=True)
            else:
                #nover2folder=os.path.join(Path(inps_dict['save_folder']).parent, inps_dict['repr_str'])
                init_param = np.load(os.path.join(inps_dict['nover2folder'],'init_param_for_2N_{}.npy'.format(args.ini)))
                self.w = nn.Parameter(torch.from_numpy(init_param).to(device), requires_grad=True)
        else:
            assert 1==2
        self.sigmoid_layer = nn.Sigmoid()
    # forward propagate input
    def forward(self):
        self.thetas = self.sigmoid_layer(self.sg_slope*self.w)
        return self.thetas
    
    #return (is_converged, number of not yet convergeds)
    def is_converged(self):
        theta_ts = np.asarray(self().tolist())
        return np.logical_or(theta_ts<=0.02,theta_ts>=0.98).all(), (1-np.logical_or(theta_ts<=0.02,theta_ts>=0.98)).sum()