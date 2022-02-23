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
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    working_dir = '/home/yesiloglu/projects/3d_tracking_nerp'
    if args is None: #if slurm:
        args = parameters.get_arguments(params_dict)
    repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos], wantShort=True, params_dict=params_dict)
    print('**Start up**')
    check_gpu(args.gpu_id)
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=7)
    print_freq = 2500 # print thetas once in "print_frequency" epochs 
    write_freq = 500 # print thetas once in "print_frequency" epochs
    cudnn.benchmark = True
    
    save_folder = os.path.join(working_dir, 'detailed_results', 'vol_'+str(im_ind), repr_str)
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except:
            raise
                
    runset_folder = os.path.join(working_dir, 'runset_summaries', 'vol_'+str(im_ind), args.runsetName)
    if not os.path.exists(os.path.join(runset_folder, 'ind_runs')):
        try:
            os.makedirs(os.path.join(runset_folder, 'ind_runs'))
        except: 
            raise
            
    if Path(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo))).is_file():
        ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "r+")
        prev_sit_log = ind_run_sit_log.read()
        ind_run_sit_log.close()
    else:
        prev_sit_log = ""
    ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
    ind_run_sit_log.write(prev_sit_log+'Ind Run: {}, Conf: {}, Situ: Just started'.format(args.indRunNo, repr_str))
    ind_run_sit_log.close()
    update_runset_summary(args, runset_folder)
    preallruns_dict = wr_acts.preallruns_actions({'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'device':device})
    for run_number in range(args.noRuns):
        torch.cuda.empty_cache()
        sum_log_base = prev_sit_log+'\nInd Run: {}, Conf: {}, Situ: '.format(args.indRunNo, repr_str)
        for prev_run_ind in range(0,run_number):
            sum_log_base += 'Tr {} Res: {:.8f}, '.format(prev_run_ind, preallruns_dict['final_psnrs'][prev_run_ind])
        last_log = 'Tr. {} starting'.format(run_number)
        ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
        ind_run_sit_log.write(sum_log_base + last_log)
        ind_run_sit_log.close()
        update_runset_summary(args, runset_folder)
        
        preruni_dict = wr_acts.prerun_i_actions({'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'device':device}, preallruns_dict)
        print('**Just after preruni_actions**')
        check_gpu(args.gpu_id)
        psnrs_r = [] 
        losses_r = []
        print('before init psnr calc')
        check_gpu(args.gpu_id)
        start_time = time.time()
        
        with torch.no_grad():
            deformed_grid = preruni_dict['grid'] + (preruni_dict['model'](preruni_dict['train_embedding']))  # [B, C, H, W, 1]
            deformed_prior = preruni_dict['model_Pus'](preruni_dict['encoder_Pus'].embedding(deformed_grid))
            test_loss = preruni_dict['mse_loss_fn'](deformed_prior, preruni_dict['model_Ius'](preruni_dict['train_embedding_Ius']))
            test_psnr = - 10 * torch.log10(test_loss).item()
            print('STARTING MODEL PSNR: {:.5f}'.format(test_psnr))
            
            test_loss = test_loss.item()
        torch.cuda.empty_cache()
        print('after init psnr calc')
        check_gpu(args.gpu_id)
        for t in tqdm(range(args.max_iter)):
            print('T=',t)
            preruni_dict['model'].train()
            preruni_dict['optim'].zero_grad()
            # print('LATEST')
            # check_gpu(args.gpu_id)
            deformed_grid = preruni_dict['grid'] + (preruni_dict['model'](preruni_dict['train_embedding']))  # [B, C, H, W, 1]
            deformed_prior = preruni_dict['model_Pus'](preruni_dict['encoder_Pus'].embedding(deformed_grid))
            train_loss = preruni_dict['mse_loss_fn'](deformed_prior, preruni_dict['model_Ius'](preruni_dict['train_embedding_Ius']))
                
            train_loss.backward()
            print('after i backward')
            check_gpu(args.gpu_id)
            preruni_dict['optim'].step()
            # Add loss to the losses list for r
            losses_r.append(train_loss.item())
            
            with torch.no_grad():
                deformed_grid = preruni_dict['grid'] + (preruni_dict['model'](preruni_dict['train_embedding']))  # [B, C, H, W, 1]
                deformed_prior = preruni_dict['model_Pus'](preruni_dict['encoder_Pus'].embedding(deformed_grid))
                test_loss = preruni_dict['mse_loss_fn'](deformed_prior, preruni_dict['model_Ius'](preruni_dict['train_embedding_Ius']))
                test_psnr = - 10 * torch.log10(test_loss).item()
                #test_psnr2 = PSNR(test_output, preruni_dict['test_data'][1]).item()
                #print('Test psnr: {:.5f}, test psnr2: {:.5f}, equal: {}'.format(test_psnr, test_psnr2, test_psnr==test_psnr2))
                #test_loss = test_loss.item()
            psnrs_r.append(test_psnr)
            if test_psnr == max(psnrs_r):
                # Save the test output and the model:
                for filename in glob.glob(os.path.join(save_folder, 'savedmodel_run{}*'.format(run_number))):
                    os.remove(filename)
                for filename in glob.glob(os.path.join(save_folder, 'savedrec_run{}*'.format(run_number))):
                    os.remove(filename)
                #np.save(os.path.join(save_folder,'savedrec_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), test_output.detach().cpu().numpy())
                model_name = os.path.join(save_folder, 'savedmodel_run{}_ep{}_{:.4g}dB.pt'.format(run_number, t+1, test_psnr))
                torch.save({'net': preruni_dict['model'].state_dict(), \
                        'enc': preruni_dict['encoder'].B, \
                        'opt': preruni_dict['optim'].state_dict()}, \
                        model_name)
                
            # Print
            if (t+1) % print_freq == 0:
                wr_acts.print_freq_actions({'args':args, 't':t, 'losses_r':losses_r})
            if (t+1) % write_freq == 0:
                last_log = 'Tr {} Ep {} Tr Ls: {:.8f}'.format(run_number, t+1, losses_r[-1])
                ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
                ind_run_sit_log.write(sum_log_base + last_log)
                ind_run_sit_log.close()
                update_runset_summary(args, runset_folder)
                
                write_freq_dict = wr_acts.write_freq_actions({'args':args, 't':t, 'start_time':start_time, 'psnrs_r':psnrs_r, \
                    'save_folder': save_folder, 'run_number':run_number,'losses_r':losses_r}, preruni_dict)
                start_time = write_freq_dict['start_time']
                # Save final model
            if (t + 1) % args.image_save_iter == 0:
                model_name = os.path.join(preruni_dict['checkpoint_directory'], 'model_%06d.pt' % (t + 1))
                torch.save({'net': preruni_dict['model'].state_dict(), \
                            'enc': preruni_dict['encoder'].B, \
                            'opt': preruni_dict['optim'].state_dict(), \
                            }, model_name)
        wr_acts.postrun_i_actions({'args':args, 'run_number':run_number, 'save_folder':save_folder, 'losses_r':losses_r, 'psnrs_r':psnrs_r,'device':device, \
                                  't':t},preallruns_dict, preruni_dict)
        last_log = 'Tr {} Res: {:.8f}'.format(run_number, preallruns_dict['final_psnrs'][run_number])
        ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
        ind_run_sit_log.write(sum_log_base + last_log)
        ind_run_sit_log.close()
        update_runset_summary(args, runset_folder)
    wr_acts.postallruns_actions({'args':args, 'save_folder':save_folder}, preallruns_dict)
    mean_log = ' Mean Res: {:.8f}'.format(preallruns_dict['final_psnrs'].mean())
    ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
    ind_run_sit_log.write(sum_log_base + last_log + mean_log)
    ind_run_sit_log.close()
    update_runset_summary(args, runset_folder)
if __name__ == "__main__":
    main() 
