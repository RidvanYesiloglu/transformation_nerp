import os
import torch
import numpy as np

import time

from networks import Positional_Encoder, FFN, SIREN
from utils import prepare_sub_folder, mri_fourier_transform_3d, complex2real, random_sample_uniform_mask, random_sample_gaussian_mask, save_image_3d, PSNR, check_gpu

from torchnufftexample import create_radial_mask, project_radial, backproject_radial

# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'device':device}
# outs_dict: {'mean_f_zk':mean_f_zk, 'log_mean_f_zk':log_mean_f_zk, 'final_f_zk_vals':final_f_zk_vals, 'final_thetas':final_thetas}
def preallruns_actions(inps_dict):
    args = inps_dict['args']
    mean_psnr = 0
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write('Runset Name: {}, Individual Run No: {}\n'.format(args.runsetName, args.indRunNo))
    main_logs.write('Configuration: {}\n'.format(inps_dict['repr_str']))
    if torch.cuda.is_available():
        main_logs.write('GPU Total Memory [GB]: {}\n'.format(torch.cuda.get_device_properties(0).total_memory/1e9))
    else:
        main_logs.write('Using CPU.\n')
    main_logs.close()
        
    final_psnrs = np.zeros((args.noRuns))
    np.save(os.path.join(inps_dict['save_folder'],'final_psnrs'),final_psnrs)

    preallruns_dict = {'mean_psnr':mean_psnr, 'final_psnrs':final_psnrs}
    return preallruns_dict


# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'model': model}
# outs_dict: 
def prerun_i_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    device = inps_dict['device']
    
    
    # Setup output folder
    #output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    if args.pretrain: 
        output_subfolder = args.data + '_pretrain'
    else:
        output_subfolder = args.data
    model_name = os.path.join(inps_dict['save_folder'], output_subfolder + '/img{}_af{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
        .format(args.img_size, args.sampler['af'], \
            args.model, args.net['network_input_size'], args.net['network_width'], \
            args.net['network_depth'], args.lr, args.encoder['embedding']))
    if not(args.encoder['embedding'] == 'none'):
        model_name += '_scale{}_size{}'.format(args.encoder['scale'], args.encoder['embedding_size'])
    print(model_name)
    
    # train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_path = '/home/yesiloglu/projects/3d_tracking_nerp/output_path'
    output_directory = os.path.join(output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    #shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


    # Setup input encoder:
    encoder = Positional_Encoder(args)
    print('Encoder B[65:68,:]: ', encoder.B[65:68,:])
    
    # Setup model
    if args.model == 'SIREN':
        model = SIREN(args.net)
    elif args.model == 'FFN':
        model = FFN(args.net).cuda(args.gpu_id)
    else:
        raise NotImplementedError
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print('Model param: {} megabytes, buffers: {} megabytes, total: {} megabytes'.format(mem_params/1000000.0,mem_bufs/1000000.0, mem/1000000.0))
    
    model.cuda(args.gpu_id)
    model.train()
    # Setup optimizer
    if args.optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    else:
        NotImplementedError
    # print('GPU 3 before model load:')
    # check_gpu(3)
    # Load pretrain model
    if args.pretrain:
        model_path = args.pretrain_model_path#.format(config['data'], config['img_size'], \
                        #config['model'], config['net']['network_width'], config['net']['network_depth'], \
                        #config['encoder']['scale'])
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        model.load_state_dict(state_dict['net'])
        encoder.B = state_dict['enc'].cuda(args.gpu_id)
        model = model.cuda(args.gpu_id)
        #optim.load_state_dict(state_dict['opt'])
        print('Load pretrain model: {}'.format(model_path))
    # print('GPU 3 aftr model load:')
    # check_gpu(3)
    
    
    # Setup loss function
    mse_loss_fn = torch.nn.MSELoss()
    
    def spec_loss_fn(pred_spec, gt_spec):
        '''
        spec: [B, H, W, C]
        '''
        loss = torch.mean(torch.abs(pred_spec - gt_spec))
        return loss
    
    
    # Setup data loader
    #print('Load image: {}'.format(args.img_path))
    img_path = '../../data73/ims_tog.npy' if args.togOrSepNorm==1 else '../../data73/ims_sep.npy'
    #data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])
    
    
    args.img_size = (args.img_size, args.img_size, args.img_size) if type(args.img_size) == int else tuple(args.img_size)
    slice_idx = list(range(0, args.img_size[0], int(args.img_size[0]/args.display_image_num)))
    
    
    # Input coordinates (x, y, z) grid and target image
    #grid = grid.cuda()  # [bs, c, h, w, 3], [0, 1]
    #image = image.cuda()  # [bs, c, h, w, 1], [0, 1]
    
    image = torch.from_numpy(np.expand_dims(np.load(args.img_path)[args.im_ind],(0,-1))).cuda(args.gpu_id)
    grid_np = np.asarray([(x,y,z) for x in range(128) for y in range(128) for z in range(64)]).reshape((1,128,128,64,3))
    grid_np = (grid_np/np.array([128,128,64])) + np.array([1/256.0,1/256.0,1/128.0])
    grid = torch.from_numpy(grid_np.astype('float32')).cuda(args.gpu_id)
    print('Image min: {} and max: {}'.format(image.min(), image.max()))
    print('Image and grid created. Image shape: {}, grid shape: {}'.format(image.shape, grid.shape))
    print('Image element size: {} and neleements: {}, size in megabytes: {}'.format(image.element_size(), image.nelement(), (image.element_size()*image.nelement())/1000000.0))
    print('Grid element size: {} and neleements: {}, size in megabytes: {}'.format(grid.element_size(), grid.nelement(), (grid.element_size()*grid.nelement())/1000000.0))
    # Randomly sample mask for each image
    #mask = random_sample_gaussian_mask(args.img_size, 1.0/args.sampler['af'])#.cuda()
    #mask = torch.from_numpy(np.fft.fftshift(mask, (0,1,2))).cuda(args.gpu_id)
    #spectrum = mri_fourier_transform_3d(image)  # 3D FFT, [bs, c, h, w, 1]
    
    #np.save(os.path.join(inps_dict['save_folder'], 'maskk'), mask.detach().cpu().numpy())
    #np.save(os.path.join(inps_dict['save_folder'], 'spectrumm'), spectrum.detach().cpu().numpy())
    #ds_spectrum = spectrum * mask[None, ..., None]  # downsample spectrum. [1, c, h, w, 1]
    ktraj, im_size, grid_size = create_radial_mask(args.nproj, (64,1,128,128), args.gpu_id, plot=False)
    kdata = project_radial(image, ktraj, im_size, grid_size)
    print('Ktraj shape ', ktraj.shape, 'kdata.shape ', kdata.shape)
    # print('Mask shape: {}, spectrum shape: {}, ds_spectrum shape: {}'.format(mask.shape, spectrum.shape, ds_spectrum.shape))
    # print('Mask eleement size: {} and neleements: {}, size in megabytes: {}'.format(mask.element_size(), mask.nelement(), (mask.element_size()*mask.nelement())/1000000.0))
    print('Image eleement size: {} and neleements: {}, size in megabytes: {}'.format(image.element_size(), image.nelement(), (image.element_size()*image.nelement())/1000000.0))
    # Data loading
    test_data = (grid, image)
    train_data = (grid, kdata)
    
    print('grid device')
    print(grid.get_device())
    
    # save_image_3d(test_data[1], slice_idx, os.path.join(image_directory, "test.png"))
    # save_image_3d(complex2real(train_data[1]), slice_idx, os.path.join(image_directory, "train.png"))
    # save_image_3d(complex2real(spectrum), slice_idx, os.path.join(image_directory, "spec.png"))
    
    train_embedding = encoder.embedding(train_data[0])  # [B, C, H, W, embedding*2]
    test_embedding = encoder.embedding(test_data[0])
    print('Train embedding shape: {}, test embedding shape: {}'.format(train_embedding.shape, test_embedding.shape))
    print('Train embedding eleement size: {} and neleements: {}, size in megabytes: {}'.format(train_embedding.element_size(), train_embedding.nelement(), (train_embedding.element_size()*train_embedding.nelement())/1000000.0))
    print('Test embedding eleement size: {} and neleements: {}, size in megabytes: {}'.format(test_embedding.element_size(), test_embedding.nelement(), (test_embedding.element_size()*test_embedding.nelement())/1000000.0))
    # print('VERIYORUM GAZI')
    # train_output = model(train_embedding)
    # print('VERDIM GAZI')
    init_thetas_str = "Run no: {}\n".format(inps_dict['run_number']) + '\n'
    init_psnr_str = 'Initial psnr: {:.4f} \n'.format(1)
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Runset Name: {}, Individual Run No: {}\n'.format(args.runsetName, args.indRunNo))
    r_logs.write('Configuration: {}\n'.format(inps_dict['repr_str']))
    r_logs.write(init_thetas_str)
    r_logs.write(init_psnr_str)
    r_logs.close()
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write(init_thetas_str)
    main_logs.write(init_psnr_str)
    main_logs.close()
    preruni_dict={'model':model, 'train_embedding':train_embedding, 'test_embedding':test_embedding, 'ktraj':ktraj, 'im_size':im_size, 'grid_size':grid_size, 'spec_loss_fn':spec_loss_fn, \
                  'train_data':train_data, 'mse_loss_fn':mse_loss_fn, 'test_data':test_data, 'slice_idx':slice_idx, 'image_directory':image_directory, \
                      'checkpoint_directory':checkpoint_directory, 'encoder':encoder, 'optim':optim}
    if args.pretrain:
        with torch.no_grad():
            test_output = preruni_dict['model'](preruni_dict['test_embedding'])
    
            test_loss = 0.5 * preruni_dict['mse_loss_fn'](test_output, preruni_dict['test_data'][1])
            test_psnr = - 10 * torch.log10(2 * test_loss).item()
            test_psnr2 = PSNR(test_output, preruni_dict['test_data'][1]).item()
            print('PRETRAIN MODEL PSNR:')
            print('Test psnr: {:.5f}, test psnr2: {:.5f}, equal: {}'.format(test_psnr, test_psnr2, test_psnr==test_psnr2))
            
            test_loss = test_loss.item()
        np.save(os.path.join(inps_dict['save_folder'], 'pretrainmodel_out'), test_output.detach().cpu().numpy())
    return preruni_dict

def print_freq_actions(inps_dict):
    args =inps_dict['args']
    # train_writer.add_scalar('train_loss', train_loss, iterations + 1)
    print("[Epoch: {}/{}] Train loss: {:.4g} ".format(inps_dict['t']+1, args.max_iter, inps_dict['losses_r'][-1]))
    
def write_freq_actions(inps_dict, preruni_dict):
    args = inps_dict['args']
    end_time = time.time()
    
    preruni_dict['model'].eval()
    with torch.no_grad():
        test_output = preruni_dict['model'](preruni_dict['test_embedding'])

        test_loss = 0.5 * preruni_dict['mse_loss_fn'](test_output, preruni_dict['test_data'][1])
        test_psnr = - 10 * torch.log10(2 * test_loss).item()
        test_psnr2 = PSNR(test_output, preruni_dict['test_data'][1]).item()
        print('Test psnr: {:.5f}, test psnr2: {:.5f}, equal: {}'.format(test_psnr, test_psnr2, test_psnr==test_psnr2))
        test_loss = test_loss.item()

    # train_writer.add_scalar('test_loss', test_loss, iterations + 1)
    # train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
    # Must transfer to .cpu() tensor firstly for saving images
    #save_image_3d(test_output, preruni_dict['slice_idx'], os.path.join(preruni_dict['image_directory'], "recon_{}_{:.4g}dB.png".format(inps_dict['t']+1, test_psnr)))
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Epoch: {}, Time: {}, Train Loss: {:.4f}\n'.format(inps_dict['t']+1,end_time-inps_dict['start_time'],inps_dict['losses_r'][-1]))
    to_write = "[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(inps_dict['t']+1, args.max_iter, test_loss, test_psnr)
    r_logs.write(to_write)
    start_time = time.time()
    r_logs.close()
    #plt_model.plot_change_of_objective(inps_dict['f_zks_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    #plt_model.plot_change_of_loss(inps_dict['losses_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    
    np.save(os.path.join(inps_dict['save_folder'],'psnrs_{}'.format(inps_dict['run_number'])), inps_dict['psnrs_r'])

    print(to_write)
    return {'start_time':start_time}


def postrun_i_actions(inps_dict, preallruns_dict, preruni_dict):
    args = inps_dict['args']

    preruni_dict['model'].eval()
    with torch.no_grad():
        test_output = preruni_dict['model'](preruni_dict['test_embedding'])

        test_loss = 0.5 * preruni_dict['mse_loss_fn'](test_output, preruni_dict['test_data'][1])
        test_psnr = - 10 * torch.log10(2 * test_loss).item()
        test_loss = test_loss.item()

    # train_writer.add_scalar('test_loss', test_loss, iterations + 1)
    # train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
    # Must transfer to .cpu() tensor firstly for saving images
    save_image_3d(test_output, preruni_dict['slice_idx'], os.path.join(preruni_dict['image_directory'], "recon_{}_{:.4g}dB.png".format(inps_dict['t']+1, test_psnr)))
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Epoch: {}, Train Loss: {:.4f}\n'.format(inps_dict['t']+1,inps_dict['losses_r'][-1]))
    to_write = "[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(inps_dict['t']+1, args.max_iter, test_loss, test_psnr)
    r_logs.write(to_write)
    r_logs.close()
    #plt_model.plot_change_of_objective(inps_dict['f_zks_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    #plt_model.plot_change_of_loss(inps_dict['losses_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    
    print('**************')
    print('FINAL RES.: ' + to_write)
    preallruns_dict['mean_psnr'] += test_psnr/args.noRuns
    print('******************************************')
    
    preallruns_dict['final_psnrs'][inps_dict['run_number']] = test_psnr
    np.save(os.path.join(inps_dict['save_folder'],'final_psnrs'), preallruns_dict['final_psnrs'])
    
    np.save(os.path.join(inps_dict['save_folder'],'psnrs_{}'.format(inps_dict['run_number'])), inps_dict['psnrs_r'])
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write(to_write)
    main_logs.close()
    
    print('Encoder B[65:68,:]: ',preruni_dict['encoder'].B[65:68,:])
    

def postallruns_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    save_folder = inps_dict['save_folder']
    final_psnrs = preallruns_dict['final_psnrs']
    print("All psnr values", final_psnrs)
    main_log = 'Mean psnr over all runs: {:.6f} \n'.format(preallruns_dict['mean_psnr'])
    print(main_log)
    print('********************************TRAINING ENDED**********************************************')
    main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
    main_logs.write(main_log)
    main_logs.close()
    
