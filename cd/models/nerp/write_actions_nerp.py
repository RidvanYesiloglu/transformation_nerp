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

    #########################
    # Setup input encoder:
    encoder = Positional_Encoder(args)
    # Setup model
    model = SIREN(args.net)
    model.cuda(args.gpu_id)
    model.train()
    # Setup optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
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
    ########################
    args.net['network_output_size']=1
    # Setup input encoder:
    encoder_Pus = Positional_Encoder(args)
    # Setup model
    model_Pus = SIREN(args.net)
    model_Pus.cuda(args.gpu_id)
    # Load pretrain model
    model_path = args.Pus_model_path#.format(config['data'], config['img_size'], \
                    #config['model'], config['net']['network_width'], config['net']['network_depth'], \
                    #config['encoder']['scale'])
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
    model_Pus.load_state_dict(state_dict['net'])
    encoder_Pus.B = state_dict['enc'].cuda(args.gpu_id)
    model_Pus = model_Pus.cuda(args.gpu_id)
    #optim.load_state_dict(state_dict['opt'])
    print('Load pretrain model: {}'.format(model_path))
    for param in model_Pus.parameters():
        param.requires_grad = False
    ########################
    # Setup input encoder:
    encoder_Ius = Positional_Encoder(args)
    # Setup model
    model_Ius = SIREN(args.net)
    model_Ius.cuda(args.gpu_id)
    # Load pretrain model
    model_path = args.Ius_model_path#.format(config['data'], config['img_size'], \
                    #config['model'], config['net']['network_width'], config['net']['network_depth'], \
                    #config['encoder']['scale'])
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
    model_Ius.load_state_dict(state_dict['net'])
    encoder_Ius.B = state_dict['enc'].cuda(args.gpu_id)
    model_Ius = model_Ius.cuda(args.gpu_id)
    #optim.load_state_dict(state_dict['opt'])
    print('Load pretrain model: {}'.format(model_path))
    for param in model_Ius.parameters():
        param.requires_grad = False
    args.net['network_output_size']=3
    # Setup loss function
    mse_loss_fn = torch.nn.MSELoss()
    
    def spec_loss_fn(pred_spec, gt_spec):
        '''
        spec: [B, H, W, C]
        '''
        loss = torch.mean(torch.abs(pred_spec - gt_spec))
        return loss
    
    
    # Setup data loader
    print('Load image: {}'.format(args.img_path))
    #data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])
    
    
    args.img_size = (args.img_size, args.img_size, args.img_size) if type(args.img_size) == int else tuple(args.img_size)
    slice_idx = list(range(0, args.img_size[0], int(args.img_size[0]/args.display_image_num)))
    
    
    grid_np = np.asarray([(x,y,z) for x in range(128) for y in range(128) for z in range(64)]).reshape((1,128,128,64,3))
    grid_np = (grid_np/np.array([128,128,64])) + np.array([1/256.0,1/256.0,1/128.0])
    grid = torch.from_numpy(grid_np.astype('float32')).cuda(args.gpu_id)
    print('Grid element size: {} and neleements: {}, size in megabytes: {}'.format(grid.element_size(), grid.nelement(), (grid.element_size()*grid.nelement())/1000000.0))

        
    # save_image_3d(test_data[1], slice_idx, os.path.join(image_directory, "test.png"))
    # save_image_3d(complex2real(train_data[1]), slice_idx, os.path.join(image_directory, "train.png"))
    # save_image_3d(complex2real(spectrum), slice_idx, os.path.join(image_directory, "spec.png"))
    print('**before emb**')
    check_gpu(args.gpu_id)
    #train_embedding = encoder.embedding(grid)  # [B, C, H, W, embedding*2]
    #test_embedding = encoder.embedding(grid)
    print('**before emb 2**')
    check_gpu(args.gpu_id)
    #train_embedding_Pus = encoder_Pus.embedding(grid)  # [B, C, H, W, embedding*2]
    #test_embedding_Pus = encoder_Pus.embedding(grid)
    print('**before emb3**')
    check_gpu(args.gpu_id)
    train_embedding_Ius = encoder_Ius.embedding(grid)  # [B, C, H, W, embedding*2]
    # test_embedding_Ius = encoder_Ius.embedding(grid)
    print('**after all emb**')
    check_gpu(args.gpu_id)
    
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
    preruni_dict={'model':model, 'model_Pus':model_Pus, 'model_Ius':model_Ius, 'grid':grid, \
                  'train_embedding_Ius':train_embedding_Ius,'encoder_Pus':encoder_Pus,'spec_loss_fn':spec_loss_fn,\
                  'encoder':encoder, 'mse_loss_fn':mse_loss_fn, 'slice_idx':slice_idx, 'image_directory':image_directory, \
                      'checkpoint_directory':checkpoint_directory, 'optim':optim}
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
    
