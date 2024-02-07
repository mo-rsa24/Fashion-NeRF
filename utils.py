import os
import yaml
import json
import imageio
import numpy as np
import random
import torch
from NeRF.Vanilla_NeRF.load_blender import pose_spherical
import cv2
# --cuda False --name Rail_No_Occlusion -b 4 -j 2 --tocg_checkpoint checkpoints/Rail_RT_No_Occlusion_1/tocg_step_280000.pth
import argparse

""" WHEN YOU ARE CURRENTLY TRAINING
NOTE: Experiment tracking spreadsheet helps identify what each experiment aims to achieve and their hyperparameters
"""
run_number = 1
experiment_number = 1
experiment_run = "experiment_{experiment_number}/{run_number}"


""" WHEN YOU ARE LOADING CHECKPOINTS
NOTE: Experiment tracking spreadsheet helps identify what each experiment aims to achieve and their hyperparameters
"""
run_from_number = 1
experiment_from_number = 1
experiment_from_run = "experiment_{experiment_from_number}/{run_from_number}"

get_combination = lambda opt: "{opt.person}_.clothing}"



def set_seed(seed: int):
    """
    Set the seed for reproducibility in PyTorch.

    Parameters:
    seed (int): The seed number.
    """
    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If you are using CUDA (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python's `random` module
    random.seed(seed)

    # Set environment variables for further reproducibility (especially if using GPUs)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # or ':4096:8' for newer GPUs
    
def process_opt(opt):
    opt.rail_dir = opt.rail_dir.format(opt.dataset_name, opt.res, opt.datamode)
    opt.original_dir = opt.original_dir.format(opt.dataset_name, opt.res, opt.datamode)
    
    opt.this_viton_save_to_dir = opt.this_viton_save_to_dir.format(opt.VITON_Type, opt.VITON_Name, opt.dataset_name)
    opt.this_viton_load_from_dir = opt.this_viton_load_from_dir.format(opt.VITON_Type, opt.VITON_Name, opt.dataset_name)
    
    opt.transforms_dir = opt.transforms_dir.format(opt.dataset_name)
    opt.experiment_run = opt.experiment_run.format(opt.experiment_number, opt.run_number)
    opt.experiment_from_run = opt.experiment_from_run.format(opt.experiment_from_number, opt.run_from_number)
    opt.experiment_run_yaml = opt.experiment_run_yaml.format(opt.experiment_run, opt.this_viton_load_from_dir)
    with open(os.path.join(opt.opt_vton_yaml), 'r') as config_file:
        config = yaml.safe_load(config_file)
    opt_parser = argparse.Namespace(**config)
    if not os.path.exists(opt.experiment_run_yaml):
        os.makedirs(opt.experiment_run_yaml)
    with open(os.path.join(opt.experiment_run_yaml, f"{opt.experiment_run.replace('/','_')}_{opt.opt_vton_yaml.replace('yaml/','')}"), 'w') as outfile:
        yaml.dump(vars(opt_parser), outfile, default_flow_style=False)
        
    return opt
    
def get_opt():
    with open(os.path.join('FashionNeRF.yml'), 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    opt = argparse.Namespace(**config)

    return opt

def get_vton_opt(root_opt):
    yml_name =  f"yaml/{str(root_opt.VITON_Name).lower()}.yml"
    with open(yml_name, 'r') as config_file:
        config = yaml.safe_load(config_file)
    base_dict = vars(root_opt)
    if base_dict['node'] == 'merman':
        config['root_dir'] = base_dict['merman_root_dir']
        config['checkpoint_root_dir'] = base_dict['merman_checkpoint_root_dir']
        config['tensorboard_dir'] = base_dict['merman_tensorboard_dir']
        config['results_dir'] = base_dict['merman_results_dir']
    else:
        config['root_dir'] = base_dict['root_dir']
        config['checkpoint_root_dir'] = base_dict['checkpoint_root_dir']
        config['tensorboard_dir'] = base_dict['tensorboard_dir']
        config['results_dir'] = base_dict['results_dir']
    return config

def get_vton_sweep(root_opt):
    with open(root_opt.sweeps_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def parse_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Parse experiment and VITON related arguments.')

    # Add arguments with default values
    """_summary_
        Experiment & run numbers
    Returns:
        _type_: _description_
    """
    parser.add_argument('--debug', type=int,default=0, help='0=False, 1=True')
    parser.add_argument('--sweeps', type=int,default=0, help='0=False, 1=True')
    parser.add_argument('--seed', type=int,default=0, help='3 Seeds for Experiment 1 and Run 1')
    parser.add_argument('--run_wandb',type=int, default=0, help='0=False, 1=True') 
    parser.add_argument('--experiment_number', type=int, default=1, help='The experiment number.')
    parser.add_argument('--run_number', type=int, default=1, help='The run number.')
    parser.add_argument('--experiment_from_number', type=int, default=1, help='The starting experiment number.')
    parser.add_argument('--run_from_number', type=int, default=1, help='The starting run number.')
    
    parser.add_argument('--this_viton_save_to_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    parser.add_argument('--this_viton_load_from_dir',default='VITON/{}/{}/{}', help='Selection from directory.')
    
    
    parser.add_argument('--warp_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    parser.add_argument('--gen_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    parser.add_argument('--warp_load_from_model', type=str, default='Original', help='Which dataset are we selecting to load from.')
    parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
    
    parser.add_argument('--load_last_step', default=False, help='If true that then take the latest step in the current experiment')
    
    
    # parser.add_argument('--warp_load_from_model', type=str, default='Original', help='Which dataset are we selecting to load from.')
    # parser.add_argument('--gen_load_from_model', type=str, default='Original', help='Which dataset are we selecting to load from.')
    # Parser_Based e2e
    parser.add_argument('--warp_experiment_from_number',help='Warp_experiment_from_dir', type=int, default=1)
    parser.add_argument('--warp_run_from_number', help='Warp_experiment_from_dir',type=int, default=6)
    
    parser.add_argument('--gen_experiment_from_number',help='gen_experiment_from_dir', type=int, default=1)
    parser.add_argument('--gen_discriminator_experiment_from_number',help='gen_discriminator_experiment_from_dir', type=int, default=1)
    parser.add_argument('--gen_discriminator_run_from_number', help='gen_discriminator_experiment_from_dir', type=int, default=6)
    parser.add_argument('--gen_run_from_number', help='gen_experiment_from_dir', type=int, default=6)
    parser.add_argument('--gen_load_from_model', help='gen_experiment_from_dir', type=str, default="Original")
    parser.add_argument('--gen_discriminator_load_from_model', help='gen_discriminator_experiment_from_dir', type=str, default="Original")
    
    # Parser_Free warping
    parser.add_argument('--parser_based_warp_experiment_from_number', help='parser_based_warp_experiment_from_dir', type=int, default=1)
    parser.add_argument('--parser_based_warp_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    
    parser.add_argument('--parser_based_gen_experiment_from_number', help='parser_based_gen_experiment_from_dir',type=int, default=1)
    parser.add_argument('--parser_based_gen_run_from_number', help='parser_based_gen_experiment_from_dir',type=int, default=6)
    parser.add_argument('--parser_free_warp_load_from_model', help='parser_free_warp_load_from_model', type=str, default="Original")
    parser.add_argument('--parser_free_gen_load_from_model', help='parser_free_gen_load_from_model', type=str, default="Original")
    
    # Parser_Free e2e
    parser.add_argument('--parser_free_warp_experiment_from_number', help='parser_free_warp_experiment_from_dir',type=int, default=1)
    parser.add_argument('--parser_free_warp_run_from_number', help='parser_free_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--parser_free_gen_experiment_from_number', help='parser_free_gen_experiment_from_dir',type=int, default=6)
    parser.add_argument('--parser_free_gen_run_from_number', help='parser_free_gen_experiment_from_dir',type=int, default=6)
    parser.add_argument('--parser_gen_load_from_model', help='parser_free_gen_load_from_model', type=str, default="Original")
    
    
    parser.add_argument('--VITON_Type', default='Parser_Based', help='Type of VITON.')
    parser.add_argument('--wandb_name',default='wandb_name', help='Selection directory.') 
    parser.add_argument('--preprocessing_method',default='openpose', help='Select preprocessing method.')
    parser.add_argument('--VITON_Name', default='Ladi_VTON', help='Name of the VITON model.')
    parser.add_argument('--VITON_Model', default='', help='Name of the VITON model.')
    parser.add_argument('--gpu_ids', default='0', help='Comma-separated list of GPU ids.')
    parser.add_argument('--device', default=0, help='Which GPU we are training on')
    parser.add_argument('--person', default='alex', help='Person')
    parser.add_argument('--shuffle', default=True, help='Person')
    parser.add_argument('--clothing', default='gray_jacket', help='Person')
    parser.add_argument('--res', default='high_res', help='Resolution type.')
    parser.add_argument('--dataset_name', default='Original', help='Name of the dataset.')
    parser.add_argument('--low_res_dataset_name', default='VITON-Clean', help='Name of the low-resolution dataset.')
    parser.add_argument('--viton_workers', type=int, default=4, help='Number of workers for VITON.')
    parser.add_argument('--viton_batch_size', type=int, default=4, help='Batch size for VITON.')
    parser.add_argument('--fine_width', type=int, default=192, help='Fine Width')
    parser.add_argument('--fine_height', type=int, default=256, help='Fine Height')
    parser.add_argument('--saving',type=str, help='[person, cloth]',default='cloth')
    parser.add_argument('--datamode', default='train', help='Mode of data (train/test).')

    # General hyperparameters
    parser.add_argument('--save_count', type=int, default=100, help='Model save interval.')
    parser.add_argument('--tensorboard_count', type=int, default=100, help='TensorBoard logging interval.')
    parser.add_argument('--display_count', type=int, default=100, help='Display interval.')
    parser.add_argument('--load_step', type=int, default=0, help='Step count to load the model.')
    parser.add_argument('--save_period', type=int, default=0, help='Frequency Of Saving Checkpoints At The End Of Epochs.')
    parser.add_argument('--print_step', type=int, default=0, help='Frequency Of Print Training Results On Screen')
    parser.add_argument('--sample_step', type=int, default=0, help='Step count to print results.')
    parser.add_argument('--decay_step', type=int, default=100000, help='Learning rate decay step.')
    parser.add_argument('--niter', type=int, default=100000, help='Learning rate decay step.')
    parser.add_argument('--niter_decay', type=int, default=100000, help='Learning rate decay step.')
    parser.add_argument('--keep_step', type=int, default=300000, help='Step count to keep the model.')
    parser.add_argument('--val_count', type=int, default=100, help='Validation interval.')
    parser.add_argument("--lpips_count", type=int, default=1000)
    parser.add_argument('--no_test_visualize', type=bool, default=False, help='Flag not to visualize test results.')
    parser.add_argument('--validate', type=bool, default=False, help='Validation')
    parser.add_argument('--num_test_visualize', type=int, default=4, help='Number of test visualizations.')
    

    # ============================ ACGPN ============================ 
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
    parser.add_argument('--label_nc', type=int, default=20, help='# of input label channels')
    parser.add_argument(
    '--how_many', type=int, default=1000, help='how many test images to run')
    parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy',
                            help='the path for clustered results of encoded features')
    parser.add_argument('--use_encoded_image', action='store_true',
                            help='if specified, encode the real image to get the feature map')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--resize_or_crop', type=str, default='scale_width',help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--serial_batches', action='store_true',help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--no_flip', action='store_true',help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    # for displays
    parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
    parser.add_argument('--tf_log', action='store_true',help='if specified, use tensorboard logging. Requires tensorflow installed')

    # for generator
    parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--n_downsample_global', type=int,default=4, help='number of downsampling layers in netG')
    parser.add_argument('--n_blocks_global', type=int, default=4,help='number of residual blocks in the global generator network')
    parser.add_argument('--n_blocks_local', type=int, default=3,help='number of residual blocks in the local enhancer network')
    parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
    parser.add_argument('--niter_fix_global', type=int, default=0,help='number of epochs that we only train the outmost local enhancer')
    parser.add_argument('--continue_train', action='store_true',help='continue training: load the latest model')
    
    # for training
    parser.add_argument('--load_pretrain', type=str, default='./checkpoints/label2city',help='load the pretrained model from the specified location')
    parser.add_argument('--which_epoch', type=str, default='latest',help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

    # for discriminators
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true',help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true',help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--no_lsgan', action='store_true',help='do *not* use least square GAN, if false, use vanilla GAN')
    parser.add_argument('--pool_size', type=int, default=0,help='the size of image buffer that stores previously generated images')
    parser.add_argument('--norm', type=str, default='instance',help='instance normalization or batch normalization')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
    parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
    parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
    # ----------- G -----------  
    parser.add_argument('--g_experiment_from_number', help='g_experiment_from_number', type=int, default=1)
    parser.add_argument('--g_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--g_load_from_model', help='g_load_from_model', type=str, default="Original")
    parser.add_argument('--g_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    # ----------- G1 ----------- 
    parser.add_argument('--g1_experiment_from_number', help='g1_experiment_from_number', type=int, default=1)
    parser.add_argument('--g1_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--g1_load_from_model', help='g1_load_from_model', type=str, default="Original")
    parser.add_argument('--g1_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    # ----------- G2 ----------- 
    parser.add_argument('--g2_experiment_from_number', help='g2_experiment_from_number', type=int, default=1)
    parser.add_argument('--g2_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--g2_load_from_model', help='g2_load_from_model', type=str, default="Original")
    parser.add_argument('--g2_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    # ----------- d ----------- 
    parser.add_argument('--d_experiment_from_number', help='d_experiment_from_number', type=int, default=1)
    parser.add_argument('--d_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--d_load_from_model', help='d_load_from_model', type=str, default="Original")
    parser.add_argument('--d_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    # ----------- D1 ----------- 
    parser.add_argument('--d1_experiment_from_number', help='d1_experiment_from_number', type=int, default=1)
    parser.add_argument('--d1_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--d1_load_from_model', help='d1_load_from_model', type=str, default="Original")
    parser.add_argument('--d1_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    # ----------- D2 ----------- 
    parser.add_argument('--d2_experiment_from_number', help='d2_experiment_from_number', type=int, default=1)
    parser.add_argument('--d2_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--d2_load_from_model', help='d2_load_from_model', type=str, default="Original")
    parser.add_argument('--d2_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    # ----------- D1 ----------- 
    parser.add_argument('--d3_experiment_from_number', help='d3_experiment_from_number', type=int, default=1)
    parser.add_argument('--d3_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--d3_load_from_model', help='d3_load_from_model', type=str, default="Original")
    parser.add_argument('--d3_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    
    # ----------- Unet ----------- 
    parser.add_argument('--unet_experiment_from_number', help='unet_experiment_from_number', type=int, default=1)
    parser.add_argument('--unet_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--unet_load_from_model', help='unet_load_from_model', type=str, default="Original")
    parser.add_argument('--unet_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    # ----------- VGG ----------- 
    parser.add_argument('--vgg_experiment_from_number', help='vgg_experiment_from_number', type=int, default=1)
    parser.add_argument('--vgg_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--vgg_load_from_model', help='vgg_load_from_model', type=str, default="Original")
    parser.add_argument('--vgg_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    
    # ====================================================================================  CP_VTON ====================================================================================
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument("--stage", default = "GMM")

    
    # ----------- GMM ----------- 
    parser.add_argument('--gmm_experiment_from_number', help='gmm_experiment_from_number', type=int, default=1)
    parser.add_argument('--gmm_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--gmm_load_from_model', help='gmm_load_from_model', type=str, default="Original")
    parser.add_argument('--gmm_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    # ----------- TOM ----------- 
    parser.add_argument('--tom_experiment_from_number', help='tom_experiment_from_number', type=int, default=1)
    parser.add_argument('--tom_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--tom_load_from_model', help='tom_load_from_model', type=str, default="Original")
    parser.add_argument('--tom_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    
    # ====================================================================================  HR_VITON ====================================================================================    
    parser.add_argument("--semantic_nc", type=int, default=13)
    # ----------- tocg ----------- 
    parser.add_argument('--tocg_experiment_from_number', help='tocg_experiment_from_number', type=int, default=1)
    parser.add_argument('--tocg_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--tocg_load_from_model', help='tocg_load_from_model', type=str, default="Original")
    parser.add_argument('--tocg_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    # ----------- TOM ----------- 
    parser.add_argument('--tocg_discriminator_experiment_from_number', help='tocg_discriminator_experiment_from_number', type=int, default=1)
    parser.add_argument('--tocg_discriminator_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--tocg_discriminator_load_from_model', help='tocg_discriminator_load_from_model', type=str, default="Original")
    parser.add_argument('--tocg_discriminator_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")  
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    # Cuda availability
    # training
    parser.add_argument("--G_D_seperate", action='store_true')
    parser.add_argument("--no_GAN_loss", action='store_true')
    parser.add_argument("--lasttvonly", action='store_true')
    parser.add_argument("--interflowloss", action='store_true', help="Intermediate flow loss")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    parser.add_argument('--edgeawaretv', type=str, choices=['no_edge', 'last_only', 'weighted'], default="no_edge", help="Edge aware TV loss")
    parser.add_argument('--add_lasttv', action='store_true')
    parser.add_argument('--tvlambda_tvob', type=float, default=2)
    parser.add_argument('--tvlambda_taco', type=float, default=2)
    
    parser.add_argument('--G_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='Discriminator initial learning rate for adam')
    parser.add_argument('--CElamda', type=float, default=10, help='initial learning rate for adam')
    parser.add_argument('--GANlambda', type=float, default=1)
    parser.add_argument('--tvlambda', type=float, default=2)
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")
    parser.add_argument('--cond_G_ngf', type=int, default=96)
    parser.add_argument('--cond_G_num_layers', type=int, default=5)
    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument('--fp16',type=bool, default=False, help='use amp')
    parser.add_argument('--composition_mask',type=bool, default=True)
    parser.add_argument("--test_data_list", default="test_pairs.txt")
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                    help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                            'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
 # SEAN-related hyper-parameters
    parser.add_argument('--GMM_const', type=float, default=None, help='constraint for GMM module')
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    
    # D
    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    
    # Training
    parser.add_argument('--GT', action='store_true')
    # ====================================================================================  LADI_VITON ====================================================================================    
    parser.add_argument('--const_weight', type=float, default=0.01, help='weight for the TPS constraint loss')
    parser.add_argument("--dataset", default='vitonhd', type=str,  choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument('--dense', dest='dense', default=False, action='store_true', help='use dense uv map')
    parser.add_argument("--only_extraction", default=False, action='store_true',
                        help="only extract the images using the trained networks without training")
    parser.add_argument('--vgg_weight', type=float, default=0.25, help='weight for the VGG loss (refinement network)')
    parser.add_argument('--l1_weight', type=float, default=1, help='weight for the L1 loss (refinement network)')
    parser.add_argument('--save_path', type=str, help='path to save the warped cloth images (if not provided, '
                                                      'the images will be saved in the data folder)')
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--clip_warping", type=bool, default=False)
    parser.add_argument('--tps_experiment_from_number', help='tps_experiment_from_number', type=int, default=1)
    parser.add_argument('--tps_run_from_number', help='parser_based_warp_experiment_from_dir',type=int, default=6)
    parser.add_argument('--tps_load_from_model', help='tom_load_from_model', type=str, default="Original")
    parser.add_argument('--tps_experiment_from_dir',default='VITON/{}/{}/{}', help='Selection directory.') 
    parser.add_argument('--epochs_tps', type=int, default=50, help='number of epochs to train the TPS network')
    parser.add_argument('--epochs_refinement', type=int, default=50,
                        help='number of epochs to train the refinement network')
    # ====================================================================================  EMASC ====================================================================================    
    parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default="stabilityai/stable-diffusion-2-inpainting",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help=" Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=40001,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--use_png", default=False, action="store_true", help="Whether to use png or jpg for saving")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="Guidance scale")
    parser.add_argument("--compute_metrics", default=False, action="store_true",
                        help="Compute metrics after generation")
    parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--category", default="upper_body", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'])
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ', `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader number of workers.")
    parser.add_argument("--num_workers_test", type=int, default=8, help="Test DataLoader number of workers.")
    parser.add_argument("--test_order", type=str, default="paired", choices=["unpaired", "paired"],
                        help="Whether to use paired or unpaired test data.")
    parser.add_argument("--emasc_type", type=str, default='nonlinear', choices=["linear", "nonlinear"],
                        help="Whether to use linear or nonlinear EMASC.")
    parser.add_argument("--emasc_kernel", type=int, default=3, help="EMASC kernel size.")
    parser.add_argument("--emasc_padding", type=int, default=1, help="EMASC padding size.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # args = parser.parse_args()
    
    return args

def override_arguments(base_args, override_args):
    """
    Overrides the values in base_args with the values from override_args if they exist.
    root_opt:param base_args: The base arguments, usually from a default config or loaded config file.
    python_arg:param override_args: The arguments to override the base_args with, usually from command line.
    :return: A namespace with the overridden arguments.
    """
    # Convert both namespaces to dictionaries for easier manipulation
    base_dict = vars(base_args)
    override_dict = vars(override_args)

    # Override the base arguments with the ones provided by override arguments
    if base_dict['node'] == 'merman':
        base_dict['root_dir'] = base_dict['merman_root_dir']
        base_dict['checkpoint_root_dir'] = base_dict['merman_checkpoint_root_dir']
        base_dict['tensorboard_dir'] = base_dict['merman_tensorboard_dir']
        base_dict['results_dir'] = base_dict['merman_results_dir']
    for key, value in override_dict.items():
        # We only override if the key exists in the base_args to avoid adding unknown parameters
        if key in base_dict and value is not None:
            base_dict[key] = value
    base_dict['opt_vton_yaml'] = f"yaml/{override_dict['VITON_Name'].lower()}.yml"
    base_dict['sweeps_yaml'] = f"sweeps/{override_dict['VITON_Name'].lower()}.yml"
    
    # Convert the updated dictionary back to a Namespace
    return argparse.Namespace(**base_dict)

# Example usage
if __name__ == "__main__":
    args = parse_arguments()
    # You can now use args to access the arguments
    print(args)


def get_transforms_data(data_path,person_clothing):
    transform_string = f"{data_path}/{person_clothing}.json"
    if os.path.isfile(transform_string):
        with open(transform_string, 'r') as f:
            transform_data = json.load(f)
    else:
        transform_data = None
    return transform_data

def get_transform_matrix(transform_data, image_name):
    transform_matrix = None
    for frame in transform_data['frames']:
        file_string = frame['file_path'].split("/")[-1]
        if image_name == file_string:
            transform_matrix = frame['transform_matrix']
            break
    return transform_matrix

def load_nerf_data(dataset):
    imgs = []
    poses = []
    for data in dataset:
        imgs.append(imageio.imread(data['im_name']))
        poses.append(np.array(data['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(dataset['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    return imgs, poses, render_poses, [H, W, focal], None

similarity = lambda n1, n2: 1 - abs(n1 - n2) / (n1 + n2)
labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

def get_half_res(imgs, focal, H, W):
    H = H//2
    W = W//2
    focal = focal/2.
    depth = imgs[0].shape[-1]
    imgs_half_res = np.zeros((imgs.shape[0], H, W, depth))
    for i, img in enumerate(imgs):
        imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    imgs = imgs_half_res
    return imgs



""" ================== NeRF imports =================="""
# from NeRF.Vanilla_NeRF.test_helper import render, create_nerf, to8b
# from NeRF.Vanilla_NeRF.run_nerf import train