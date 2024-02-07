import time
import argparse
import yaml
from VITON.Parser_Free.PF_AFN.PF_AFN_train.models.networks import ResUnetGenerator, VGGLoss, save_checkpoint, load_checkpoint_parallel, load_checkpoint_part_parallel
from VITON.Parser_Free.PF_AFN.PF_AFN_train.models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import datetime
import cv2

fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)  

def CreateDataset(opt, root_opt):
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, root_opt)
    return dataset

def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    
    # Parser Based
    root_opt.parser_based_warp_experiment_from_run = root_opt.parser_based_warp_experiment_from_run.format(root_opt.parser_based_warp_experiment_from_number, root_opt.parser_based_warp_run_from_number)
    root_opt.parser_based_gen_experiment_from_run = root_opt.parser_based_gen_experiment_from_run.format(root_opt.parser_based_gen_experiment_from_number, root_opt.parser_based_gen_run_from_number)
    
    # Parser Free
    root_opt.parser_free_warp_experiment_from_run = root_opt.parser_free_warp_experiment_from_run.format(root_opt.parser_free_warp_experiment_from_number, root_opt.parser_free_warp_run_from_number)
    root_opt.parser_free_gen_experiment_from_run = root_opt.parser_free_gen_experiment_from_run.format(root_opt.parser_free_gen_experiment_from_number, root_opt.parser_free_gen_run_from_number)
    return root_opt


def get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
        
    # This experiment
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.warp_load_from_model)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    # Parser Based e2e
    root_opt.parser_based_warp_experiment_from_dir = root_opt.parser_based_warp_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.warp_load_from_model)
    root_opt.parser_based_warp_experiment_from_dir = os.path.join(root_opt.parser_based_warp_experiment_from_dir, "PB_Warp")
    
    root_opt.parser_based_gen_experiment_from_dir = root_opt.parser_based_gen_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gen_load_from_model)
    root_opt.parser_based_gen_experiment_from_dir = os.path.join(root_opt.parser_based_gen_experiment_from_dir, "PB_Gen")
    
    
    # Parser Free Warp
    root_opt.parser_free_warp_experiment_from_dir = root_opt.parser_free_warp_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.parser_free_warp_load_from_model)
    root_opt.parser_free_warp_experiment_from_dir = os.path.join(root_opt.parser_free_warp_experiment_from_dir, "PF_Warp")
    
    root_opt.parser_free_gen_experiment_from_dir = root_opt.parser_free_gen_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.parser_free_gen_load_from_model)
    root_opt.parser_free_gen_experiment_from_dir = os.path.join(root_opt.parser_free_gen_experiment_from_dir, "PF_Gen")
    
    return root_opt

def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.dataset_name = root_opt.dataset_name
    parser.datamode = root_opt.datamode
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.seed = root_opt.seed
    parser.val_count = root_opt.val_count
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    return parser

def process_opt(opt, root_opt):
    parser = opt
    parser = argparse.Namespace(**parser)
    
    root_opt = get_root_experiment_runs(root_opt)
    root_opt = get_root_opt_experiment_dir(root_opt)
    parser = get_root_opt_checkpoint_dir(parser, root_opt)
    parser, root_opt = get_root_opt_results_dir(parser, root_opt)    
    parser = copy_root_opt_to_opt(parser, root_opt)
    return parser, root_opt


def get_root_opt_checkpoint_dir(parser, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    # Parser Based Warping
    parser.pb_warp_save_step_checkpoint_dir = parser.pb_warp_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_step_checkpoint = os.path.join(parser.pb_warp_save_step_checkpoint_dir, parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_step_checkpoint.split("/")[:-1])

    parser.pb_warp_load_step_checkpoint_dir = parser.pb_warp_load_step_checkpoint_dir.format(root_opt.parser_based_warp_experiment_from_run, root_opt.parser_based_warp_experiment_from_dir)
    parser.pb_warp_load_step_checkpoint = os.path.join(parser.pb_warp_load_step_checkpoint_dir, parser.pb_warp_load_step_checkpoint)
    parser.pb_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_step_checkpoint)
    parser.pb_warp_load_step_checkpoint_dir = os.path.join("/",*parser.pb_warp_load_step_checkpoint.split("/")[:-1])

    parser.pb_warp_save_final_checkpoint_dir = parser.pb_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_final_checkpoint = os.path.join(parser.pb_warp_save_final_checkpoint_dir, parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_final_checkpoint.split("/")[:-1])

    parser.pb_warp_load_final_checkpoint_dir = parser.pb_warp_load_final_checkpoint_dir.format(root_opt.parser_based_warp_experiment_from_run, root_opt.parser_based_warp_experiment_from_dir)
    parser.pb_warp_load_final_checkpoint = os.path.join(parser.pb_warp_load_final_checkpoint_dir, parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_load_final_checkpoint.split("/")[:-1])
    
    # Parser Based Gen
    parser.pb_gen_save_step_checkpoint_dir = parser.pb_gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_step_checkpoint = os.path.join(parser.pb_gen_save_step_checkpoint_dir, parser.pb_gen_save_step_checkpoint)
    parser.pb_gen_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_step_checkpoint)
    parser.pb_gen_save_step_checkpoint_dir = os.path.join("/",*parser.pb_gen_save_step_checkpoint.split("/")[:-1])
    
    parser.pb_gen_load_step_checkpoint_dir = parser.pb_gen_load_step_checkpoint_dir.format(root_opt.parser_based_gen_experiment_from_run, root_opt.parser_based_gen_experiment_from_dir)
    parser.pb_gen_load_step_checkpoint = os.path.join(parser.pb_gen_load_step_checkpoint_dir, parser.pb_gen_load_step_checkpoint)
    parser.pb_gen_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_step_checkpoint)
    parser.pb_gen_load_step_checkpoint_dir = os.path.join("/",*parser.pb_gen_load_step_checkpoint.split("/")[:-1])

    parser.pb_gen_save_final_checkpoint_dir = parser.pb_gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_final_checkpoint = os.path.join(parser.pb_gen_save_final_checkpoint_dir, parser.pb_gen_save_final_checkpoint)
    parser.pb_gen_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_final_checkpoint)
    parser.pb_gen_save_final_checkpoint_dir = os.path.join("/",*parser.pb_gen_save_final_checkpoint.split("/")[:-1])
    
    parser.pb_gen_load_final_checkpoint_dir = parser.pb_gen_load_final_checkpoint_dir.format(root_opt.parser_based_gen_experiment_from_run, root_opt.parser_based_gen_experiment_from_dir)
    parser.pb_gen_load_final_checkpoint = os.path.join(parser.pb_gen_load_final_checkpoint_dir, parser.pb_gen_load_final_checkpoint)
    parser.pb_gen_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_final_checkpoint)
    parser.pb_gen_load_final_checkpoint_dir = os.path.join("/",*parser.pb_gen_load_final_checkpoint.split("/")[:-1])
    # Parser Free Warping
    parser.pf_warp_save_step_checkpoint_dir = parser.pf_warp_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_warp_save_step_checkpoint = os.path.join(parser.pf_warp_save_step_checkpoint_dir, parser.pf_warp_save_step_checkpoint)
    parser.pf_warp_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_save_step_checkpoint)
    parser.pf_warp_save_step_checkpoint_dir = os.path.join("/",*parser.pf_warp_save_step_checkpoint.split("/")[:-1])
    
    parser.pf_warp_load_step_checkpoint_dir = parser.pf_warp_load_step_checkpoint_dir.format(root_opt.parser_free_warp_experiment_from_run, root_opt.parser_free_warp_experiment_from_dir)
    parser.pf_warp_load_step_checkpoint = os.path.join(parser.pf_warp_load_step_checkpoint_dir, parser.pf_warp_load_step_checkpoint)
    parser.pf_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_load_step_checkpoint)
    parser.pf_warp_load_step_checkpoint_dir = os.path.join("/",*parser.pf_warp_load_step_checkpoint.split("/")[:-1])
    
    parser.pf_warp_save_final_checkpoint_dir = parser.pf_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_warp_save_final_checkpoint = os.path.join(parser.pf_warp_save_final_checkpoint_dir, parser.pf_warp_save_final_checkpoint)
    parser.pf_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_save_final_checkpoint)

    parser.pf_warp_load_final_checkpoint_dir = parser.pf_warp_load_final_checkpoint_dir.format(root_opt.parser_free_warp_experiment_from_run, root_opt.parser_free_warp_experiment_from_dir)
    parser.pf_warp_load_final_checkpoint = os.path.join(parser.pf_warp_load_final_checkpoint_dir, parser.pf_warp_load_final_checkpoint)
    parser.pf_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_load_final_checkpoint)
    parser.pf_warp_load_final_checkpoint_dir = os.path.join("/",*parser.pf_warp_load_final_checkpoint.split("/")[:-1])
    # Parser Free Gen
    parser.pf_gen_save_step_checkpoint_dir = parser.pf_gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_save_step_checkpoint = os.path.join(parser.pf_gen_save_step_checkpoint_dir, parser.pf_gen_save_step_checkpoint)
    parser.pf_gen_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_save_step_checkpoint)
    parser.pf_gen_save_step_checkpoint_dir = os.path.join("/",*parser.pf_gen_save_step_checkpoint.split("/")[:-1])
    
    parser.pf_gen_save_final_checkpoint_dir = parser.pf_gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_save_final_checkpoint = os.path.join(parser.pf_gen_save_final_checkpoint_dir, parser.pf_gen_save_final_checkpoint)
    parser.pf_gen_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_save_final_checkpoint)
    parser.pf_gen_save_final_checkpoint_dir = os.path.join("/",*parser.pf_gen_save_final_checkpoint.split("/")[:-1])
    
    parser.pf_gen_load_final_checkpoint_dir = parser.pf_gen_load_final_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.parser_free_gen_experiment_from_dir)
    parser.pf_gen_load_final_checkpoint = os.path.join(parser.pf_gen_load_final_checkpoint_dir, parser.pf_gen_load_final_checkpoint)
    parser.pf_gen_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_load_final_checkpoint)
    parser.pf_gen_load_final_checkpoint_dir = os.path.join("/",*parser.pf_gen_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        parser.pf_gen_load_step_checkpoint_dir = parser.pf_gen_load_step_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.parser_free_gen_experiment_from_dir)
    else:
        parser.pf_gen_load_step_checkpoint_dir = parser.pf_gen_load_step_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_load_step_checkpoint_dir = fix(parser.pf_gen_load_step_checkpoint_dir)
    if not last_step:
        parser.pf_gen_load_step_checkpoint_dir = os.path.join(parser.pf_gen_load_step_checkpoint_dir, parser.pf_gen_load_step_checkpoint)
    else:
        if os.path.isdir(parser.pf_gen_load_step_checkpoint_dir.format(root_opt.pf_gen_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(parser.pf_gen_load_step_checkpoint_dir.format(root_opt.pf_gen_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "pf_gen" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            parser.pf_gen_load_step_checkpoint = os.path.join(parser.pf_gen_load_step_checkpoint_dir, last_step)
    parser.pf_gen_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_load_step_checkpoint)
    parser.pf_gen_load_step_checkpoint_dir = os.path.join("/",*parser.pf_gen_load_step_checkpoint.split("/")[:-1])
    return parser



def train_pfafn_e2e_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_pfafn_e2e_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_pfafn_e2e_()
    else:
        wandb = None
        _train_pfafn_e2e_()

def _train_pfafn_e2e_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_pfafn_e2e_()
            
            
def split_dataset(dataset,train_size=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, validation_indices = train_test_split(indices, train_size=train_size)
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    return train_subset, validation_subset


def make_dirs(opt):
    if not os.path.exists(opt.pf_warp_save_step_checkpoint_dir):
        os.makedirs(opt.pf_warp_save_step_checkpoint_dir)
    if not os.path.exists(opt.pf_gen_save_final_checkpoint_dir):
        os.makedirs(opt.pf_gen_save_final_checkpoint_dir)
    if not os.path.exists(opt.pf_gen_save_step_checkpoint_dir):
        os.makedirs(opt.pf_gen_save_step_checkpoint_dir)
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

def _train_pfafn_e2e_():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    writer = SummaryWriter(opt.tensorboard_dir)
    torch.cuda.set_device(opt.device)
    device = torch.device(f'cuda:{opt.device}')
    if sweep_id is not None:
        opt = wandb.config
    epoch_num = opt.niter + opt.niter_decay
    
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    
    log_path = os.path.join(opt.results_dir, 'log.txt')
    
    with open(log_path, 'w') as file:
        file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    
    start_epoch, epoch_iter = 1, 0

    train_data = CreateDataset(opt, root_opt)
    train_dataset, validation_dataset = split_dataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=opt.viton_batch_size, shuffle=True,num_workers=root_opt.workers)
    validation_loader = DataLoader(validation_dataset, batch_size=opt.viton_batch_size, shuffle=True, num_workers=root_opt.workers)
    dataset_size = len(train_loader)

    PF_warp_model = AFWM(opt, 3)
    print(PF_warp_model)
    PF_warp_model.train()
    PF_warp_model.cuda()
    if os.path.exists(opt.pf_warp_load_final_checkpoint_dir):
        load_checkpoint_part_parallel(opt, PF_warp_model, opt.pf_warp_load_final_checkpoint)

    PF_gen_model = ResUnetGenerator(opt, 7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    if os.path.exists(opt.pf_gen_load_final_checkpoint_dir):
        load_checkpoint_parallel(opt, PF_gen_model, opt.pf_gen_load_final_checkpoint)
    print(PF_gen_model)
    PF_gen_model.train()
    PF_gen_model.cuda()

    PB_warp_model = AFWM(opt, 45)
    print(PB_warp_model)
    PB_warp_model.eval()
    PB_warp_model.cuda()
    if os.path.exists(opt.pb_warp_load_final_checkpoint_dir):
        load_checkpoint_part_parallel(opt, PB_warp_model, opt.pb_warp_load_final_checkpoint)
    

    PB_gen_model = ResUnetGenerator(opt, 8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    print(PB_gen_model)
    PB_gen_model.eval()
    PB_gen_model.cuda()
    if os.path.exists(opt.pb_gen_load_final_checkpoint_dir):
        load_checkpoint_parallel(opt, PB_gen_model, opt.pb_gen_load_final_checkpoint)

    PF_warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(PF_warp_model).to(device)
    PF_gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(PF_gen_model).to(device)

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionL2 = nn.MSELoss('sum')

    params_warp = [p for p in PF_warp_model.parameters()]
    params_gen = [p for p in PF_gen_model.parameters()]
    optimizer_warp = torch.optim.Adam(params_warp, lr=0.2 * opt.lr, betas=(opt.beta1, 0.999))
    optimizer_gen = torch.optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, 0.999))

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    writer = SummaryWriter(opt.tensorboard_dir)

    step = 0
    step_per_batch = dataset_size

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        train_batch(opt, root_opt, train_loader, 
                    PB_warp_model,PF_warp_model,PB_gen_model, PF_gen_model, total_steps,
                    epoch,epoch_iter,criterionL1,criterionVGG,optimizer_warp,optimizer_gen,
                    writer, step_per_batch,epoch_start_time)
        if epoch % opt.val_count == 0:
            with torch.no_grad():
                validate_batch(opt, root_opt, validation_loader, PB_warp_model,PF_warp_model,PB_gen_model, PF_gen_model, total_steps,
                    epoch,criterionL1,criterionVGG,writer)
    save_checkpoint(PF_gen_model, opt.pf_gen_save_final_checkpoint)
    
def validate_batch(opt, root_opt, validation_loader, PB_warp_model,PF_warp_model,PB_gen_model, PF_gen_model, total_steps,
    epoch,criterionL1,criterionVGG,writer):
    PB_warp_model.eval()
    PF_warp_model.eval()
    PB_gen_model.eval()
    PF_gen_model.eval()
    warping_loss = 0
    warping_l1 = 0
    warping_vgg = 0
    loss_gen =0 
    composition_loss = 0
    for i, data in enumerate(validation_loader):
        total_steps += 1
        if root_opt.dataset_name == 'Rail':
            t_mask = torch.FloatTensor(((data['label'] == 3) | (data['label'] == 11)).cpu().numpy().astype(np.int64))
        else:
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        edge_un = data['edge_un']
        pre_clothes_edge_un = torch.FloatTensor((edge_un.detach().numpy() > 0.5).astype(np.int))
        clothes_un = data['color_un']
        clothes_un = clothes_un * pre_clothes_edge_un
        if root_opt.dataset_name == 'Rail':
            person_clothes_edge = torch.FloatTensor(((data['label'] == 5) | (data['label'] == 6) | (data['label'] == 7)).cpu().numpy().astype(np.int64))
        else:
            person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        pose_map = data['pose_map']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1, data['densepose'].data.long().cuda(), 1.0)
        densepose_fore = data['densepose'] / 24
        if root_opt.dataset_name == 'Rail':
            face_mask = torch.FloatTensor(
            (data['label'].cpu().numpy() == 1).astype(np.int64)
            ) + torch.FloatTensor(((data['label'] == 4) | (data['label'] == 13)).cpu().numpy().astype(np.int64))
        else:
            face_mask = torch.FloatTensor(
            (data['label'].cpu().numpy() == 1).astype(np.int64)
            ) + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int64))
        if root_opt.dataset_name == 'Rail':    
            other_clothes_mask = (
                torch.FloatTensor((data['label'].cpu().numpy() == 18).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 19).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 16).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 17).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int64))
            )
        else:
            other_clothes_mask = (
                torch.FloatTensor((data['label'].cpu().numpy() == 5).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 6).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 8).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int64))
                + torch.FloatTensor((data['label'].cpu().numpy() == 10).astype(np.int64))
            )
        face_img = face_mask * real_image
        other_clothes_img = other_clothes_mask * real_image
        preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)

        concat_un = torch.cat([preserve_mask.cuda(), densepose, pose.cuda()], 1)
        flow_out_un = PB_warp_model(concat_un.cuda(), clothes_un.cuda(), pre_clothes_edge_un.cuda())
        warped_cloth_un, last_flow_un, cond_un_all, flow_un_all, delta_list_un, x_all_un, x_edge_all_un, delta_x_all_un, delta_y_all_un = flow_out_un
        warped_prod_edge_un = F.grid_sample(pre_clothes_edge_un.cuda(), last_flow_un.permute(0, 2, 3, 1),
                                            mode='bilinear', padding_mode='zeros')

        flow_out_sup = PB_warp_model(concat_un.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth_sup, last_flow_sup, cond_sup_all, flow_sup_all, delta_list_sup, x_all_sup, x_edge_all_sup, delta_x_all_sup, delta_y_all_sup = flow_out_sup

        arm_mask = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.float)) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float))
        hand_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 3).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int))
        dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 22))
        hand_img = (arm_mask * hand_mask) * real_image
        dense_preserve_mask = dense_preserve_mask.cuda() * (1 - warped_prod_edge_un)
        preserve_region = face_img + other_clothes_img + hand_img

        gen_inputs_un = torch.cat([preserve_region.cuda(), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1)
        gen_outputs_un = PB_gen_model(gen_inputs_un)
        p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
        p_rendered_un = torch.tanh(p_rendered_un)
        m_composite_un = torch.sigmoid(m_composite_un)
        m_composite_un = m_composite_un * warped_prod_edge_un
        p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

        flow_out = PF_warp_model(p_tryon_un.detach(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]
        if root_opt.dataset_name == 'Rail' and epoch >0 :
            binary_mask = (warped_prod_edge > 0.5).float()
            warped_cloth = warped_cloth * binary_mask
            
        epsilon = opt.epsilon
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_warp = 0
        loss_fea_sup_all = 0
        loss_flow_sup_all = 0

        l1_loss_batch = torch.abs(warped_cloth_sup.detach() - person_clothes.cuda())
        l1_loss_batch = l1_loss_batch.reshape(opt.viton_batch_size, 3 * 256 * 192)
        l1_loss_batch = l1_loss_batch.sum(dim=1) / (3 * 256 * 192)
        l1_loss_batch_pred = torch.abs(warped_cloth.detach() - person_clothes.cuda())
        l1_loss_batch_pred = l1_loss_batch_pred.reshape(opt.viton_batch_size, 3 * 256 * 192)
        l1_loss_batch_pred = l1_loss_batch_pred.sum(dim=1) / (3 * 256 * 192)
        weight = (l1_loss_batch < l1_loss_batch_pred).float()
        num_all = len(np.where(weight.cpu().numpy() > 0)[0])
        if num_all == 0:
            num_all = 1

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            b1, c1, h1, w1 = cond_all[num].shape
            weight_all = weight.reshape(-1, 1, 1, 1).repeat(1, 256, h1, w1)
            cond_sup_loss = ((cond_sup_all[num].detach() - cond_all[num]) ** 2 * weight_all).sum() / (256 * h1 * w1 * num_all)
            loss_fea_sup_all = loss_fea_sup_all + (5 - num) * 0.04 * cond_sup_loss
            loss_warp = loss_warp + (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth + (5 - num) * 0.04 * cond_sup_loss
            if num >= 2:
                b1, c1, h1, w1 = flow_all[num].shape
                weight_all = weight.reshape(-1, 1, 1).repeat(1, h1, w1)
                flow_sup_loss = (torch.norm(flow_sup_all[num].detach() - flow_all[num], p=2, dim=1) * weight_all).sum() / (h1 * w1 * num_all)
                loss_flow_sup_all = loss_flow_sup_all + (num + 1) * 1 * flow_sup_loss
                loss_warp = loss_warp + (num + 1) * 1 * flow_sup_loss

        loss_warp = opt.lambda_loss_smooth * loss_smooth + loss_warp

        skin_mask = warped_prod_edge_un.detach() * (1 - person_clothes_edge.cuda())
        gen_inputs = torch.cat([p_tryon_un.detach(), warped_cloth, warped_prod_edge], 1)
        gen_outputs = PF_gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite1 = m_composite * warped_prod_edge
        m_composite = person_clothes_edge.cuda() * m_composite1
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
        loss_l1_skin = criterionL1(p_rendered * skin_mask, skin_mask * real_image.cuda())
        loss_vgg_skin = criterionVGG(p_rendered * skin_mask, skin_mask * real_image.cuda())
        loss_l1 = criterionL1(p_tryon, real_image.cuda())
        loss_vgg = criterionVGG(p_tryon, real_image.cuda())
        bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
        bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())

        if epoch < opt.niter:
            loss_gen = (loss_l1 * opt.lambda_loss_l1 + loss_l1_skin * opt.lambda_loss_vgg + loss_vgg + loss_vgg_skin * opt.lambda_loss_l1_skin + bg_loss_l1 * opt.lambda_bg_loss_l1 + bg_loss_vgg + opt.lambda_loss_l1_mask * loss_mask_l1)
        else:
            loss_gen = (loss_l1 * 5 + loss_l1_skin * 60 + loss_vgg + loss_vgg_skin *opt.lambda_loss_vgg_skin + bg_loss_l1 * opt.lambda_bg_loss_l1 + bg_loss_vgg + opt.lambda_loss_l1_mask * loss_mask_l1)

        loss_all = opt.lambda_loss_warp * loss_warp + opt.lambda_loss_gen * loss_gen
        
        a = real_image.float().cuda()
        b = p_tryon_un.detach()
        c = clothes.cuda()
        d = person_clothes.cuda()
        e = torch.cat([skin_mask.cuda(), skin_mask.cuda(), skin_mask.cuda()], 1)
        f = warped_cloth
        g = p_rendered
        h = torch.cat([m_composite1, m_composite1, m_composite1], 1)
        i = p_tryon
        j = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
        combine = torch.cat([a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0]], 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)
        warping_loss += loss_warp.item()
        warping_l1 += loss_l1.item()
        warping_vgg += loss_vgg.item()
        loss_gen += loss_gen.item()
        composition_loss += loss_all.item()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(opt.results_dir, f"{epoch}.jpg"),bgr)
        
    log_losses = {'val_warping_loss': warping_loss / len(validation_loader.dataset) ,'val_warping_l1': warping_l1 / len(validation_loader.dataset) ,'val_warping_vgg': warping_vgg / len(validation_loader.dataset),
                      'val_loss_gen':loss_gen / len(validation_loader.dataset) ,'val_composition_loss': composition_loss / len(validation_loader.dataset) }
    log_images = {'Val/Image': (a[0].cpu() / 2 + 0.5), 
        'Val/Pose Image': (pose_map[0].cpu() / 2 + 0.5), 
        'Val/Clothing': (c[0].cpu() / 2 + 0.5), 
        'Val/Parse Clothing': (d[0].cpu() / 2 + 0.5), 
        'Val/Parse Clothing Mask': j[0].cpu().expand(3, -1, -1), 
        'Val/Warped Cloth': (f[0].cpu().detach() / 2 + 0.5), 
        'Val/Warped Cloth Mask': person_clothes_edge[0].cpu().detach().expand(3, -1, -1),
        "Val/Composition": i[0].cpu() / 2 + 0.5}
    log_results(log_images, log_losses, writer,wandb, epoch, train=False)
        

def log_results(log_images, log_losses,board,wandb, step,iter_start_time=None,train=True):
    table = 'Table' if train else 'Val_Table'
    for key,value in log_losses.items():
        board.add_scalar(key, value, step+1)
    wandb_images = []
    for key,value in log_images.items():
        board.add_image(key, value, step+1)
        if wandb is not None:
            wandb_images.append(get_wandb_image(value, wandb=wandb))
    if wandb is not None:
        my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth','Warped Cloth Mask', 'Composition'])
        my_table.add_data(*wandb_images)
        wandb.log({table: my_table, **log_losses})
    if train and iter_start_time is not None:
        t = time.time() - iter_start_time
        print('training step: %8d, time: %.3f, composition_loss: %4f warping loss: %4f warping_l1 loss: %4f warping_vgg loss: %4f gen loss: %4f' % (step+1, t, log_losses['composition_loss'],log_losses['warping_loss'],log_losses['warping_l1'],log_losses['warping_vgg'], log_losses['loss_gen']), flush=True)
    else:
        print('validation step: %8d, time: %.3f, composition_loss: %4f warping loss: %4f warping_l1 loss: %4f warping_vgg loss: %4f gen loss: %4f' % (step+1, t, log_losses['val_composition_loss'],log_losses['val_warping_loss'],log_losses['val_warping_l1'],log_losses['val_warping_vgg'], log_losses['val_loss_gen']), flush=True)
        
def train_batch(opt, root_opt, train_loader, PB_warp_model,PF_warp_model,PB_gen_model, PF_gen_model, total_steps,
                epoch,epoch_iter,criterionL1,criterionVGG,optimizer_warp,optimizer_gen,
                writer, step_per_batch,epoch_start_time):
    PB_warp_model.train()
    PF_warp_model.train()
    PB_gen_model.train()
    PF_gen_model.train()
    dataset_size = len(train_loader)
    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        edge_un = data['edge_un']
        pre_clothes_edge_un = torch.FloatTensor((edge_un.detach().numpy() > 0.5).astype(np.int))
        clothes_un = data['color_un']
        clothes_un = clothes_un * pre_clothes_edge_un
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        pose_map = data['pose_map']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1, data['densepose'].data.long().cuda(), 1.0)
        densepose_fore = data['densepose'] / 24
        face_mask = torch.FloatTensor((data['label'].cpu().numpy() == 1).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy() == 5).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy() == 6).astype(np.int)) \
                            + torch.FloatTensor((data['label'].cpu().numpy() == 8).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int)) \
                            + torch.FloatTensor((data['label'].cpu().numpy() == 10).astype(np.int))
        face_img = face_mask * real_image
        other_clothes_img = other_clothes_mask * real_image
        preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)

        concat_un = torch.cat([preserve_mask.cuda(), densepose, pose.cuda()], 1)
        flow_out_un = PB_warp_model(concat_un.cuda(), clothes_un.cuda(), pre_clothes_edge_un.cuda())
        warped_cloth_un, last_flow_un, cond_un_all, flow_un_all, delta_list_un, x_all_un, x_edge_all_un, delta_x_all_un, delta_y_all_un = flow_out_un
        warped_prod_edge_un = F.grid_sample(pre_clothes_edge_un.cuda(), last_flow_un.permute(0, 2, 3, 1),
                                            mode='bilinear', padding_mode='zeros')

        flow_out_sup = PB_warp_model(concat_un.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth_sup, last_flow_sup, cond_sup_all, flow_sup_all, delta_list_sup, x_all_sup, x_edge_all_sup, delta_x_all_sup, delta_y_all_sup = flow_out_sup

        arm_mask = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.float)) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float))
        hand_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 3).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int))
        dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 22))
        hand_img = (arm_mask * hand_mask) * real_image
        dense_preserve_mask = dense_preserve_mask.cuda() * (1 - warped_prod_edge_un)
        preserve_region = face_img + other_clothes_img + hand_img

        gen_inputs_un = torch.cat([preserve_region.cuda(), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1)
        gen_outputs_un = PB_gen_model(gen_inputs_un)
        p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
        p_rendered_un = torch.tanh(p_rendered_un)
        m_composite_un = torch.sigmoid(m_composite_un)
        m_composite_un = m_composite_un * warped_prod_edge_un
        p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

        flow_out = PF_warp_model(p_tryon_un.detach(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]
        if root_opt.dataset_name == 'Rail' and epoch >0 :
            binary_mask = (warped_prod_edge > 0.5).float()
            warped_cloth = warped_cloth * binary_mask
            
        epsilon = opt.epsilon
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_warp = 0
        loss_fea_sup_all = 0
        loss_flow_sup_all = 0

        l1_loss_batch = torch.abs(warped_cloth_sup.detach() - person_clothes.cuda())
        l1_loss_batch = l1_loss_batch.reshape(opt.viton_batch_size, 3 * 256 * 192)
        l1_loss_batch = l1_loss_batch.sum(dim=1) / (3 * 256 * 192)
        l1_loss_batch_pred = torch.abs(warped_cloth.detach() - person_clothes.cuda())
        l1_loss_batch_pred = l1_loss_batch_pred.reshape(opt.viton_batch_size, 3 * 256 * 192)
        l1_loss_batch_pred = l1_loss_batch_pred.sum(dim=1) / (3 * 256 * 192)
        weight = (l1_loss_batch < l1_loss_batch_pred).float()
        num_all = len(np.where(weight.cpu().numpy() > 0)[0])
        if num_all == 0:
            num_all = 1

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            b1, c1, h1, w1 = cond_all[num].shape
            weight_all = weight.reshape(-1, 1, 1, 1).repeat(1, 256, h1, w1)
            cond_sup_loss = ((cond_sup_all[num].detach() - cond_all[num]) ** 2 * weight_all).sum() / (256 * h1 * w1 * num_all)
            loss_fea_sup_all = loss_fea_sup_all + (5 - num) * 0.04 * cond_sup_loss
            loss_warp = loss_warp + (num+1) * loss_l1 + (num+1) * opt.lambda_loss_vgg * loss_vgg + (num+1) * opt.lambda_loss_edge * loss_edge + (num+1) * opt.lambda_loss_second_smooth * loss_second_smooth
            if num >= 2:
                b1, c1, h1, w1 = flow_all[num].shape
                weight_all = weight.reshape(-1, 1, 1).repeat(1, h1, w1)
                flow_sup_loss = (torch.norm(flow_sup_all[num].detach() - flow_all[num], p=2, dim=1) * weight_all).sum() / (h1 * w1 * num_all)
                loss_flow_sup_all = loss_flow_sup_all + (num + 1) * 1 * flow_sup_loss
                loss_warp = loss_warp + (num + 1) * 1 * flow_sup_loss

        loss_warp = opt.lambda_loss_smooth * loss_smooth + loss_warp

        skin_mask = warped_prod_edge_un.detach() * (1 - person_clothes_edge.cuda())
        gen_inputs = torch.cat([p_tryon_un.detach(), warped_cloth, warped_prod_edge], 1)
        gen_outputs = PF_gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite1 = m_composite * warped_prod_edge
        m_composite = person_clothes_edge.cuda() * m_composite1
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
        loss_l1_skin = criterionL1(p_rendered * skin_mask, skin_mask * real_image.cuda())
        loss_vgg_skin = criterionVGG(p_rendered * skin_mask, skin_mask * real_image.cuda())
        loss_l1 = criterionL1(p_tryon, real_image.cuda())
        loss_vgg = criterionVGG(p_tryon, real_image.cuda())
        bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
        bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())

        if epoch < opt.niter:
            loss_gen = (loss_l1 * opt.lambda_loss_l1 + loss_l1_skin * opt.lambda_loss_l1_skin + loss_vgg + loss_vgg_skin * 2 + bg_loss_l1 * opt.lambda_bg_loss_l1 + bg_loss_vgg + opt.lambda_loss_l1_mask * loss_mask_l1)
        else:
            loss_gen = (loss_l1 * 5 + loss_l1_skin * 60 + loss_vgg + loss_vgg_skin * 4 + bg_loss_l1 * 5 + bg_loss_vgg + 1 * loss_mask_l1)

        loss_all = opt.lambda_loss_warp * loss_warp + opt.lambda_loss_gen * loss_gen
        
        optimizer_warp.zero_grad()
        optimizer_gen.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        optimizer_gen.step()

        if (epoch + 1) % opt.display_count == 0:
            a = real_image.float().cuda()
            b = p_tryon_un.detach()
            c = clothes.cuda()
            d = person_clothes.cuda()
            e = torch.cat([skin_mask.cuda(), skin_mask.cuda(), skin_mask.cuda()], 1)
            f = warped_cloth
            g = p_rendered
            h = torch.cat([m_composite1, m_composite1, m_composite1], 1)
            i = p_tryon
            j = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
            combine = torch.cat([a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0]], 2).squeeze()
            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            rgb = (cv_img * 255).astype(np.uint8)
            log_losses = {'warping_loss': loss_warp.item() ,'warping_l1': loss_l1.item(),'warping_vgg': loss_vgg.item(),
                        'loss_gen':loss_gen.item(),'composition_loss': loss_all.item()}
            log_images = {'Image': (a[0].cpu() / 2 + 0.5), 
            'Pose Image': (pose_map[0].cpu() / 2 + 0.5), 
            'Clothing': (c[0].cpu() / 2 + 0.5), 
            'Parse Clothing': (d[0].cpu() / 2 + 0.5), 
            'Parse Clothing Mask': j[0].cpu().expand(3, -1, -1), 
            'Warped Cloth': (f[0].cpu().detach() / 2 + 0.5), 
            'Warped Cloth Mask': person_clothes_edge[0].cpu().detach().expand(3, -1, -1),
            "Composition": i[0].cpu() / 2 + 0.5}
            log_results(log_images, log_losses, writer,wandb, epoch, iter_start_time=iter_start_time, train=True)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.results_dir, f"{epoch}.jpg"),bgr)

        if epoch_iter >= dataset_size:
            break
        break

    # end of epoch


    if epoch % opt.save_period == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        save_checkpoint(PF_warp_model,opt.pf_warp_save_step_checkpoint % (epoch+1))
        save_checkpoint(PF_gen_model,opt.pf_gen_save_step_checkpoint % (epoch+1))

    if epoch > opt.niter:
        PF_warp_model.update_learning_rate_warp(optimizer_warp)
        PF_gen_model.update_learning_rate(optimizer_gen)
