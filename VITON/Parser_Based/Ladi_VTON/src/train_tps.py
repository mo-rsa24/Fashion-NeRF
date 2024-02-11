import argparse
import os
import random
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# from dataset.dresscode import DressCodeDataset
from VITON.Parser_Based.Ladi_VTON.src.dataset.dresscode import DressCodeDataset
from VITON.Parser_Based.Ladi_VTON.src.dataset.vitonhd import VitonHDDataset
from VITON.Parser_Based.Ladi_VTON.src.models.ConvNet_TPS import ConvNet_TPS
from VITON.Parser_Based.Ladi_VTON.src.models.UNet import UNetVanilla
from VITON.Parser_Based.Ladi_VTON.src.utils.vgg_loss import VGGLoss
from VITON.Parser_Based.Ladi_VTON.src.utils.env import tps_process_opt
fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None
   
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     


def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_metric(dataloader: DataLoader, tps: ConvNet_TPS, criterion_l1: nn.L1Loss, criterion_vgg: VGGLoss,
                   refinement: UNetVanilla = None, height: int = 512, width: int = 384) -> tuple[
    float, float, list[list]]:
    """
    Perform inference on the given dataloader and compute the L1 and VGG loss between the warped cloth and the
    ground truth image.
    """
    tps.eval()
    if refinement:
        refinement.eval()

    running_loss = 0.
    vgg_running_loss = 0
    for step, inputs in enumerate(tqdm(dataloader)):
        cloth = inputs['cloth'].to(device)
        image = inputs['image'].to(device)
        im_cloth = inputs['im_cloth'].to(device)
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)

        # TPS parameters prediction
        # For sake of performance, the TPS parameters are predicted on a low resolution image
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)

        # TPS parameters prediction
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)

        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(height, width),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)
        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

        if refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)

        # Compute the loss
        loss = criterion_l1(warped_cloth, im_cloth)
        running_loss += loss.item()
        if criterion_vgg:
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)
            vgg_running_loss += vgg_loss.item()
        break
    visual = [[image, cloth, im_cloth, warped_cloth.clamp(-1, 1)]]
    loss = running_loss / (step + 1)
    vgg_loss = vgg_running_loss / (step + 1)
    return loss, vgg_loss, visual


def training_loop_tps(dataloader: DataLoader, tps: ConvNet_TPS, optimizer_tps: torch.optim.Optimizer,
                      criterion_l1: nn.L1Loss, scaler: torch.cuda.amp.GradScaler, const_weight: float, epoch: int) -> tuple[
    float, float, float, list[list]]:
    """
    Training loop for the TPS network. Note that the TPS is trained on a low resolution image for sake of performance.
    """
    tps.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_const_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):  # Yield images with low resolution (256x192)
        low_cloth = inputs['cloth'].to(device, non_blocking=True)
        low_image = inputs['image'].to(device, non_blocking=True)
        low_im_cloth = inputs['im_cloth'].to(device, non_blocking=True)
        low_im_mask = inputs['im_mask'].to(device, non_blocking=True)

        low_pose_map = inputs.get('dense_uv')
        if low_pose_map is None:  # If the dataset does not provide dense UV maps, use the pose map (keypoints) instead
            low_pose_map = inputs['pose_map']
        low_pose_map = low_pose_map.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # TPS parameters prediction
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)
            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

            # Warp the cloth using the predicted TPS parameters
            low_warped_cloth = F.grid_sample(low_cloth, low_grid, padding_mode='border')

            # Compute the loss
            l1_loss = criterion_l1(low_warped_cloth, low_im_cloth)
            const_loss = torch.mean(rx + ry + cx + cy + rg + cg)

            loss = l1_loss + const_loss * const_weight

        # Update the parameters
        optimizer_tps.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_tps)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_const_loss += const_loss.item()

    visual = [[low_image, low_cloth, low_im_cloth, low_warped_cloth.clamp(-1, 1)]]
    log_images = {'Image': (low_image[0].cpu() / 2 + 0.5), 
            'Pose Image': (low_pose_map[0].cpu() / 2 + 0.5), 
            'Clothing': (low_cloth[0].cpu() / 2 + 0.5), 
            'Parse Clothing': (low_im_cloth[0].cpu() / 2 + 0.5), 
            'Parse Clothing Mask': low_im_mask[0].cpu().expand(3, -1, -1), 
            'Warped Cloth': (low_warped_cloth[0].cpu().detach() / 2 + 0.5), 
            'Warped Cloth Mask': low_im_mask[0].cpu().detach().expand(3, -1, -1)}
    loss = running_loss  / len(dataloader.dataset)
    l1_loss = running_l1_loss / len(dataloader.dataset)
    const_loss = running_const_loss / len(dataloader.dataset)
    log_losses = {'warping_loss': running_loss ,'warping_l1': running_l1_loss,
                  'const_loss':const_loss}
    return loss, l1_loss, const_loss, visual,log_images, log_losses


def training_loop_refinement(dataloader: DataLoader, tps: ConvNet_TPS, refinement: UNetVanilla,
                             optimizer_ref: torch.optim.Optimizer, criterion_l1: nn.L1Loss, criterion_vgg: VGGLoss,
                             l1_weight: float, vgg_weight: float, scaler: torch.cuda.amp.GradScaler, height=512,
                             width=384) -> tuple[float, float, float, list[list]]:
    """
    Training loop for the refinement network. Note that the refinement network is trained on a high resolution image
    """
    tps.eval()
    refinement.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_vgg_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):
        cloth = inputs['cloth'].to(device)
        image = inputs['image'].to(device)
        im_cloth = inputs['im_cloth'].to(device)
        im_mask = inputs['im_mask'].to(device)

        pose_map = inputs.get('dense_uv')
        if pose_map is None:  # If the dataset does not provide dense UV maps, use the pose map (keypoints) instead
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)

        # Resize the inputs to the low resolution for the TPS network
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)

        with torch.cuda.amp.autocast():
            # TPS parameters prediction
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)

            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)
            low_warped_cloth = F.grid_sample(cloth, low_grid, padding_mode='border')

            # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
            highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                    size=(height, width),
                                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                    antialias=True).permute(0, 2, 3, 1)

            warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)

            # Compute the loss
            l1_loss = criterion_l1(warped_cloth, im_cloth)
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)

            loss = l1_loss * l1_weight + vgg_loss * vgg_weight
        # Update the parameters
        optimizer_ref.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_ref)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_vgg_loss += vgg_loss.item()

    visual = [[image, cloth, im_cloth, low_warped_cloth.clamp(-1, 1)]]
    loss = running_loss / len(dataloader.dataset)
    l1_loss = running_l1_loss / len(dataloader.dataset)
    vgg_loss = running_vgg_loss / len(dataloader.dataset)
    
    log_images = {'Image': (image[0].cpu() / 2 + 0.5), 
            'Pose Image': (pose_map[0].cpu() / 2 + 0.5), 
            'Clothing': (cloth[0].cpu() / 2 + 0.5), 
            'Parse Clothing': (im_cloth[0].cpu() / 2 + 0.5), 
            'Parse Clothing Mask': im_mask[0].cpu().expand(3, -1, -1), 
            'Warped Cloth': (warped_cloth[0].cpu().detach() / 2 + 0.5), 
            'Warped Cloth Mask': im_mask[0].cpu().detach().expand(3, -1, -1)}
    log_losses = {'warping_loss': loss ,'warping_l1': l1_loss,
                  'const_loss':vgg_loss}
    
    
    return loss, l1_loss, vgg_loss, visual, log_images, log_losses


@torch.no_grad()
def extract_images(dataloader: DataLoader, tps: ConvNet_TPS, refinement: UNetVanilla, save_path: str, height: int = 512,
                   width: int = 384) -> None:
    """
    Extracts the images using the trained networks and saves them to the save_path
    """
    tps.eval()
    refinement.eval()

    # running_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):
        c_name = inputs['c_name']
        im_name = inputs['im_name']
        cloth = inputs['cloth'].to(device)
        category = inputs.get('category')
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)

        # Resize the inputs to the low resolution for the TPS network
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)

        # TPS parameters prediction
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)

        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(height, width),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

        # Refine the warped cloth using the refinement network
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
        warped_cloth = refinement(warped_cloth)

        warped_cloth = (warped_cloth + 1) / 2
        warped_cloth = warped_cloth.clamp(0, 1)

        # Save the images
        for cname, iname, warpclo, cat in zip(c_name, im_name, warped_cloth, category):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            save_image(warpclo, os.path.join(save_path, cat, iname.replace(".jpg", "") + "_" + cname),
                       quality=95)
        break


def _train_ladi_vton_tps__sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_ladi_vton_tps_()
            
def train_ladi_vton_tps_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = tps_process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_ladi_vton_tps__sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_ladi_vton_tps_()
    else:
        wandb = None
        _train_ladi_vton_tps_()
             
def make_dirs(opt):
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    if not os.path.exists(opt.tocg_save_final_checkpoint_dir):
        os.makedirs(opt.tocg_save_final_checkpoint_dir)
    if not os.path.exists(opt.tocg_discriminator_save_final_checkpoint_dir):
        os.makedirs(opt.tocg_discriminator_save_final_checkpoint_dir)
    if not os.path.exists(os.path.join(opt.results_dir,'val')):
        os.makedirs(os.path.join(opt.results_dir,'val'))
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    if not os.path.exists(opt.tocg_save_step_checkpoint_dir):
        os.makedirs(opt.tocg_save_step_checkpoint_dir)
    if not os.path.exists(opt.tocg_discriminator_save_step_checkpoint_dir):
        os.makedirs(opt.tocg_discriminator_save_step_checkpoint_dir)

def log_results(log_images, log_losses, board,wandb, step, iter_start_time=None, train=True):
    table = 'Table' if train else 'Val_Table'
    wandb_images = []
    for key,value in log_losses.items():
        board.add_scalar(key, value, step+1)
        
    for key,value in log_images.items():
        board.add_image(key, value, step+1)
        if wandb is not None:
            wandb_images.append(get_wandb_image(value, wandb=wandb))

    if wandb is not None:
        my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth','Warped Cloth Mask'])
        my_table.add_data(*wandb_images)
        wandb.log({table: my_table, **log_losses})
    if train and iter_start_time is not None:
        t = time.time() - iter_start_time
        print("training step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, Const loss %.4f"
                % (step + 1, t, log_losses['warping_loss'], log_losses['warping_l1'], log_losses['const_loss']), flush=True)
    else:
        print("validation step: %8d, loss G: %.4f, L1_cloth loss: %.4f, Const loss %.4f"
                % (step + 1,  log_losses['val_warping_loss'], log_losses['val_warping_l1'], log_losses['val_const_loss']), flush=True)

def _train_ladi_vton_tps_():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    board = SummaryWriter(log_dir = opt.tensorboard_dir)
    torch.cuda.set_device(f'cuda:{opt.device}')
    if sweep_id is not None:
        opt.lr = wandb.config.lr
        
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    with open(log_path, 'w') as file:
        file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")

    dataset_output_list = ['c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'im_mask', 'pose_map', 'category']
    if opt.dense:
        dataset_output_list.append('dense_uv')

    # Training dataset and dataloader
    dataroot = os.path.join(root_opt.root_dir, root_opt.original_dir)
    if opt.dataset == "vitonhd":
        dataset_train = VitonHDDataset(opt, root_opt, phase='train',
                                       outputlist=dataset_output_list,
                                       dataroot_path=dataroot,
                                       size=(opt.height, opt.width))
    elif opt.dataset == "dresscode":
        dataset_train = DressCodeDataset(dataroot_path=opt.dresscode_dataroot,
                                         phase='train',
                                         outputlist=dataset_output_list,
                                         size=(opt.height, opt.width))
    else:
        raise NotImplementedError("Dataset should be either vitonhd or dresscode")

    dataset_train.__getitem__(0)
    dataloader_train = DataLoader(batch_size=opt.viton_batch_size,
                                  dataset=dataset_train,
                                  shuffle=True,
                                  num_workers=opt.viton_workers)

    # Validation dataset and dataloader
    if opt.dataset == "vitonhd":
        dataset_test_paired = VitonHDDataset(opt, root_opt, phase='test',
                                             dataroot_path=dataroot,
                                             outputlist=dataset_output_list, size=(opt.height, opt.width))

        dataset_test_unpaired = VitonHDDataset(opt, root_opt, phase='test',
                                               order='unpaired',
                                               dataroot_path=dataroot,
                                               outputlist=dataset_output_list, size=(opt.height, opt.width))

    elif opt.dataset == "dresscode":
        dataset_test_paired = DressCodeDataset(dataroot_path=opt.dresscode_dataroot,
                                               phase='test',
                                               outputlist=dataset_output_list, size=(opt.height, opt.width))

        dataset_test_unpaired = DressCodeDataset(phase='test',
                                                 order='unpaired',
                                                 dataroot_path=opt.dresscode_dataroot,
                                                 outputlist=dataset_output_list, size=(opt.height, opt.width))

    else:
        raise NotImplementedError("Dataset should be either vitonhd or dresscode")

    dataloader_test_paired = DataLoader(batch_size=opt.viton_batch_size,
                                        dataset=dataset_test_paired,
                                        shuffle=True,
                                        num_workers=opt.viton_workers, drop_last=True)

    dataloader_test_unpaired = DataLoader(batch_size=opt.viton_batch_size,
                                          dataset=dataset_test_unpaired,
                                          shuffle=True,
                                          num_workers=opt.viton_workers, drop_last=True)

    # Define TPS and refinement network
    input_nc = 5 if opt.dense else 21
    n_layer = 3
    tps = ConvNet_TPS(256, 192, input_nc, n_layer).to(device)

    refinement = UNetVanilla(
        n_channels=8 if opt.dense else 24,
        n_classes=3,
        bilinear=True).to(device)

    # Define optimizer, scaler and loss
    optimizer_tps = torch.optim.Adam(tps.parameters(), lr=opt.lr, betas=(0.5, 0.99))
    optimizer_ref = torch.optim.Adam(list(refinement.parameters()), lr=opt.lr, betas=(0.5, 0.99))

    scaler = torch.cuda.amp.GradScaler()
    criterion_l1 = nn.L1Loss()

    if opt.vgg_weight > 0:
        criterion_vgg = VGGLoss().to(device)
    else:
        criterion_vgg = None

    start_epoch = 0
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    
    if last_step:
        print('Loading full checkpoint')
        print_log(log_path, f'Load pretrained model from {opt.tps_load_step_checkpoint}')
        state_dict = torch.load(opt.tps_load_step_checkpoint)
        tps.load_state_dict(state_dict['tps'])
        refinement.load_state_dict(state_dict['refinement'])
        optimizer_tps.load_state_dict(state_dict['optimizer_tps'])
        optimizer_ref.load_state_dict(state_dict['optimizer_ref'])
        start_epoch = state_dict['epoch']
    elif os.path.exists(opt.tps_load_final_checkpoint):
        print('Loading full checkpoint')
        print_log(log_path, f'Load pretrained model from {opt.tps_load_final_checkpoint}')
        state_dict = torch.load(opt.tps_load_final_checkpoint)
        tps.load_state_dict(state_dict['tps'])
        refinement.load_state_dict(state_dict['refinement'])
        optimizer_tps.load_state_dict(state_dict['optimizer_tps'])
        optimizer_ref.load_state_dict(state_dict['optimizer_ref'])
        start_epoch = state_dict['epoch']

        if opt.only_extraction:
            print("Extracting warped cloth images...")
            extraction_dataset_paired = torch.utils.data.ConcatDataset([dataset_test_paired, dataset_train])
            extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                                      dataset=extraction_dataset_paired,
                                                      shuffle=False,
                                                      num_workers=opt.viton_workers,
                                                      drop_last=False)

            
            warped_cloth_root = opt.results_dir

            save_name_paired = os.path.join(warped_cloth_root, 'warped_cloths', opt.dataset)

            extract_images(extraction_dataloader_paired, tps, refinement, save_name_paired, opt.height, opt.width)

            extraction_dataset = dataset_test_unpaired
            extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                                      dataset=extraction_dataset,
                                                      shuffle=False,
                                                      num_workers=opt.viton_workers)

            save_name_unpaired = os.path.join(warped_cloth_root, 'warped_cloths_unpaired', opt.dataset)
            extract_images(extraction_dataloader_paired, tps, refinement, save_name_unpaired, opt.height, opt.width)
            exit()

    if opt.only_extraction and not os.path.exists(
            opt.tps_load_final_checkpoint):
        print("No checkpoint found, before extracting warped cloth images, please train the model first.")
        exit()
        
    # Set training dataset height and width to (256, 192) since the TPS is trained using a lower resolution
    dataset_train.height = 256
    dataset_train.width = 192
    # for e in range(start_epoch, opt.epochs_tps):
    for e in range(start_epoch, opt.epochs_tps):
        print(f"Epoch {e}/{opt.epochs_tps}")
        iter_start_time = time.time()
        train_loss, train_l1_loss, train_const_loss, visual,log_images, log_losses = training_loop_tps(
            dataloader_train,
            tps,
            optimizer_tps,
            criterion_l1,
            scaler,
            opt.const_weight)
        # [[low_image, low_cloth, low_im_cloth, low_warped_cloth.clamp(-1, 1)]]
        if (e + 1) % opt.display_count == 0:
            log_results(log_images, log_losses, board,wandb, e, iter_start_time=iter_start_time, train=True)
            
        imgs = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True,
                                           range=None, scale_each=False, pad_value=0)

        imgs_unpaired = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2,
                                                    normalize=True, range=None,
                                                    scale_each=False, pad_value=0)
        if (e + 1) % opt.save_period == 0:
            os.makedirs(opt.tps_save_step_checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': e + 1,
                'tps': tps.state_dict(),
                'refinement': refinement.state_dict(),
                'optimizer_tps': optimizer_tps.state_dict(),
                'optimizer_ref': optimizer_ref.state_dict(),
            }, opt.tps_save_step_checkpoint % (e + 1))
        
    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler again for refinement

    # Training loop for refinement
    # Set training dataset height and width to (opt.height, opt.width) since the refinement is trained using a higher resolution
    dataset_train.height = opt.height
    dataset_train.width = opt.width
    for e in range(max(start_epoch, opt.epochs_tps), max(start_epoch, opt.epochs_tps) + opt.epochs_refinement):
        print(f"Epoch {e}/{max(start_epoch, opt.epochs_tps) + opt.epochs_refinement}")
        train_loss, train_l1_loss, train_vgg_loss, visual, log_images, log_losses = training_loop_refinement(
            dataloader_train,
            tps,
            refinement,
            optimizer_ref,
            criterion_l1,
            criterion_vgg,
            opt.l1_weight,
            opt.vgg_weight,
            scaler,
            opt.height,
            opt.width)

        imgs = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True,
                                           range=None, scale_each=False, pad_value=0)

        imgs_unpaired = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2,
                                                    normalize=True, range=None,
                                                    scale_each=False, pad_value=0)
        if (e + 1) % opt.display_count == 0:
            log_results(log_images, log_losses, board,wandb, e, iter_start_time=iter_start_time, train=True)
        if (e + 1) % opt.save_period == 0:
            os.makedirs(opt.tps_save_step_checkpoint_dir, exist_ok=True)        
            torch.save({
                'epoch': e + 1,
                'tps': tps.state_dict(),
                'refinement': refinement.state_dict(),
                'optimizer_tps': optimizer_tps.state_dict(),
                'optimizer_ref': optimizer_ref.state_dict(),
            }, opt.tps_save_step_checkpoint % (e + 1))
    print("Extracting warped cloth images...")
    extraction_dataset_paired = torch.utils.data.ConcatDataset([dataset_test_paired, dataset_train])
    extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                              dataset=extraction_dataset_paired,
                                              shuffle=False,
                                              num_workers=opt.viton_workers,
                                              drop_last=False)

    warped_cloth_root = opt.results_dir

    # save_name_paired = warped_cloth_root / 'warped_cloths' / opt.dataset
    save_name_paired = os.path.join(warped_cloth_root, 'warped_cloths', opt.dataset)
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_paired, opt.height, opt.width)

    extraction_dataset = dataset_test_unpaired
    extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                              dataset=extraction_dataset,
                                              shuffle=False,
                                              num_workers=opt.viton_workers)

    # save_name_unpaired = warped_cloth_root / 'warped_cloths_unpaired' / opt.dataset
    save_name_unpaired = os.path.join(warped_cloth_root, 'warped_cloths_unpaired', opt.dataset)
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_unpaired, opt.height, opt.width)
    os.makedirs(opt.tps_save_final_checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': e + 1,
        'tps': tps.state_dict(),
        'refinement': refinement.state_dict(),
        'optimizer_tps': optimizer_tps.state_dict(),
        'optimizer_ref': optimizer_ref.state_dict(),
    }, opt.tps_save_final_checkpoint)


