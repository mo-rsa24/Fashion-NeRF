# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import os
import time
from VITON.Parser_Based.CP_VTON_plus.cp_dataset import CPDataset, CPDataLoader
from VITON.Parser_Based.CP_VTON_plus.networks import GicLoss, GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from VITON.Parser_Based.CP_VTON_plus.utils import process_opt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from VITON.Parser_Based.CP_VTON_plus.visualization import board_add_image, board_add_images

opt,root_opt,wandb,sweep_id =None, None, None,None

def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)
            
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     


def train_cpvton_plus_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_cpvton_plus_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_cpvton_plus_()
    else:
        wandb = None
        _train_cpvton_plus_()
    
def _train_cpvton_plus_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_cpvton_plus_()
      
def split_dataset(dataset,train_size=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, validation_indices = train_test_split(indices, train_size=train_size)
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    return train_subset, validation_subset      
            
def make_dirs(opt):
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    if not os.path.exists(opt.tom_save_final_checkpoint_dir):
        print("dir: ", opt.tom_save_final_checkpoint_dir)
        os.makedirs(opt.tom_save_final_checkpoint_dir)    
    if not os.path.exists(opt.gmm_save_step_checkpoint_dir):
        os.makedirs(opt.gmm_save_step_checkpoint_dir)
    if not os.path.exists(opt.tom_save_step_checkpoint_dir):
        os.makedirs(opt.tom_save_step_checkpoint_dir)
    if not os.path.exists(opt.gmm_save_final_checkpoint_dir):
        os.makedirs(opt.gmm_save_final_checkpoint_dir)

def _train_cpvton_plus_():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    board = SummaryWriter(log_dir = opt.tensorboard_dir)
    if sweep_id is not None:
        opt.niter = wandb.config.niter
        opt.niter_decay = wandb.config.niter_decay
        opt.lr = wandb.config.lr
        opt.init_type = wandb.config.init_type
        opt.Lgic = wandb.config.Lgic
        
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    # print("Start to train stage: %s" % opt.VITON_Name)
    log_path = os.path.join(opt.results_dir, 'log.txt')
    with open(log_path, 'w') as file:
        file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    # create dataset
    train_dataset = CPDataset(root_opt, opt)
    train_dataset, validation_dataset = split_dataset(train_dataset)
    
    train_loader = CPDataLoader(opt, train_dataset)
    validation_loader = CPDataLoader(opt, validation_dataset)
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
        if last_step:
            load_checkpoint(model, opt.gmm_load_step_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.gmm_load_step_checkpoint}')
        elif os.path.exists(opt.gmm_load_final_checkpoint):
            load_checkpoint(model, opt.gmm_load_final_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.gmm_load_final_checkpoint}')
        train_gmm(opt, train_loader, validation_loader, model, board, wandb=wandb)
        save_checkpoint(model,opt.gmm_save_final_checkpoint)
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(
            26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        gmm_model = GMM(opt)
        last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
        if last_step:
            load_checkpoint(gmm_model, opt.gmm_load_step_checkpoint)
            print_log(log_path, f'Load pretrained gmm_model from {opt.gmm_load_step_checkpoint}')
        elif os.path.exists(opt.gmm_load_final_checkpoint):
            load_checkpoint(gmm_model, opt.gmm_load_final_checkpoint)
            print_log(log_path, f'Load pretrained gmm_model from {opt.gmm_load_final_checkpoint}')
            
        last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
        if last_step:
            load_checkpoint(model, opt.tom_load_step_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.tom_load_step_checkpoint}')
        elif os.path.exists(opt.tom_load_final_checkpoint):
            load_checkpoint(model, opt.tom_load_final_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.tom_load_final_checkpoint}')
        train_tom(opt, train_loader, gmm_model, model, board, wandb=wandb)
        save_checkpoint(model, opt.tom_save_final_checkpoint)
    
    print('Finished training named:' )


def train_gmm(opt, train_loader, validation_loader, model, board, wandb=None):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for step in tqdm(range(opt.niter + opt.niter_decay)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()
        im_g = inputs['grid_image'].cuda()

        grid, theta = model(agnostic, cm)    # can be added c too for new training
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        if opt.clip_warping:
            warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        # loss for warped cloth
        Lwarp = criterionL1(warped_cloth, im_c)    # changing to previous code as it corresponds to the working code
        Lgic = gicloss(grid)
        # 200x200 = 40.000 * 0.001
        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

        loss = Lwarp + opt.Lgic * Lgic    # total GMM loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            log_images = {
                'Image': im[0].cpu().detach() / 2 + 0.5,
                'Pose Image': im_pose[0].cpu().detach() / 2 + 0.5,
                'Clothing': c[0].cpu().detach() / 2 + 0.5,
                'Parse Clothing': im_c[0].cpu().detach() / 2 + 0.5,
                'Parse Clothing Mask': im_cm[0].cpu().expand(3, -1, -1),
                'Warped Cloth': warped_cloth[0].cpu().detach() / 2 + 0.5,
                'Warped Cloth Mask': warped_mask[0].cpu().detach() / 2 + 0.5,
            }
            log_losses = {'warping_loss':loss, 'warping_l1':Lwarp}
            log_results(log_images,log_losses, board,wandb, step, iter_start_time=iter_start_time, train=True)
            # break
        if (step + 1) % opt.val_count == 0:
            validate_gmm(validation_loader, model, board,step, wandb=wandb)
            model.train()
        if (step+1) % opt.save_period == 0:
            t = time.time() - iter_start_time
            print('Saving checkpoint: %8d, time: %.3f, checkpoint: %s' % (step+1, t, opt.gmm_save_step_checkpoint % (step+1)), flush=True)
            save_checkpoint(model, opt.gmm_save_step_checkpoint % (step+1))

    
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
        my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth','Warped Cloth Mask'])
        my_table.add_data(*wandb_images)
        wandb.log({table: my_table, **log_losses})
    if train and iter_start_time is not None:
        t = time.time() - iter_start_time
        print('training step: %8d, time: %.3f, loss: %4f' % (step+1, t, log_losses['warping_loss']), flush=True)
    else:
        print('validation step: %8d, loss: %4f' % (step+1, log_losses['val_warping_loss']), flush=True)
  
def log_tom_results(log_images, log_losses,board,wandb, step,iter_start_time=None,train=True):
    table = 'Table' if train else 'Val_Table'
    for key,value in log_losses.items():
        board.add_scalar(key, value, step+1)
    wandb_images = []
    for key,value in log_images.items():
        board.add_image(key, value, step+1)
        if wandb is not None:
            wandb_images.append(get_wandb_image(value, wandb=wandb))
    if wandb is not None:
        my_table = wandb.Table(columns=['Image','Pose Image','Parse Clothing','Warped Cloth','Warped Cloth Mask','Rendered Image','Composition'])
        my_table.add_data(*wandb_images)
        wandb.log({table: my_table, **log_losses})
    if train and iter_start_time is not None:
        t = time.time() - iter_start_time
        print('training step: %8d, time: %.3f, composition_loss: %.4f, l1_composition_loss: %.4f, vgg_composition_loss: %.4f, mask_composition_loss: %.4f' 
                    % (step+1, t, log_losses['composition_loss'], log_losses['l1_composition_loss'], log_losses['vgg_composition_loss'], log_losses['mask_composition_loss']), flush=True)
    else:
        print('validation step: %8d, composition_loss: %.4f, l1_composition_loss: %.4f, vgg_composition_loss: %.4f, mask_composition_loss: %.4f' 
                    % (step+1, log_losses['val_composition_loss'], log_losses['val_l1_composition_loss'], log_losses['val_vgg_composition_loss'], log_losses['val_mask_composition_loss']), flush=True)  
def validate_gmm(validation_loader,model, board, step, wandb=wandb):
    model.cuda()
    model.eval()
    # criterion
    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)
    total_batches = len(validation_loader.dataset) // opt.viton_batch_size
    processed_batches = 0
    iter_start_time = time.time()
    val_loss = 0
    val_Lwarp = 0
    with torch.no_grad():
        while processed_batches < total_batches:
            inputs = validation_loader.next_batch()
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image'].cuda()
            im_h = inputs['head'].cuda()
            shape = inputs['shape'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_c = inputs['parse_cloth'].cuda()
            im_cm = inputs['parse_cloth_mask'].cuda()
            im_g = inputs['grid_image'].cuda()

            grid, theta = model(agnostic, cm)    # can be added c too for new training
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
            if opt.clip_warping:
                warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
            visuals = [[im_h, shape, im_pose],
                        [c, warped_cloth, im_c],
                        [warped_grid, (warped_cloth+im)*0.5, im]]

            # loss for warped cloth
            Lwarp = criterionL1(warped_cloth, im_c)    # changing to previous code as it corresponds to the working code
            # Actual loss function as in the paper given below (comment out previous line and uncomment below to train as per the paper)
            # grid regularization loss
            Lgic = gicloss(grid)
            # 200x200 = 40.000 * 0.001
            Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

            loss = Lwarp + 40 * Lgic    # total GMM loss
            val_loss += loss.item()
            val_Lwarp += Lwarp.item()
            processed_batches += 1
        val_loss = val_loss / len(validation_loader.dataset)  
        val_Lwarp = val_Lwarp / len(validation_loader.dataset)  
        log_losses = {'val_warping_loss':val_loss, 'val_warping_l1':val_Lwarp}
        log_images = {
                    'Val/Image': im[0].cpu().detach() / 2 + 0.5,
                    'Val/Pose Image': im_pose[0].cpu().detach() / 2 + 0.5,
                    'Val/Clothing': c[0].cpu().detach() / 2 + 0.5,
                    'Val/Parse Clothing': im_c[0].cpu().detach() / 2 + 0.5,
                    'Val/Parse Clothing Mask': im_cm[0].cpu().expand(3, -1, -1),
                    'Val/Warped Cloth': warped_cloth[0].cpu().detach() / 2 + 0.5,
                    'Val/Warped Cloth Mask': warped_mask[0].cpu().detach() / 2 + 0.5,
                }
        log_results(log_images, log_losses, board,wandb, step, train=False)
    
def train_tom(opt, train_loader, validation_loader,  gmm_model, model, board, wandb=None):
    model.cuda()
    model.train()

    gmm_model.cuda()
    gmm_model.eval()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))


    for step in range(opt.niter + opt.niter_decay):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        im_c =  inputs['parse_cloth'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()

        grid, theta = gmm_model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        
        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, warped_cloth, warped_mask], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        """visuals = [[im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]"""  # CP-VTON

        visuals = [[im_h, shape, im_pose],
                   [warped_cloth, pcm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]  # CP-VTON+
        if opt.clip_warping:
            warped_cloth = warped_cloth * pcm + torch.ones_like(warped_cloth) * (1 - pcm)
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        # loss_mask = criterionMask(m_composite, cm)  # CP-VTON
        loss_mask = criterionMask(m_composite, pcm)  # CP-VTON+
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            log_images = {'Image', im[0].cpu().detach() / 2 + 0.5,
              'Pose Image', im_pose[0].cpu().detach() / 2 + 0.5,
              'Parse Clothing', im_c[0].cpu().detach() / 2 + 0.5,
              'Warped Cloth', warped_cloth[0].cpu().detach() / 2 + 0.5,
              'Warped Cloth Mask', (warped_mask*2-1)[0],
              'Rendered Image', p_rendered[0].cpu().detach() / 2 + 0.5,
              'Composition', p_tryon[0].cpu().detach() / 2 + 0.5}
            log_losses = {'composition_loss', loss.item(),'l1_composition_loss', loss_l1.item(),
            'vgg_composition_loss', loss_vgg.item(),'mask_composition_loss', loss_mask.item()}
            log_tom_results(log_images,log_losses, board,wandb, step,iter_start_time=iter_start_time,train=True)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step+1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        if (step + 1) % opt.val_count == 0:
            validate_tom(validation_loader, model,gmm_model, board, step, wandb=wandb)
            model.train()
        if (step+1) % opt.save_period == 0:
            print_log(os.path.join(opt.results_dir, 'log.txt'), f'Save pretrained model to {opt.tom_save_step_checkpoint}')
            save_checkpoint(model, opt.tom_save_step_checkpoint % (step+1))
            
def validate_tom(validation_loader,model,gmm_model,board, step, wandb=wandb):
    model.cuda()
    model.eval()

    gmm_model.cuda()
    gmm_model.eval()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    total_batches = len(validation_loader.dataset) // opt.viton_batch_size
    processed_batches = 0
    val_composition_loss = 0
    val_l1_composition_loss = 0
    val_vgg_composition_loss = 0
    val_mask_composition_loss = 0
    with torch.no_grad():
        while processed_batches < total_batches:
            inputs = validation_loader.next_batch()

            im = inputs['image'].cuda()
            im_pose = inputs['pose_image']
            im_h = inputs['head']
            shape = inputs['shape']
            im_c =  inputs['parse_cloth'].cuda()
            im_cm =  inputs['parse_cloth_mask'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            pcm = inputs['parse_cloth_mask'].cuda()

            grid, theta = gmm_model(agnostic, cm)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
            
            # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
            outputs = model(torch.cat([agnostic, warped_cloth, warped_mask], 1))  # CP-VTON+
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            """visuals = [[im_h, shape, im_pose],
                    [c, cm*2-1, m_composite*2-1],
                    [p_rendered, p_tryon, im]]"""  # CP-VTON

            visuals = [[im_h, shape, im_pose],
                    [warped_cloth, pcm*2-1, m_composite*2-1],
                    [p_rendered, p_tryon, im]]  # CP-VTON+

            loss_l1 = criterionL1(p_tryon, im)
            loss_vgg = criterionVGG(p_tryon, im)
            # loss_mask = criterionMask(m_composite, cm)  # CP-VTON
            loss_mask = criterionMask(m_composite, pcm)  # CP-VTON+
            loss = loss_l1 + loss_vgg + loss_mask
            loss = loss_l1 + loss_vgg + loss_mask
            val_composition_loss += loss
            val_l1_composition_loss += loss_l1
            val_vgg_composition_loss += loss_vgg
            val_mask_composition_loss += loss_mask
            if opt.clip_warping:
                warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
            processed_batches += 1
        log_images = {'Val/Image', im[0].cpu().detach() / 2 + 0.5,
            'Val/Pose Image', im_pose[0].cpu().detach() / 2 + 0.5,
            'Val/Parse Clothing', im_c[0].cpu().detach() / 2 + 0.5,
            'Val/Warped Cloth', warped_cloth[0].cpu().detach() / 2 + 0.5,
            'Val/Warped Cloth Mask', (warped_mask*2-1)[0],
            'Val/Rendered Image', p_rendered[0].cpu().detach() / 2 + 0.5,
            'Val/Composition', p_tryon[0].cpu().detach() / 2 + 0.5}
        log_losses = {'val_composition_loss', val_composition_loss / len(validation_loader.dataset),'val_l1_composition_loss', val_l1_composition_loss / len(validation_loader.dataset),
        'val_vgg_composition_loss', val_vgg_composition_loss / len(validation_loader.dataset),'val_mask_composition_loss', val_mask_composition_loss / len(validation_loader.dataset)}
        log_results(log_images,log_losses, board,wandb, step,train=False)
    
    

