import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
from VITON.Parser_Based.SD_VITON.networks import make_grid as mkgrid
from VITON.Parser_Based.SD_VITON.networks import make_grid_3d as mkgrid_3d
import argparse
import os
import time
from dataset import FashionDataLoader, FashionNeRFDataset
from VITON.Parser_Based.SD_VITON.networks import ConditionGenerator, VGGLoss, GANLoss, load_checkpoint, save_checkpoint, define_D
from tqdm import tqdm
from VITON.Parser_Based.SD_VITON.utils import *
from VITON.Parser_Based.HR_VITON.utils import condition_process_opt

fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None
to = lambda inputs,name,i: inputs[name][i].permute(1,2,0).cpu().numpy()
to2 = lambda inputs,name1,name2, i: inputs[name1][name2][i].permute(1,2,0).cpu().numpy()
to3 = lambda inputs,i: inputs[i].permute(1,2,0).detach().cpu().numpy()

def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)   

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def train_model(opt, train_loader, validation_loader, board, tocg, D, wandb=None):
    # Model
    tocg.cuda()
    tocg.train()
    D.cuda()
    D.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    if opt.fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor)

    # optimizer
    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))


    for step in tqdm(range(opt.niter + opt.niter_decay)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input1
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        openpose = inputs['pose'].cuda()
        # GT
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['pcm'].cuda()  # L1
        im_c = inputs['parse_cloth'].cuda()  # VGG
        # visualization
        im = inputs['image'].cuda()
        # tucked-out shirts style
        lower_clothes_mask = inputs['lower_clothes_mask'].cuda()
        clothes_no_loss_mask = inputs['clothes_no_loss_mask'].cuda()

        # inputs
        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        # forward
        flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)

        # warped cloth mask one hot         
        warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()

        # fake segmap cloth channel * warped clothmask
        cloth_mask = torch.ones_like(fake_segmap.detach())
        cloth_mask[:, 3:4, :, :] = warped_clothmask_paired_taco
        fake_segmap = fake_segmap * cloth_mask

        if opt.occlusion:
            warped_clothmask_paired_taco = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired_taco)
            warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_paired_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_paired_taco)

            warped_clothmask_paired_tvob = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired_tvob)
            warped_cloth_paired_tvob = warped_cloth_paired_tvob * warped_clothmask_paired_tvob + torch.ones_like(warped_cloth_paired_tvob) * (1-warped_clothmask_paired_tvob)            
        
        if opt.clip_warping:
            warped_cloth_paired_taco = warped_cloth_paired_taco * parse_cloth_mask + torch.ones_like(warped_cloth_paired_taco) * (1 - parse_cloth_mask)
            warped_cloth_paired_tvob = warped_cloth_paired_tvob * parse_cloth_mask + torch.ones_like(warped_cloth_paired_tvob) * (1 - parse_cloth_mask)
        # generated fake cloth mask & misalign mask
        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = fake_clothmask - warped_clothmask_paired_taco_onehot
        misalign[misalign < 0.0] = 0.0
        
        # loss warping
        loss_l1_cloth = criterionL1(warped_clothmask_paired_taco, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired_taco, im_c)

        ## Eq.8 & Eq.9 of SD-VITON
        inv_lower_clothes_mask = lower_clothes_mask * clothes_no_loss_mask
        inv_lower_clothes_mask = 1. - inv_lower_clothes_mask
        loss_l1_cloth += criterionL1(warped_clothmask_paired_tvob*inv_lower_clothes_mask, parse_cloth_mask*inv_lower_clothes_mask)
        loss_vgg += criterionVGG(warped_cloth_paired_tvob*inv_lower_clothes_mask, im_c*inv_lower_clothes_mask)

        ## Eq.12 of SD-VITON
        roi_mask = torch.nn.functional.interpolate(parse_cloth_mask, scale_factor=0.5, mode='nearest')
        non_roi_mask = 1. - roi_mask

        flow_taco = flow_list_taco[-1]
        z_gt_non_roi = -1
        z_gt_roi = 1
        z_src_coordinate = -1
        z_dist_loss_non_roi = (torch.abs(z_src_coordinate + flow_taco[:,0:1,:,:,2] + z_gt_non_roi) * non_roi_mask).mean()
        z_dist_loss_roi = (torch.abs(z_src_coordinate + flow_taco[:,0:1,:,:,2] + z_gt_roi) * roi_mask).mean()
        
        loss_tv_tvob = 0
        loss_tv_taco = 0
        if not opt.lasttvonly:
            for flow in flow_list_taco:
                y_tv = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).mean()
                loss_tv_taco = loss_tv_taco + y_tv + x_tv 

            for flow in flow_list_tvob:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv_tvob = loss_tv_tvob + y_tv + x_tv
        else:
            for flow in flow_list_taco[-1:]:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv_taco = loss_tv_taco + y_tv + x_tv

            for flow in flow_list_tvob[-1:]:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv_tvob = loss_tv_tvob + y_tv + x_tv


        N, _, iH, iW = c_paired.size()
        # Intermediate flow loss
        if opt.interflowloss:
            layers_max_idx = len(flow_list_tvob)-1
            for i in range(len(flow_list_tvob)-1):
                flow = flow_list_tvob[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW)
                grid_3d = mkgrid_3d(N, iH, iW)
                
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size = c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)

                ## Eq.8 & Eq.9 of SD-VITON
                loss_l1_cloth += criterionL1(warped_cm*inv_lower_clothes_mask, parse_cloth_mask*inv_lower_clothes_mask) / (2 ** (layers_max_idx-i))
                loss_vgg += criterionVGG(warped_c*inv_lower_clothes_mask, im_c*inv_lower_clothes_mask) / (2 ** (layers_max_idx-i))


        # loss segmentation
        # generator
        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        fake_segmap_softmax = torch.softmax(fake_segmap, 1)
        pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
        loss_G_GAN = criterionGAN(pred_segmap, True)
        
        # discriminator
        fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
        real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
        loss_D_fake = criterionGAN(fake_segmap_pred, False)
        loss_D_real = criterionGAN(real_segmap_pred, True)

        # loss sum
        loss_G = (opt.loss_l1_cloth_lambda * loss_l1_cloth + loss_vgg + opt.tvlambda_tvob * loss_tv_tvob + opt.tvlambda_taco * loss_tv_taco) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda) + z_dist_loss_non_roi + z_dist_loss_roi
        loss_D = loss_D_fake + loss_D_real

        # step
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()    
        # display
        if (step + 1) % opt.display_count == 0:

            a_0 = c_paired[0].cuda()
            b_0 = im[0].cuda()
            c_0 = warped_cloth_paired_tvob[0]
            d_0 = warped_cloth_paired_taco[0]
            
            e_0 = lower_clothes_mask
            e_0 = torch.cat((e_0[0],e_0[0],e_0[0]), dim=0) 

            f_0 = densepose[0].cuda()

            g_0 = clothes_no_loss_mask
            g_0 = torch.cat((g_0[0],g_0[0],g_0[0]), dim=0) 

            h_0 = lower_clothes_mask*clothes_no_loss_mask
            h_0 = torch.cat((h_0[0],h_0[0],h_0[0]), dim=0) 

            i_0 = inv_lower_clothes_mask
            i_0 = torch.cat((i_0[0],i_0[0],i_0[0]), dim=0) 

            combine = torch.cat((a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0), dim=2)
            log_losses = {'warping_loss': loss_G.item(),    
            'warping_l1': loss_l1_cloth.item(),'warping_vgg': loss_vgg.item(),'warping_cross_entropy_loss': CE_loss.item(),
            'TV_tvob':loss_tv_tvob.item(),
            'TV_taco':loss_tv_taco.item(),
            'warping_cross_entropy_loss': CE_loss.item(), 
            'gan': loss_G_GAN.item(),
            'discriminator':loss_D.item(), 
            'pred_real': loss_D_real.item(), 
            'pred_fake':loss_D_fake.item(), 
            'z_non_roi':z_dist_loss_non_roi,'z_roi': z_dist_loss_roi
            }
            log_images = {
            'Image': (b_0.cpu() / 2 + 0.5),
            'Pose Image': (openpose[0].cpu() / 2 + 0.5),
            'Clothing': (a_0.cpu() / 2 + 0.5),
            'Parse Clothing': (im_c[0].cpu() / 2 + 0.5),
            'Parse Clothing Mask': parse_cloth_mask[0].cpu().expand(3, -1, -1),
            'Warped Cloth TVOB': (c_0.cpu().detach() / 2 + 0.5),
            'Warped Cloth TACO': (d_0.cpu().detach() / 2 + 0.5),
            'Warped Cloth Mask TVOB': warped_clothmask_paired_tvob[0].cpu().detach().expand(3, -1, -1),
            'Warped Cloth Mask TACO': warped_clothmask_paired_taco[0].cpu().detach().expand(3, -1, -1)}
            log_results(log_images, log_losses, board,wandb, step, iter_start_time=iter_start_time, train=True)
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.results_dir, f'warped_cloth_paired_{step}.png'),bgr)
        if (step + 1) % opt.val_count == 0:
            validate_tocg(opt, step, tocg,D, validation_loader,board,wandb)
            tocg.train()    
        if (step + 1) % opt.save_period == 0:
            save_checkpoint(tocg,opt.tocg_save_step_checkpoint % (step + 1))
            save_checkpoint(D,opt.tocg_discriminator_save_step_checkpoint % (step + 1))

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
        my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth TVOB','Warped Cloth TACO','Warped Cloth Mask TVOB','Warped Cloth Mask TACO'])
        my_table.add_data(*wandb_images)
        wandb.log({table: my_table, **log_losses})
    if train and iter_start_time is not None:
        t = time.time() - iter_start_time
        print("training step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV_tvob loss: %.4f, TV_taco loss: %.4f, CE: %.4f, G GAN: %.4f\n z_non_roi: %.4f, z_roi: %.4f"
      % (step + 1, t, 
         log_losses['warping_loss'], 
         log_losses['warping_l1'], 
         log_losses['warping_vgg'], 
         log_losses['TV_tvob'], 
         log_losses['TV_taco'], 
         log_losses['warping_cross_entropy_loss'],
         log_losses['gan'],
         log_losses['z_non_roi'],log_losses['z_roi']), flush=True)
    else:
        print("validation step: %8d, \nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV_tvob loss: %.4f, TV_taco loss: %.4f, CE: %.4f, G GAN: %.4f\n z_non_roi: %.4f, z_roi: %.4f"
      % (step + 1, 
         log_losses['val_warping_loss'], 
         log_losses['val_warping_l1'], 
         log_losses['val_warping_vgg'], 
         log_losses['val_TV_tvob'], 
         log_losses['val_TV_taco'], 
         log_losses['val_warping_cross_entropy_loss'],
         log_losses['val_gan'],
         log_losses['val_z_non_roi'],log_losses['val_z_roi']), flush=True)
        
def validate_tocg(opt, step, tocg,D, validation_loader,board,wandb):
    # Model
    tocg.eval()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    if opt.fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor)
    val_warping_loss =  0
    val_warping_l1 = 0
    val_warping_vgg= 0 
    val_warping_cross_entropy_loss =0
    val_TV_tvob =0
    val_TV_taco =0
    val_warping_cross_entropy_loss =0
    val_gan =0
    val_z_non_roi =0
    val_z_roi =0
    total_batches = len(validation_loader.dataset) // opt.viton_batch_size
    processed_batches = 0
    with torch.no_grad():
        while processed_batches < total_batches:
            inputs = validation_loader.next_batch()

            # input1
            c_paired = inputs['cloth']['paired'].cuda()
            cm_paired = inputs['cloth_mask']['paired'].cuda()
            cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
            # input2
            parse_agnostic = inputs['parse_agnostic'].cuda()
            densepose = inputs['densepose'].cuda()
            openpose = inputs['pose'].cuda()
            # GT
            label_onehot = inputs['parse_onehot'].cuda()  # CE
            label = inputs['parse'].cuda()  # GAN loss
            parse_cloth_mask = inputs['pcm'].cuda()  # L1
            im_c = inputs['parse_cloth'].cuda()  # VGG
            # visualization
            im = inputs['image']
            # tucked-out shirts style
            lower_clothes_mask = inputs['lower_clothes_mask'].cuda()
            clothes_no_loss_mask = inputs['clothes_no_loss_mask'].cuda()

            # inputs
            input1 = torch.cat([c_paired, cm_paired], 1)
            input2 = torch.cat([parse_agnostic, densepose], 1)

            # forward
            flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)

            # warped cloth mask one hot         
            warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()

            # fake segmap cloth channel * warped clothmask
            cloth_mask = torch.ones_like(fake_segmap.detach())
            cloth_mask[:, 3:4, :, :] = warped_clothmask_paired_taco
            fake_segmap = fake_segmap * cloth_mask

            if opt.occlusion:
                warped_clothmask_paired_taco = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired_taco)
                warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_paired_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_paired_taco)

                warped_clothmask_paired_tvob = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired_tvob)
                warped_cloth_paired_tvob = warped_cloth_paired_tvob * warped_clothmask_paired_tvob + torch.ones_like(warped_cloth_paired_tvob) * (1-warped_clothmask_paired_tvob)            
            
            # generated fake cloth mask & misalign mask
            fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
            misalign = fake_clothmask - warped_clothmask_paired_taco_onehot
            misalign[misalign < 0.0] = 0.0
            
            # loss warping
            loss_l1_cloth = criterionL1(warped_clothmask_paired_taco, parse_cloth_mask)
            loss_vgg = criterionVGG(warped_cloth_paired_taco, im_c)

            ## Eq.8 & Eq.9 of SD-VITON
            inv_lower_clothes_mask = lower_clothes_mask * clothes_no_loss_mask
            inv_lower_clothes_mask = 1. - inv_lower_clothes_mask
            loss_l1_cloth += criterionL1(warped_clothmask_paired_tvob*inv_lower_clothes_mask, parse_cloth_mask*inv_lower_clothes_mask)
            loss_vgg += criterionVGG(warped_cloth_paired_tvob*inv_lower_clothes_mask, im_c*inv_lower_clothes_mask)

            ## Eq.12 of SD-VITON
            roi_mask = torch.nn.functional.interpolate(parse_cloth_mask, scale_factor=0.5, mode='nearest')
            non_roi_mask = 1. - roi_mask

            flow_taco = flow_list_taco[-1]
            z_gt_non_roi = -1
            z_gt_roi = 1
            z_src_coordinate = -1
            z_dist_loss_non_roi = (torch.abs(z_src_coordinate + flow_taco[:,0:1,:,:,2] + z_gt_non_roi) * non_roi_mask).mean()
            z_dist_loss_roi = (torch.abs(z_src_coordinate + flow_taco[:,0:1,:,:,2] + z_gt_roi) * roi_mask).mean()
            
            loss_tv_tvob = 0
            loss_tv_taco = 0
            if not opt.lasttvonly:
                for flow in flow_list_taco:
                    y_tv = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).mean()
                    loss_tv_taco = loss_tv_taco + y_tv + x_tv 

                for flow in flow_list_tvob:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv_tvob = loss_tv_tvob + y_tv + x_tv
            else:
                for flow in flow_list_taco[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv_taco = loss_tv_taco + y_tv + x_tv

                for flow in flow_list_tvob[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv_tvob = loss_tv_tvob + y_tv + x_tv


            N, _, iH, iW = c_paired.size()
            # Intermediate flow loss
            if opt.interflowloss:
                layers_max_idx = len(flow_list_tvob)-1
                for i in range(len(flow_list_tvob)-1):
                    flow = flow_list_tvob[i]
                    N, fH, fW, _ = flow.size()
                    grid = mkgrid(N, iH, iW)
                    grid_3d = mkgrid_3d(N, iH, iW)
                    
                    flow = F.interpolate(flow.permute(0, 3, 1, 2), size = c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                    warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                    warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                    warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)

                    ## Eq.8 & Eq.9 of SD-VITON
                    loss_l1_cloth += criterionL1(warped_cm*inv_lower_clothes_mask, parse_cloth_mask*inv_lower_clothes_mask) / (2 ** (layers_max_idx-i))
                    loss_vgg += criterionVGG(warped_c*inv_lower_clothes_mask, im_c*inv_lower_clothes_mask) / (2 ** (layers_max_idx-i))


            # loss segmentation
            # generator
            CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
            fake_segmap_softmax = torch.softmax(fake_segmap, 1)
            pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
            loss_G_GAN = criterionGAN(pred_segmap, True)
            
            # loss sum
            loss_G = (opt.loss_l1_cloth_lambda * loss_l1_cloth + loss_vgg + opt.tvlambda_tvob * loss_tv_tvob + opt.tvlambda_taco * loss_tv_taco) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda) + z_dist_loss_non_roi + z_dist_loss_roi

            a_0 = c_paired[0].cuda()
            b_0 = im[0].cuda()
            c_0 = warped_cloth_paired_tvob[0]
            d_0 = warped_cloth_paired_taco[0]
            
            e_0 = lower_clothes_mask
            e_0 = torch.cat((e_0[0],e_0[0],e_0[0]), dim=0) 

            f_0 = densepose[0].cuda()

            g_0 = clothes_no_loss_mask
            g_0 = torch.cat((g_0[0],g_0[0],g_0[0]), dim=0) 

            h_0 = lower_clothes_mask*clothes_no_loss_mask
            h_0 = torch.cat((h_0[0],h_0[0],h_0[0]), dim=0) 

            i_0 = inv_lower_clothes_mask
            i_0 = torch.cat((i_0[0],i_0[0],i_0[0]), dim=0)
            val_warping_loss += loss_G.item()
            val_warping_l1 += loss_l1_cloth.item()
            val_warping_vgg += loss_vgg.item()
            val_warping_cross_entropy_loss += CE_loss.item()
            val_TV_tvob += loss_tv_tvob.item()
            val_TV_taco += loss_tv_taco.item()
            val_gan +=  loss_G_GAN.item()
            val_z_non_roi += z_dist_loss_non_roi.item()
            val_z_roi += z_dist_loss_roi.item()
            processed_batches += 1
            combine = torch.cat((a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0), dim=2)
        log_losses = {'val_warping_loss': val_warping_loss / len(validation_loader.dataset),    
        'val_warping_l1': val_warping_l1 / len(validation_loader.dataset),'val_warping_vgg': val_warping_vgg / len(validation_loader.dataset),
        'val_warping_cross_entropy_loss': val_warping_cross_entropy_loss / len(validation_loader.dataset),
        'val_TV_tvob': val_TV_tvob / len(validation_loader.dataset),
        'val_TV_taco': val_TV_taco / len(validation_loader.dataset),
        'val_gan': val_gan / len(validation_loader.dataset),
        'val_z_non_roi':val_z_non_roi / len(validation_loader.dataset),'val_z_roi': val_z_roi / len(validation_loader.dataset)
        }
        log_images = {
        'Val/Image': (b_0.cpu() / 2 + 0.5),
        'Val/Pose Image': (openpose[0].cpu() / 2 + 0.5),
        'Val/Clothing': (a_0.cpu() / 2 + 0.5),
        'Val/Parse Clothing': (im_c[0].cpu() / 2 + 0.5),
        'Val/Parse Clothing Mask': parse_cloth_mask[0].cpu().expand(3, -1, -1),
        'Val/Warped Cloth TVOB': (c_0.cpu().detach() / 2 + 0.5),
        'Val/Warped Cloth TACO': (d_0.cpu().detach() / 2 + 0.5),
        'Val/Warped Cloth Mask TVOB': warped_clothmask_paired_tvob[0].cpu().detach().expand(3, -1, -1),
        'Val/Warped Cloth Mask TACO': warped_clothmask_paired_taco[0].cpu().detach().expand(3, -1, -1)}
        log_results(log_images, log_losses, board,wandb, step, train=False)
        cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
        rgb=(cv_img*255).astype(np.uint8)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(opt.results_dir, f'warped_cloth_paired_{step}.png'),bgr)

        
            
            
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
       
       
def _train_sd_viton_tocg_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_sd_viton_tocg_()
            
def train_sd_viton_tocg_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = condition_process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_sd_viton_tocg_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_sd_viton_tocg_()
    else:
        wandb = None
        _train_sd_viton_tocg_()
         
def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)     
            
def _train_sd_viton_tocg_():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    board = SummaryWriter(log_dir = opt.tensorboard_dir)
    torch.cuda.set_device(opt.device)
    if sweep_id is not None:
        opt.lr = wandb.config.lr
        opt.momentum = wandb.config.momentum
        opt.segment_anything = wandb.config.segment_anything
        opt.flow_self_attention = wandb.config.flow_self_attention
        opt.flow_spatial_attention = wandb.config.flow_spatial_attention
        opt.flow_channel_attention = wandb.config.flow_channel_attention
        opt.feature_pyramid_self_attention = wandb.config.feature_pyramid_self_attention
        opt.feature_pyramid_spatial_attention = wandb.config.feature_pyramid_spatial_attention
        opt.feature_pyramid_channel_attention = wandb.config.feature_pyramid_channel_attention
        opt.G_lr = wandb.config.G_lr
        opt.D_lr = wandb.config.D_lr
        opt.CElamda = wandb.config.CElamda
        opt.GANlambda = wandb.config.GANlambda
        opt.loss_l1_cloth_lambda = wandb.config.loss_l1_cloth_lambda
        opt.occlusion = wandb.config.occlusion
        opt.norm_G = wandb.config.norm_G
        opt.num_D = wandb.config.num_D
        opt.init_type = wandb.config.init_type
        opt.num_upsampling_layers = wandb.config.num_upsampling_layers
        opt.lambda_l1 = wandb.config.lambda_l1
        opt.lambda_vgg = wandb.config.lambda_vgg
        opt.lambda_feat = wandb.config.lambda_feat

    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    with open(log_path, 'w') as file:
        file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
            
    train_dataset = FashionNeRFDataset(root_opt, opt, viton=True, model='viton')
    train_loader = FashionDataLoader(train_dataset, root_opt.viton_batch_size, root_opt.viton_workers, True)

    test_dataset = FashionNeRFDataset(root_opt, opt, viton=True, mode='test', model='viton')
    test_loader = FashionDataLoader(test_dataset, opt.num_test_visualize, 1, False)
    validation_dataset = Subset(test_dataset, np.arange(50))
    validation_loader = FashionDataLoader(validation_dataset, opt.num_test_visualize, root_opt.viton_workers, False)
    
    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=opt.cond_G_ngf, norm_layer=nn.BatchNorm2d, num_layers=opt.cond_G_num_layers) # num_layers: training condition network w/ fine_height 256 -> 5, - w/ fine_height 512 -> 6, - w/ fine_height 1024 -> 7
    
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=(opt.cond_G_num_layers-2), spectral = opt.spectral, num_D = opt.num_D) # n_layers_D: training condition network w/ fine_height 256 -> 3, - w/ fine_height 512 -> 4, - w/ fine_height 1024 -> 5

    # Load Checkpoint
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    if last_step:
        load_checkpoint(tocg, opt.tocg_load_step_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.tocg_load_step_checkpoint}')
        
        load_checkpoint(D, opt.tocg_discriminator_load_step_checkpoint)
        print_log(log_path, f'Load pretrained discimrinator model from {opt.tocg_discriminator_load_step_checkpoint}')
    elif os.path.exists(opt.tocg_load_final_checkpoint):
        load_checkpoint(tocg, opt.tocg_load_final_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.tocg_load_final_checkpoint}')
        
        load_checkpoint(D, opt.tocg_discriminator_load_final_checkpoint)
        print_log(log_path, f'Load pretrained dicriminator model from {opt.tocg_discriminator_load_final_checkpoint}')

    # Train
    # train(opt, train_loader, tocg, D)
    train_model(opt, train_loader, validation_loader, board, tocg, D, wandb=wandb)
    # Save Checkpoint
    if wandb is not None:
        wandb.finish()
    save_checkpoint(tocg,opt.tocg_save_final_checkpoint, opt)
    save_checkpoint(D,opt.tocg_discriminator_save_final_checkpoint , opt)
    print("Finished training !" )
