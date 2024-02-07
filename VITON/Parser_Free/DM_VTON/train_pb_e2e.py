import datetime
import time
from pathlib import Path
import os
import cupy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from VITON.Parser_Free.DM_VTON.dataloader.viton_dataset import LoadVITONDataset
from VITON.Parser_Free.DM_VTON.losses.tv_loss import TVLoss
from VITON.Parser_Free.DM_VTON.losses.vgg_loss import VGGLoss
from VITON.Parser_Free.DM_VTON.models.generators.res_unet import ResUnetGenerator
from VITON.Parser_Free.DM_VTON.models.warp_modules.style_afwm import StyleAFWM as PBAFWM
# from opt.train_opt import TrainOptions
from VITON.Parser_Free.DM_VTON.utils.general import AverageMeter, print_log
from VITON.Parser_Free.DM_VTON.utils.lr_utils import MyLRScheduler
from VITON.Parser_Free.DM_VTON.utils.torch_utils import get_ckpt, load_ckpt, select_device, smart_optimizer, smart_resume
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)   

def train_batch(
   opt, root_opt, data, models, optimizers, criterions, device, writer, global_step, wandb=None, epoch=0
):
    batch_start_time = time.time()

    warp_model, gen_model = models['warp'], models['gen']
    warp_optimizer, gen_optimizer = optimizers['warp'], optimizers['gen']
    criterionL1, criterionVGG = criterions['L1'], criterions['VGG']

    if root_opt.dataset_name == 'Rail':
        t_mask = torch.FloatTensor(((data['label'] == 3) | (data['label'] == 11)).cpu().numpy().astype(np.int64))
    else:
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))

    data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
    edge = data['edge']
    pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
    clothes = data['color']
    clothes = clothes * pre_clothes_edge
    if root_opt.dataset_name == 'Rail':
        person_clothes_edge = torch.FloatTensor(((data['label'] == 5) | (data['label'] == 6) | (data['label'] == 7)).cpu().numpy().astype(np.int64))
    else:
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
    person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
    real_image = data['image']
    person_clothes = real_image * person_clothes_edge
    pose = data['pose']
    pose_map = data['pose_map']
    size = data['label'].size()
    oneHot_size1 = (size[0], 25, size[2], size[3])
    densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1), device=device).zero_()
    densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
    densepose_fore = data['densepose'] / 24.0
    
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
    preserve_region = face_img + other_clothes_img
    preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)
    concat = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)
    if root_opt.dataset_name == 'Rail':
        arm_mask = torch.FloatTensor(
            (data['label'].cpu().numpy() == 14).astype(np.float64)
        ) + torch.FloatTensor((data['label'].cpu().numpy() == 15).astype(np.float64))
        hand_mask = torch.FloatTensor(
            (data['densepose'].cpu().numpy() == 3).astype(np.int64)
        ) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
    else:
        arm_mask = torch.FloatTensor(
            (data['label'].cpu().numpy() == 11).astype(np.float64)
        ) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float64))
        hand_mask = torch.FloatTensor(
            (data['densepose'].cpu().numpy() == 3).astype(np.int64)
        ) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
    hand_mask = arm_mask * hand_mask
    hand_img = hand_mask * real_image
    dense_preserve_mask = (
        torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int64))
        + torch.FloatTensor(data['densepose'].cpu().numpy() == 22)
    )
    dense_preserve_mask = dense_preserve_mask.to(device) * (1 - person_clothes_edge.to(device))
    preserve_region = face_img + other_clothes_img + hand_img

    with cupy.cuda.Device(int(device.split(':')[-1])):
        flow_out = warp_model(concat.to(device), clothes.to(device), pre_clothes_edge.to(device))
    (
        warped_cloth,
        last_flow,
        cond_fea_all,
        warp_fea_all,
        flow_all,
        delta_list,
        x_all,
        x_edge_all,
        delta_x_all,
        delta_y_all,
    ) = flow_out

    epsilon = opt.epsilon
    loss_smooth = sum([TVLoss(x) for x in delta_list])
    loss_warp = 0

    for num in range(5):
        cur_person_clothes = F.interpolate(
            person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear'
        )
        cur_person_clothes_edge = F.interpolate(
            person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear'
        )
        loss_l1 = criterionL1(x_all[num], cur_person_clothes.to(device))
        loss_vgg = criterionVGG(x_all[num], cur_person_clothes.to(device))
        loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.to(device))
        b, c, h, w = delta_x_all[num].shape
        loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
        loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
        loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
        loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
        loss_second_smooth = loss_flow_x + loss_flow_y
        loss_warp = (
            loss_warp
            + (num + 1) * loss_l1
            + (num + 1) * opt.lambda_loss_vgg * loss_vgg
            + (num + 1) * opt.lambda_loss_edge * loss_edge
            + (num + 1) * opt.lambda_loss_second_smooth * loss_second_smooth
        )

    loss_warp = opt.lambda_loss_smooth * loss_smooth + loss_warp

    warped_prod_edge = x_edge_all[4]
    if root_opt.dataset_name == 'Rail' and epoch >0 :
        binary_mask = (warped_prod_edge > 0.5).float()
        warped_cloth = warped_cloth * binary_mask
    gen_inputs = torch.cat(
        [preserve_region.to(device), warped_cloth, warped_prod_edge, dense_preserve_mask], 1
    )

    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_prod_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    # TUNGPNT2
    loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
    loss_l1 = criterionL1(p_tryon, real_image.to(device))
    loss_vgg = criterionVGG(p_tryon, real_image.to(device))
    bg_loss_l1 = criterionL1(p_rendered, real_image.to(device))
    bg_loss_vgg = criterionVGG(p_rendered, real_image.to(device))
    
    loss_gen = loss_l1 * opt.lambda_loss_l1 + loss_vgg + bg_loss_l1 * opt.lambda_bg_loss_l1 + bg_loss_vgg + loss_mask_l1

    # loss_mask_l1 = criterionL1(person_clothes_edge.to(device), m_composite)
    # loss_l1 = criterionL1(p_tryon, real_image.to(device))
    # loss_vgg = criterionVGG(p_tryon,real_image.to(device))
    # loss_gen = (loss_l1 * 5 + loss_vgg + loss_mask_l1)

    loss_all = opt.lambda_loss_warp * loss_warp + opt.lambda_loss_gen * loss_gen

    warp_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    loss_all.backward()
    warp_optimizer.step()
    gen_optimizer.step()

    train_batch_time = time.time() - batch_start_time

    # Visualize
    if (epoch + 1) % opt.display_count == 0:
        a = real_image.float().to(device)
        b = person_clothes.to(device)
        c = clothes.to(device)
        d = torch.cat(
            [densepose_fore.to(device), densepose_fore.to(device), densepose_fore.to(device)], 1
        )
        e = warped_cloth
        f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
        g = preserve_region.to(device)
        h = torch.cat([dense_preserve_mask, dense_preserve_mask, dense_preserve_mask], 1)
        i = p_rendered
        j = torch.cat([m_composite, m_composite, m_composite], 1)
        k = p_tryon
        combine = torch.cat(
            [a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0], j[0], k[0]], 2
        ).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        writer.add_image('combine', (combine.data + 1) / 2.0, global_step)
        log_losses = {'warping_loss': loss_warp.item() ,'warping_l1': loss_l1.item(),'warping_vgg': loss_vgg.item(),
                      'loss_gen':loss_gen.item(),'composition_loss': loss_all.item()}
        log_images = {'Image': (a[0].cpu() / 2 + 0.5), 
        'Pose Image': (pose_map[0].cpu() / 2 + 0.5), 
        'Clothing': (c[0].cpu() / 2 + 0.5), 
        'Parse Clothing': (b[0].cpu() / 2 + 0.5), 
        'Parse Clothing Mask': person_clothes_edge[0].cpu().expand(3, -1, -1), 
        'Warped Cloth': (e[0].cpu().detach() / 2 + 0.5), 
        'Warped Cloth Mask': f[0].cpu().detach().expand(3, -1, -1),
        "Composition": k[0].cpu() / 2 + 0.5}
        log_results(log_images, log_losses, writer,wandb, epoch, iter_start_time=batch_start_time, train=True)
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(opt.results_dir, f"{global_step}.jpg"), bgr)

    return loss_all.item(), loss_warp.item(), loss_gen.item(), train_batch_time

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

def validate_batch(opt, root_opt,validation_loader,models,criterions,device,writer,global_step,wandb=None,epoch=0):
    warp_model, gen_model = models['warp'], models['gen']
    warp_model.eval()
    gen_model.eval()
    criterionL1, criterionVGG = criterions['L1'], criterions['VGG']
    warping_loss = 0
    warping_l1 = 0
    warping_vgg = 0
    loss_gen =0 
    composition_loss = 0
    for i, data in enumerate(validation_loader):
        if root_opt.dataset_name == 'Rail':
            t_mask = torch.FloatTensor(((data['label'] == 3) | (data['label'] == 11)).cpu().numpy().astype(np.int64))
        else:
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))

        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        if root_opt.dataset_name == 'Rail':
            person_clothes_edge = torch.FloatTensor(((data['label'] == 5) | (data['label'] == 6) | (data['label'] == 7)).cpu().numpy().astype(np.int64))
        else:
            person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        pose_map = data['pose_map']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1), device=device).zero_()
        densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
        densepose_fore = data['densepose'] / 24.0
        
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
        preserve_region = face_img + other_clothes_img
        preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)
        concat = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)
        if root_opt.dataset_name == 'Rail':
            arm_mask = torch.FloatTensor(
                (data['label'].cpu().numpy() == 14).astype(np.float64)
            ) + torch.FloatTensor((data['label'].cpu().numpy() == 15).astype(np.float64))
            hand_mask = torch.FloatTensor(
                (data['densepose'].cpu().numpy() == 3).astype(np.int64)
            ) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
        else:
            arm_mask = torch.FloatTensor(
                (data['label'].cpu().numpy() == 11).astype(np.float64)
            ) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float64))
            hand_mask = torch.FloatTensor(
                (data['densepose'].cpu().numpy() == 3).astype(np.int64)
            ) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
        hand_mask = arm_mask * hand_mask
        hand_img = hand_mask * real_image
        dense_preserve_mask = (
            torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int64))
            + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int64))
            + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int64))
            + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int64))
            + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int64))
            + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int64))
            + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int64))
            + torch.FloatTensor(data['densepose'].cpu().numpy() == 22)
        )
        dense_preserve_mask = dense_preserve_mask.to(device) * (1 - person_clothes_edge.to(device))
        preserve_region = face_img + other_clothes_img + hand_img

        with cupy.cuda.Device(int(device.split(':')[-1])):
            flow_out = warp_model(concat.to(device), clothes.to(device), pre_clothes_edge.to(device))
        (
            warped_cloth,
            last_flow,
            cond_fea_all,
            warp_fea_all,
            flow_all,
            delta_list,
            x_all,
            x_edge_all,
            delta_x_all,
            delta_y_all,
        ) = flow_out

        epsilon = opt.epsilon
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_warp = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(
                person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear'
            )
            cur_person_clothes_edge = F.interpolate(
                person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear'
            )
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.to(device))
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.to(device))
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.to(device))
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_warp = (
                loss_warp
                + (num + 1) * loss_l1
                + (num + 1) * opt.lambda_loss_vgg * loss_vgg
                + (num + 1) * opt.lambda_loss_edge * loss_edge
                + (num + 1) * opt.lambda_loss_second_smooth * loss_second_smooth
            )

        loss_warp = opt.lambda_loss_smooth * loss_smooth + loss_warp

        warped_prod_edge = x_edge_all[4]
        if root_opt.dataset_name == 'Rail' and epoch >0 :
            binary_mask = (warped_prod_edge > 0.5).float()
            warped_cloth = warped_cloth * binary_mask
        gen_inputs = torch.cat(
            [preserve_region.to(device), warped_cloth, warped_prod_edge, dense_preserve_mask], 1
        )

        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_prod_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        # TUNGPNT2
        loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
        loss_l1 = criterionL1(p_tryon, real_image.to(device))
        loss_vgg = criterionVGG(p_tryon, real_image.to(device))
        bg_loss_l1 = criterionL1(p_rendered, real_image.to(device))
        bg_loss_vgg = criterionVGG(p_rendered, real_image.to(device))
        
        loss_gen = loss_l1 * opt.lambda_loss_l1 + loss_vgg + bg_loss_l1 * opt.lambda_bg_loss_l1 + bg_loss_vgg + loss_mask_l1

        loss_all = opt.lambda_loss_warp * loss_warp + opt.lambda_loss_gen * loss_gen
        a = real_image.float().to(device)
        b = person_clothes.to(device)
        c = clothes.to(device)
        d = torch.cat(
            [densepose_fore.to(device), densepose_fore.to(device), densepose_fore.to(device)], 1
        )
        e = warped_cloth
        f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
        g = preserve_region.to(device)
        h = torch.cat([dense_preserve_mask, dense_preserve_mask, dense_preserve_mask], 1)
        i = p_rendered
        j = torch.cat([m_composite, m_composite, m_composite], 1)
        k = p_tryon
        combine = torch.cat(
            [a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0], j[0], k[0]], 2
        ).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)
        warping_loss += loss_warp.item()
        warping_l1 += loss_l1.item()
        warping_vgg += loss_vgg.item()
        loss_gen += loss_gen.item()
        composition_loss += loss_all.item()
    log_losses = {'val_warping_loss': warping_loss / len(validation_loader.dataset) ,'val_warping_l1': warping_l1 / len(validation_loader.dataset) ,'val_warping_vgg': warping_vgg / len(validation_loader.dataset),
                      'val_loss_gen':loss_gen / len(validation_loader.dataset) ,'val_composition_loss': composition_loss / len(validation_loader.dataset) }
    log_images = {'Val/Image': (a[0].cpu() / 2 + 0.5), 
    'Val/Pose Image': (pose_map[0].cpu() / 2 + 0.5), 
    'Val/Clothing': (c[0].cpu() / 2 + 0.5), 
    'Val/Parse Clothing': (b[0].cpu() / 2 + 0.5), 
    'Val/Parse Clothing Mask': person_clothes_edge[0].cpu().expand(3, -1, -1), 
    'Val/Warped Cloth': (e[0].cpu().detach() / 2 + 0.5), 
    'Val/Warped Cloth Mask': f[0].cpu().detach().expand(3, -1, -1),
    "Val/Composition": k[0].cpu() / 2 + 0.5}
    log_results(log_images, log_losses, writer,wandb, global_step, train=False)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(opt.results_dir, 'val', f"{global_step}.jpg"), bgr)

    return loss_all.item(), loss_warp.item(), loss_gen.item()



def make_dirs(opt):
    if not os.path.exists(os.path.join(opt.results_dir, 'val')):
        os.makedirs(os.path.join(opt.results_dir, 'val'))
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    if not os.path.exists(opt.pb_warp_save_step_checkpoint_dir):
        os.makedirs(opt.pb_warp_save_step_checkpoint_dir)
    if not os.path.exists(opt.pb_gen_save_step_checkpoint_dir):
        os.makedirs(opt.pb_gen_save_step_checkpoint_dir)
        
def train_pb_e2e():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    writer = SummaryWriter(opt.tensorboard_dir)
    if sweep_id is not None:
        opt.lambda_loss_second_smooth = wandb.config.lambda_loss_second_smooth
        opt.lambda_loss_vgg = wandb.config.lambda_loss_vgg
        opt.lambda_loss_vgg_skin = wandb.config.lambda_loss_vgg_skin
        opt.lambda_loss_edge = wandb.config.lambda_loss_edge
        opt.lambda_loss_smooth = wandb.config.lambda_loss_smooth
        opt.lambda_loss_l1 = wandb.config.lambda_loss_l1
        opt.lambda_bg_loss_l1 = wandb.config.lambda_bg_loss_l1
        opt.lambda_loss_warp = wandb.config.lambda_loss_warp
        opt.lambda_loss_gen = wandb.config.lambda_loss_gen
        opt.lambda_cond_sup_loss = wandb.config.lambda_cond_sup_loss
        opt.lambda_warp_sup_loss = wandb.config.lambda_warp_sup_loss
        opt.lambda_loss_l1_skin = wandb.config.lambda_loss_l1_skin
        opt.lambda_loss_l1_mask = wandb.config.lambda_loss_l1_mask
        opt.align_corners = wandb.config.align_corners
        opt.optimizer = wandb.config.optimizer
        opt.epsilon = wandb.config.epsilon
        opt.momentum = wandb.config.momentum
        opt.lr = wandb.config.lr
        opt.pb_gen_lr = wandb.config.pb_gen_lr
    epoch_num = opt.niter + opt.niter_decay
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)

    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    with open(log_path, 'w') as file:
        file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    # Device
    device = select_device(opt.device, batch_size=opt.viton_batch_size)

    # Model
    warp_model = PBAFWM(45, opt.align_corners).to(device)
        
    if os.path.exists(opt.pb_warp_load_final_checkpoint):
        warp_ckpt = get_ckpt(opt.pb_warp_load_final_checkpoint)
        load_ckpt(warp_model, warp_ckpt)
        print_log(log_path, f'Load pretrained parser-based warp from {opt.pb_warp_load_final_checkpoint}')
        
    gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d).to(device)    
    if eval(root_opt.load_last_step):
        gen_ckpt = get_ckpt(opt.pb_gen_load_step_checkpoint)
        load_ckpt(gen_model, gen_ckpt)
        print_log(log_path, f'Load pretrained parser-based gen from {opt.pb_gen_load_step_checkpoint}')
    elif os.path.exists(opt.pb_gen_load_final_checkpoint):
        gen_ckpt = get_ckpt(opt.pb_gen_load_final_checkpoint)
        load_ckpt(gen_model, gen_ckpt)
        print_log(log_path, f'Load pretrained parser-based gen from {opt.pb_gen_load_final_checkpoint}')
    

    # Optimizer
    warp_optimizer = smart_optimizer(
        model=warp_model, name=opt.optimizer, lr=opt.pb_gen_lr * opt.lr, momentum=opt.momentum
    )
    gen_optimizer = smart_optimizer(
        model=gen_model, name=opt.optimizer, lr=opt.lr, momentum=opt.momentum
    )

    # Resume
    if opt.resume:
        if warp_ckpt:
            _ = smart_resume(warp_ckpt, warp_optimizer, opt.pb_warp_load_final_checkpoint, epoch_num=epoch_num)
        if gen_ckpt:  # resume with information of gen_model
            start_epoch = smart_resume(
                gen_ckpt, gen_optimizer, opt.pb_gen_load_final_checkpoint, epoch_num=epoch_num
            )
    else:
        start_epoch = 1

    # Scheduler
    last_epoch = start_epoch - 1
    gen_scheduler = MyLRScheduler(gen_optimizer, last_epoch, opt.niter, opt.niter_decay, False)
    warp_scheduler = MyLRScheduler(warp_optimizer, last_epoch, opt.niter, opt.niter_decay, False)

    # Dataloader
    if root_opt.dataset_name == 'Rail':
        dataset_dir = os.path.join(root_opt.root_dir, root_opt.rail_dir)
    else:
        dataset_dir = os.path.join(root_opt.root_dir, root_opt.original_dir)
    train_data = LoadVITONDataset(root_opt, path=dataset_dir, phase='train', size=(256, 192))
    train_dataset, validation_dataset = split_dataset(train_data)
    # train_data.__getitem__(0)
    train_loader = DataLoader(train_data, batch_size=root_opt.viton_batch_size, shuffle=True, num_workers=root_opt.workers)
    validation_loader = DataLoader(validation_dataset, batch_size=opt.viton_batch_size, shuffle=True, num_workers=root_opt.workers)
    # Loss
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(device=device)

    # Start training
    nb = len(train_loader)  # number of batches
    total_steps = epoch_num * nb
    eta_meter = AverageMeter()
    global_step = 1
    t0 = time.time()
    train_warp_loss = 0
    train_gen_loss = 0
    train_loss = 0
    val_warp_loss = 0
    val_gen_loss = 0
    val_loss = 0
    steps_warp_loss = 0
    steps_gen_loss = 0
    steps_loss = 0
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        for idx, data in enumerate(train_loader):  # batch -----------------------------------------
            warp_model.train()
            gen_model.train()
            loss_all, loss_warp, loss_gen, train_batch_time = train_batch(
                opt, root_opt,
                data,
                models={'warp': warp_model, 'gen': gen_model},
                optimizers={'warp': warp_optimizer, 'gen': gen_optimizer},
                criterions={'L1': criterionL1, 'VGG': criterionVGG},
                device=device,
                writer=writer,
                global_step=global_step,
                wandb=wandb,
                epoch=epoch
            )
            train_warp_loss += loss_warp
            train_gen_loss += loss_gen
            train_loss += loss_all
            steps_warp_loss += loss_warp
            steps_gen_loss += loss_gen
            steps_loss += loss_all

            # Logging
            eta_meter.update(train_batch_time)
            now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
            if global_step % opt.print_step == 0:
                eta_sec = ((epoch_num + 1 - epoch) * len(train_loader) - idx - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                strs = '[{}]: [epoch-{}/{}]--[global_step-{}/{}-{:.2%}]--[loss-{:.6f}: warp-{:.6f}, gen-{:.6f}]--[lr: warp-{}, gen-{}]--[eta-{}]'.format(  # noqa: E501
                    now,
                    epoch,
                    epoch_num,
                    global_step,
                    total_steps,
                    global_step / total_steps,
                    steps_loss / opt.display_count,
                    steps_warp_loss / opt.display_count,
                    steps_gen_loss / opt.display_count,
                    ['%.6f' % group['lr'] for group in warp_optimizer.param_groups],
                    ['%.6f' % group['lr'] for group in gen_optimizer.param_groups],
                    eta_sec_format,
                )
                print_log(log_path, strs)

                steps_warp_loss = 0
                steps_gen_loss = 0
                steps_loss = 0
            break
        if epoch % opt.val_count == 0:
            val_loss_all, val_loss_warp, val_loss_gen= validate_batch(
                opt, root_opt,
                validation_loader,
                models={'warp': warp_model, 'gen': gen_model},
                criterions={'L1': criterionL1, 'VGG': criterionVGG},
                device=device,
                writer=writer,
                global_step=global_step,
                wandb=wandb,
                epoch=epoch
            )
            
            global_step += 1
            # break
            # end batch ---------------------------------------------------------------------------
        
        # Scheduler
        warp_scheduler.step()
        gen_scheduler.step()

        # Save model
        warp_ckpt = {
            'epoch': epoch,
            'model': warp_model.state_dict(),
            'optimizer': warp_optimizer.state_dict(),
        }
        gen_ckpt = {
            'epoch': epoch,
            'model': gen_model.state_dict(),
            'optimizer': gen_optimizer.state_dict(),
        }
        if epoch % opt.save_period == 0:
            torch.save(warp_ckpt, opt.pb_warp_save_step_checkpoint % epoch)
            torch.save(gen_ckpt, opt.pb_gen_save_step_checkpoint % epoch)
            print_log(
                log_path,
                'Save the model at the end of epoch %d, iters %d' % (epoch, global_step - 1),
            )
        # del warp_ckpt, gen_ckpt

        print_log(
            log_path,
            'End of epoch %d / %d: train_loss: %.3f \t time: %d sec'
            % (epoch, opt.niter + opt.niter_decay, train_loss, time.time() - epoch_start_time),
        )

        train_warp_loss = 0
        train_gen_loss = 0
        train_loss = 0
        
        val_warp_loss = 0
        val_gen_loss = 0
        val_loss = 0
        # break
        # end epoch -------------------------------------------------------------------------------
    # end training --------------------------------------------------------------------------------
    torch.save(warp_ckpt, opt.pb_warp_save_final_checkpoint)
    torch.save(gen_ckpt, opt.pb_gen_save_final_checkpoint)
    print_log(
        log_path,
        (f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.'),
    )
    print_log(log_path, f'Results are saved at {opt.results_dir}')

    with torch.cuda.device(device):
        torch.cuda.empty_cache()

def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)   


import argparse
def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    
    root_opt.warp_experiment_from_run = root_opt.warp_experiment_from_run.format(root_opt.warp_experiment_from_number, root_opt.warp_run_from_number)
    root_opt.gen_experiment_from_run = root_opt.gen_experiment_from_run.format(root_opt.gen_experiment_from_number, root_opt.gen_run_from_number)
    
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
    root_opt.warp_experiment_from_dir = root_opt.warp_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.warp_load_from_model)
    root_opt.warp_experiment_from_dir = os.path.join(root_opt.warp_experiment_from_dir, "PB_Warp")
    
    root_opt.gen_experiment_from_dir = root_opt.gen_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gen_load_from_model)
    root_opt.gen_experiment_from_dir = os.path.join(root_opt.gen_experiment_from_dir, "PB_Gen")
    
    
    return root_opt

def get_root_opt_checkpoint_dir(parser, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    parser.pb_warp_save_step_checkpoint_dir = parser.pb_warp_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_step_checkpoint = os.path.join(parser.pb_warp_save_step_checkpoint_dir, parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_step_checkpoint.split("/")[:-1])
    
    parser.pb_warp_save_final_checkpoint_dir = parser.pb_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_final_checkpoint = os.path.join(parser.pb_warp_save_final_checkpoint_dir, parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_final_checkpoint.split("/")[:-1])
    
    parser.pb_warp_load_final_checkpoint_dir = parser.pb_warp_load_final_checkpoint_dir.format(root_opt.experiment_from_run, root_opt.warp_experiment_from_dir)
    parser.pb_warp_load_final_checkpoint = os.path.join(parser.pb_warp_load_final_checkpoint_dir, parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        parser.pb_warp_load_step_checkpoint_dir = parser.pb_warp_load_step_checkpoint_dir.format(root_opt.warp_experiment_from_run, root_opt.warp_experiment_from_dir)
    else:
        parser.pb_warp_load_step_checkpoint_dir = parser.pb_warp_load_step_checkpoint_dir.format(root_opt.warp_experiment_from_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_load_step_checkpoint_dir = fix(parser.pb_warp_load_step_checkpoint_dir)
    if not last_step:
        parser.pb_warp_load_step_checkpoint = os.path.join(parser.pb_warp_load_step_checkpoint_dir, parser.pb_warp_load_step_checkpoint)
    else:
        if os.path.isdir(parser.pb_warp_load_step_checkpoint_dir.format(root_opt.warp_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(parser.pb_warp_load_step_checkpoint_dir.format(root_opt.warp_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "warp" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            parser.pb_warp_load_step_checkpoint = os.path.join(parser.pb_warp_load_step_checkpoint_dir, last_step)
    parser.pb_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_step_checkpoint)
    parser.pb_warp_load_step_checkpoint_dir = os.path.join("/",*parser.pb_warp_load_step_checkpoint.split("/")[:-1])

    
    # gen
    parser.pb_gen_save_step_checkpoint_dir = parser.pb_gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_step_checkpoint = os.path.join(parser.pb_gen_save_step_checkpoint_dir, parser.pb_gen_save_step_checkpoint)
    parser.pb_gen_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_step_checkpoint)
    parser.pb_gen_save_step_checkpoint_dir = os.path.join("/",*parser.pb_gen_save_step_checkpoint.split("/")[:-1])
    
    
    parser.pb_gen_save_final_checkpoint_dir = parser.pb_gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_final_checkpoint = os.path.join(parser.pb_gen_save_final_checkpoint_dir, parser.pb_gen_save_final_checkpoint)
    parser.pb_gen_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_final_checkpoint)
    parser.pb_gen_save_final_checkpoint_dir = os.path.join("/",*parser.pb_gen_save_final_checkpoint.split("/")[:-1])
    
    parser.pb_gen_load_final_checkpoint_dir = parser.pb_gen_load_final_checkpoint_dir.format(root_opt.experiment_from_run, root_opt.warp_experiment_from_dir)
    parser.pb_gen_load_final_checkpoint = os.path.join(parser.pb_gen_load_final_checkpoint_dir, parser.pb_gen_load_final_checkpoint)
    parser.pb_gen_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_final_checkpoint)
    parser.pb_gen_load_final_checkpoint_dir = os.path.join("/",*parser.pb_gen_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        parser.pb_gen_load_step_checkpoint_dir = parser.pb_gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.gen_experiment_from_dir)
    else:
        parser.pb_gen_load_step_checkpoint_dir = parser.pb_gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_load_step_checkpoint_dir = fix(parser.pb_gen_load_step_checkpoint_dir)
    if not last_step:
        parser.pb_gen_load_step_checkpoint = os.path.join(parser.pb_gen_load_step_checkpoint_dir, parser.pb_gen_load_step_checkpoint)
    else:
        if os.path.isdir(parser.pb_gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(parser.pb_gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "gen" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            parser.pb_gen_load_step_checkpoint = os.path.join(parser.pb_gen_load_step_checkpoint_dir, last_step)
    parser.pb_gen_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_step_checkpoint)
    parser.pb_gen_load_step_checkpoint_dir = os.path.join("/",*parser.pb_gen_load_step_checkpoint.split("/")[:-1])
    return parser 


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
    parser.save_period = root_opt.save_period
    parser.val_count = root_opt.val_count
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

import yaml 
def train_pb_e2e_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            train_pb_e2e()
            
def train_pb_e2e_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,train_pb_e2e_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        train_pb_e2e()
    else:
        wandb = None
        train_pb_e2e()


def split_dataset(dataset,train_size=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, validation_indices = train_test_split(indices, train_size=train_size)
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    return train_subset, validation_subset

