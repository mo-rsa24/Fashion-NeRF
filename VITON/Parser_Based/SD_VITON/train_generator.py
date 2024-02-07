import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
from tensorboardX import SummaryWriter
import argparse
import os
import time
from networks import ConditionGenerator, VGGLoss, load_checkpoint, save_checkpoint, make_grid, make_grid_3d
from network_generator import SPADEGenerator, MultiscaleDiscriminator, GANLoss, Projected_GANs_Loss, set_requires_grad
from dataset import FashionDataLoader, FashionNeRFDataset
from sync_batchnorm import DataParallelWithCallback
from utils import create_network
import sys
from VITON.Parser_Based.HR_VITON.utils import generator_process_opt
from tqdm import tqdm

import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import transforms
import eval_models as models
import torchgeometry as tgm

from pg_modules.discriminator import ProjectedDiscriminator
import cv2
fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None
def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     


def train(opt, train_loader,validation_loader, test_loader,board,gen, generator, discriminator, model, wandb=None):
    """
        Train Generator
    """

    # Model
    tocg.cuda()
    tocg.eval()
    generator.train()
    discriminator.train()
    if not opt.composition_mask:
        discriminator.feature_network.requires_grad_(False)
    discriminator.cuda()
    model.eval()

    # criterion
    criterionGAN = None
    if opt.fp16:
        if opt.composition_mask:
            criterionGAN = GANLoss('hinge', tensor=torch.cuda.HalfTensor)
        else:
            criterionGAN = Projected_GANs_Loss(tensor=torch.cuda.HalfTensor)
    else:
        if opt.composition_mask:
            criterionGAN = GANLoss('hinge', tensor=torch.cuda.FloatTensor)
        else:
            criterionGAN = Projected_GANs_Loss(tensor=torch.cuda.FloatTensor)
    
    criterionL1 = nn.L1Loss()
    criterionFeat = nn.L1Loss()
    criterionVGG = VGGLoss()

    # optimizer
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=opt.G_lr, betas=(0, 0.9))
    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.D_lr, betas=(0, 0.9))
    scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))

    if opt.fp16:
        from apex import amp
        [tocg, generator, discriminator], [optimizer_gen, optimizer_dis] = amp.initialize(
            [tocg, generator, discriminator], [optimizer_gen, optimizer_dis], opt_level='O1', num_losses=2)
        
    if len(opt.gpu_ids) > 0:
        tocg = DataParallelWithCallback(tocg, device_ids=opt.gpu_ids)
        generator = DataParallelWithCallback(generator, device_ids=opt.gpu_ids)
        discriminator = DataParallelWithCallback(discriminator, device_ids=opt.gpu_ids)
        criterionGAN = DataParallelWithCallback(criterionGAN, device_ids=opt.gpu_ids)
        criterionFeat = DataParallelWithCallback(criterionFeat, device_ids=opt.gpu_ids)
        criterionVGG = DataParallelWithCallback(criterionVGG, device_ids=opt.gpu_ids)
        criterionL1 = DataParallelWithCallback(criterionL1, device_ids=opt.gpu_ids)
        
    upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()

    for step in tqdm(range(opt.niter + opt.niter_decay)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input
        agnostic = inputs['agnostic'].cuda()
        parse_GT = inputs['parse'].cuda()
        pose = inputs['densepose'].cuda()
        parse_cloth = inputs['parse_cloth'].cuda()
        parse_agnostic = inputs['parse_agnostic'].cuda()
        pcm = inputs['pcm'].cuda()
        cm = inputs['cloth_mask']['paired'].cuda()
        c_paired = inputs['cloth']['paired'].cuda()
        openpose = inputs['pose'].cuda()
        
        # target
        im = inputs['image'].cuda()

        with torch.no_grad():

            # Warping Cloth
            # down
            pre_clothes_mask_down = F.interpolate(cm, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            clothes_down = F.interpolate(c_paired, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
            densepose_down = F.interpolate(pose, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
            
            # multi-task inputs
            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

            # forward
            flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)
            
            # warped cloth mask one hot 
            warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
            
            # fake segmap cloth channel * warped clothmask
            cloth_mask = torch.ones_like(fake_segmap)
            cloth_mask[:,3:4, :, :] = warped_clothmask_paired_taco
            fake_segmap = fake_segmap * cloth_mask
                    
            # warped cloth
            N, _, iH, iW = c_paired.shape
            N, flow_iH, flow_iW, _ = flow_list_tvob[-1].shape

            flow_tvob = F.interpolate(flow_list_tvob[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
            flow_tvob_norm = torch.cat([flow_tvob[:, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_tvob[:, :, :, 1:2] / ((flow_iH - 1.0) / 2.0)], 3)

            grid = make_grid(N, iH, iW)
            grid_3d = make_grid_3d(N, iH, iW)

            warped_grid_tvob = grid + flow_tvob_norm
            warped_cloth_tvob = F.grid_sample(c_paired, warped_grid_tvob, padding_mode='border')
            warped_clothmask_tvob = F.grid_sample(cm, warped_grid_tvob, padding_mode='border')

            flow_taco = F.interpolate(flow_list_taco[-1].permute(0, 4, 1, 2, 3), size=(2,iH,iW), mode='trilinear').permute(0, 2, 3, 4, 1)
            flow_taco_norm = torch.cat([flow_taco[:, :, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_taco[:, :, :, :, 1:2] / ((flow_iH - 1.0) / 2.0), flow_taco[:, :, :, :, 2:3]], 4)
            warped_cloth_tvob = warped_cloth_tvob.unsqueeze(2)
            warped_cloth_paired_taco = F.grid_sample(torch.cat((warped_cloth_tvob, torch.zeros_like(warped_cloth_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
            warped_cloth_paired_taco = warped_cloth_paired_taco[:,:,0,:,:]

            warped_clothmask_tvob = warped_clothmask_tvob.unsqueeze(2)
            warped_clothmask_taco = F.grid_sample(torch.cat((warped_clothmask_tvob, torch.zeros_like(warped_clothmask_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
            warped_clothmask_taco = warped_clothmask_taco[:,:,0,:,:]

            # make generator input parse map
            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

            # occlusion
            if opt.occlusion:
                warped_clothmask_taco = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask_taco)
                warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_taco)
                warped_cloth_paired_taco = warped_cloth_paired_taco.detach()
                
            old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
            old_parse.scatter_(1, fake_parse, 1.0)
            if opt.clip_warping:
                warped_cloth_paired_taco = warped_cloth_paired_taco * pcm + torch.ones_like(warped_cloth_paired_taco) * (1 - pcm)
                warped_cloth_paired_tvob = warped_cloth_paired_tvob * pcm + torch.ones_like(warped_cloth_paired_tvob) * (1 - pcm)
            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]
                    
            parse = parse.detach()

        # --------------------------------------------------------------------------------------------------------------
        #                                              Train the generator
        # --------------------------------------------------------------------------------------------------------------
        G_losses = {}
        # GANs w/ composition mask
        if opt.composition_mask:
            output_paired_rendered, output_paired_comp = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
            output_paired_comp1 = output_paired_comp * warped_clothmask_taco
            output_paired_comp = parse[:,2:3,:,:] * output_paired_comp1
            output_paired = warped_cloth_paired_taco * output_paired_comp + output_paired_rendered * (1 - output_paired_comp)

            fake_concat = torch.cat((parse, output_paired_rendered), dim=1)
            real_concat = torch.cat((parse, im), dim=1)
            pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

            # the prediction contains the intermediate outputs of multiscale GAN,
            # so it's usually a list
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
            
            # Adversarial loss
            G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False)

            # Feature matching loss
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

            # VGG perceptual loss
            G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg + criterionVGG(output_paired_rendered, im) * opt.lambda_vgg

            # Image-level L1 loss
            G_losses['L1'] = criterionL1(output_paired_rendered, im) * opt.lambda_l1 + criterionL1(output_paired, im) * opt.lambda_l1

            # Composition mask loss
            G_losses['Composition_Mask'] = torch.mean(torch.abs(1 - output_paired_comp))

            loss_gen = sum(G_losses.values()).mean()

        # GANs w/o composition mask & w/ projected discriminator
        else:
            set_requires_grad(discriminator, False)
            output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)

            pred_fake, feats_fake = discriminator(output_paired)
            pred_real, feats_real = discriminator(im)

            # Adversarial loss
            G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False) * 0.5

            # Feature matching loss
            num_D = len(feats_fake)
            GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(feats_fake[i])
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = criterionFeat(feats_fake[i][j], feats_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

            # VGG perceptual loss
            G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg

            # Image-level L1 loss
            G_losses['L1'] = criterionL1(output_paired, im) * opt.lambda_l1

            loss_gen = sum(G_losses.values()).mean()

        optimizer_gen.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_gen, optimizer_gen, loss_id=0) as loss_gen_scaled:
                loss_gen_scaled.backward()
        else:
            loss_gen.backward()
        optimizer_gen.step()

        # --------------------------------------------------------------------------------------------------------------
        #                                            Train the discriminator
        # --------------------------------------------------------------------------------------------------------------
        D_losses = {}
        # GANs w/ composition mask
        if opt.composition_mask:
            with torch.no_grad():            
                output_paired_rendered, output_comp = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)

                output_comp1 = output_comp * warped_clothmask_taco
                output_comp = parse[:,2:3,:,:] * output_comp1
                output = warped_cloth_paired_taco * output_comp + output_paired_rendered * (1 - output_comp)

                output_comp = output_comp.detach()
                output = output.detach()
                output_comp.requires_grad_()
                output.requires_grad_()

            fake_concat = torch.cat((parse, output_paired_rendered), dim=1)
            real_concat = torch.cat((parse, im), dim=1)
            pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

            # the prediction contains the intermediate outputs of multiscale GAN,
            # so it's usually a list
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
            
            # Adversarial loss
            D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
            D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)
            
            loss_dis = sum(D_losses.values()).mean()

        # GANs w/o composition mask & w/ projected discriminator
        else:
            set_requires_grad(discriminator, True)
            discriminator.module.feature_network.requires_grad_(False)

            with torch.no_grad():
                output = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
                output = output.detach()
                output.requires_grad_()

            pred_fake, _ = discriminator(output)
            pred_real, _ = discriminator(im)

            # Adversarial loss
            D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
            D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)

            loss_dis = sum(D_losses.values()).mean()

        optimizer_dis.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_dis, optimizer_dis, loss_id=1) as loss_dis_scaled:
                loss_dis_scaled.backward()
        else:
            loss_dis.backward()
        optimizer_dis.step()
        
        if not opt.composition_mask:
            set_requires_grad(discriminator, False)
        if (step + 1) % opt.val_count == 0:
            validate_gen(opt, step, generator, discriminator, gen,gauss, criterionL1,criterionFeat,criterionGAN,criterionVGG,validation_loader,board,wandb)
            generator.train()
        if (step + 1) % opt.display_count == 0:
            a_0 = im.cuda()[0]
            b_0 = output.cuda()[0]
            c_0 = warped_cloth_paired_taco.cuda()[0]
            combine = torch.cat((a_0, b_0, c_0), dim=2)
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            i = 0
            out = output[i].cpu() / 2 + 0.5
            board.add_images('train_images', grid.unsqueeze(0), step + 1)
            log_images = {'Image': (im[0].cpu() / 2 + 0.5),
            'Pose Image': (openpose[0].cpu() / 2 + 0.5),
            'Parse Clothing': (parse_cloth[0].cpu() / 2 + 0.5),
            'Parse Clothing Mask': pcm[0].cpu().expand(3, -1, -1),
            'Warped Cloth TVOB': (warped_cloth_paired_tvob[0].cpu().detach() / 2 + 0.5),
            'Warped Cloth TACO': (c_0.cpu().detach() / 2 + 0.5),
            'Warped Cloth Mask TVOB': warped_clothmask_paired_tvob[0].cpu().detach().expand(3, -1, -1),
            'Warped Cloth Mask TACO': warped_clothmask_paired_taco[0].cpu().detach().expand(3, -1, -1),
            'Composition': out}
            log_losses = {'composition_loss': loss_gen.item(),
            'gan_composition_loss': G_losses['GAN'].mean().item(),
            'feat_composition_loss': G_losses['GAN_Feat'].mean().item(),
            'vgg_composition_loss': G_losses['VGG'].mean().item(),
            'l1_composition_loss' : G_losses['L1'].mean().item(),
            'composition_loss_disc': loss_dis.item(),
            'fake_composition_loss_disc': D_losses['D_Fake'].mean().item(),
            'real_composition_loss_disc': D_losses['D_Real'].mean().item()}
            log_results(log_images, log_losses, board,wandb, step, iter_start_time=iter_start_time, train=True)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.results_dir, f'out_{step}.png'),bgr)
            

        # --------------------------------------------------------------------------------------------------------------
        #                                            Evaluate the generator
        # --------------------------------------------------------------------------------------------------------------
        if (step + 1) % opt.display_count == 0:
            generator.eval()
            T2 = transforms.Compose([transforms.Resize((128, 128))])
            lpips_list = []
            avg_distance = 0.0
            
            with torch.no_grad():
                print("LPIPS")
                for i in tqdm(range(500)):
                    inputs = test_loader.next_batch()
                    # input
                    agnostic = inputs['agnostic'].cuda()
                    parse_GT = inputs['parse'].cuda()
                    pose = inputs['densepose'].cuda()
                    parse_cloth = inputs['parse_cloth'].cuda()
                    parse_agnostic = inputs['parse_agnostic'].cuda()
                    pcm = inputs['pcm'].cuda()
                    cm = inputs['cloth_mask']['paired'].cuda()
                    c_paired = inputs['cloth']['paired'].cuda()
                    
                    # target
                    im = inputs['image'].cuda()
                                
                    with torch.no_grad():
                        # Warping Cloth
                        # down
                        pre_clothes_mask_down = F.interpolate(cm, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                        input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                        clothes_down = F.interpolate(c_paired, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
                        densepose_down = F.interpolate(pose, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
                        
                        # multi-task inputs
                        input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                        input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                        # forward
                        flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)
                        
                        # warped cloth mask one hot 
                        warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                        
                        cloth_mask = torch.ones_like(fake_segmap)
                        cloth_mask[:,3:4, :, :] = warped_clothmask_paired_taco
                        fake_segmap = fake_segmap * cloth_mask
                                
                        # warped cloth
                        N, _, iH, iW = c_paired.shape
                        N, flow_iH, flow_iW, _ = flow_list_tvob[-1].shape

                        flow_tvob = F.interpolate(flow_list_tvob[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                        flow_tvob_norm = torch.cat([flow_tvob[:, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_tvob[:, :, :, 1:2] / ((flow_iH - 1.0) / 2.0)], 3)

                        grid = make_grid(N, iH, iW)
                        grid_3d = make_grid_3d(N, iH, iW)

                        warped_grid_tvob = grid + flow_tvob_norm
                        warped_cloth_tvob = F.grid_sample(c_paired, warped_grid_tvob, padding_mode='border')
                        warped_clothmask_tvob = F.grid_sample(cm, warped_grid_tvob, padding_mode='border')

                        flow_taco = F.interpolate(flow_list_taco[-1].permute(0, 4, 1, 2, 3), size=(2, iH, iW), mode='trilinear').permute(0, 2, 3, 4, 1)
                        flow_taco_norm = torch.cat([flow_taco[:, :, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_taco[:, :, :, :, 1:2] / ((flow_iH - 1.0) / 2.0), flow_taco[:, :, :, :, 2:3]], 4)
                        warped_cloth_tvob = warped_cloth_tvob.unsqueeze(2)
                        warped_cloth_paired_taco = F.grid_sample(torch.cat((warped_cloth_tvob, torch.zeros_like(warped_cloth_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
                        warped_cloth_paired_taco = warped_cloth_paired_taco[:,:,0,:,:]

                        warped_clothmask_tvob = warped_clothmask_tvob.unsqueeze(2)
                        warped_clothmask_taco = F.grid_sample(torch.cat((warped_clothmask_tvob, torch.zeros_like(warped_clothmask_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
                        warped_clothmask_taco = warped_clothmask_taco[:,:,0,:,:]

                        # make generator input parse map
                        fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                        fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                        # occlusion
                        if opt.occlusion:
                            warped_clothmask_taco = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask_taco)
                            warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_taco)
                            warped_cloth_paired_taco = warped_cloth_paired_taco.detach()
                            
                        old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
                        old_parse.scatter_(1, fake_parse, 1.0)

                        labels = {
                            0:  ['background',  [0]],
                            1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                            2:  ['upper',       [3]],
                            3:  ['hair',        [1]],
                            4:  ['left_arm',    [5]],
                            5:  ['right_arm',   [6]],
                            6:  ['noise',       [12]]
                        }
                        parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                        for i in range(len(labels)):
                            for label in labels[i][1]:
                                parse[:, i] += old_parse[:, label]
                                
                        parse = parse.detach()
                    
                    # GANs w/ composition mask
                    if opt.composition_mask:
                        output_paired_rendered, output_paired_comp = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
                        
                        output_paired_comp1 = output_paired_comp * warped_clothmask_taco
                        output_paired_comp = parse[:,2:3,:,:] * output_paired_comp1
                        output_paired = warped_cloth_paired_taco * output_paired_comp + output_paired_rendered * (1 - output_paired_comp)

                    # GANs w/o composition mask & w/ projected discriminator
                    else:
                        output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)

                    avg_distance += model.forward(T2(im), T2(output_paired))
                    
            avg_distance = avg_distance / 500
            print(f"LPIPS{avg_distance}")
                
            generator.train()

        if (step + 1) % opt.save_period == 0:
            save_checkpoint(generator,opt.gen_save_step_checkpoint % (step + 1))    
            save_checkpoint(discriminator,opt.gen_discriminator_save_step_checkpoint % (step + 1))

        if (step + 1) % opt.save_period == 0:
            scheduler_gen.step()
            scheduler_dis.step()

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
        my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth TVOB','Warped Cloth TACO','Warped Cloth Mask TVOB','Warped Cloth Mask TACO','Composition'])
        my_table.add_data(*wandb_images)
        wandb.log({table: my_table, **log_losses})
    if train and iter_start_time is not None:
        t = time.time() - iter_start_time
        print("training step: %8d, time: %.3f\n composition_loss: %.4f, gan_composition_loss: %.4f, feat_composition_loss: %.4f, vgg_composition_loss: %.4f composition_loss_disc: %.4f, fake_composition_loss_disc: %.4f\n real_composition_loss_disc: %.4f"
                % (step + 1, t, log_losses['composition_loss'], log_losses['gan_composition_loss'], log_losses['feat_composition_loss'], log_losses['vgg_composition_loss'], log_losses['composition_loss_disc'], log_losses['fake_composition_loss_disc'], log_losses['real_composition_loss_disc']), flush=True)
    else:
        print("validation step: %8d, \n composition_loss: %.4f, gan_composition_loss: %.4f, feat_composition_loss: %.4f, vgg_composition_loss: %.4f"
                % (step + 1, log_losses['val_composition_loss'], log_losses['val_gan_composition_loss'], log_losses['val_feat_composition_loss'], log_losses['val_vgg_composition_loss']), flush=True)
        
def validate_gen(opt, step, generator,discriminator, tocg,gauss,criterionL1, criterionFeat,criterionGAN,criterionVGG, validation_loader,board,wandb):
    generator.eval()
    T2 = transforms.Compose([transforms.Resize((128, 128))])
    lpips_list = []
    avg_distance = 0.0
    total_batches = len(validation_loader.dataset) // opt.viton_batch_size
    processed_batches = 0
    composition_loss = 0
    gan_composition_loss = 0
    feat_composition_loss = 0
    vgg_composition_loss = 0
    with torch.no_grad():
        while processed_batches < total_batches:
            inputs = validation_loader.next_batch()

            # input
            agnostic = inputs['agnostic'].cuda()
            parse_GT = inputs['parse'].cuda()
            pose = inputs['densepose'].cuda()
            parse_cloth = inputs['parse_cloth'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            pcm = inputs['pcm'].cuda()
            cm = inputs['cloth_mask']['paired'].cuda()
            c_paired = inputs['cloth']['paired'].cuda()
            openpose = inputs['pose'].cuda()
            
            # target
            im = inputs['image'].cuda()

            with torch.no_grad():

                # Warping Cloth
                # down
                pre_clothes_mask_down = F.interpolate(cm, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
                clothes_down = F.interpolate(c_paired, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
                densepose_down = F.interpolate(pose, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
                
                # multi-task inputs
                input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                # forward
                flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)
                
                # warped cloth mask one hot 
                warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                
                # fake segmap cloth channel * warped clothmask
                cloth_mask = torch.ones_like(fake_segmap)
                cloth_mask[:,3:4, :, :] = warped_clothmask_paired_taco
                fake_segmap = fake_segmap * cloth_mask
                        
                # warped cloth
                N, _, iH, iW = c_paired.shape
                N, flow_iH, flow_iW, _ = flow_list_tvob[-1].shape

                flow_tvob = F.interpolate(flow_list_tvob[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                flow_tvob_norm = torch.cat([flow_tvob[:, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_tvob[:, :, :, 1:2] / ((flow_iH - 1.0) / 2.0)], 3)

                grid = make_grid(N, iH, iW)
                grid_3d = make_grid_3d(N, iH, iW)

                warped_grid_tvob = grid + flow_tvob_norm
                warped_cloth_tvob = F.grid_sample(c_paired, warped_grid_tvob, padding_mode='border')
                warped_clothmask_tvob = F.grid_sample(cm, warped_grid_tvob, padding_mode='border')

                flow_taco = F.interpolate(flow_list_taco[-1].permute(0, 4, 1, 2, 3), size=(2,iH,iW), mode='trilinear').permute(0, 2, 3, 4, 1)
                flow_taco_norm = torch.cat([flow_taco[:, :, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_taco[:, :, :, :, 1:2] / ((flow_iH - 1.0) / 2.0), flow_taco[:, :, :, :, 2:3]], 4)
                warped_cloth_tvob = warped_cloth_tvob.unsqueeze(2)
                warped_cloth_paired_taco = F.grid_sample(torch.cat((warped_cloth_tvob, torch.zeros_like(warped_cloth_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
                warped_cloth_paired_taco = warped_cloth_paired_taco[:,:,0,:,:]

                warped_clothmask_tvob = warped_clothmask_tvob.unsqueeze(2)
                warped_clothmask_taco = F.grid_sample(torch.cat((warped_clothmask_tvob, torch.zeros_like(warped_clothmask_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
                warped_clothmask_taco = warped_clothmask_taco[:,:,0,:,:]

                # make generator input parse map
                fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                # occlusion
                if opt.occlusion:
                    warped_clothmask_taco = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask_taco)
                    warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_taco)
                    warped_cloth_paired_taco = warped_cloth_paired_taco.detach()
                    
                old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
                old_parse.scatter_(1, fake_parse, 1.0)

                labels = {
                    0:  ['background',  [0]],
                    1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                    2:  ['upper',       [3]],
                    3:  ['hair',        [1]],
                    4:  ['left_arm',    [5]],
                    5:  ['right_arm',   [6]],
                    6:  ['noise',       [12]]
                }
                parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                for i in range(len(labels)):
                    for label in labels[i][1]:
                        parse[:, i] += old_parse[:, label]
                        
                parse = parse.detach()

            # --------------------------------------------------------------------------------------------------------------
            #                                              Train the generator
            # --------------------------------------------------------------------------------------------------------------
            G_losses = {}
            # GANs w/ composition mask
            if opt.composition_mask:
                output_paired_rendered, output_paired_comp = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)
                output_paired_comp1 = output_paired_comp * warped_clothmask_taco
                output_paired_comp = parse[:,2:3,:,:] * output_paired_comp1
                output_paired = warped_cloth_paired_taco * output_paired_comp + output_paired_rendered * (1 - output_paired_comp)
                pred_fake = []
                pred_real = []            
                # Adversarial loss
                G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False)

                # Feature matching loss
                num_D = len(pred_fake)
                GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

                # VGG perceptual loss
                G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg + criterionVGG(output_paired_rendered, im) * opt.lambda_vgg

                # Image-level L1 loss
                G_losses['L1'] = criterionL1(output_paired_rendered, im) * opt.lambda_l1 + criterionL1(output_paired, im) * opt.lambda_l1

                # Composition mask loss
                G_losses['Composition_Mask'] = torch.mean(torch.abs(1 - output_paired_comp))

                loss_gen = sum(G_losses.values()).mean()

            # GANs w/o composition mask & w/ projected discriminator
            else:
                set_requires_grad(discriminator, False)
                output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired_taco), dim=1), parse)

                pred_fake, feats_fake = discriminator(output_paired)
                pred_real, feats_real = discriminator(im)

                # Adversarial loss
                G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False) * 0.5

                # Feature matching loss
                num_D = len(feats_fake)
                GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(feats_fake[i])
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = criterionFeat(feats_fake[i][j], feats_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

                # VGG perceptual loss
                G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg

                # Image-level L1 loss
                G_losses['L1'] = criterionL1(output_paired, im) * opt.lambda_l1

                loss_gen = sum(G_losses.values()).mean()
            
            composition_loss += loss_gen.item()
            gan_composition_loss += G_losses['GAN'].mean().item()
            feat_composition_loss += G_losses['GAN_Feat'].mean().item()
            vgg_composition_loss += G_losses['VGG'].mean().item()
            l1_composition_loss += G_losses['L1'].mean().item()
            a_0 = im.cuda()[0]
            b_0 = output_paired.cuda()[0]
            c_0 = warped_cloth_paired_taco.cuda()[0]
            combine = torch.cat((a_0, b_0, c_0), dim=2)
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            i = 0
            out = output_paired[i].cpu() / 2 + 0.5
            processed_batches += 1
        log_images = {'Val/Image': (im[0].cpu() / 2 + 0.5),
        'Val/Pose Image': (openpose[0].cpu() / 2 + 0.5),
        'Val/Parse Clothing': (parse_cloth[0].cpu() / 2 + 0.5),
        'Val/Parse Clothing Mask': pcm[0].cpu().expand(3, -1, -1),
        'Val/Warped Cloth TVOB': (warped_cloth_paired_tvob[0].cpu().detach() / 2 + 0.5),
        'Val/Warped Cloth TACO': (c_0.cpu().detach() / 2 + 0.5),
        'Val/Warped Cloth Mask TVOB': warped_clothmask_paired_tvob[0].cpu().detach().expand(3, -1, -1),
        'Val/Warped Cloth Mask TACO': warped_clothmask_paired_taco[0].cpu().detach().expand(3, -1, -1),
        'Val/Composition': out}
        log_losses = {'val_composition_loss': composition_loss / len(validation_loader.dataset) ,
        'val_gan_composition_loss': gan_composition_loss / len(validation_loader.dataset) ,
        'val_feat_composition_loss': feat_composition_loss / len(validation_loader.dataset),
        'val_vgg_composition_loss': vgg_composition_loss / len(validation_loader.dataset),
        'val_l1_composition_loss' : l1_composition_loss / len(validation_loader.dataset)}
        log_results(log_images, log_losses, board,wandb, step, train=False)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(opt.results_dir, f'out_{step}.png'),bgr)
            
def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)
            
def make_dirs(opt):
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    if not os.path.exists(opt.gen_save_final_checkpoint_dir):
        os.makedirs(opt.gen_save_final_checkpoint_dir)
    if not os.path.exists(os.path.join(opt.results_dir,'val')):
        os.makedirs(os.path.join(opt.results_dir,'val'))
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    if not os.path.exists(opt.gen_save_step_checkpoint_dir):
        os.makedirs(opt.gen_save_step_checkpoint_dir)
    if not os.path.exists(opt.gen_discriminator_save_step_checkpoint_dir):
        os.makedirs(opt.gen_discriminator_save_step_checkpoint_dir)
    if not os.path.exists(opt.gen_discriminator_save_final_checkpoint_dir):
        os.makedirs(opt.gen_discriminator_save_final_checkpoint_dir)
        
def _train_tryon_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_tryon_()
            
    
def train_tryon_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = generator_process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_tryon_sweep,count=5)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_tryon_()
    else:
        wandb = None
        _train_tryon_()
        
def _train_tryon_():
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
    num_cuda_devices = torch.cuda.device_count()
    print("Number of available CUDA devices:", num_cuda_devices)
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
        with open(log_path, 'w') as file:
            file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    opt.fine_height = 1024
    opt.fine_width = 768
    # print("Start to train %s!" % opt.name)
    train_dataset = FashionNeRFDataset(root_opt, opt, viton=True, model='viton')
    train_loader = FashionDataLoader(train_dataset, opt.viton_batch_size, opt.viton_workers, True)
    test_dataset = FashionNeRFDataset(root_opt, opt, viton=True, mode='test', model='viton')
    test_loader = FashionDataLoader(test_dataset, opt.num_test_visualize, 1, False)
    validation_dataset = Subset(test_dataset, np.arange(50))
    validation_loader = FashionDataLoader(validation_dataset, opt.num_test_visualize, root_opt.viton_workers, False)
    
    # warping-seg Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=13, ngf=opt.cond_G_ngf, norm_layer=nn.BatchNorm2d, num_layers=opt.cond_G_num_layers) # num_layers: training condition network w/ fine_height 256 -> 5, - w/ fine_height 512 -> 6, - w/ fine_height 1024 -> 7
    # Load Checkpoint
    if os.path.exists(opt.tocg_load_final_checkpoint):
        load_checkpoint(tocg, opt.tocg_load_final_checkpoint)

    # generator model
    generator = SPADEGenerator(opt, 3+3+3)
    generator.print_network()
    generator.init_weights(opt.init_type, opt.init_variance)

    # discriminator model
    discriminator = None
    if opt.composition_mask:
        discriminator = create_network(MultiscaleDiscriminator, opt)
    else:
        discriminator = ProjectedDiscriminator(interp224=False)

    # lpips
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

    # Load Checkpoint
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    if last_step:
        load_checkpoint(generator, opt.gen_load_step_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.gen_load_step_checkpoint}')
    elif os.path.exists(opt.gen_load_final_checkpoint):
        load_checkpoint(generator, opt.gen_load_final_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.gen_load_final_checkpoint}')     

    if last_step:
        load_checkpoint(discriminator, opt.gen_discriminator_load_step_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.gen_discriminator_load_step_checkpoint}')
    elif os.path.exists(opt.gen_discriminator_load_final_checkpoint):
        load_checkpoint(discriminator, opt.gen_discriminator_load_final_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.gen_discriminator_load_final_checkpoint}')

    # Train
    train(opt, train_loader,validation_loader, test_loader,board,tocg, generator, discriminator, model, wandb=wandb)

    # Save Checkpoint
    save_checkpoint(generator,opt.gen_save_final_checkpoint)
    save_checkpoint(discriminator,opt.gen_discriminator_save_final_checkpoint)

    print("Finished training !")


