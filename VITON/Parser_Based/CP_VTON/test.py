#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import argparse
import os
import time
from VITON.Parser_Based.CP_VTON.cp_dataset import CPDataset, CPDataLoader, CPDataTestLoader
from VITON.Parser_Based.CP_VTON.networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from VITON.Parser_Based.CP_VTON.visualization import board_add_image, board_add_images, save_images
from VITON.Parser_Based.CP_VTON.utils import *

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    save_dir = os.path.join(opt.results_dir, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)


    prediction_dir = os.path.join(opt.results_dir, 'prediction')
    ground_truth_dir = os.path.join(opt.results_dir, 'ground_truth')
    ground_truth_mask_dir = os.path.join(opt.results_dir, 'ground_truth_mask')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    if not os.path.exists(ground_truth_mask_dir):
        os.makedirs(ground_truth_mask_dir)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        # if opt.clip_warping:
        #     warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
            
        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 
        
        image_name = os.path.join(prediction_dir, inputs['im_name'][0])
        ground_truth_image_name = os.path.join(ground_truth_dir, inputs['im_name'][0])
        ground_truth_mask_name = os.path.join(ground_truth_mask_dir, inputs['im_name'][0])
        save_image(warped_cloth.cpu().detach() / 2 + 0.5, image_name)
        save_image(im_c.cpu().detach() / 2 + 0.5, ground_truth_image_name)
        save_image(warped_mask, ground_truth_mask_name)
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
        


def test_tom(opt, test_loader, gmm_model, model, board):
    model.cuda()
    model.eval()
    
    gmm_model.cuda()
    gmm_model.eval()
    save_dir = os.path.join(opt.results_dir, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    prediction_dir = os.path.join(opt.results_dir, 'prediction')
    ground_truth_dir = os.path.join(opt.results_dir, 'ground_truth')
    ground_truth_mask_dir = os.path.join(opt.results_dir, 'ground_truth_mask')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    if not os.path.exists(ground_truth_mask_dir):
        os.makedirs(ground_truth_mask_dir)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
    
        
        grid, theta = gmm_model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        
        if opt.clip_warping:
            warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
        
        outputs = model(torch.cat([agnostic, warped_cloth],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        
        visuals = [ [im_h, shape, im_pose], 
                   [warped_cloth, warped_mask*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
        
        save_images(p_tryon, im_names, try_on_dir) 
        
        image_name = os.path.join(prediction_dir, inputs['im_name'][0])
        ground_truth_image_name = os.path.join(ground_truth_dir, inputs['im_name'][0])
        save_image(im.cpu().detach() / 2 + 0.5, ground_truth_image_name)
        save_image(p_tryon.cpu().detach() / 2 + 0.5, image_name)
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

def test_cpvton_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    print("Start to test %s!")
    _test_cpvton_(opt, root_opt)

def _test_cpvton_(opt, root_opt):
    print("Start to test stage: %s" % opt.VITON_Model)
   
    # create dataset 
    test_dataset = CPDataset(root_opt, opt)
    test_dataset.__getitem__(0)
    # create dataloader
    test_loader = CPDataTestLoader(opt, test_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.VITON_Model))
   
    # create model & train
    if opt.VITON_Model == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.gmm_load_final_checkpoint)
        with torch.no_grad():
            test_gmm(opt, test_loader, model, board)
    elif opt.VITON_Model == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        gmm_model = GMM(opt)
        load_checkpoint(model, opt.tom_load_final_checkpoint)
        load_checkpoint(gmm_model, opt.gmm_load_final_checkpoint)
        with torch.no_grad():
            test_tom(opt, test_loader,gmm_model, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.VITON_Model)
  
    print('Finished test %s !' % opt.VITON_Model)

