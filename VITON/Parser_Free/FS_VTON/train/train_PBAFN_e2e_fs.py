import time
import argparse
import yaml
from VITON.Parser_Free.PF_AFN.PF_AFN_train.models.networks import VGGLoss, save_checkpoint
from VITON.Parser_Free.FS_VTON.train.models.afwm import TVLoss,AFWM
from VITON.Parser_Free.FS_VTON.train.models.networks import ResUnetGenerator
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PBAFN_e2e import process_opt
from VITON.Parser_Free.DM_VTON.utils.general import AverageMeter, print_log
from torch.utils.data.distributed import DistributedSampler
from VITON.Parser_Free.DM_VTON.utils.torch_utils import get_ckpt, load_ckpt, select_device
from tensorboardX import SummaryWriter
import cv2
import datetime

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

def select_device(device='', batch_size=0):
    cpu = device == 'cpu'
    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        arg = f'cuda:{device}'
    else:  # revert to CPU
        arg = 'cpu'
    return arg
  
def train_fsvton_pb_gen_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_fsvton_pb_gen_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_fsvton_pb_gen_()
    else:
        wandb = None
        _train_fsvton_pb_gen_()
    
    
def _train_fsvton_pb_gen_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_fsvton_pb_gen_()
            
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
    if not os.path.exists(os.path.join(opt.results_dir, 'val')):
        os.makedirs(os.path.join(opt.results_dir, 'val'))
    if not os.path.exists(opt.pb_warp_save_step_checkpoint_dir):
        os.makedirs(opt.pb_warp_save_step_checkpoint_dir)
    if not os.path.exists(opt.pb_gen_save_step_checkpoint_dir):
        os.makedirs(opt.pb_gen_save_step_checkpoint_dir)
    if not os.path.exists(opt.pb_gen_save_final_checkpoint_dir):
        os.makedirs(opt.pb_gen_save_final_checkpoint_dir)
    if not os.path.exists(opt.pb_warp_save_final_checkpoint_dir):
        os.makedirs(opt.pb_warp_save_final_checkpoint_dir) 
        
def _train_fsvton_pb_gen_():
  global opt, root_opt, wandb,sweep_id
  make_dirs(opt)
  device = select_device(opt.device, batch_size=root_opt.viton_batch_size)
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
    
  experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
  with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
    yaml.dump(vars(opt), outfile, default_flow_style=False)    
  torch.cuda.set_device(opt.device)


  start_epoch, epoch_iter = 1, 0

  train_data = CreateDataset(opt, root_opt)
  train_dataset, validation_dataset = split_dataset(train_data)
  train_loader = DataLoader(train_dataset, batch_size=opt.viton_batch_size, shuffle=True,num_workers=root_opt.workers)
  validation_loader = DataLoader(validation_dataset, batch_size=opt.viton_batch_size, shuffle=True, num_workers=root_opt.workers)
  dataset_size = len(train_loader)
  print('#training images = %d' % dataset_size)

  warp_model = AFWM(opt, 45)
  print(warp_model)
  warp_model.train()
  warp_model.cuda()
  log_path = os.path.join(opt.results_dir, 'log.txt')
  with open(log_path, 'w') as file:
    file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
  if os.path.exists(opt.pb_warp_load_final_checkpoint):
      warp_ckpt = get_ckpt(opt.pb_warp_load_final_checkpoint)
      load_ckpt(warp_model, warp_ckpt)
      print_log(log_path, f'Load pretrained parser-based warp from {opt.pb_warp_load_final_checkpoint}')

  gen_model = ResUnetGenerator(8, 4, 5, opt, ngf=64, norm_layer=nn.BatchNorm2d)
  print(gen_model)
  gen_model.train()
  gen_model.cuda()

  # warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)
  # gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)

  if eval(root_opt.load_last_step):
        gen_ckpt = get_ckpt(opt.pb_gen_load_step_checkpoint)
        load_ckpt(gen_model, gen_ckpt)
        print_log(log_path, f'Load pretrained parser-based gen from {opt.pb_gen_load_step_checkpoint}')
  elif os.path.exists(opt.pb_gen_load_final_checkpoint):
        gen_ckpt = get_ckpt(opt.pb_gen_load_final_checkpoint)
        load_ckpt(gen_model, gen_ckpt)
        print_log(log_path, f'Load pretrained parser-based gen from {opt.pb_gen_load_final_checkpoint}')  

  criterionL1 = nn.L1Loss()
  criterionVGG = VGGLoss()
  # optimizer
  params_warp = [p for p in warp_model.parameters()]
  params_gen = [p for p in gen_model.parameters()]
  optimizer_warp = torch.optim.Adam(params_warp, lr=0.2*opt.lr, betas=(opt.beta1, 0.999))
  optimizer_gen = torch.optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, 0.999))

  total_steps = (start_epoch-1) * dataset_size + epoch_iter
  total_valid_steps = (start_epoch-1) * dataset_size + epoch_iter
  step = 0
  step_per_batch = dataset_size

  for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
      epoch_start_time = time.time()
      if epoch != start_epoch:
          epoch_iter = epoch_iter % dataset_size
      train_batch(opt, train_loader, 
                    gen_model, warp_model,total_steps,epoch,epoch_iter, criterionL1,criterionVGG,optimizer_gen,optimizer_warp,
                    writer, step_per_batch)
      if epoch % opt.val_count == 0:
            validate_batch(opt, validation_loader,gen_model, warp_model,total_valid_steps, epoch,criterionL1,criterionVGG,optimizer_gen,optimizer_warp, writer, step_per_batch)
  save_checkpoint(warp_model, opt.pb_warp_save_final_checkpoint)
  save_checkpoint(gen_model, opt.pb_gen_save_final_checkpoint)

def train_batch(opt, train_loader,model_gen, model,total_steps, epoch,epoch_iter,criterionL1,criterionVGG,optimizer_gen,optimizer_warp, writer, step_per_batch):
  model_gen.train()
  model.train()
  total_loss_warping = 0
  dataset_size = len(train_loader)
  for i, data in enumerate(train_loader):
    iter_start_time = time.time()

    total_steps += 1
    epoch_iter += 1
    if root_opt.dataset_name == 'Rail':
        t_mask = torch.FloatTensor(((data['label'] == 3) | (data['label'] == 11)).cpu().numpy().astype(np.int64))
    else:
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
    data['label'] = data['label']*(1-t_mask)+t_mask*4
    edge = data['edge']
    pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
    clothes = data['color']
    clothes = clothes * pre_clothes_edge
    if root_opt.dataset_name == 'Rail':
        person_clothes_edge = torch.FloatTensor(((data['label'] == 5) | (data['label'] == 6) | (data['label'] == 7)).cpu().numpy().astype(np.int64))
    else:
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
    real_image = data['image']
    person_clothes = real_image*person_clothes_edge
    pose = data['pose']
    pose_map = data['pose_map']
    size = data['label'].size()
    oneHot_size1 = (size[0], 25, size[2], size[3])
    densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
    densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
    densepose_fore = data['densepose']/24.0
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
    preserve_mask = torch.cat([face_mask, other_clothes_mask],1)
    concat = torch.cat([preserve_mask.cuda(), densepose, pose.cuda()],1)
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
    hand_mask = arm_mask*hand_mask
    hand_img = hand_mask*real_image
    dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy()==15).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==16).astype(np.int))\
                          +torch.FloatTensor((data['densepose'].cpu().numpy()==17).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==18).astype(np.int))\
                          +torch.FloatTensor((data['densepose'].cpu().numpy()==19).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==20).astype(np.int))\
                          +torch.FloatTensor((data['densepose'].cpu().numpy()==21).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==22))
    dense_preserve_mask = dense_preserve_mask.cuda()*(1-person_clothes_edge.cuda())
    preserve_region = face_img + other_clothes_img +hand_img

    flow_out = model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
    warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out

    epsilon = opt.epsilon
    loss_smooth = sum([TVLoss(x) for x in delta_list])
    warp_loss = 0

    for num in range(5):
        cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
        cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
        loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
        loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
        loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
        b,c,h,w = delta_x_all[num].shape
        loss_flow_x = (delta_x_all[num].pow(2) + epsilon*epsilon).pow(0.45)
        loss_flow_x = torch.sum(loss_flow_x) / (b*c*h*w)
        loss_flow_y = (delta_y_all[num].pow(2) + epsilon*epsilon).pow(0.45)
        loss_flow_y = torch.sum(loss_flow_y) / (b*c*h*w)
        loss_second_smooth = loss_flow_x + loss_flow_y
        warp_loss = warp_loss + (num+1) * loss_l1 + (num+1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num+1) * 6 * loss_second_smooth

    warp_loss = opt.lambda_loss_smooth * loss_smooth + warp_loss

    warped_prod_edge = x_edge_all[4]
    if root_opt.dataset_name == 'Rail' and epoch >0 :
        binary_mask = (warped_prod_edge > 0.5).float()
        warped_cloth = warped_cloth * binary_mask
    gen_inputs = torch.cat([preserve_region.cuda(), warped_cloth, warped_prod_edge, dense_preserve_mask], 1)

    gen_outputs = model_gen(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite1 = m_composite * warped_prod_edge
    m_composite =  person_clothes_edge.cuda()*m_composite1
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
    loss_l1 = criterionL1(p_tryon, real_image.cuda())
    loss_vgg = criterionVGG(p_tryon,real_image.cuda())
    bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
    bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())
    gen_loss = (loss_l1 * opt.lambda_loss_l1 + loss_vgg + bg_loss_l1 * opt.lambda_bg_loss_l1 + bg_loss_vgg + loss_mask_l1)

    

    loss_all = opt.lambda_loss_warp * warp_loss + opt.lambda_loss_gen * gen_loss
    total_loss_warping += loss_all
    

    optimizer_warp.zero_grad()
    optimizer_gen.zero_grad()
    loss_all.backward()
    optimizer_warp.step()
    optimizer_gen.step()

    if (epoch + 1) % opt.display_count == 0:
      a = real_image.float().cuda()
      b = person_clothes.cuda()
      c = clothes.cuda()
      d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
      e = warped_cloth
      f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
      g = preserve_region.cuda()
      h = torch.cat([dense_preserve_mask,dense_preserve_mask,dense_preserve_mask],1)
      i = p_rendered
      j = torch.cat([m_composite1,m_composite1,m_composite1],1)
      k = p_tryon
      combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0],g[0],h[0],i[0],j[0],k[0]], 2).squeeze()
      cv_img = (combine.permute(1,2,0).detach().cpu().numpy()+1)/2
      log_losses = {'warping_loss': warp_loss.item() ,'warping_l1': loss_l1.item(),'warping_vgg': loss_vgg.item(),
                  'loss_gen':gen_loss.item(),'composition_loss': loss_all.item()}
      log_images = {'Image': (a[0].cpu() / 2 + 0.5), 
      'Pose Image': (pose_map[0].cpu() / 2 + 0.5), 
      'Clothing': (c[0].cpu() / 2 + 0.5), 
      'Parse Clothing': (b[0].cpu() / 2 + 0.5), 
      'Parse Clothing Mask': person_clothes_edge[0].cpu().expand(3, -1, -1), 
      'Warped Cloth': (e[0].cpu().detach() / 2 + 0.5), 
      'Warped Cloth Mask': f[0].cpu().detach().expand(3, -1, -1),
      "Composition": k[0].cpu() / 2 + 0.5}
      log_results(log_images, log_losses, writer,wandb, epoch, iter_start_time=iter_start_time, train=True)
      rgb = (cv_img*255).astype(np.uint8)
      bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(opt.results_dir, str(epoch)+'.jpg'),bgr)

    iter_end_time = time.time()
    iter_delta_time = iter_end_time - iter_start_time
    step_delta = (step_per_batch-epoch%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
    eta = iter_delta_time*step_delta
    eta = str(datetime.timedelta(seconds=int(eta)))
    time_stamp = datetime.datetime.now()

    if epoch_iter >= dataset_size:
        break

    iter_end_time = time.time()

    ### save model for this epoch
    if epoch % opt.save_period == 0:
      print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
      save_checkpoint(model, opt.pb_warp_save_step_checkpoint % (epoch+1))
      save_checkpoint(model_gen, opt.pb_gen_save_step_checkpoint % (epoch+1))
    
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
        print('validation step: %8d, composition_loss: %4f warping loss: %4f warping_l1 loss: %4f warping_vgg loss: %4f gen loss: %4f' % (step+1, log_losses['val_composition_loss'],log_losses['val_warping_loss'],log_losses['val_warping_l1'],log_losses['val_warping_vgg'], log_losses['val_loss_gen']), flush=True)
        
def validate_batch(opt, validation_loader,model_gen, model,total_steps, epoch,criterionL1,criterionVGG,optimizer_gen,optimizer_warp, writer, step_per_batch):
  model_gen.eval()
  model.eval()
  total_loss_warping = 0
  dataset_size = len(validation_loader)
  warping_loss = 0
  warping_l1 = 0
  warping_vgg = 0
  loss_gen =0 
  composition_loss = 0
  for i, data in enumerate(validation_loader):
    iter_start_time = time.time()

    total_steps += 1

    if root_opt.dataset_name == 'Rail':
        t_mask = torch.FloatTensor(((data['label'] == 3) | (data['label'] == 11)).cpu().numpy().astype(np.int64))
    else:
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
    data['label'] = data['label']*(1-t_mask)+t_mask*4
    edge = data['edge']
    pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
    clothes = data['color']
    clothes = clothes * pre_clothes_edge
    if root_opt.dataset_name == 'Rail':
        person_clothes_edge = torch.FloatTensor(((data['label'] == 5) | (data['label'] == 6) | (data['label'] == 7)).cpu().numpy().astype(np.int64))
    else:
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
    real_image = data['image']
    person_clothes = real_image*person_clothes_edge
    pose = data['pose']
    pose_map = data['pose_map']
    size = data['label'].size()
    oneHot_size1 = (size[0], 25, size[2], size[3])
    densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
    densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
    densepose_fore = data['densepose']/24.0
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
    preserve_mask = torch.cat([face_mask, other_clothes_mask],1)
    concat = torch.cat([preserve_mask.cuda(), densepose, pose.cuda()],1)
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
    hand_mask = arm_mask*hand_mask
    hand_img = hand_mask*real_image
    dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy()==15).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==16).astype(np.int))\
                          +torch.FloatTensor((data['densepose'].cpu().numpy()==17).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==18).astype(np.int))\
                          +torch.FloatTensor((data['densepose'].cpu().numpy()==19).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==20).astype(np.int))\
                          +torch.FloatTensor((data['densepose'].cpu().numpy()==21).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==22))
    dense_preserve_mask = dense_preserve_mask.cuda()*(1-person_clothes_edge.cuda())
    preserve_region = face_img + other_clothes_img +hand_img

    flow_out = model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
    warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out

    epsilon = opt.epsilon
    loss_smooth = sum([TVLoss(x) for x in delta_list])
    warp_loss = 0

    for num in range(5):
        cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
        cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
        loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
        loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
        loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
        b,c,h,w = delta_x_all[num].shape
        loss_flow_x = (delta_x_all[num].pow(2) + epsilon*epsilon).pow(0.45)
        loss_flow_x = torch.sum(loss_flow_x) / (b*c*h*w)
        loss_flow_y = (delta_y_all[num].pow(2) + epsilon*epsilon).pow(0.45)
        loss_flow_y = torch.sum(loss_flow_y) / (b*c*h*w)
        loss_second_smooth = loss_flow_x + loss_flow_y
        warp_loss = warp_loss + (num+1) * loss_l1 + (num+1) * opt.lambda_loss_vgg * loss_vgg + (num+1) * opt.lambda_loss_edge * loss_edge + (num+1) * opt.lambda_loss_second_smooth * loss_second_smooth

    warp_loss = opt.lambda_loss_smooth * loss_smooth + warp_loss



    warped_prod_edge = x_edge_all[4]
    
    if root_opt.dataset_name == 'Rail' and epoch >0 :
        binary_mask = (warped_prod_edge > 0.5).float()
        warped_cloth = warped_cloth * binary_mask
    gen_inputs = torch.cat([preserve_region.cuda(), warped_cloth, warped_prod_edge, dense_preserve_mask], 1)

    gen_outputs = model_gen(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite1 = m_composite * warped_prod_edge
    m_composite =  person_clothes_edge.cuda()*m_composite1
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
    loss_l1 = criterionL1(p_tryon, real_image.cuda())
    loss_vgg = criterionVGG(p_tryon,real_image.cuda())
    bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
    bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())
    gen_loss = (loss_l1 * opt.lambda_loss_l1 + loss_vgg + bg_loss_l1 * opt.lambda_bg_loss_l1 + bg_loss_vgg + loss_mask_l1)

    loss_all = opt.lambda_loss_warp * warp_loss + opt.lambda_loss_gen * gen_loss
    total_loss_warping += loss_all
    

    optimizer_warp.zero_grad()
    optimizer_gen.zero_grad()
    loss_all.backward()
    optimizer_warp.step()
    optimizer_gen.step()
    a = real_image.float().cuda()
    b = person_clothes.cuda()
    c = clothes.cuda()
    d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
    e = warped_cloth
    f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
    g = preserve_region.cuda()
    h = torch.cat([dense_preserve_mask,dense_preserve_mask,dense_preserve_mask],1)
    i = p_rendered
    j = torch.cat([m_composite1,m_composite1,m_composite1],1)
    k = p_tryon
    combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0],g[0],h[0],i[0],j[0],k[0]], 2).squeeze()
    cv_img = (combine.permute(1,2,0).detach().cpu().numpy()+1)/2
    rgb = (cv_img * 255).astype(np.uint8)
    warping_loss += warp_loss.item()
    warping_l1 += loss_l1.item()
    warping_vgg += loss_vgg.item()
    loss_gen += gen_loss.item()
    composition_loss += loss_all.item()
    rgb = (cv_img*255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
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
  log_results(log_images, log_losses, writer,wandb, epoch, train=False)
  
