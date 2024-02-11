# File based on https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
import yaml
import argparse
import logging
import os
import shutil
from pathlib import Path
import subprocess
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# command = "source ~/.bashrc && conda activate ladi-vton && conda env list | grep ladi"
# subprocess.run(command, shell=True, executable='/bin/bash')
import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from tensorboardX import SummaryWriter
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
from tqdm.auto import tqdm
import numpy as np
from VITON.Parser_Based.Ladi_VTON.src.dataset.dresscode import DressCodeDataset
from VITON.Parser_Based.Ladi_VTON.src.dataset.vitonhd import VitonHDDataset
from VITON.Parser_Based.Ladi_VTON.src.models.AutoencoderKL import AutoencoderKL
from VITON.Parser_Based.Ladi_VTON.src.models.emasc import EMASC
from VITON.Parser_Based.Ladi_VTON.src.utils.data_utils import mask_features
from VITON.Parser_Based.Ladi_VTON.src.utils.image_from_pipe import extract_save_vae_images
from VITON.Parser_Based.Ladi_VTON.src.utils.set_seeds import set_seed
from VITON.Parser_Based.Ladi_VTON.src.utils.val_metrics import compute_metrics
from VITON.Parser_Based.Ladi_VTON.src.utils.vgg_loss import VGGLoss
from VITON.Parser_Based.Ladi_VTON.src.utils.env import emasc_process_opt
fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

fix = lambda path: os.path.normpath(path)

   
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
        
def train_ladi_vton_emasc_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = emasc_process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_ladi_vton_emasc_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_ladi_vton_emasc_()
    else:
        wandb = None
        _train_ladi_vton_emasc_()

def _train_ladi_vton_emasc_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_ladi_vton_emasc_()
            
def _train_ladi_vton_emasc_():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    board = SummaryWriter(log_dir = opt.tensorboard_dir)
    torch.cuda.set_device(f'cuda:{opt.device}')
    if sweep_id is not None:
        opt.lr = wandb.config.lr
        
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)

    log_path = os.path.join(opt.results_dir, 'log.txt')
    with open(log_path, 'w') as file:
        file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        log_with=opt.report_to,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if opt.seed is not None:
        set_seed(opt.seed)

    # Load VAE model.
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
    vae.eval()

    # Define EMASC model.
    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]
    int_layers = [1, 2, 3, 4, 5]

    emasc = EMASC(in_feature_channels,
                  out_feature_channels,
                  kernel_size=opt.emasc_kernel,
                  padding=opt.emasc_padding,
                  stride=1,
                  type=opt.emasc_type)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if opt.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        emasc.parameters(),
        lr=opt.learning_rate,
        betas=(opt.adam_beta1, opt.adam_beta2),
        weight_decay=opt.adam_weight_decay,
        eps=eval(opt.adam_epsilon),
    )
    dataroot = os.path.join(root_opt.root_dir, root_opt.original_dir)
    # Define datasets and dataloaders.
    if opt.dataset == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=opt.dresscode_dataroot,
            phase='train',
            order='paired',
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )

        test_dataset = DressCodeDataset(
            dataroot_path=opt.dresscode_dataroot,
            phase='test',
            order=opt.test_order,
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )
    elif opt.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            opt,root_opt,
            dataroot_path=dataroot,
            phase='train',
            order='paired',
            radius=5,
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )

        test_dataset = VitonHDDataset(
            opt,root_opt,
            dataroot_path=dataroot,
            phase='test',
            order=opt.test_order,
            radius=5,
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )
    else:
        raise NotImplementedError("Dataset not implemented")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=opt.train_batch_size,
        num_workers=opt.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=opt.test_batch_size,
        num_workers=opt.num_workers_test,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / opt.gradient_accumulation_steps)
    if opt.max_train_steps is None:
        opt.max_train_steps = opt.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        opt.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=opt.lr_warmup_steps * opt.gradient_accumulation_steps,
        num_training_steps=opt.max_train_steps * opt.gradient_accumulation_steps,
    )

    # Define VGG loss when vgg_weight > 0
    if opt.vgg_weight > 0:
        criterion_vgg = VGGLoss()
    else:
        criterion_vgg = None

    # Prepare everything with our `accelerator`.
    emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg = accelerator.prepare(
        emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / opt.gradient_accumulation_steps)
    if overrode_max_train_steps:
        opt.max_train_steps = opt.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    opt.num_train_epochs = math.ceil(opt.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("LaDI_VTON_EMASC", config=vars(opt),
                                  init_kwopt={"wandb": {"name": os.path.basename(opt.results_dir)}})
        if opt.report_to == 'wandb':
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.name = os.path.basename(opt.results_dir)
    # Train!
    total_batch_size = opt.train_batch_size * accelerator.num_processes * opt.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {opt.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {opt.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if opt.resume_from_checkpoint:
        try:
            if opt.resume_from_checkpoint != "latest":
                path = os.path.basename(os.path.join("checkpoint", opt.resume_from_checkpoint))
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(os.path.join(opt.results_dir, "checkpoint"))
                dirs = [d for d in dirs if d.startswith("emasc")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            accelerator.print(f"Resuming from checkpoint {path}")

            accelerator.load_state(os.path.join(opt.results_dir, "checkpoint", path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
        except Exception as e:
            print("Failed to load checkpoint, training from scratch:")
            print(e)
            resume_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, opt.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    for epoch in range(first_epoch, opt.num_train_epochs):
        emasc.train()
        train_loss = 0.0
        training_loss = 0 
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if opt.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % opt.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(emasc):
                # Convert images to latent space
                with torch.no_grad():
                    # take latents from the encoded image and intermediate features from the encoded masked image
                    posterior_im, _ = vae.encode(batch["image"])
                    _, intermediate_features = vae.encode(batch["im_mask"])

                    intermediate_features = [intermediate_features[i] for i in int_layers]

                # Use EMASC to process the intermediate features
                processed_intermediate_features = emasc(intermediate_features)

                # Mask the features
                processed_intermediate_features = mask_features(processed_intermediate_features, batch["inpaint_mask"])

                # Decode the image from the latent space use the EMASC module
                latents = posterior_im.latent_dist.sample()
                reconstructed_image = vae.decode(z=latents,
                                                 intermediate_features=processed_intermediate_features,
                                                 int_layers=int_layers).sample

                # Compute the loss
                with accelerator.autocast():
                    loss = F.l1_loss(reconstructed_image, batch["image"], reduction="mean")
                    if criterion_vgg:
                        loss += opt.vgg_weight * (criterion_vgg(reconstructed_image, batch["image"]))
                    
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(opt.train_batch_size)).mean()
                train_loss += avg_loss.item() / opt.gradient_accumulation_steps
                training_loss += avg_loss.item()
                # Backpropagate and update gradients
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(emasc.parameters(), opt.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint every checkpointing_steps steps
                if global_step % opt.checkpointing_steps == 0:
                    # Validation Step
                    emasc.eval()
                    if accelerator.is_main_process:
                        # Save model checkpoint
                        # os.makedirs(os.path.join(opt.results_dir, "checkpoint"), exist_ok=True)
                        accelerator_state_path = opt.emasc_vitonhd_save_step_checkpoint_dir % global_step
                        accelerator.save_state(accelerator_state_path)

                        # Unwrap the EMASC model
                        unwrapped_emasc = accelerator.unwrap_model(emasc, keep_fp32_wrapper=True)
                        with torch.no_grad():
                            # Extract the images
                            with torch.cuda.amp.autocast():
                                extract_save_vae_images(vae, unwrapped_emasc, test_dataloader, int_layers,
                                                        opt.results_dir, opt.test_order,
                                                        save_name=f"imgs_step_{global_step}",
                                                        emasc_type=opt.emasc_type)

                            # Compute the metrics
                            metrics = compute_metrics(
                                os.path.join(opt.results_dir, f"imgs_step_{global_step}_{opt.test_order}"),
                                opt.test_order, opt.dataset, 'all', ['all'], opt.dresscode_dataroot,
                                dataroot)

                            print(metrics, flush=True)
                            accelerator.log(metrics, step=global_step)

                            dirs = os.listdir(opt.emasc_vitonhd_save_step_checkpoint_dir)
                            dirs = [d for d in dirs if d.startswith("emasc")]
                            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                            emasc_path = os.path.join(opt.emasc_vitonhd_save_step_checkpoint_dir, f"emasc_{global_step}.pth")
                            accelerator.save(unwrapped_emasc.state_dict(), emasc_path)
                            del unwrapped_emasc

                        emasc.train()
            if epoch % opt.display_count == 0:
                pass
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= opt.max_train_steps:
                break

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()