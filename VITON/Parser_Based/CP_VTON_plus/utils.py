import os 
import argparse
import os
fix = lambda path: os.path.normpath(path)
def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    root_opt.gmm_experiment_from_run = root_opt.gmm_experiment_from_run.format(root_opt.gmm_experiment_from_number, root_opt.gmm_run_from_number)
    root_opt.tom_experiment_from_run = root_opt.tom_experiment_from_run.format(root_opt.tom_experiment_from_number, root_opt.tom_run_from_number)
    return root_opt

def get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)    
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
    # Current model
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.this_viton_load_from_dir)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    # GMM
    root_opt.gmm_experiment_from_dir = root_opt.gmm_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gmm_load_from_model)
    root_opt.gmm_experiment_from_dir = os.path.join(root_opt.gmm_experiment_from_dir, "GMM")
    
    # GMM
    root_opt.tom_experiment_from_dir = root_opt.tom_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tom_load_from_model)
    root_opt.tom_experiment_from_dir = os.path.join(root_opt.tom_experiment_from_dir, 'TOM')    
    return root_opt


def get_root_opt_checkpoint_dir(opt, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    # ================================= GMM =================================
    opt.gmm_save_step_checkpoint_dir = opt.gmm_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gmm_save_step_checkpoint_dir = fix(opt.gmm_save_step_checkpoint_dir)
    opt.gmm_save_step_checkpoint = os.path.join(opt.gmm_save_step_checkpoint_dir, opt.gmm_save_step_checkpoint)
    opt.gmm_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_save_step_checkpoint)
    opt.gmm_save_step_checkpoint_dir = os.path.join("/",*opt.gmm_save_step_checkpoint.split("/")[:-1])
    
    opt.gmm_save_final_checkpoint_dir = opt.gmm_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gmm_save_final_checkpoint_dir = fix(opt.gmm_save_final_checkpoint_dir)
    opt.gmm_save_final_checkpoint = os.path.join(opt.gmm_save_final_checkpoint_dir, opt.gmm_save_final_checkpoint)
    opt.gmm_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_save_final_checkpoint)
    opt.gmm_save_final_checkpoint_dir = os.path.join("/",*opt.gmm_save_final_checkpoint.split("/")[:-1])
    
    opt.gmm_load_final_checkpoint_dir = opt.gmm_load_final_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.gmm_experiment_from_dir)
    opt.gmm_load_final_checkpoint_dir = fix(opt.gmm_load_final_checkpoint_dir)
    opt.gmm_load_final_checkpoint = os.path.join(opt.gmm_load_final_checkpoint_dir, opt.gmm_load_final_checkpoint)
    opt.gmm_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_load_final_checkpoint)
    opt.gmm_load_final_checkpoint_dir = os.path.join("/",*opt.gmm_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        opt.gmm_load_step_checkpoint_dir = opt.gmm_load_step_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.gmm_experiment_from_dir)
    else:
        opt.gmm_load_step_checkpoint_dir = opt.gmm_load_step_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.gmm_load_step_checkpoint_dir = fix(opt.gmm_load_step_checkpoint_dir)
    if not last_step:
        opt.gmm_load_step_checkpoint = os.path.join(opt.gmm_load_step_checkpoint_dir, opt.gmm_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.gmm_load_step_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "gmm" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.gmm_load_step_checkpoint = os.path.join(opt.gmm_load_step_checkpoint_dir, last_step)
    opt.gmm_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_load_step_checkpoint)    
    opt.gmm_load_step_checkpoint_dir = os.path.join("/",*opt.gmm_load_step_checkpoint.split("/")[:-1])
    # ================================= TOM =================================
    opt.tom_save_step_checkpoint_dir = opt.tom_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tom_save_step_checkpoint_dir = fix(opt.tom_save_step_checkpoint_dir)
    opt.tom_save_step_checkpoint = os.path.join(opt.tom_save_step_checkpoint_dir, opt.tom_save_step_checkpoint)
    opt.tom_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_save_step_checkpoint)    
    opt.tom_save_step_checkpoint_dir = os.path.join("/",*opt.tom_save_step_checkpoint.split("/")[:-1])
    
    opt.tom_save_final_checkpoint_dir = opt.tom_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tom_save_final_checkpoint_dir = fix(opt.tom_save_final_checkpoint_dir)
    opt.tom_save_final_checkpoint = os.path.join(opt.tom_save_final_checkpoint_dir, opt.tom_save_final_checkpoint)
    opt.tom_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_save_final_checkpoint)    
    opt.tom_save_final_checkpoint_dir = os.path.join("/",*opt.tom_save_final_checkpoint.split("/")[:-1])
    
    opt.tom_load_final_checkpoint_dir = opt.tom_load_final_checkpoint_dir.format(root_opt.vgg_experiment_from_run, root_opt.tom_experiment_from_dir)
    opt.tom_load_final_checkpoint_dir = fix(opt.tom_load_final_checkpoint_dir)
    opt.tom_load_final_checkpoint = os.path.join(opt.tom_load_final_checkpoint_dir, opt.tom_load_final_checkpoint)
    opt.tom_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_load_final_checkpoint)    
    opt.tom_load_final_checkpoint_dir = os.path.join("/",*opt.tom_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        opt.tom_load_step_checkpoint_dir = opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.tom_experiment_from_dir)
    else:
        opt.tom_load_step_checkpoint_dir = opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tom_load_step_checkpoint_dir = fix(opt.tom_load_step_checkpoint_dir)
    if not last_step:
        opt.tom_load_step_checkpoint = os.path.join(opt.tom_load_step_checkpoint_dir, opt.tom_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tom" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tom_load_step_checkpoint = os.path.join(opt.tom_load_step_checkpoint_dir, last_step)
    opt.tom_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_load_step_checkpoint)    
    opt.tom_load_step_checkpoint_dir = os.path.join("/",*opt.tom_load_step_checkpoint.split("/")[:-1])
    return opt

def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.datamode = root_opt.datamode
    parser.dataset_name = root_opt.dataset_name
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.stage = root_opt.stage
    parser.seed = root_opt.seed
    parser.val_count = root_opt.val_count
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
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