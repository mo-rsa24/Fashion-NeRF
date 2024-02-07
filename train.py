def train_acgpn(opt, root_opt, run_wandb, sweeps):
    from VITON.Parser_Based.ACGPN.ACGPN_train.train import train_acgpn_
    train_acgpn_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    

def train_dmvton(opt, root_opt, run_wandb, sweeps, VITON_Model):
    from VITON.Parser_Free.DM_VTON.train_pb_warp import train_pb_warp_
    from VITON.Parser_Free.DM_VTON.train_pb_e2e import train_pb_e2e_
    from VITON.Parser_Free.DM_VTON.train_pf_warp import train_pf_warp_
    from VITON.Parser_Free.DM_VTON.train_pf_e2e import train_pf_e2e_
    
    if VITON_Model == 'PB_Warp':
        train_pb_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'PB_Gen':
        train_pb_e2e_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'PF_Warp':
        train_pf_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'PF_Gen':
        train_pf_e2e_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)    
        
        
def train_cp_vton(opt, root_opt, run_wandb, sweeps): # Warping
    from VITON.Parser_Based.CP_VTON.train import train_cpvton_
    train_cpvton_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    
    
def train_cp_vton_plus(opt, root_opt, run_wandb, sweeps):
    from VITON.Parser_Based.CP_VTON_plus.train import train_cpvton_plus_
    train_cpvton_plus_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    
    
def train_pfafn(opt, root_opt, run_wandb, sweeps, VITON_Model):
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PBAFN_stage1 import train_pfafn_pb_warp_
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PBAFN_e2e import train_pfafn_pb_gen_
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PFAFN_stage1 import train_pfafn_pf_warp_
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PFAFN_e2e import train_pfafn_e2e_
    if VITON_Model == 'PB_Warp':
        train_pfafn_pb_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'PB_Gen':
        train_pfafn_pb_gen_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'PF_Warp':
        train_pfafn_pf_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'PF_Gen':
        train_pfafn_e2e_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)   
        
def train_fsvton(opt, root_opt, run_wandb, sweeps, VITON_Model):
    from VITON.Parser_Free.FS_VTON.train.train_PBAFN_stage1_fs import train_fsvton_pb_warp_
    from VITON.Parser_Free.FS_VTON.train.train_PBAFN_e2e_fs import train_fsvton_pb_gen_
    if VITON_Model == 'PB_Warp':
        train_fsvton_pb_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'PB_Gen':
        train_fsvton_pb_gen_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
        

def train_hrviton(opt, root_opt, run_wandb, sweeps, VITON_Model):
    from VITON.Parser_Based.HR_VITON.train_condition import train_hrviton_tocg_
    from VITON.Parser_Based.HR_VITON.train_generator import train_tryon_
    if VITON_Model == 'TOCG':
        train_hrviton_tocg_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)    
    elif VITON_Model == 'GEN':
        train_tryon_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)

def train_ladi_vton(opt, root_opt, run_wandb, sweeps, VITON_Model):
    from VITON.Parser_Based.Ladi_VTON.src.train_tps import train_ladi_vton_tps_
    if VITON_Model == 'TPS':
        train_ladi_vton_tps_(opt, root_opt, run_wandb=run_wandb)
    
    
def train_sdviton(opt, root_opt, run_wandb, sweeps, VITON_Model):
    from VITON.Parser_Based.SD_VITON.train_condition import train_sd_viton_tocg_
    from VITON.Parser_Based.SD_VITON.train_generator import train_tryon_
    if VITON_Model == 'TOCG':
        train_sd_viton_tocg_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    elif VITON_Model == 'GEN':
        train_tryon_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
        
def train_viton(opt, root_opt, run_wandb, sweeps,VITON_Name, VITON_Model):
    if VITON_Name == 'HR_VITON':
        train_hrviton(opt, root_opt, run_wandb, sweeps, VITON_Model)
    elif VITON_Name == 'ACGPN':
        train_acgpn(opt, root_opt, run_wandb, sweeps, VITON_Model)
    elif VITON_Name == 'SD_VITON':
        train_sdviton(opt, root_opt, run_wandb, sweeps, VITON_Model)
    elif VITON_Name == 'DM_VTON':
        train_dmvton(opt, root_opt, run_wandb, sweeps, VITON_Model)
    elif VITON_Name == 'PF_AFN':
        train_pfafn(opt, root_opt, run_wandb, sweeps, VITON_Model)
    elif VITON_Name == "FS_VTON":
        train_fsvton(opt, root_opt, run_wandb, sweeps, VITON_Model)
    elif VITON_Name == 'CP_VTON':
        train_cp_vton(opt, root_opt, run_wandb, sweeps)
    elif VITON_Name == 'CP_VTON_plus':
        train_cp_vton_plus(opt, root_opt, run_wandb, sweeps)