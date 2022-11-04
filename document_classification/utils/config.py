"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from document_classification.utils.utils import mkdir_if_missing


def create_config(config_file_env, config_file_exp, tb_run, make_dirs=True):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, tb_run)
    pretext_dir = os.path.join(base_dir, 'pretext')

    if make_dirs:
        mkdir_if_missing(base_dir)
        mkdir_if_missing(pretext_dir)

    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')
    cfg['topk_furthest_train_path'] = os.path.join(pretext_dir, 'topk-train-furthest.npy')
    cfg['topk_furthest_val_path'] = os.path.join(pretext_dir, 'topk-val-furthest.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel', 'simpred']:
        base_dir = os.path.join(root_dir, tb_run)
        scan_dir = os.path.join(base_dir, 'scan')
        simpred_dir = os.path.join(base_dir, 'simpred')
        selflabel_dir = os.path.join(base_dir, 'selflabel')

        if make_dirs:
            mkdir_if_missing(base_dir)
            mkdir_if_missing(scan_dir)
            mkdir_if_missing(simpred_dir)
            mkdir_if_missing(selflabel_dir)

        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
        cfg['simpred_dir'] = simpred_dir
        cfg['simpred_checkpoint'] = os.path.join(simpred_dir, 'checkpoint.pth.tar')
        cfg['simpred_model'] = os.path.join(simpred_dir, 'model.pth.tar')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')
        cfg['scan_tb_dir'] = os.path.join(base_dir, 'tb_scan')
        cfg['simpred_tb_dir'] = os.path.join(base_dir, 'tb_simpred')
        cfg['selflabel_tb_dir'] = os.path.join(base_dir, 'tb_selflabel')

    return cfg 
