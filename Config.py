import torch


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def DE_360IQA_config():
    config = Config({
        # model setting
        'num_vps': 8,                       # number of viewports in a sequence.
        'img_channels': 3,
        'img_size': 224,
        'dim': 64,                          # dimension after Stem module.
        'depths': (2, 2, 5, 3),             # number of maxvit block in each stage.
        'channels': (128, 256, 512, 512),     # channels in each stage.
        'num_heads': (2, 4, 8, 16),          # number of head in each stage.
        
        'mlp_ratio': 3,
        'drop_rate': 0.,
        'pos_drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.,              # droppath rate in encoder block.
        'kernel_size': 7,
        'layer_scale': None,
        'dilations': None,
        'qkv_bias': True,
        'qk_scale': None,
        'select_rate': 0.5,                 # the rate of select feature from all viewport features.
        'num_classes': 4,
        'hidden_dim': 1152,                   
        
        
        # resource setting
        'root_dir':'/mnt/10T/zjy/database/OIQ_10k',
        'csv_path':'/mnt/10T/zjy/OIQ-10k/train.csv',
        'test_csv_path':'/mnt/10T/zjy/OIQ-10k/test.csv',
        # train setting
        'seed': 42,
        'dataset_name': 'OIQ-10K',
        'epochs': 60,
        'batch_size': 8,
        'num_workers': 8,
        'learning_rate': 1e-4,
        'lrf': 0.01,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'p': 1,
        'q': 2,
        'use_tqdm': False,
        'T_max':10,
        'eta_min':0,
        'use_tensorboard': False,
        'batch_print': False,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'val_freq':1,
        
         # load & save checkpoint
        "model_name": "MMBO_rei-t",
        "type_name": "MMBO_OIQ-10K",
        "ckpt_path": "/mnt/10T/zjy/output/models/",  # directory for saving checkpoint
        "log_path": "/mnt/10T/zjy/output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/"
    })  
        
    return config