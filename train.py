import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from PIL import Image
from Config  import DE_360IQA_config
from torch.utils.tensorboard import SummaryWriter
import logging
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from MMBO.dataset import DE_t_Dataset,DE_t_vp_Dataset,DE_t_re_Dataset,DE_t_new_ERP_Dataset
from util import *
from MMBO.MMBO_model.ECB_ECB_IJCV import MMBO


print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    cpu_num = 8
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    setup_seed(20)
    cfg=DE_360IQA_config()
    
    
    cfg.log_file = cfg.model_name + ".log"
    cfg.tensorboard_path = os.path.join(cfg.tensorboard_path, cfg.type_name)
    cfg.tensorboard_path = os.path.join(cfg.tensorboard_path, cfg.model_name)

    cfg.ckpt_path = os.path.join(cfg.ckpt_path, cfg.type_name)
    cfg.ckpt_path = os.path.join(cfg.ckpt_path, cfg.model_name)

    cfg.log_path = os.path.join(cfg.log_path, cfg.type_name)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    if not os.path.exists(cfg.tensorboard_path):
        os.makedirs(cfg.tensorboard_path)

    set_logging(cfg)
    logging.info(cfg)

    writer = SummaryWriter(cfg.tensorboard_path)
    
    dataset=DE_t_Dataset# 创建数据集实例
    train_dataset = dataset(
        cfg=DE_360IQA_config(),
        csv_path=cfg.csv_path,
        transform= transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),

            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
transform1 = transforms.Compose([
            transforms.Resize((224, 224),interpolation=BICUBIC),
            # transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
)
    
    

    test_dataset = dataset(
        cfg=DE_360IQA_config(),
        csv_path=cfg.test_csv_path,
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
        transform1 = transforms.Compose([
            transforms.Resize((224, 224),interpolation=BICUBIC),
            # transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    )


    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(test_dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,drop_last=True, num_workers=cfg.num_workers,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, drop_last=True,num_workers=cfg.num_workers,shuffle=False)
    model=MMBO()

    model = model.to(device)
        # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)

     # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0
    for epoch in range(0, cfg.epochs):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, srcc, plcc = train_clip_epoch(epoch, model, criterion, optimizer, scheduler, train_loader)
        print("[train epoch %d/%d] loss: %.6f, srcc: %.4f, plcc: %.4f, lr: %.6f, time: %.2f min" % \
                (epoch+1, cfg.epochs, loss_val, srcc, plcc, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", srcc, epoch)
        writer.add_scalar("PLCC", plcc, epoch)

        if (epoch + 1) % cfg.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, srcc_v, plcc_v = eval_clip_epoch(cfg, epoch, model, criterion, test_loader)
            print("[val epoch %d/%d] loss: %.6f, srcc_v: %.4f, plcc_v: %.4f, lr: %.6f, time: %.2f min" % \
                (epoch+1, cfg.epochs, loss, srcc_v, plcc_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            logging.info('Eval done...')

            if srcc_v + plcc_v > main_score:
                main_score = srcc_v + plcc_v
                best_srocc = srcc_v
                
                best_plcc = plcc_v

                logging.info('======================================================================================')
                logging.info(
                    '============================== best main score is {} ================================='.format(
                        main_score))
                logging.info('======================================================================================')

                # save weights
                model_name = "best_epoch.pt"
                model_save_path = os.path.join(cfg.ckpt_path, model_name)
                torch.save(model.state_dict(), model_save_path)
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))



