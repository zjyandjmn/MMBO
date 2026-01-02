import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from tqdm import tqdm
import logging
import numpy as np
from scipy.stats import spearmanr, pearsonr
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_squared_error(actual, predicted, squared=True):
    """MSE or RMSE (squared=False)"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    error = predicted - actual
    res = np.mean(error**2)
    
    if squared==False:
        res = np.sqrt(res)
    
    return res

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
def train_clip_epoch(epoch, model, criterion, optimizer, scheduler, train_loader):
    losses = []
    model.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    for image1,patch,position,labels,text in tqdm(train_loader):
        image1=image1.cuda()
        patch = patch.cuda()
        position = position.cuda()
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        pred_d = model(image1,patch,position,text)
        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rmse = mean_squared_error(np.squeeze(pred_epoch),np.squeeze(labels_epoch),squared=False)

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p,rmse))

    return ret_loss, rho_s, rho_p


def eval_clip_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        for image1,patch,position,labels,text in tqdm(test_loader):
            pred = 0
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
            pred = net(image1.to(device),patch.to(device),position.to(device),text)
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = mean_squared_error(np.squeeze(pred_epoch),np.squeeze(labels_epoch),squared=False)
        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ==== RMSE{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p,rmse))
        return np.mean(losses), rho_s, rho_p
    
