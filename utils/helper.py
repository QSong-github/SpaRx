import random
import torch
import numpy as np
import os

def seed_everything(seed):
    # seed = 10
    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False 


def print_loss_stat(loss_stat, epoch, total_epoch):
    msg ='%d/%d'%(epoch, total_epoch)
    for k, v in loss_stat.items():
        msg += '\t%s:%.4f'%(k, v.item())
    print(msg)


def eval_results(pred, gt, verbose=False):
    # TP, TN, FP, FN = 0,0,0,0
    TP = ((pred == 1) & (gt == 1)).sum()
    TN = ((pred == 0) & (gt == 0)).sum()
    FP = ((pred == 1) & (gt == 0)).sum()
    FN = ((pred == 0) & (gt == 1)).sum()

    label_tp = (gt == 1).sum()
    label_tn = (gt == 0).sum()

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    sensitive = recall
    specity = TN / (TN + FP) if (TN + FP) else 0
    acc = (TP + TN) / (label_tp + label_tn)

    F1 = (2 * precision * recall) / (precision + recall)
    message = '\n------------------------results----------------------\n'
    message += '{:>10d}\t{:>10d}\n'.format(TP,label_tp)
    message += '{:>10d}\t{:>10d}\n'.format(TN,label_tn)
    message += '{:>10}\t{:>10.4f}\n'.format('acc:', acc)
    message += '{:>10}\t{:>10.4f}\n'.format('precision:', precision)
    message += '{:>10}\t{:>10.4f}\n'.format('recall:', recall)
    message += '{:>10}\t{:>10.4f}\n'.format('Specificity:', specity)
    message += '{:>10}\t{:>10.4f}\n'.format('Sensitivity:', sensitive)
    message += '{:>10}\t{:>10.4f}\n'.format('F1-measure:', F1)
    message += '------------------------------------------------------\n'
    if verbose:
        print(message)
    return F1