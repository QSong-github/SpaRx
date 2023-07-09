import torch
import os
import math
import numpy as np
import pandas as pd
import yaml
from utils.helper import print_loss_stat, seed_everything, eval_results

from loader import load_data, load_unsuper_data
from model import TLModel
from tqdm import tqdm

@torch.no_grad()
def val(cfg, model, tar):
    model.set_input(tar, tar)
    pred = model.inference()
    pred = pred.detach().cpu()
    label = tar.y.detach().cpu()

    f1 = eval_results(pred, label, cfg['TEST']['Verbose'])
    # print("f1 is %.4f"%(f1))
    return f1

@torch.no_grad()
def test(cfg):
    tar_data_root = cfg['DATASET']['target_data_root']
    tar_label_root = cfg['DATASET']['target_label_root']
    tar_adj_root = cfg['DATASET']['target_adj_root']
    tar, tar_index = load_unsuper_data(root=tar_data_root,
                            adj_root=tar_adj_root,
                            label_root=tar_label_root, return_index=True)

    model = TLModel(cfg)
    model.load(cfg['TEST']['Save_path'])

    pred = model.set_input(tar, tar)
    pred = model.inference()
    # src_emb, tar_emb = model.get_emb()
    pred = pred.detach().cpu()

    label = tar.y.detach().cpu()
    f1 = eval_results(pred, label)

    # save pred to csv
    True_label = label.numpy()

    df = pd.DataFrame(index=tar_index)
    df['pred'] = pred.numpy()
    df['true_label'] = True_label

    # if not os.path.exists(os.path.join('results', cfg['Name'])):
    #     os.makedirs(os.path.join('results', cfg['Name']))
    
    df.to_csv(os.path.join(cfg['TEST']['Pred'],'prediction.csv'))        
    return f1

@torch.no_grad()
def save_emb(cfg):
    src_data_root = cfg['DATASET']['Source_data_root']
    src_label_root = cfg['DATASET']['Source_label_root']
    src_adj_root = cfg['DATASET']['Source_adj_root']

    src, src_index = load_unsuper_data(root=src_data_root,
                            adj_root=src_adj_root,
                            label_root=src_label_root, return_index=True)

    tar_data_root = cfg['DATASET']['target_data_root']
    tar_label_root = cfg['DATASET']['target_label_root']
    tar_adj_root = cfg['DATASET']['target_adj_root']
    tar, tar_index = load_unsuper_data(root=tar_data_root,
                            adj_root=tar_adj_root,
                            label_root=tar_label_root, return_index=True)


    model = TLModel(cfg)
    model.load(cfg['TEST']['Save_path'])

    pred = model.set_input(src, tar)
    src_emb, tar_emb = model.get_emb()

    src_emb = src_emb.detach().cpu().numpy()
    tar_emb = tar_emb.detach().cpu().numpy()

    df = pd.DataFrame(src_emb, index=src_index)
    # if not os.path.exists(os.path.join('results', cfg['Name'])):
    #     os.makedirs(os.path.join('results', cfg['Name']))
    df.to_csv(os.path.join(cfg['TEST']['Pred'],'src_emb.csv'))        

    df = pd.DataFrame(tar_emb, index=tar_index)
    df.to_csv(os.path.join(cfg['TEST']['Pred'],'tar_emb.csv'))        
    
def train(cfg):
    # build data
    src_data_root = cfg['DATASET']['Source_data_root']
    src_label_root = cfg['DATASET']['Source_label_root']
    src_adj_root = cfg['DATASET']['Source_adj_root']

    src, src_train_idx, src_test_idx = load_data(root=src_data_root,
                                    adj_root=src_adj_root,
                                    label_root=src_label_root,
                                    num_val=0.2, random_state=cfg['SEED'])
    
    tar_data_root = cfg['DATASET']['target_data_root']
    tar_label_root = cfg['DATASET']['target_label_root']
    tar_adj_root = cfg['DATASET']['target_adj_root']
    tar = load_unsuper_data(root=tar_data_root,
                            adj_root=tar_adj_root,
                            label_root=tar_label_root)


    best_loss = 10000000
    # best_f1 = 0
    best_epoch = 0
    # # build up model
    model = TLModel(cfg)
    for epoch in tqdm(range(cfg['TRAIN']['Epochs'])):
        # print('best_F1 is %f in Epoch %d' %(best_f1, best_epoch))
        model.set_input(src, tar)
        model.update_parameters()
        loss_stat = model.get_current_loss()
        loss = loss_stat['overall_loss'].item()

        # print_loss_stat(loss_stat, epoch, cfg['TRAIN']['Epochs'])

        # f1 = val(cfg, model, tar)
        # if best_f1 < f1:
        #     model.save('best_f1')
        #     best_f1 = f1
        #     best_epoch = epoch
        if best_loss > loss:
            model.save('best_loss')
            best_loss = loss
            best_epoch = epoch
    #src_emb, tar_emb = model.get_emb()
    f1 = val(cfg, model, tar)
    return best_loss

if __name__ == '__main__':
    yaml_file = 'parameter_tune.yml'
    txt = open('para_tune_log.txt', 'a')
    with open(yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(cfg['SEED'])

    best_loss = train(cfg)
    # test(cfg)
    #best_f1s.append(f1)
    ## save_emb
    # save_emb(cfg)


    d1 = cfg['MODEL']['NUM_HIDDENS'][0]
    d2 = cfg['MODEL']['NUM_HIDDENS'][1]
    momentum = cfg['TRAIN']['Momentum']
    Wd = cfg['TRAIN']['Weight_decay']
    gd = cfg['TRAIN']['grad_clip']
    msg = 'd1: %d, d2: %d, momentum: %.2f, Wd: %.2f, gd: %.2f, best_loss: %.4f, f1: %.4f\n'%(d1, d2, momentum, Wd, gd, best_loss)
    txt.write(msg)
