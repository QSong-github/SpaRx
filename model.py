import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, LayerNorm, GATConv, GCNConv
from torch_geometric.utils import softmax
import torch.nn.functional as F

from torch.autograd import Function
import os

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Feature_extractor(nn.Module):
    def __init__(self, in_dim=15962, num_hiddens=[512, 64], ConvFunc=TransformerConv):
        super().__init__()
        d1, d2 = num_hiddens[0], num_hiddens[1]
        self.conv1 = ConvFunc(in_dim, d1)
        self.conv2 = ConvFunc(d1, d2)
        self.activate = F.elu

    def forward(self, x, edge_index):
        h1 = self.activate(self.conv1(x, edge_index))
        h2 = self.activate(self.conv2(h1, edge_index))
        return h2 


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, last=False, ConvFunc=TransformerConv):
        super().__init__()
        self.fc = ConvFunc(in_dim, out_dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.last = last
        
    def forward(self, x, edge_index):
        h = self.fc(x, edge_index)
        if self.last:
            return h
        else:
            out = self.dropout(self.relu(h))
            return out

class Base_classfier(nn.Module):
    def __init__(self, num_hiddens=[64, 32, 512, 512], out_dim=2, ConvFunc=TransformerConv):
        super().__init__()
        self.layer_list = nn.ModuleList()
        for i in range(1, len(num_hiddens)):
            self.layer_list.append(MLP(num_hiddens[i-1], num_hiddens[i], ConvFunc=ConvFunc))
        self.layer_list.append(MLP(num_hiddens[-1], out_dim, ConvFunc=ConvFunc))

    def forward(self, x, edge_index):
        for layer in self.layer_list:
            x = layer(x, edge_index)
        return x

class SRC_classifier(nn.Module):
    def __init__(self, num_classes=2, in_dim=15962, num_hiddens=[512, 64], ConvFunc=TransformerConv):
        super().__init__()
        # self.shareNet = Feature_extractor(in_dim, num_hiddens)
        self.shareNet = Feature_extractor(in_dim, num_hiddens=num_hiddens, ConvFunc=ConvFunc)

        # self.bottleneck = MLP(num_hiddens[-1], 32)
        # here make some change to the original DAAN
        self.classifier =Base_classfier(num_hiddens=[64, 32, 16], out_dim=2, ConvFunc=ConvFunc)

    def forward(self, x, edge_index):
        emb = self.shareNet(x, edge_index)
        # emb2 = self.bottleneck(emb)
        pred = self.classifier(emb, edge_index)
        return pred


class TL_classifier(nn.Module):
    def __init__(self, num_classes=2, in_dim=15962, num_hiddens=[512, 64], ConvFunc=TransformerConv):
        super().__init__()
        self.shareNet = Feature_extractor(in_dim, num_hiddens=num_hiddens,ConvFunc=ConvFunc)
        d1, d2 = num_hiddens[0], num_hiddens[1]
        self.src_classifier = Base_classfier(num_hiddens=[d2, d2//2, d2//4], out_dim=num_classes, ConvFunc=ConvFunc)

        self.global_domain_classifier = Base_classfier(num_hiddens=[d2, d2//2, d2//4], out_dim=num_classes, ConvFunc=ConvFunc)

        self.dcis = nn.ModuleList()
        for nc in range(num_classes):
            # out_dim = 2, one for src, one for target
            adv_classifier = Base_classfier(num_hiddens=[d2, d2//2, d2//4], out_dim=2, ConvFunc=ConvFunc)
            self.dcis.append(adv_classifier)

        self.softmax = nn.Softmax(dim=1)
        self.classes = num_classes

    def forward(self, src_x, src_edge, tar_x, tar_edge, alpha=0.0):
        src_emb = self.shareNet(src_x, src_edge)
        src_pred = self.src_classifier(src_emb, src_edge)

        tar_emb = self.shareNet(tar_x, tar_edge)
        tar_pred = self.src_classifier(tar_emb, tar_edge)

        s_out = []
        t_out = []

        p_src = self.softmax(src_pred)
        p_tar = self.softmax(tar_pred)

        if self.training == True:
            # rev Grad
            src_rev_feat = ReverseLayerF.apply(src_emb, alpha)
            tar_rev_feat = ReverseLayerF.apply(tar_emb, alpha)

            src_domain_out = self.global_domain_classifier(src_rev_feat, src_edge)
            tar_domain_out = self.global_domain_classifier(tar_rev_feat, tar_edge)

            for i in range(self.classes):
                ps = p_src[:, i].reshape((src_emb.shape[0], 1))
                fs = ps * src_rev_feat

                pt = p_tar[:, i].reshape((tar_emb.shape[0], 1))
                ft = pt * tar_rev_feat

                outsi = self.dcis[i](fs, src_edge)
                s_out.append(outsi)

                outti = self.dcis[i](ft, tar_edge)
                t_out.append(outti)
        else:
            src_domain_out = 0
            tar_domain_out = 0
            s_out = [0] * self.classes
            t_out = [0] * self.classes

        return src_pred, src_domain_out, tar_domain_out, s_out, t_out, src_emb, tar_emb


class TLModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg['MODEL']['Conv'] == "GCNConv":
            print('Using GCNConv')
            conv = GCNConv
        elif cfg['MODEL']['Conv'] == "GATConv":
            print('Using GATConv')
            conv = GATConv
        elif cfg['MODEL']['Conv'] == "TransformerConv":
            print('Using TransformerConv')
            conv = TransformerConv
        else:
            raise NotImplementedError

        self.model = TL_classifier(num_classes=cfg['MODEL']['num_classes'], 
                            in_dim=cfg['MODEL']['INPUT_DIM'], num_hiddens=cfg['MODEL']['NUM_HIDDENS'],ConvFunc=conv)
        if cfg['Use_CUDA']:
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(cfg['TRAIN']['lr']), 
                    momentum=float(cfg['TRAIN']['Momentum']), weight_decay=float(cfg['TRAIN']['Weight_decay']))

        self.loss_stat = {}
        self.use_cuda = cfg['Use_CUDA']
        self.grad_clip = cfg['TRAIN']['grad_clip']
        # create save path
        self.save_path = cfg['TRAIN']['Save_path']
        self.name = cfg['Name']
        self.classes = cfg['MODEL']['num_classes']

        self.celoss = nn.CrossEntropyLoss()

        self.dm = 0
        self.dc = 0
        self.mu = 0.5
        self.length = 0

    def set_input(self, src, tar):
        self.length = src.x.shape[0]
        if self.use_cuda:
            self.src = src.to(self.device, non_blocking=True)
            self.tar = tar.to(self.device, non_blocking=True)
        else:
            self.src = src
            self.tar = tar

    def inference(self):
        self.model.eval()
        self.pred, self.src_domain_out, self.tar_domain_out, self.s_out, self.t_out, self.src_emb, self.tar_emb = self.model(self.tar.x, self.tar.edge_index, self.tar.x, self.tar.edge_index)
        return self.pred.argmax(dim=1)

    def forward(self):
        self.model.train()
        self.pred, self.src_domain_out, self.tar_domain_out, self.s_out, self.t_out, self.src_emb, self.tar_emb = self.model(self.src.x, self.src.edge_index, self.tar.x, self.tar.edge_index)
    
    def get_emb(self):
        self.model.eval()
        _, _, _, _,_, src_emb, tar_emb = self.model(self.src.x, self.src.edge_index, self.tar.x, self.tar.edge_index)
        return src_emb, tar_emb

    def compute_loss(self):
        # global domain loss
        sdomain_label = torch.zeros(self.src_domain_out.shape[0]).long().to(self.device)
        src_global_domain_loss = self.celoss(self.src_domain_out, sdomain_label)
        tdomain_label = torch.ones(self.tar_domain_out.shape[0]).long().to(self.device)
        tar_global_domain_loss = self.celoss(self.tar_domain_out, tdomain_label)
        
        # local domain loss
        loss_s = 0
        loss_t = 0
        tmpd_c = 0

        # dc dm
        dc = 0
        dm = 0
        for i in range(self.classes):
            loss_si = self.celoss(self.s_out[i], sdomain_label)
            loss_ti = self.celoss(self.t_out[i], tdomain_label)

            loss_s += loss_si
            loss_t += loss_ti
            tmpd_c += 2 * (1 - 2 * (loss_si + loss_ti))
        tmpd_c /= self.classes

        global_loss = 0.05 * (src_global_domain_loss + tar_global_domain_loss)
        local_loss = 0.01 * (loss_s + loss_t)

        joint_loss = (1- self.mu) * global_loss + self.mu * local_loss
        # soft_loss for src_classifier
        train_mask = self.src.train_mask
        soft_loss = self.celoss(self.pred[train_mask], self.src.y.long()[train_mask])

        dc = dc + tmpd_c.cpu().item()
        dm = dm + 2 * (1 - 2 * global_loss.cpu().item())

        self.loss = soft_loss - joint_loss

        self.loss_stat['overall_loss'] = self.loss
        self.loss_stat['src_celoss'] = soft_loss
        self.loss_stat['global_loss'] = global_loss
        self.loss_stat['local_loss'] = local_loss

        self.dc = dc / self.length
        self.dm = dm / self.length
        self.mu = 1 - self.dm / (self.dm + self.dc)
        return self.loss

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def update_parameters(self):
        self.forward()
        self.compute_loss()
        self.backward()

    def get_current_loss(self):
        return self.loss_stat

    def save(self, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, name+'.pth'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class Src_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = SRC_classifier(num_classes=cfg['MODEL']['num_classes'], 
                            in_dim=cfg['MODEL']['INPUT_DIM'])
        if cfg['Use_CUDA']:
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(cfg['TRAIN']['lr']), 
                    momentum=float(cfg['TRAIN']['Momentum']), weight_decay=float(cfg['TRAIN']['Weight_decay']))

        self.loss_stat = {}
        self.use_cuda = cfg['Use_CUDA']
        self.grad_clip = cfg['TRAIN']['grad_clip']
        self.save_path = cfg['TRAIN']['Save_path']
        self.name = cfg['Name']

        self.celoss = nn.CrossEntropyLoss()


    def set_input(self, src):
        if self.use_cuda:
            self.src = src.to(self.device, non_blocking=True)
        # self.src_x = src.x
        # self.src_edge = src.edge_index
        # self.src_y = src.y.long()

    def inference(self):
        self.model.eval()
        self.pred = self.model(self.src.x, self.src.edge_index)
        return self.pred.argmax(dim=1)

    def forward(self):
        self.model.train()
        self.pred = self.model(self.src.x, self.src.edge_index)
        # print(self.pred.shape)
        # print(self.pred.detach().cpu())
        # exit(0)

    def compute_loss(self):
        # the label should be in long tensor
        # print(self.pred)
        # exit(0)
        # self.pred = softmax(self.pred)
        self.loss = self.celoss(self.pred[self.src.train_mask], self.src.y.long()[self.src.train_mask])
        # self.loss = self.celoss(self.pred, self.src_y)
        # self.loss = self.soft_loss

        # self.loss_stat['soft_loss'] = self.soft_loss
        self.loss_stat['loss'] = self.loss

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def update_parameters(self):
        self.forward()
        self.compute_loss()
        self.backward()

    def get_current_loss(self):
        return self.loss_stat

    def save(self, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, name+'.pth'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class DAANNet(nn.Module):
    def __init__(self, num_classes=2, in_dim=15962, num_hiddens=[512, 64], ConvFunc=TransformerConv):
        super().__init__()
        self.shareNet = Feature_extractor(in_dim, num_hiddens=num_hiddens, ConvFunc=ConvFunc)

        self.bottleneck = nn.Linear(num_hiddens[-1], 32)
        self.source_fc = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classes = num_classes

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(32, 512))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(512, 512))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(512, 2))

        # local domain discriminator
        self.dcis = nn.Sequential()
        self.dci = {}
        for i in range(num_classes):
            self.dci[i] = nn.Sequential()
            self.dci[i].add_module('fc1', nn.Linear(32, 512))
            self.dci[i].add_module('relu1', nn.ReLU(True))
            self.dci[i].add_module('dpt1', nn.Dropout())
            self.dci[i].add_module('fc2', nn.Linear(512, 512))
            self.dci[i].add_module('relu2', nn.ReLU(True))
            self.dci[i].add_module('dpt2', nn.Dropout())
            self.dci[i].add_module('fc3', nn.Linear(512, 2))
            self.dcis.add_module('dci_'+str(i), self.dci[i])

    def forward(self, source, target, alpha=0.0):
        source_share = self.shareNet(source.x, source.edge_index)
        source_share = self.bottleneck(source_share)
        source_emb = self.source_fc(source_share)
        p_source = self.softmax(source_emb)

        target_share = self.shareNet(target.x, target.edge_index)
        target_share = self.bottleneck(target_share)
        t_label = self.source_fc(target_share)
        p_target = self.softmax(t_label)
        t_label = t_label.data.max(1)[1]

        s_out = []
        t_out = []
        if self.trainning == True:
            # RevGrad
            s_reverse_feature = ReverseLayerF.apply(source_share, alpha)
            t_reverse_feature = ReverseLayerF.apply(target_share, alpha)
            s_domain_output = self.domain_classifier(s_reverse_feature)
            t_domain_output = self.domain_classifier(t_reverse_feature)
            # p*feature-> classifier_i ->loss_i
            for i in range(self.classes):
                ps = p_source[:, i].reshape((target_share.shape[0],1))
                fs = ps * s_reverse_feature
                pt = p_target[:, i].reshape((target_share.shape[0],1))
                ft = pt * t_reverse_feature
                outsi = self.dcis[i](fs)
                s_out.append(outsi)
                outti = self.dcis[i](ft)
                t_out.append(outti)
        else:
            s_domain_output = 0
            t_domain_output = 0
            s_out = [0]*self.classes
            t_out = [0]*self.classes

        return source_emb, s_domain_output, t_domain_output, s_out, t_out
