import random
import pickle
import numpy as np
import torch
from torch import Tensor, device, dtype
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.data import CoraGraphDataset, CoraFullDataset, register_data_args, RedditDataset, CoauthorCSDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
import copy
from sklearn.metrics import roc_auc_score, average_precision_score

class Linear_IL(nn.Linear):
    def forward(self, input: Tensor, n_cls=10000, normalize = True) -> Tensor:
        if normalize:
            return F.linear(F.normalize(input,dim=-1), F.normalize(self.weight[0:n_cls],dim=-1), bias=None)
        else:
            return F.linear(input, self.weight[0:n_cls], bias=None)

def accuracy(logits, labels, cls_balance=True, ids_per_cls=None):
    if cls_balance:
        logi = logits.cpu().numpy()
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def mean_AP(args,logits, labels, cls_balance=True, ids_per_cls=None):
    eval_ogb = Evaluator(args.dataset)
    pos = (F.sigmoid(logits)>0.5)
    APs = 0
    if cls_balance:
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        input_dict = {"y_true": labels, "y_pred": logits}

        eval_result_ogb = eval_ogb.eval(input_dict)
        for c,ids in enumerate(ids_per_cls):
            TP_ = (pos[ids,c]*labels[ids,c]).sum()
            FP_ = (pos[ids,c]*(labels[ids, c]==False)).sum()
            med0 = TP_ + FP_ + 0.0001
            med1 = TP_ / med0
            APs += med1
        med2 = APs/labels.shape[1]

            #mAP_per_cls.append((TP / (TP+FP)).mean().item())
        #return (TP / (TP+FP)).mean().item()

        return med2.item()

def evaluate_batch_ncil(args,model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, p=None):
    model.eval()
    with torch.no_grad():
        old_protos = torch.cat([m_std[0] for m_std in p], dim=0)
        
        collator = dgl.dataloading.NodeCollator(g.cpu(), list(range(labels.shape[0])), args.nb_sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, 
                                                 collate_fn=collator.collate,
                                                 batch_size=args.batch_size, 
                                                 shuffle=False, 
                                                 drop_last=False)
        output = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions, _ = model.forward_batch(blocks, input_features)
            output_predictions = torch.nn.functional.normalize(output_predictions, dim=1) @ torch.nn.functional.normalize(old_protos, dim=1).T # 
            output = torch.cat((output,output_predictions),dim=0)
            output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels.cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluate_batch_yooop(args,model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, p=None):
    model.eval()
    with torch.no_grad():
        old_protos = p
        
        collator = dgl.dataloading.NodeCollator(g.cpu(), list(range(labels.shape[0])), args.nb_sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, 
                                                 collate_fn=collator.collate,
                                                 batch_size=args.batch_size, 
                                                 shuffle=False, 
                                                 drop_last=False)
        output = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            _ = model.forward_batch(blocks, input_features)
            output_predictions = model.second_last_h
            output_predictions = torch.nn.functional.normalize(output_predictions, dim=1) @ torch.nn.functional.normalize(old_protos, dim=1).T # 
            output = torch.cat((output,output_predictions),dim=0)
            output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels.cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluate_batch(args,model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None):
    model.eval()
    with torch.no_grad():
        collator = dgl.dataloading.NodeCollator(g.cpu(), list(range(labels.shape[0])), args.nb_sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, 
                                                 collate_fn=collator.collate,
                                                 batch_size=args.batch_size, 
                                                 shuffle=False, 
                                                 drop_last=False)
        output = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions, _ = model.forward_batch(blocks, input_features)
            output = torch.cat((output,output_predictions),dim=0)
            output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels.cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluate(model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        output, _ = model(g, features)
        logits = output[:, label_offset1:label_offset2]
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def yooop_evaluate(model, protos, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        _ = model(g, features)
        n = protos.size(0)
        output = model.second_last_h
        logits = torch.nn.functional.normalize(output, p=2, dim=1) @ torch.nn.functional.normalize(model.gat_layers[-1].weight[:n], p=2, dim=1).T 
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def ncil_evaluate(model, protos, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        _ = model(g, features)
        output = model.second_last_h
        ps = torch.cat([m_std[0] for m_std in protos], dim=0)
        logits = torch.nn.functional.normalize(output, dim=1) @ torch.nn.functional.normalize(ps, dim=1).T
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def _maha_dist(vectors, init_means, class_means, norm_cov_mat, t, total_classes):
    vectors = torch.tensor(vectors).to('cuda')
    if t > 0:
        vectors = torch.pow(vectors, 0.5)
    maha_dist = []
    for class_index in range(total_classes):
        if t == 0:
            dist = _mahalanobis(vectors, init_means[class_index], t) # 
        else:
            dist = _mahalanobis(vectors, class_means[class_index], t, norm_cov_mat[class_index]) # 
 
        maha_dist.append(dist)
    maha_dist = torch.tensor(np.array(maha_dist)).to('cuda')  # [nb_classes, N]  
    return maha_dist

def _mahalanobis(vectors, class_means, t, cov=None):
    if t > 0:
        class_means = torch.pow(class_means, 0.5)
    x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
    if cov is None:
        cov = torch.eye(vectors.size(-1))  # identity covariance matrix for euclidean distance
    
    inv_covmat = torch.linalg.pinv(cov).float().to('cuda')
    left_term = torch.matmul(x_minus_mu, inv_covmat)
    mahal = torch.matmul(left_term, x_minus_mu.T)
    return torch.diagonal(mahal, 0).cpu().numpy()
        
def fecam_evaluate(model, protos, g, features, labels, t, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    _init_protos, _protos, _common_cov = protos
    model.eval()
    with torch.no_grad():
        _ = model(g, features)
        vectors = model.second_last_h.cpu().data.numpy()
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
        dists = _maha_dist(vectors, _init_protos, _protos, _common_cov, t, label_offset2)
        logits = -dists.T  # [N, nb_classes], choose the one with the smallest distance
        
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        
def evaluate_batch_fecam(args, model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, p=None, t=None):
    _init_protos, _protos, _norm_cov_mat = p
    model.eval()
    with torch.no_grad():
        collator = dgl.dataloading.NodeCollator(g.cpu(), list(range(labels.shape[0])), args.nb_sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, 
                                                 collate_fn=collator.collate,
                                                 batch_size=args.batch_size, 
                                                 shuffle=False, 
                                                 drop_last=False)
        output = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            _ = model.forward_batch(blocks, input_features)
            vectors = model.second_last_h.cpu().data.numpy()
            
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
            dists = _maha_dist(vectors, _init_protos, _protos, _norm_cov_mat, t, label_offset2)
            logits = -dists.T  # [N, nb_classes], choose the one with the smallest distance
                
            output = torch.cat((output, logits),dim=0)
            output_l = torch.cat((output_l, output_labels), dim=0)

        logits = output
        if cls_balance:
            return accuracy(logits, labels.cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)

class incremental_graph_trans_(nn.Module):
    def __init__(self, dataset, n_cls):
        super().__init__()
        # transductive setting
        self.graph, self.labels = dataset[0]
        #self.graph = dgl.add_reverse_edges(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1] # 节点特征维度
        self.n_cls = n_cls
        self.n_nodes = self.labels.shape[0] # 节点数
        self.tr_va_te_split = dataset[1] # 数据集的划分

    def get_graph(self, tasks_to_retain=[], node_ids = None, remove_edges=True): # 列表中包含需要保存的类别
        # get the partial graph
        # tasks-to-retain: classes retained in the partial graph
        # tasks-to-infer: classes to predict on the partial graph
        node_ids_ = copy.deepcopy(node_ids)
        node_ids_retained = []
        ids_train_old, ids_valid_old, ids_test_old = [], [], []
        if len(tasks_to_retain) > 0:
            # retain nodes according to classes
            for t in tasks_to_retain:
                ids_train_old.extend(self.tr_va_te_split[t][0])
                ids_valid_old.extend(self.tr_va_te_split[t][1])
                ids_test_old.extend(self.tr_va_te_split[t][2])
                node_ids_retained.extend(self.tr_va_te_split[t][0] + self.tr_va_te_split[t][1] + self.tr_va_te_split[t][2])
            subgraph_0 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True)
            if node_ids_ is None:
                subgraph = subgraph_0
        if node_ids_ is not None:
            # retrain the given nodes
            if not isinstance(node_ids_[0],list):
                # if nodes are not divided into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_, store_ids=True)
                if remove_edges:
                    # to facilitate the methods like ER-GNN to only retrieve nodes
                    n_edges = subgraph_1.edges()[0].shape[0]
                    subgraph_1.remove_edges(list(range(n_edges)))
            elif isinstance(node_ids_[0],list):
                # if nodes are diveded into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_[0], store_ids=True) # load the subgraph containing nodes of the first task
                node_ids_.pop(0)
                for ids in node_ids_:
                    # merge the remaining nodes
                    subgraph_1 = dgl.batch([subgraph_1,dgl.node_subgraph(self.graph, ids, store_ids=True)])

            if len(tasks_to_retain)==0:
                subgraph = subgraph_1

        if len(tasks_to_retain)>0 and node_ids is not None:
            subgraph = dgl.batch([subgraph_0,subgraph_1])

        old_ids = subgraph.ndata['_ID'].cpu() # 原始的在原图中的节点序号（构建子图后，节点的序号会从0开始重新排列）
        ids_train = [(old_ids == i).nonzero()[0][0].item() for i in ids_train_old] # 节点在新的子图中的序号
        ids_val = [(old_ids == i).nonzero()[0][0].item() for i in ids_valid_old]
        ids_test = [(old_ids == i).nonzero()[0][0].item() for i in ids_test_old]
        node_ids_per_task_reordered = []
        for c in tasks_to_retain:
            ids = (subgraph.ndata['label'] == c).nonzero()[:, 0].view(-1).tolist()
            node_ids_per_task_reordered.append(ids)
        subgraph = dgl.add_self_loop(subgraph)

        return subgraph, node_ids_per_task_reordered, [ids_train, ids_val, ids_test]

def train_valid_test_split(ids,ratio_valid_test):
    va_te_ratio = sum(ratio_valid_test)
    train_ids, va_te_ids = train_test_split(ids, test_size=va_te_ratio)
    return [train_ids] + train_test_split(va_te_ids, test_size=ratio_valid_test[1]/va_te_ratio)

class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name='ogbn-arxiv',IL='class',default_split=False,ratio_valid_test=None,args=None):
        r""""
        name: name of the dataset
        IL: use task- or class-incremental setting
        default_split: if True, each class is split according to the splitting of the original dataset, which may cause the train-val-test ratio of different classes greatly different
        ratio_valid_test: in form of [r_val,r_test] ratio of validation and test set, train set ratio is directly calculated by 1-r_val-r_test
        """

        # return an incremental graph instance that can return required subgraph upon request
        # 都没有加自环
        if name[0:4] == 'ogbn':
            data = DglNodePropPredDataset(name, root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name in ['CoraFullDataset', 'CoraFull','corafull', 'CoraFull-CL','Corafull-CL']:
            data = CoraFullDataset(raw_dir=f'{args.ori_data_path}')
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['reddit','Reddit','Reddit-CL']:
            data = RedditDataset(self_loop=False, raw_dir=f'{args.ori_data_path}')
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name == 'Products-CL':
            data = DglNodePropPredDataset('ogbn-products', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name == 'CS-CL': 
            data = CoauthorCSDataset(raw_dir=f'{args.ori_data_path}')
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        else:
            print('invalid data name')
        n_cls = data.num_classes
        cls = [i for i in range(n_cls)]
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls} # {0: [1,3,5]} 类别和对应的节点索引
        cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map} # {0: 3} 类别和对应的节点数量
        for c in cls_sizes:
            if cls_sizes[c] < 2:
                cls.remove(c) # remove classes with less than 2 examples, which cannot be split into train, val, test sets
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls} # 过滤后的类别以及对应的节点索引
        n_cls = len(cls) # 过滤后的类别数
        if default_split: # 默认划分数据集
            split_idx = data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
                "test"].tolist()
            tr_va_te_split = {c: [list(set(cls_id_map[c]).intersection(set(train_idx))),
                                  list(set(cls_id_map[c]).intersection(set(valid_idx))),
                                  list(set(cls_id_map[c]).intersection(set(test_idx)))] for c in cls}

        elif not default_split:
            split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            try:
                tr_va_te_split = pickle.load(open(split_name, 'rb')) # could use same split across different experiments for consistency
            except:
                if ratio_valid_test[1] > 0:
                    tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
                                      for c in
                                      cls}
                    print(f'splitting is {ratio_valid_test}')
                elif ratio_valid_test[1] == 0:
                    tr_va_te_split = {c: [cls_id_map[c], [], []] for c in
                                      cls}
                with open(split_name, 'wb') as f:
                    pickle.dump(tr_va_te_split, f) # 保存数据集划分
        super().__init__([[graph, label], tr_va_te_split], n_cls)
        
