import torch
import dgl
import copy
import itertools
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import time

class SupConLoss(torch.nn.Module):
    def __init__(self, args, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.args = args
        self.temperature = temperature
    def forward(self, features, protos, samples, labels_f, labels_p, repeats_num):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        mask = torch.eq(labels_f.unsqueeze(1), labels_p.unsqueeze(0)).float().to(device) # (B,K)
        samples_labels = torch.repeat_interleave(labels_p, repeats=repeats_num)
        m1 = torch.ones_like(mask)
        m2 = (~torch.eq(labels_f.unsqueeze(1), samples_labels.unsqueeze(0))).float().to(device)
        m12 = torch.cat([m1, m2], dim=1)
        mask_ = torch.zeros_like(m2)
        mask = torch.cat([mask, mask_], dim=1)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, torch.cat([protos, samples], dim=0).T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) * m12
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = -1 * mean_log_prob_pos
        return loss.mean()

def solve_ranking_leaked(adj_matrix):
    cal_zero = adj_matrix.sum(1) == 0
    adj_matrix[cal_zero] = 1

def calc_out_degree_ratio(adj_matrix):
    out_degree = torch.sum(adj_matrix, dim=1, keepdim=True)
    adj_matrix /= out_degree

def pagerank(adj_matrix, alpha=0.85, tol=1e-6):
    pr_vec = torch.ones(adj_matrix.shape[1], device=adj_matrix.device) / adj_matrix.shape[1]
    jump_value = (1 - alpha) / adj_matrix.shape[1] 
    jump_vec = jump_value * torch.ones(adj_matrix.shape[1], device=adj_matrix.device) 
    for n_iter in range(1, 201):
        pr_new = alpha * (pr_vec @ adj_matrix) + jump_vec

        if torch.norm(pr_new - pr_vec, p=1) < tol:
            # print(n_iter)
            break
        pr_vec = pr_new
    return pr_vec

def calc_pagerank(adj_matrix):
    solve_ranking_leaked(adj_matrix)
    calc_out_degree_ratio(adj_matrix)
    return pagerank(adj_matrix)

class Linear(torch.nn.Module):
    def __init__(self, feature_dim):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.b = nn.Parameter(torch.zeros(feature_dim))
    def forward(self, input):
        return nn.functional.linear(input, self.w, self.b)
    
def norm(x):
    return nn.functional.normalize(x, dim=1)

def relation_distillation(features, features_old, protos, args): # [old_struc_protos, old_low_protos, old_high_protos]
    size = features.shape[0] if features.shape[0] <= args.ncil_args['S'] else args.ncil_args['S'] # 100
    labels = torch.repeat_interleave(torch.tensor([i for i in range(protos[0].shape[0])]).to(features.device), repeats=size)
    indices = torch.randperm(features.shape[0])[:size]
    loss_ = 0.
    for j in range(len(protos)):
        pseudo_features_n, pseudo_features_o = [], []
        for i in range(protos[j].shape[0]):
            lam = np.random.beta(9, 21, size)
            lam[lam > 0.4] = 0.2 # 0.4
            lam = torch.from_numpy(lam).unsqueeze(1).to(device='cuda:{}'.format(args.gpu)) # (B,1)
            temp_n = (1 - lam) * protos[j][i].unsqueeze(0) + lam * features[indices]
            temp_o = (1 - lam) * protos[j][i].unsqueeze(0) + lam * features_old[indices]
            pseudo_features_n.append(temp_n)
            pseudo_features_o.append(temp_o)
        pseudo_features_n = torch.cat(pseudo_features_n, dim=0).float()
        pseudo_features_o = torch.cat(pseudo_features_o, dim=0).float()
        indices_ = torch.argmax(norm(pseudo_features_o) @ norm(protos[j]).T, dim=1) == labels
        h_n = norm(pseudo_features_n[indices_])
        h_o = norm(pseudo_features_o[indices_])
        loss_ += torch.dist(h_n @ norm(protos[j]).T, h_o @ norm(protos[j]).T) 
    
    return loss_ / len(protos)

class NET(torch.nn.Module):

    """
    Bare model baseline for NCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """

    def __init__(self,
                 model,
                 task_manager,
                 args):
        """
        The initialization of the baseline

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.
        """
        super(NET, self).__init__()

        self.task_manager = task_manager

        # backbone model
        self.net = model

        # setup optimizer
        self.args = args
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup loss
        self.ce = torch.nn.functional.cross_entropy
        
        self.flag = True
        self.epochs = 0
        self.protos = []
        
        self.pr_vec = None
        self.supcon = SupConLoss(args)
        self.k = args.ncil_args['K'] # 10
        
        self.g = None

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the class-IL setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
        :param dataset: The entire dataset (not in use in the current baseline).

        """
        if t == 1 and self.flag: 
            for params_group in self.opt.param_groups:
                params_group['lr'] = 1e-4
            self.flag = False
        
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        self.net.zero_grad()
        
        if last_epoch == 1:
            adj = dgl.remove_self_loop(g).adjacency_matrix().to_dense()
            pr_vec = calc_pagerank(adj)
            self.pr_vec = pr_vec[train_ids]
 
        offset1, offset2 = self.task_manager.get_label_offset(t) 
        output_labels = labels[train_ids]
        output = self.net(g, features)
        if isinstance(output,tuple):
            output = output[0]
        output = output[train_ids]
        h = output
        
        online_protos, boundary_samples, repeats_num = [], [], []
        for c in args.task_seq[t]:
            h_c = h[output_labels == c]
            pr_c = self.pr_vec[output_labels == c]
            pr_c = pr_c / pr_c.sum()
            proto_c = torch.sum(h_c * pr_c.unsqueeze(-1), dim=0, keepdim=True) # (1,C)
            online_protos.append(proto_c)
        online_protos = torch.cat(online_protos, dim=0)
        
        if t == 0:
            for c in args.task_seq[t]:
                h_c = h[output_labels == c]
                probablity = (norm(h_c) @ norm(online_protos).T).softmax(1)
                # 熵
                entropy = -torch.sum(probablity * torch.log(probablity + 1e-6), dim=-1) # (B)
                # Margin-based
                # one_hot_label = torch.nn.functional.one_hot(torch.tensor([c], device=h_c.device), num_classes=online_protos.size(0)).float()
                # entropy = torch.norm(probablity - one_hot_label, dim=-1)
                # Energy-based
                # probablity = norm(h_c) @ norm(online_protos).T
                # entropy = -torch.logsumexp(probablity, dim=-1)
                
                _, indices = torch.topk(entropy, k=self.k if h_c.shape[0] > self.k else h_c.shape[0], largest=True)
                sample_feats = h_c[indices]
                boundary_samples.append(sample_feats)
                repeats_num.append(sample_feats.shape[0])
            boundary_samples = torch.cat(boundary_samples, dim=0)
            repeats_num = torch.tensor(repeats_num, device=features.device)
        
        if t > 0:
            prev_classes = offset2 - args.n_cls_per_task
            old_protos, old_virtual_samples = [], []
            for i, m_std in enumerate(self.protos):
                old_protos.append(m_std[0])
                noise = torch.randn([self.k, old_protos[0].shape[-1]]).to(features.device)
                virtual_samples = m_std[1] * noise + m_std[0]
                old_virtual_samples.append(virtual_samples)
            old_protos = torch.cat(old_protos, dim=0)
            old_virtual_samples = torch.cat(old_virtual_samples, dim=0)
            old_repeats_num = torch.tensor([self.k for _ in range(prev_classes)]).to(features.device)
            old_labels = torch.tensor([i for i in range(prev_classes)], dtype=labels.dtype).to(features.device)
            new_labels = torch.tensor(args.task_seq[t], dtype=output_labels.dtype, device=output_labels.device)
            
            all_protos = torch.cat([online_protos, old_protos], dim=0)
            
            for c in args.task_seq[t]:
                h_c = h[output_labels == c]
                probablity = (norm(h_c) @ norm(all_protos).T).softmax(1)
                # 熵
                entropy = -torch.sum(probablity * torch.log(probablity + 1e-6), dim=-1) # (B)
                # Margin-based
                # one_hot_label = torch.nn.functional.one_hot(torch.tensor([c], device=h_c.device), num_classes=all_protos.size(0)).float()
                # entropy = torch.norm(probablity - one_hot_label, dim=-1)
                # Energy-based
                # probablity = norm(h_c) @ norm(all_protos).T
                # entropy = -torch.logsumexp(probablity, dim=-1)
                
                _, indices = torch.topk(entropy, k=self.k if h_c.shape[0] > self.k else h_c.shape[0], largest=True)
                sample_feats = h_c[indices]
                boundary_samples.append(sample_feats)
                repeats_num.append(sample_feats.shape[0])
            boundary_samples = torch.cat(boundary_samples, dim=0)
            repeats_num = torch.tensor(repeats_num, device=features.device)
            
            loss = self.supcon(norm(output), norm(torch.cat([online_protos, old_protos],dim=0)), norm(torch.cat([boundary_samples, old_virtual_samples], dim=0)), output_labels, torch.cat([new_labels, old_labels], dim=0), torch.cat([repeats_num, old_repeats_num]))
            
            with torch.no_grad():
                old_h = prev_model.forward(g, features)[0][train_ids]
            
            loss_dist = relation_distillation(h, old_h, [old_protos], args)

            loss += args.ncil_args['alpha']*loss_dist 
            
            # loss += args.ncil_args['alpha'] * torch.dist(h, old_h, 2)

        else:
            new_labels = torch.tensor(args.task_seq[t], dtype=output_labels.dtype, device=output_labels.device)
            loss = self.supcon(norm(output), norm(online_protos), norm(boundary_samples), output_labels, new_labels, repeats_num)

        loss.backward()
        self.opt.step()
            
        if last_epoch == 0: 
            if t > 0:
                with torch.no_grad():
                    n_h = self.net(g, features)[0][train_ids]
                    o_h = prev_model.forward(g, features)[0][train_ids]
                delta = n_h - o_h

                for i, m_std in enumerate(self.protos):
                    w = torch.exp((norm(o_h) * norm(m_std[0])).sum(1)) # (B)
                    w /= w.sum()
                    self.protos[i] = [m_std[0] + args.ncil_args['beta'] * torch.sum(w.unsqueeze(1) * delta, dim=0, keepdim=True), m_std[1]] # 0.1
        
            # 保存类原型
            self.net.eval()
            with torch.no_grad():
                h = self.net(g, features)[0][train_ids]
            h_labels = labels[train_ids]
            for c in args.task_seq[t]:
                h_c = h[h_labels == c]
                pr_c = self.pr_vec[h_labels == c]
                pr_c = pr_c / pr_c.sum()
                proto_c = torch.sum(h_c * pr_c.unsqueeze(-1), dim=0, keepdim=True) # (1,C)
                proto_std = torch.sqrt(torch.mean(torch.sum(pr_c.unsqueeze(-1) * (h_c - proto_c)**2, dim=0)))
                self.protos.append([proto_c, proto_std])

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the class-IL setting with mini-batch training.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param dataloader: The data loader for mini-batch training.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
        :param dataset: The entire dataset (currently not in use).

        """
        if t == 1 and self.flag:
            for params_group in self.opt.param_groups:
                params_group['lr'] = 1e-4
            self.flag = False
        
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        
        offset1, offset2 = self.task_manager.get_label_offset(t) 
        for idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            self.net.train()
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            pr_c = blocks[-1].dstdata['pr_vec'].squeeze()
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            
            h = output_predictions
            
            online_protos, boundary_samples, repeats_num, new_labels = [], [], [], []
            for c in args.task_seq[t]:
                if (output_labels == c).sum() != 0:
                    h_c = h[output_labels == c]
                    pr_c_ = pr_c[output_labels == c]
                    pr_c_ = pr_c_ / pr_c_.sum()
                    proto_c = torch.sum(h_c * pr_c_.unsqueeze(-1), dim=0, keepdim=True) # (1,C)

                    online_protos.append(proto_c)
                    new_labels.append(c)
            online_protos = torch.cat(online_protos, dim=0)
            
            if t == 0:
                for c in args.task_seq[t]:
                    if (output_labels == c).sum() != 0:
                        h_c = h[output_labels == c]
                        probablity = (norm(h_c) @ norm(online_protos).T).softmax(1)
                        # 熵
                        entropy = -torch.sum(probablity * torch.log(probablity + 1e-6), dim=-1) # (B)
                        # Margin-based
                        # newnew_labels = torch.tensor(new_labels, dtype=output_labels.dtype, device=output_labels.device)
                        # pos = (newnew_labels == c).nonzero(as_tuple=False)[0]
                        # one_hot_label = torch.nn.functional.one_hot(pos, num_classes=online_protos.size(0)).float()
                        # entropy = torch.norm(probablity - one_hot_label, dim=-1)
                        # Energy-based
                        # probablity = norm(h_c) @ norm(online_protos).T
                        # entropy = -torch.logsumexp(probablity, dim=-1)
                        
                        _, indices = torch.topk(entropy, k=self.k if h_c.shape[0] > self.k else h_c.shape[0], largest=True)
                        sample_feats = h_c[indices]
                        boundary_samples.append(sample_feats)
                        repeats_num.append(sample_feats.shape[0])
                boundary_samples = torch.cat(boundary_samples, dim=0)
                repeats_num = torch.tensor(repeats_num, device=input_features.device)
            
            if t > 0:
                prev_classes = offset2 - args.n_cls_per_task
                old_protos, old_virtual_samples = [], []
                for i, m_std in enumerate(self.protos):
                    old_protos.append(m_std[0])
                    noise = torch.randn([self.k, old_protos[0].shape[-1]]).to(input_features.device)
                    virtual_samples = m_std[1] * noise + m_std[0]
                    old_virtual_samples.append(virtual_samples)
                old_protos = torch.cat(old_protos, dim=0)
                old_virtual_samples = torch.cat(old_virtual_samples, dim=0)
                old_repeats_num = torch.tensor([self.k for _ in range(prev_classes)]).to(input_features.device)
                old_labels = torch.tensor([i for i in range(prev_classes)], dtype=labels.dtype).to(input_features.device)
                new_labels = torch.tensor(new_labels, dtype=output_labels.dtype, device=output_labels.device)
                
                all_protos = torch.cat([online_protos, old_protos], dim=0)
                
                for c in args.task_seq[t]:
                    if (output_labels == c).sum() != 0:
                        h_c = h[output_labels == c]
                        probablity = (norm(h_c) @ norm(all_protos).T).softmax(1)
                        # 熵
                        entropy = -torch.sum(probablity * torch.log(probablity + 1e-6), dim=-1) # (B)
                        # Margin-based
                        # ap_labels = torch.cat([new_labels, old_labels])
                        # pos = (ap_labels == c).nonzero(as_tuple=False)[0]
                        # one_hot_label = torch.nn.functional.one_hot(pos, num_classes=all_protos.size(0)).float()
                        # entropy = torch.norm(probablity - one_hot_label, dim=-1)
                        # Energy-based
                        # probablity = norm(h_c) @ norm(all_protos).T
                        # entropy = -torch.logsumexp(probablity, dim=-1)
                        
                        _, indices = torch.topk(entropy, k=self.k if h_c.shape[0] > self.k else h_c.shape[0], largest=True)
                        sample_feats = h_c[indices]
                        boundary_samples.append(sample_feats)
                        repeats_num.append(sample_feats.shape[0])
                boundary_samples = torch.cat(boundary_samples, dim=0)
                repeats_num = torch.tensor(repeats_num, device=input_features.device)
                
                loss = self.supcon(norm(output_predictions), norm(torch.cat([online_protos, old_protos],dim=0)), norm(torch.cat([boundary_samples, old_virtual_samples], dim=0)), output_labels, torch.cat([new_labels, old_labels], dim=0), torch.cat([repeats_num, old_repeats_num]))
                
                with torch.no_grad():
                    old_h = prev_model.forward_batch(blocks, input_features)[0]
                
                loss_dist = relation_distillation(h, old_h, [old_protos], args)

                loss += args.ncil_args['alpha']*loss_dist 
                
                # loss += args.ncil_args['alpha'] * torch.dist(h, old_h, 2)

            else:
                new_labels = torch.tensor(new_labels, dtype=output_labels.dtype, device=output_labels.device)
                loss = self.supcon(norm(output_predictions), norm(online_protos), norm(boundary_samples), output_labels, new_labels, repeats_num)
            
            loss.backward()
            self.opt.step()

        if last_epoch == 0:
            if t > 0:
                with torch.no_grad():
                    n_h, o_h = [], []
                    for input_nodes, output_nodes, blocks in dataloader:
                        blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                        input_features = blocks[0].srcdata['feat']
                        output_labels = blocks[-1].dstdata['label'].squeeze()
                        n_h.append(self.net.forward_batch(blocks, input_features)[0])
                        o_h.append(prev_model.forward_batch(blocks, input_features)[0])
                    n_h = torch.cat(n_h, dim=0)
                    o_h = torch.cat(o_h, dim=0)
                delta = n_h - o_h

                for i, m_std in enumerate(self.protos):
                    w = torch.exp((norm(o_h) * norm(m_std[0])).sum(1)) # (B)
                    w /= w.sum()
                    self.protos[i] = [m_std[0] + args.ncil_args['beta'] * torch.sum(w.unsqueeze(1) * delta, dim=0, keepdim=True), m_std[1]] # 0.1
            
            # 保存类原型
            self.net.eval()
            with torch.no_grad():
                n_h, lbls, pr_c = [], [], []
                for input_nodes, output_nodes, blocks in dataloader:
                    blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                    input_features = blocks[0].srcdata['feat']
                    output_labels = blocks[-1].dstdata['label'].squeeze()
                    n_h.append(self.net.forward_batch(blocks, input_features)[0])
                    lbls.append(output_labels)
                    pr_c.append(blocks[-1].dstdata['pr_vec'].squeeze()) 
                h = torch.cat(n_h, dim=0)
                h_labels = torch.cat(lbls, dim=0)
                pr_vec = torch.cat(pr_c, dim=0)
                
            for c in args.task_seq[t]:
                h_c = h[h_labels == c]
                pr_c = pr_vec[h_labels == c]
                pr_c = pr_c / pr_c.sum()
                proto_c = torch.sum(h_c * pr_c.unsqueeze(-1), dim=0, keepdim=True) # (1,C)
                proto_std = torch.sqrt(torch.mean(torch.sum(pr_c.unsqueeze(-1) * (h_c - proto_c)**2, dim=0)))
                self.protos.append([proto_c, proto_std])
