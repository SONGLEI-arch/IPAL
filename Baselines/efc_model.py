import torch
import abc
import copy
import itertools
import torch.nn as nn
import einops
from torch.distributions import MultivariateNormal
from copy import deepcopy
import numpy as np
from random import sample, shuffle  

# class EmpiricalFeatureMatrix:
#     def __init__(self, args, device):
#         self.args = args
#         self.empirical_feat_mat = None
#         self.device = device
    
#     def create(self):
#         self.empirical_feat_mat = torch.zeros((self.args.GCN_args['h_dims'][-1], self.args.GCN_args['h_dims'][-1]), requires_grad=False).to(self.device)

#     def get(self):
#         return self.empirical_feat_mat

#     def compute(self, model, dataloader, offsets):
#         self.compute_efm(model, dataloader, offsets)
                
#     def compute_efm(self, model, dataloader, offsets):
#         # print("Evaluating Empirical Feature Matrix")
#         # Compute empirical feature matrix for specified number of samples -- rounded to the batch size
#         model.eval() 
#         model.zero_grad()   
#         self.create() # (C,C)
#         with torch.no_grad():
#             out, gap_out = [], []
#             for input_nodes, output_nodes, blocks in dataloader:
#                 blocks = [b.to('cuda') for b in blocks]
#                 input_features = blocks[0].srcdata['feat']
#                 output_labels = blocks[-1].dstdata['label'].squeeze()
#                 out_ = model.forward_batch(blocks, input_features)[0]
#                 out.append(out_[:,offsets[0]:offsets[1]])
#                 gap_out.append(model.second_last_h)
#             out = torch.cat(out, dim=0)
#             gap_out = torch.cat(gap_out, dim=0)
#             out_size = out.shape[1]
#             # compute the efm using the closed formula    
#             log_p =  nn.LogSoftmax(dim=1)(out)
#             identity = torch.eye(out.shape[1], device=self.device) # (K,K)
#             der_log_softmax =  einops.repeat(identity, 'n m -> b n m', b=gap_out.shape[0]) - einops.repeat(torch.exp(log_p), 'n p -> n a p', a=out.shape[1]) # (B,K,K) - (B,K,K)
#             weight_matrix = model.gat_layers[-1].weight[offsets[0]:offsets[1]] # (K,C)
#             # closed formula jacobian matrix
#             jac  = torch.bmm(der_log_softmax, einops.repeat(weight_matrix, 'n m -> b n m', b=gap_out.shape[0])) # (B,K,K) @ (B,K,C) = (B,K,C)
#             efm_per_batch = torch.zeros((self.args.batch_size, self.args.GCN_args['h_dims'][-1], self.args.GCN_args['h_dims'][-1]), device=self.device) # (B,C,C)
#             p = torch.exp(log_p)
#             # jac =  jacobian_in_batch(log_p, gap_out).detach() # equivalent formulation with gradient computation, with torch.no_grad() should be removed
#             num = 0
#             for batch_p, batch_jac in zip(torch.split(p, self.args.batch_size, dim=0), torch.split(jac, self.args.batch_size, dim=0)):
#                 if batch_p.shape[0] == self.args.batch_size:
#                     for c in range(out_size):
#                         efm_per_batch +=  batch_p[:, c].view(batch_p.shape[0], 1, 1) * torch.bmm(batch_jac[:,c,:].unsqueeze(1).permute(0,2,1), batch_jac[:,c,:].unsqueeze(1)) # (B,1,1) + (B,C,1) @ (B,1,C) = (B,C,C)
#                     self.empirical_feat_mat +=  torch.sum(efm_per_batch, dim=0)   
#                     num += batch_p.shape[0]
#             self.empirical_feat_mat /= num
            
#         # if isPSD(self.empirical_feat_mat):
#         #     print("EFM is semidefinite positive")
#         return self.empirical_feat_mat # (C,C)

class EmpiricalFeatureMatrix:
    def __init__(self, args, device):
        self.args = args
        self.empirical_feat_mat = None
        self.device = device
    
    def create(self):
        self.empirical_feat_mat = torch.zeros((self.args.GCN_args['h_dims'][-1], self.args.GCN_args['h_dims'][-1]), requires_grad=False).to(self.device)

    def get(self):
        return self.empirical_feat_mat

    def compute(self, model, g, features, train_ids, offsets):
        self.compute_efm(model, g, features, train_ids, offsets)
                
    def compute_efm(self, model, g, features, train_ids, offsets):
        # print("Evaluating Empirical Feature Matrix")
        # Compute empirical feature matrix for specified number of samples -- rounded to the batch size
        model.eval() 
        model.zero_grad()   
        self.create() # (C,C)
        with torch.no_grad():
            out = model(g, features)[0][train_ids][:,offsets[0]:offsets[1]]
            gap_out = model.second_last_h[train_ids]
            out_size = out.shape[1]
            # compute the efm using the closed formula    
            log_p =  nn.LogSoftmax(dim=1)(out)
            identity = torch.eye(out.shape[1], device=self.device) # (K,K)
            der_log_softmax =  einops.repeat(identity, 'n m -> b n m', b=gap_out.shape[0]) - einops.repeat(torch.exp(log_p), 'n p -> n a p', a=out.shape[1]) # (B,K,K) - (B,K,K)
            weight_matrix = model.gat_layers[-1].weight[offsets[0]:offsets[1]] # (K,C)
            # closed formula jacobian matrix
            jac  = torch.bmm(der_log_softmax, einops.repeat(weight_matrix, 'n m -> b n m', b=gap_out.shape[0])) # (B,K,K) @ (B,K,C) = (B,K,C)
            if self.args.dataset == 'Arxiv-CL':
                efm_per_batch = torch.zeros((2000, self.args.GCN_args['h_dims'][-1], self.args.GCN_args['h_dims'][-1]), device=self.device) # (Batch,C,C)
            else:
                efm_per_batch = torch.zeros((out.shape[0], self.args.GCN_args['h_dims'][-1], self.args.GCN_args['h_dims'][-1]), device=self.device) # (B,C,C)
            p = torch.exp(log_p)
            # jac =  jacobian_in_batch(log_p, gap_out).detach() # equivalent formulation with gradient computation, with torch.no_grad() should be removed
            if self.args.dataset == 'Arxiv-CL':
                num = 0
                for batch_p, batch_jac in zip(torch.split(p, 2000, dim=0), torch.split(jac, 2000, dim=0)):
                    if batch_p.shape[0] == 2000:
                        for c in range(out_size):
                            efm_per_batch +=  batch_p[:, c].view(batch_p.shape[0], 1, 1) * torch.bmm(batch_jac[:,c,:].unsqueeze(1).permute(0,2,1), batch_jac[:,c,:].unsqueeze(1)) # (B,1,1) + (B,C,1) @ (B,1,C) = (B,C,C)
                        self.empirical_feat_mat +=  torch.sum(efm_per_batch, dim=0)   
                        num += batch_p.shape[0]
                self.empirical_feat_mat /= num
                    
            else:
                for c in range(out_size):
                    efm_per_batch +=  p[:, c].view(out.shape[0], 1, 1) * torch.bmm(jac[:,c,:].unsqueeze(1).permute(0,2,1), jac[:,c,:].unsqueeze(1)) # (B,1,1) + (B,C,1) @ (B,1,C) = (B,C,C)
                self.empirical_feat_mat +=  torch.mean(efm_per_batch, dim=0)   
            
        # if isPSD(self.empirical_feat_mat):
        #     print("EFM is semidefinite positive")
        return self.empirical_feat_mat # (C,C)
    
class ProtoManager(metaclass=abc.ABCMeta):
    def __init__(self, device):
        self.device = device 
        self.prototype = []
        self.variances = []
        self.class_label = []
    @abc.abstractmethod 
    def compute(self, *args):
        pass 
    @abc.abstractmethod 
    def perturbe(self, *args):
        pass 
    @abc.abstractmethod
    def update(self, *args):
        pass 

class ProtoGenerator(ProtoManager):
    def __init__(self, device):
        super(ProtoGenerator, self).__init__(device)
        self.running_proto = None 
        self.running_proto_variance = []
        self.gaussians = {}
        
    def compute(self, args, model, dataloader, t):
        model.eval()
        with torch.no_grad():
            feats, label_list = [], []
            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label'].squeeze()
                _ = model.forward_batch(blocks, input_features)
                feats.append(model.second_last_h)
                label_list.append(output_labels)
            features_list = torch.cat(feats, dim=0)
            label_list = torch.cat(label_list, dim=0)

        for label in args.task_seq[t]:
            mask = (label_list == label)
            feature_classwise = features_list[mask]
            proto = feature_classwise.mean(dim=0) # (C)
            covariance = torch.cov(feature_classwise.T) # (C,C)
            
            # print(label, torch.isnan(proto).any(), feature_classwise.size(0))
            
            self.running_proto_variance.append(covariance)
            self.prototype.append(proto.cpu())
            self.class_label.append(label)
            self.gaussians[label] = MultivariateNormal(
                                                    proto.cpu(),
                                                    covariance_matrix=covariance.cpu() + 1e-1 * torch.eye(covariance.shape[0]).cpu(), # 1e-1/10e-1 # long: 0
                                    )
        self.running_proto = deepcopy(self.prototype)

    def update_gaussian(self, proto_label, mean, var):
        self.gaussians[proto_label] = MultivariateNormal(
                                                    mean.cpu(),
                                                    covariance_matrix=var.cpu() + 1e-1 * torch.eye(var.shape[0]).cpu(),
                                    )

    def perturbe(self, prev_classes, args):
        proto_aug = []
        for class_idx in range(prev_classes):
            proto_aug.append(self.gaussians[class_idx].sample((args.efc_args['budget'],))) # (10,C)
        proto_aug = torch.cat(proto_aug, dim=0) # (10*K,C)
        proto_aug_label = torch.repeat_interleave(torch.tensor(self.class_label, device=self.device), repeats=torch.tensor([args.efc_args['budget'] for _ in range(prev_classes)], device=self.device))
        return proto_aug, proto_aug_label

    def update(self, *args):
        pass 

# class  ProtoGenerator(ProtoManager):
#     def __init__(self, device):
#         super(ProtoGenerator, self).__init__(device)
#         self.running_proto = None 
#         self.running_proto_variance = []
#         self.gaussians = {}
        
#     def compute(self, args, model, g ,feats, train_ids, labels, t):
#         model.eval()
#         with torch.no_grad():
#             _ = model(g, feats)
#             features_list = model.second_last_h[train_ids]
#             label_list = labels[train_ids]

#         for label in args.task_seq[t]:
#             mask = (label_list == label)
#             feature_classwise = features_list[mask]
#             proto = feature_classwise.mean(dim=0) # (C)
#             covariance = torch.cov(feature_classwise.T) # (C,C)
#             self.running_proto_variance.append(covariance)
#             self.prototype.append(proto.cpu())
#             self.class_label.append(label)
#             self.gaussians[label] = MultivariateNormal(
#                                                     proto.cpu(),
#                                                     covariance_matrix=covariance.cpu() + 10e-1 * torch.eye(covariance.shape[0]).cpu(), # 1e-1/10e-1
#                                     )
#         self.running_proto = deepcopy(self.prototype)

#     def update_gaussian(self, proto_label, mean, var):
#         self.gaussians[proto_label] = MultivariateNormal(
#                                                     mean.cpu(),
#                                                     covariance_matrix=var.cpu() + 10e-1 * torch.eye(var.shape[0]).cpu(),
#                                     )

#     def perturbe(self, prev_classes, args):
#         proto_aug = []
#         for class_idx in range(prev_classes):
#             proto_aug.append(self.gaussians[class_idx].sample((args.efc_args['budget'],))) # (10,C)
#         proto_aug = torch.cat(proto_aug, dim=0) # (10*K,C)
#         proto_aug_label = torch.repeat_interleave(torch.tensor(self.class_label, device=self.device), repeats=torch.tensor([args.efc_args['budget'] for _ in range(prev_classes)], device=self.device))
#         return proto_aug, proto_aug_label

#     def update(self, *args):
#         pass 

def isPSD(A, tol=1e-7):
    import numpy as np 
    A = A.cpu().numpy()
    E = np.linalg.eigvalsh(A)
    print("Maximum eigenvalue {}".format(np.max(E)))
    return np.all(E > -tol)

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
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup loss
        self.ce = torch.nn.functional.cross_entropy
        
        self.flag = True
        self.epochs = 0
        self.protos = []
        self.previous_efm = None
        self.proto_generator = ProtoGenerator(device='cuda:{}'.format(args.gpu))
        self.sigma_proto_update = 0.2
        self.prevbatch_block = None

    def forward(self, features):
        output = self.net(features)
        return output

    def efm_loss(self, features, features_old):
        features = features.unsqueeze(1) # (B,1,C)
        features_old = features_old.unsqueeze(1) # (B,1,C)
        matrix_reg = 10.0 *  self.previous_efm + 0.1 * torch.eye(self.previous_efm.shape[0], device=features.device) 
        efc_loss = torch.mean(torch.bmm(torch.bmm((features - features_old), matrix_reg.expand(features.shape[0], -1, -1)), (features - features_old).permute(0,2,1))) # (B,1,C) @ (B,C,C) @ (B,C,1) = (B,1,1)
        return  efc_loss
    
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
        offset1, offset2 = self.task_manager.get_label_offset(t) 
        output = self.net(g, features)
        if isinstance(output,tuple):
            output = output[0]
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], output_labels, weight=loss_w_)
            
        if t > 0:
            prev_classes = offset2 - args.n_cls_per_task
            with torch.no_grad():
                proto_aug, proto_aug_label = self.proto_generator.perturbe(prev_classes, args)   
                proto_aug, proto_aug_label = proto_aug.to(device='cuda:{}'.format(args.gpu)), proto_aug_label.to(device='cuda:{}'.format(args.gpu))
            pred_old = self.net.gat_layers[-1](proto_aug)
            loss_old = self.ce(pred_old[:,offset1:offset2], proto_aug_label)
                
            new_h = self.net.second_last_h[train_ids]
            with torch.no_grad():
                _ = prev_model.forward(g, features)
                old_h = prev_model.second_last_h[train_ids]
            loss_kd = self.efm_loss(new_h, old_h)
            
            of1, of2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
            loss_current = self.ce(output[train_ids, of1:of2], output_labels - of1, weight=loss_w_[of1:of2])
            
            loss += loss_old + loss_current + loss_kd
        
        loss.backward()
        self.opt.step()

        if last_epoch == 0:
            if t > 0:
                with torch.no_grad():
                    _ = self.net(g, features)
                    _ = prev_model.forward(g, features)
                    new_features = self.net.second_last_h[train_ids]
                    old_features = prev_model.second_last_h[train_ids]
                drift = self.compute_drift(new_features, old_features,  device="cpu")
                drift = drift.cpu()
                for i, (p, var, proto_label) in enumerate(zip(self.proto_generator.prototype, 
                                                            self.proto_generator.running_proto_variance,
                                                            self.proto_generator.class_label)):
            
                    mean = p.cpu() + drift[i] 
                    self.proto_generator.update_gaussian(proto_label, mean, var)
                    # final update the mean
                    self.proto_generator.prototype[i] = mean 
                self.proto_generator.running_proto = deepcopy(self.proto_generator.prototype)
            
            efm_matrix = EmpiricalFeatureMatrix(args, 'cuda:{}'.format(args.gpu))
            efm_matrix.compute(self.net, g, features, train_ids, [offset1, offset2])
            self.previous_efm = efm_matrix.get() # (C,C)        
            
            self.proto_generator.compute(args, self.net, g, features, train_ids, labels, t)
            
    
    def compute_drift(self, new_features, old_features, device): 
        DY = (new_features - old_features).to(device)
        new_features =  new_features.to(device)
        old_features =  old_features.to(device)
        running_prototypes = torch.stack(self.proto_generator.running_proto, dim=0) # (K,C)

        running_prototypes = running_prototypes.to(device)
        distance = torch.zeros(len(running_prototypes), new_features.shape[0]) # (K,B)
        
        for i in range(running_prototypes.shape[0]): # K
            # we use the EFM to update prototypes
            curr_diff = (old_features - running_prototypes[i, :].unsqueeze(0)).unsqueeze(1).to(device) # (B,1,C) ??????????????????? running_prototypes有问题
        
            distance[i] = -torch.bmm(torch.bmm(curr_diff, self.previous_efm.expand(curr_diff.shape[0], -1, -1).cpu()), curr_diff.permute(0,2,1)).flatten().cpu() # (B)
            # print(i, torch.isnan(distance[i]).any())

        scaled_distance = (distance- distance.min())/(distance.max() - distance.min())

        W  = torch.exp(scaled_distance/(2*self.sigma_proto_update ** 2)) # (K,B)
        normalization_factor  = torch.sum(W, axis=1)[:, None] # (K,1)
        W_norm = W/torch.tile(normalization_factor, [1, W.shape[1]])

        displacement = torch.zeros((running_prototypes.shape[0], DY.shape[1])) # (K,C)

        for i in range(running_prototypes.shape[0]):  
            displacement[i] = torch.sum((W_norm[i].unsqueeze(1) * DY),dim=0)
    
        return displacement

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
        
        if last_epoch == 1:
            self.prevbatch_block = None
        
        offset1, offset2 = self.task_manager.get_label_offset(t) # 0，包括当前task的之前的所有tasks的类别数量
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=loss_w_)
                
            if t > 0:
                prev_classes = offset2 - args.n_cls_per_task
                with torch.no_grad():
                    proto_aug, proto_aug_label = self.proto_generator.perturbe(prev_classes, args)   
                    proto_aug, proto_aug_label = proto_aug.to(device='cuda:{}'.format(args.gpu)), proto_aug_label.to(device='cuda:{}'.format(args.gpu))
                pred_old = self.net.gat_layers[-1](proto_aug)
                loss_old = self.ce(pred_old[:,offset1:offset2], proto_aug_label)
                    
                new_h = self.net.second_last_h
                with torch.no_grad():
                    _ = prev_model.forward_batch(blocks, input_features)
                    old_h = prev_model.second_last_h
                loss_kd = self.efm_loss(new_h, old_h)
                
                if self.prevbatch_block != None:
                    blocks_prev = [b.to(device='cuda:{}'.format(args.gpu)) for b in self.prevbatch_block]
                    input_features_prev = blocks_prev[0].srcdata['feat']
                    output_labels_prev = blocks_prev[-1].dstdata['label'].squeeze()
                    if args.cls_balance:
                        n_per_cls = [(output_labels_prev == j).sum() for j in range(args.n_cls)]
                        loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                    else:
                        loss_w_ = [1. for i in range(args.n_cls)]
                    loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                    output_predictions_prev, _ = self.net.forward_batch(blocks_prev, input_features_prev)
                    of1, of2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
                    loss_current = self.ce(output_predictions_prev[:, of1:of2], output_labels_prev - of1, weight=loss_w_[of1:of2])
                else:
                    loss_current = 0.
                
                loss += loss_old + loss_current + loss_kd
            
            loss.backward()
            self.opt.step()
            
            if t > 0:
                self.prevbatch_block = blocks
        
        if last_epoch == 0:
            if t > 0:
                with torch.no_grad():   
                    news, olds = [], []
                    for input_nodes, output_nodes, blocks in dataloader:
                        blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                        input_features = blocks[0].srcdata['feat']
                        output_labels = blocks[-1].dstdata['label'].squeeze()
                        
                        _ = self.net.forward_batch(blocks, input_features)
                        _ = prev_model.forward_batch(blocks, input_features)
                        news.append(self.net.second_last_h)
                        olds.append(prev_model.second_last_h)
                    new_features = torch.cat(news, dim=0)
                    old_features = torch.cat(olds, dim=0)
                    
                drift = self.compute_drift(new_features, old_features,  device="cpu")
                drift = drift.cpu()
                for i, (p, var, proto_label) in enumerate(zip(self.proto_generator.prototype, 
                                                            self.proto_generator.running_proto_variance,
                                                            self.proto_generator.class_label)):
            
                    mean = p.cpu() + drift[i] 
                    self.proto_generator.update_gaussian(proto_label, mean, var)
                    # final update the mean
                    self.proto_generator.prototype[i] = mean 
                self.proto_generator.running_proto = deepcopy(self.proto_generator.prototype)
            
            efm_matrix = EmpiricalFeatureMatrix(args, 'cuda:{}'.format(args.gpu))
            efm_matrix.compute(self.net, dataloader, [offset1, offset2])
            self.previous_efm = efm_matrix.get() # (C,C)        
            
            self.proto_generator.compute(args, self.net, dataloader, t)