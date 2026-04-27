import os
import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluate, NodeLevelDataset, evaluate_batch, ncil_evaluate, evaluate_batch_ncil, yooop_evaluate, evaluate_batch_yooop, fecam_evaluate, evaluate_batch_fecam
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl
from tqdm import tqdm
import random
import time

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
            break
        pr_vec = pr_new
    return pr_vec

def calc_pagerank(adj_matrix):
    solve_ranking_leaked(adj_matrix)
    calc_out_degree_ratio(adj_matrix)
    return pagerank(adj_matrix)

class NegativeSampler(object):
    def __init__(self, edge_index, k):
        self.ei = edge_index
        self.k = k
    def __call__(self, g, eids):
        p = len(eids)
        n = p*self.k
        id = [i for i in range(self.ei.shape[1])]
        sample = random.sample(id, n)
        edges = self.ei[:,sample] # [2,m]
        return edges[0], edges[1]

joint_alias = ['joint', 'Joint', 'joint_replay_all', 'jointtrain']
def get_pipeline(args):
    # choose the pipeline for the chosen setting
    if args.minibatch:
        if args.ILmode == 'classIL':
            if args.method in joint_alias:
                return pipeline_class_IL_no_inter_edge_minibatch_joint
            else:
                return pipeline_class_IL_no_inter_edge_minibatch
    else:
        if args.ILmode == 'classIL':
            if args.method in joint_alias:
                return pipeline_class_IL_no_inter_edge_joint
            else:
                return pipeline_class_IL_no_inter_edge

def data_prepare(args):
    """
    check whether the processed data exist or create new processed data
    if args.load_check is True, loading data will be tried, else, will only check the existence of the files
    """
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    if args.dataset == 'CoraFull-CL': 
        base_cls = [list(range(0, 30))]
        cls = base_cls + [list(range(i+30, i+30 + args.n_cls_per_task)) for i in range(0, args.n_cls-30-1, args.n_cls_per_task)]
    elif args.dataset == 'Arxiv-CL':
        base_cls = [list(range(0, 20))]
        cls = base_cls + [list(range(i+20, i+20 + args.n_cls_per_task)) for i in range(0, args.n_cls-20-1, args.n_cls_per_task)]
    elif args.dataset == 'Reddit-CL': 
        base_cls = [list(range(0, 20))]
        cls = base_cls + [list(range(i+20, i+20 + args.n_cls_per_task)) for i in range(0, args.n_cls-20-1, args.n_cls_per_task)]
    elif args.dataset == 'CS-CL': 
        base_cls = [list(range(0, 5))]
        cls = base_cls + [list(range(i+5, i+5 + args.n_cls_per_task)) for i in range(0, args.n_cls-5-1, args.n_cls_per_task)]
    
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    n_cls_so_far = 0
    # check whether the preprocessed data exist and can be loaded
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    run_times = []
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        try:
            if args.load_check:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
            else:
                if f'{args.dataset}_{task_cls}.pkl' not in os.listdir(f'{args.data_path}/{str_int_tsk}'):
                    subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                        f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
        except:
            # if not exist or cannot be loaded correctly, create new processed data
            print(f'preparing data for task {task}')
            mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
            mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
            if args.inter_task_edges:
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                    tasks_to_retain=cls_retain)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls) 
                
                if args.dataset in ['Arxiv-CL', 'Reddit-CL']:
                    adj = dgl.remove_self_loop(subgraph).adjacency_matrix().to_dense()
                    pr_vec = calc_pagerank(adj)
                    subgraph.ndata['pr_vec'] = pr_vec.unsqueeze(-1)
                
                with open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)

def pipeline_class_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        save_proto_path = f'{args.result_path}/{subfolder_c}val_protos/{save_model_name}.pkl'
        n_cls_so_far+=len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1, label_offset2 = task_manager.get_label_offset(task) 

        if args.method == 'fecam':
            if valid:
                if task == 0:
                    for epoch in tqdm(range(epochs), ncols=100, desc='training_gnns', colour='red'):
                        life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
                else:
                    life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
        else:
            for epoch in tqdm(range(epochs), ncols=100, desc='training_gnns', colour='red'):
                if args.method in ['lwf', 'ncil', 'polo', 'efc', 'yooop']:
                    if args.method == 'yooop':
                        life_model_ins.generate_memory(task_cls, args.GCN_args['h_dims'][-1], subgraph, features, labels, train_ids) # 
                        if prev_model is not None:
                            life_model_ins.get_R_matrxi(task_cls)
                        life_model_ins.observe(args, subgraph, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
                        if life_model_ins.epochs % args.epochs == 0:
                            life_model_ins.generate_memory(task_cls, args.GCN_args['h_dims'][-1], subgraph, features, labels, train_ids, True)
                    else:
                        life_model_ins.observe(args, subgraph, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
                    
                else:
                    life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
                    torch.cuda.empty_cache()
        
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
                if args.method in ['ncil', 'yooop', 'fecam']:
                    protos = pickle.load(open(save_proto_path,'rb'))
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        # test
        for t in range(task+1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                if args.method == 'ncil':
                    acc = ncil_evaluate(model, life_model_ins.protos if valid else protos, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
                elif args.method == 'yooop':
                    acc = yooop_evaluate(model, life_model_ins.memory[life_model_ins.initial_memory==False] if valid else protos, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
                elif args.method == 'fecam':
                    acc = fecam_evaluate(model, [life_model_ins._init_protos, life_model_ins._protos, life_model_ins._norm_cov_mat] if valid else protos, subgraph, features, labels, task, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
                else:
                    acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, args.n_cls, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)

            acc_matrix[task][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")
            
        accs = acc_mean[:task+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)
        acc_mean = round(np.mean(acc_mean)*100,2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            if args.method in ['ncil', 'yooop', 'fecam']:
                mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_protos')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
                if args.method == 'ncil':
                    with open(save_proto_path, 'wb') as f:
                        pickle.dump(life_model_ins.protos, f)
                elif args.method == 'yooop':
                    with open(save_proto_path, 'wb') as f:
                        pickle.dump(life_model_ins.memory[life_model_ins.initial_memory==False], f)
                elif args.method == 'fecam':
                    with open(save_proto_path, 'wb') as f:
                        pickle.dump([life_model_ins._init_protos, life_model_ins._protos, life_model_ins._norm_cov_mat], f)
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))
                
        prev_model = copy.deepcopy(model).cuda()
        
    if args.method == 'yooop' and valid:
        if life_model_ins.all_process is not None:
            for tmp_process in life_model_ins.all_process:
                tmp_process.terminate()
                tmp_process.join() 

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        for epoch in range(epochs):
            life_model_ins.observe(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f: 
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_minibatch_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        # build the dataloader for mini batch training
        train_ids_entire = []
        n_existing_nodes = 0
        for ids, g in zip(train_idss, subgraphs):
            new_ids = [i + n_existing_nodes for i in ids]
            train_ids_entire.extend(new_ids)
            n_existing_nodes += g.num_nodes()

        graph_entire = dgl.batch(subgraphs)
        collator = dgl.dataloading.NodeCollator(graph_entire, train_ids_entire, args.nb_sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, 
                                                 collate_fn=collator.collate,
                                                 batch_size=args.batch_size, 
                                                 shuffle=args.batch_shuffle, 
                                                 drop_last=False)

        for epoch in tqdm(range(epochs), desc='training gnn', ncols=100, colour='red'):
            life_model_ins.observe_class_IL_batch(args, subgraphs, dataloader, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        save_proto_path = f'{args.result_path}/{subfolder_c}val_protos/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl',
                 'rb'))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        # build the dataloader for mini batch training
        collator = dgl.dataloading.NodeCollator(subgraph, train_ids, args.nb_sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, 
                                                 collate_fn=collator.collate,
                                                 batch_size=args.batch_size, 
                                                 shuffle=args.batch_shuffle, 
                                                 drop_last=False)
        
        if args.method == 'fecam':
            if valid:
                if task == 0:
                    for epoch in tqdm(range(epochs), desc='training gnns', ncols=100, colour='red'):
                        life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                else:
                    life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
        else:
            for epoch in tqdm(range(epochs), desc='training gnns', ncols=100, colour='red'):
                if args.method in ['lwf', 'polo', 'efc', 'ncil', 'yooop']:
                    if args.method == 'yooop':
                        life_model_ins.generate_memory_minibatch(task_cls, args.GCN_args['h_dims'][-1], dataloader)
                        if prev_model is not None:
                            life_model_ins.get_R_matrxi(task_cls)
                        life_model_ins.observe_class_IL_batch(args, dataloader, task, prev_model)
                        if life_model_ins.epochs % args.epochs == 0:
                            life_model_ins.generate_memory_minibatch(task_cls, args.GCN_args['h_dims'][-1], dataloader, True)
                    else:
                        life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model, train_ids, ids_per_cls,
                                            dataset)
                
                else:
                    life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                    torch.cuda.empty_cache()  # tracemalloc.stop()

        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        # test
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
                if args.method in ['ncil', 'yooop', 'fecam']:
                    protos = pickle.load(open(save_proto_path,'rb'))
                else:
                    pass
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.method == 'ncil':
                acc = evaluate_batch_ncil(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                                cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test, p=life_model_ins.protos if valid else protos)
            elif args.method == 'yooop':
                acc = evaluate_batch_yooop(args, model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                                cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test, p=life_model_ins.memory[life_model_ins.initial_memory==False] if valid else protos)
            elif args.method == 'fecam':
                acc = evaluate_batch_fecam(args, model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                                cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test, p=[life_model_ins._init_protos, life_model_ins._protos, life_model_ins._norm_cov_mat] if valid else protos, t=task)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                                cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            if args.method in ['ncil', 'yooop', 'fecam']:
                mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_protos')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
                if args.method == 'ncil':
                    with open(save_proto_path, 'wb') as f:
                        pickle.dump(life_model_ins.protos, f) # save the best model for each hyperparameter composition
                elif args.method == 'yooop':
                    with open(save_proto_path, 'wb') as f:
                        pickle.dump(life_model_ins.memory[life_model_ins.initial_memory==False], f) # save the best model for each hyperparameter composition
                elif args.method == 'fecam':
                    with open(save_proto_path, 'wb') as f:
                        pickle.dump([life_model_ins._init_protos, life_model_ins._protos, life_model_ins._norm_cov_mat], f) # save the best model for each hyperparameter composition
                else:
                    pass
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))
        prev_model = copy.deepcopy(model).cuda()

    if args.method == 'yooop' and valid:
        if life_model_ins.all_process is not None:
            for tmp_process in life_model_ins.all_process:
                tmp_process.terminate()
                tmp_process.join() 
    
    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix
