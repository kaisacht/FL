# -*- coding = utf-8 -*-
import numpy as np
import torch
import copy
import time
import hdbscan
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import KernelPCA
def cos(a, b):
    # res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def fltrust(params, central_param, global_parameters, args):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        # 计算cos相似度得分和向量长度裁剪值
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                # 乘得分 再乘裁剪值
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        # 除以所以客户端的信任得分总和
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


def multi_krum(gradients, n_attackers, args, multi_k=False):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        print(scores)
        args.krum_distance.append(scores)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    args.turn+=1
    if multi_k == False:
        if candidate_indices[0] < num_malicious_clients:
            args.wrong_mal += 1
            
    print(candidate_indices)
    
    print('Proportion of malicious are selected:'+str(args.wrong_mal/args.turn))

    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]
    
    return np.array(candidate_indices)



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2



def RLR(global_model, agent_updates_list, args):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    # args.robustLR_threshold = 6
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs 
    #return sm_of_signs.to(args.gpu)
   
    


def flame(local_model, update_params, global_model, args, epochs):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 +1,min_samples=1,allow_single_cluster=True).fit(cos_ij)
    print("clusterer_labels:",clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_model_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1
    proportion_of_malicious_are_selected = args.wrong_mal/(num_malicious_clients*args.turn)
    proportion_of_benign_are_selected = args.right_ben/(num_benign_clients*args.turn)
    print('proportion of malicious are selected:',proportion_of_malicious_are_selected)
    print('proportion of benign are selected:',proportion_of_benign_are_selected)
    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model, proportion_of_malicious_are_selected, proportion_of_benign_are_selected

def computr_gradient(global_model, single_local_model):
    global_model - single_local_model

def test_gradient(local_model, update_params, global_model, args, epochs):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    


# def flame_fake(local_model, update_params, global_model, args, epochs):
#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     # for i in range(len(local_model_vector)):
#     #     cos_i = []
#     #     for j in range(len(local_model_vector)):
#     #         cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#     #         # cos_i.append(round(cos_ij.item(),4))
#     #         cos_i.append(cos_ij.item())
#     #     cos_list.append(cos_i)
#     print(len(local_model_vector[0]), len(local_model_vector[1]), len(local_model_vector[3]))
#     numpy_local_vector = [tensor.detach().cpu().numpy() for tensor in local_model_vector]
#     numpy_local_vector = torch.tensor(numpy_local_vector).cuda()
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(numpy_local_vector.cpu())
#     print("clusterer_labels:",clusterer.labels_)
#     benign_client = []
#     norm_list = np.array([])

#     max_num_in_cluster=0
#     max_cluster_index=0
#     if clusterer.labels_.max() < 0:
#         for i in range(len(local_model)):
#             benign_client.append(i)
#             norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     else:
#         for index_cluster in range(clusterer.labels_.max()+1):
#             if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
#                 max_cluster_index = index_cluster
#                 max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
#         for i in range(len(clusterer.labels_)):
#             if clusterer.labels_[i] == max_cluster_index:
#                 benign_client.append(i)
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
#     # print(benign_client)
#     # print('num_malicious_clients: ',num_malicious_clients)
#     # for i in range(len(benign_client)):
#     #     print(benign_client[i])
#     if epochs % 10 == 0:
#         lengths = [torch.norm(vecto) for vecto in local_model_vector]
#         lengths_cuda = [tensor.cuda() for tensor in lengths]
#         print(lengths_cuda)
#         coss = []
#         weight_1 = torch.ones_like(local_model_vector[0]).cuda()
#         for i in range(len(local_model_vector)):
#             cos_0i = torch.cosine_similarity(weight_1.unsqueeze(0), local_model_vector[i].unsqueeze(0))
#             coss.append(cos_0i)
#         coss_cuda = [tensor.cuda() for tensor in coss]
#         print(coss_cuda)
        
#         lengths_numpy = torch.stack(lengths_cuda).flatten().cpu().numpy()
#         coss_numpy = torch.stack(coss_cuda).flatten().cpu().numpy()
#         print(lengths_numpy * coss_numpy)
        
#         plt.figure()
#         plt.xlabel('x')
#         plt.ylabel('y')
        
#         i = 0
#         check = False
#         labels = []
#         label_red_show = False
#         label_green_show = False
#         label_yellow_show = False
#         label_blue_show = False
#         lengths_tensor = torch.from_numpy(lengths_numpy)
#         coss_tensor = torch.from_numpy(coss_numpy)

#         for x, y in zip(lengths_tensor * coss_tensor, lengths_tensor * torch.sqrt(1 - coss_tensor ** 2)):
#             if i < num_malicious_clients:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         label = 'Tấn công nhưng được chọn'
#                         plt.plot([0, x], [0, y], 'r', linewidth=0.5, label=label if not label_red_show else None)
#                         label_red_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'green', linewidth=0.5, label="Tấn công và bị loại" if not label_green_show else None)
#                     label_green_show = True
#             else:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         plt.plot([0, x], [0, y], 'yellow', linewidth=0.5, label='Đúng và được chọn' if not label_yellow_show else None)
#                         label_yellow_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'b', linewidth=0.5, label='Đúng nhưng bị loại' if not label_blue_show else None)
#                     label_blue_show = True
#             check = False
#             i += 1
        
#         plt.legend()
#         plt.savefig('./' + args.save + '/' + str(epochs) + str(args.attack)+ str(args.defence) + 'vu.pdf', format='pdf', bbox_inches='tight')

    
#     for i in range(len(benign_client)):
#         if benign_client[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     proportion_of_malicious_are_selected = args.wrong_mal/(num_malicious_clients*args.turn)
#     proportion_of_benign_are_selected = args.right_ben/(num_benign_clients*args.turn)
#     print('proportion of malicious are selected:',proportion_of_malicious_are_selected)
#     print('proportion of benign are selected:',proportion_of_benign_are_selected)
#     clip_value = np.median(norm_list)
#     for i in range(len(benign_client)):
#         gama = clip_value/norm_list[i]
#         if gama < 1:
#             for key in update_params[benign_client[i]]:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#                 update_params[benign_client[i]][key] *= gama
#     global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
#     return global_model, proportion_of_malicious_are_selected, proportion_of_benign_are_selected


# def k_means(local_model, update_params, global_model, args, epochs):
#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     for i in range(len(local_model_vector)):
#         cos_i = []
#         for j in range(len(local_model_vector)):
#             cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#             # cos_i.append(round(cos_ij.item(),4))
#             cos_i.append(cos_ij.item())
#         cos_list.append(cos_i)
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
    
#     clusterer = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(cos_list)
#     label_counts = np.bincount(clusterer.labels_)
#     count_label_0 = label_counts[0]
#     count_label_1 = label_counts[1]
#     benign_label = 1
#     if(count_label_0 >= count_label_1):
#         benign_label = 0
#     print("clusterer_labels:",clusterer.labels_)
#     benign_client = []
#     norm_list = np.array([])

#     max_num_in_cluster=0
#     max_cluster_index=0
    
#     if benign_label == 0:
#         benign_client = np.where(clusterer.labels_ == 0)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     elif benign_label == 1:
#         benign_client = np.where(clusterer.labels_ == 1)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    
#     # if clusterer.labels_.max() < 1:
#     #     for i in range(len(local_model)):
#     #         benign_client.append(i)
#     #         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     # else:
#     #     for index_cluster in range(clusterer.labels_.max()+1):
#     #         if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
#     #             max_cluster_index = index_cluster
#     #             max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
#     #     for i in range(len(clusterer.labels_)):
#     #         if clusterer.labels_[i] == max_cluster_index:
#     #             benign_client.append(i)
#     if epochs % 10 == 0:
#         lengths = [torch.norm(vecto).cuda() for vecto in local_model_vector]
#         lengths_cuda = [tensor.cuda() for tensor in lengths]
#         print(lengths_cuda)
#         coss = []
#         weight_1 = torch.ones_like(local_model_vector[0]).cuda()
#         for i in range(len(local_model_vector)):
#             cos_0i = torch.cosine_similarity(weight_1.unsqueeze(0), local_model_vector[i].unsqueeze(0))
#             coss.append(cos_0i)
#         coss_cuda = [tensor.cuda() for tensor in coss]
#         print(coss_cuda)
        
#         lengths_numpy = torch.stack(lengths_cuda).flatten().cpu().numpy()
#         coss_numpy = torch.stack(coss_cuda).flatten().cpu().numpy()
#         print(lengths_numpy * coss_numpy)
        
#         plt.figure()
#         plt.xlabel('x')
#         plt.ylabel('y')
        
#         i = 0
#         check = False
#         labels = []
#         label_red_show = False
#         label_green_show = False
#         label_yellow_show = False
#         label_blue_show = False
#         lengths_tensor = torch.from_numpy(lengths_numpy)
#         coss_tensor = torch.from_numpy(coss_numpy)

#         for x, y in zip(lengths_tensor * coss_tensor, lengths_tensor * torch.sqrt(1 - coss_tensor ** 2)):
#             if i < num_malicious_clients:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         label = 'Tấn công nhưng được chọn'
#                         plt.plot([0, x], [0, y], 'r', linewidth=0.5, label=label if not label_red_show else None)
#                         label_red_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'green', linewidth=0.5, label="Tấn công và bị loại" if not label_green_show else None)
#                     label_green_show = True
#             else:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         plt.plot([0, x], [0, y], 'yellow', linewidth=0.5, label='Đúng và được chọn' if not label_yellow_show else None)
#                         label_yellow_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'b', linewidth=0.5, label='Đúng nhưng bị loại' if not label_blue_show else None)
#                     label_blue_show = True
#             check = False
#             i += 1
        
#         plt.legend()
#         plt.savefig('./' + args.save + '/' + str(epochs) + str(args.attack) + 'vu.pdf', format='pdf', bbox_inches='tight')

    
#     for i in range(len(benign_client)):
#         if benign_client[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     proportion_of_malicious_are_selected = args.wrong_mal/(num_malicious_clients*args.turn)
#     proportion_of_benign_are_selected = args.right_ben/(num_benign_clients*args.turn)
#     print('proportion of malicious are selected:',proportion_of_malicious_are_selected)
#     print('proportion of benign are selected:',proportion_of_benign_are_selected)
#     clip_value = np.median(norm_list)
#     for i in range(len(benign_client)):
#         gama = clip_value/norm_list[i]
#         if gama < 1:
#             for key in update_params[benign_client[i]]:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#                 update_params[benign_client[i]][key] *= gama
#     global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
#     return global_model, proportion_of_malicious_are_selected, proportion_of_benign_are_selected

# def kmeans_hdbscan(local_model, update_params, global_model, args, epochs):
#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     for i in range(len(local_model_vector)):
#         cos_i = []
#         for j in range(len(local_model_vector)):
#             cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#             # cos_i.append(round(cos_ij.item(),4))
#             cos_i.append(cos_ij.item())
#         cos_list.append(cos_i)
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
#     clusterer = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(cos_list)
#     label_counts = np.bincount(clusterer.labels_)
#     count_label_0 = label_counts[0]
#     count_label_1 = label_counts[1]
#     benign_label = 1
#     if(count_label_0 >= count_label_1):
#         benign_label = 0
#     # print("clusterer_labels:",clusterer.labels_)
#     benign_client = []
#     norm_list = np.array([])

#     max_num_in_cluster=0
#     max_cluster_index=0
    
#     if benign_label == 0:
#         benign_client = np.where(clusterer.labels_ == 0)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     elif benign_label == 1:
#         benign_client = np.where(clusterer.labels_ == 1)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     cos_list_kmeans =[]
#     for i in range(len(benign_client)):
#         # print(benign_client[i])
#         cos_kmeans_i = []
#         for j in range(len(benign_client)):
#             cos_kmeans_ij = 1 - cos(local_model_vector[benign_client[i]], local_model_vector[benign_client[j]])
#             cos_kmeans_i.append(cos_kmeans_ij.item())
#         cos_list_kmeans.append(cos_kmeans_i)
#     clusterer_hdbscan = hdbscan.HDBSCAN(min_cluster_size =  int(len(benign_client)*0.75), min_samples=1,allow_single_cluster=True).fit(cos_list_kmeans)
    
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    
#     benign_client_hdbscan = []
#     if clusterer_hdbscan.labels_.max() < 0:
#         for i in range(len(benign_client)):
#             benign_client_hdbscan.append(benign_client[i])
#             norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[benign_client[i]]),p=2).item())
#     else:
#         for index_cluster in range(clusterer_hdbscan.labels_.max()+1):
#             if len(clusterer_hdbscan.labels_[clusterer_hdbscan.labels_==index_cluster]) > max_num_in_cluster:
#                 max_cluster_index = index_cluster
#                 max_num_in_cluster = len(clusterer_hdbscan.labels_[clusterer_hdbscan.labels_==index_cluster])
#         for i in range(len(clusterer_hdbscan.labels_)):
#             if clusterer_hdbscan.labels_[i] == max_cluster_index:
#                 benign_client_hdbscan.append(benign_client[i])
    
    
#     if epochs % 10 == 0:
#         lengths = [torch.norm(vecto).cuda() for vecto in local_model_vector]
#         lengths_cuda = [tensor.cuda() for tensor in lengths]
#         # print(lengths_cuda)
#         coss = []
#         weight_1 = torch.ones_like(local_model_vector[0]).cuda()
#         for i in range(len(local_model_vector)):
#             cos_0i = torch.cosine_similarity(weight_1.unsqueeze(0), local_model_vector[i].unsqueeze(0))
#             coss.append(cos_0i)
#         coss_cuda = [tensor.cuda() for tensor in coss]
#         # print(coss_cuda)
        
#         lengths_numpy = torch.stack(lengths_cuda).flatten().cpu().numpy()
#         coss_numpy = torch.stack(coss_cuda).flatten().cpu().numpy()
#         # print(lengths_numpy * coss_numpy)
        
#         plt.figure()
#         plt.xlabel('x')
#         plt.ylabel('y')
        
#         i = 0
#         check = False
#         labels = []
#         label_red_show = False
#         label_green_show = False
#         label_yellow_show = False
#         label_blue_show = False
#         lengths_tensor = torch.from_numpy(lengths_numpy)
#         coss_tensor = torch.from_numpy(coss_numpy)

#         for x, y in zip(lengths_tensor * coss_tensor, lengths_tensor * torch.sqrt(1 - coss_tensor ** 2)):
#             if i < num_malicious_clients:
#                 for j in range(len(benign_client_hdbscan)):
#                     if benign_client_hdbscan[j] == i:
#                         label = 'Tấn công nhưng được chọn'
#                         plt.plot([0, x], [0, y], 'r', linewidth=0.5, label=label if not label_red_show else None)
#                         label_red_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'green', linewidth=0.5, label="Tấn công và bị loại" if not label_green_show else None)
#                     label_green_show = True
#             else:
#                 for j in range(len(benign_client_hdbscan)):
#                     if benign_client_hdbscan[j] == i:
#                         plt.plot([0, x], [0, y], 'yellow', linewidth=0.5, label='Đúng và được chọn' if not label_yellow_show else None)
#                         label_yellow_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'b', linewidth=0.5, label='Đúng nhưng bị loại' if not label_blue_show else None)
#                     label_blue_show = True
#             check = False
#             i += 1
        
#         plt.legend()
#         plt.savefig('./' + args.save + '/' + str(epochs) + str(args.attack) + 'vu.pdf', format='pdf', bbox_inches='tight')
#     print(benign_client)
#     print(benign_client_hdbscan)
#     for i in range(len(benign_client_hdbscan)):
#         # print( benign_client_hdbscan[i])
#         if benign_client_hdbscan[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     proportion_of_malicious_are_selected = args.wrong_mal/(num_malicious_clients*args.turn)
#     proportion_of_benign_are_selected = args.right_ben/(num_benign_clients*args.turn)
#     print('proportion of malicious are selected:',proportion_of_malicious_are_selected)
#     print('proportion of benign are selected:',proportion_of_benign_are_selected)
#     clip_value = np.median(norm_list)
#     for i in range(len(benign_client)):
#         gama = clip_value/norm_list[i]
#         if gama < 1:
#             for key in update_params[benign_client[i]]:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#                 update_params[benign_client[i]][key] *= gama
#     global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
#     return global_model, proportion_of_malicious_are_selected, proportion_of_benign_are_selected

# def kmeans_nocos(local_model, update_params, global_model, args, epochs):
#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     # for i in range(len(local_model_vector)):
#     #     cos_i = []
#     #     for j in range(len(local_model_vector)):
#     #         cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#     #         # cos_i.append(round(cos_ij.item(),4))
#     #         cos_i.append(cos_ij.item())
#     #     cos_list.append(cos_i)
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
#     numpy_local_vector = [tensor.detach().cpu().numpy() for tensor in local_model_vector]
#     numpy_local_vector_cuda = torch.tensor(numpy_local_vector).cuda()
#     clusterer = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(numpy_local_vector)
#     label_counts = np.bincount(clusterer.labels_)
#     count_label_0 = label_counts[0]
#     count_label_1 = label_counts[1]
#     benign_label = 1
#     if(count_label_0 >= count_label_1):
#         benign_label = 0
#     print("clusterer_labels:",clusterer.labels_)
#     benign_client = []
#     norm_list = np.array([])

#     max_num_in_cluster=0
#     max_cluster_index=0
    
#     if benign_label == 0:
#         benign_client = np.where(clusterer.labels_ == 0)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     elif benign_label == 1:
#         benign_client = np.where(clusterer.labels_ == 1)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    
#     # if clusterer.labels_.max() < 1:
#     #     for i in range(len(local_model)):
#     #         benign_client.append(i)
#     #         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     # else:
#     #     for index_cluster in range(clusterer.labels_.max()+1):
#     #         if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
#     #             max_cluster_index = index_cluster
#     #             max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
#     #     for i in range(len(clusterer.labels_)):
#     #         if clusterer.labels_[i] == max_cluster_index:
#     #             benign_client.append(i)
#     if epochs % 10 == 0:
#         lengths = [torch.norm(vecto).cuda() for vecto in local_model_vector]
#         lengths_cuda = [tensor.cuda() for tensor in lengths]
#         # print(lengths_cuda)
#         coss = []
#         weight_1 = torch.ones_like(local_model_vector[0]).cuda()
#         for i in range(len(local_model_vector)):
#             cos_0i = torch.cosine_similarity(weight_1.unsqueeze(0), local_model_vector[i].unsqueeze(0))
#             coss.append(cos_0i)
#         coss_cuda = [tensor.cuda() for tensor in coss]
#         # print(coss_cuda)
        
#         lengths_numpy = torch.stack(lengths_cuda).flatten().cpu().numpy()
#         coss_numpy = torch.stack(coss_cuda).flatten().cpu().numpy()
#         # print(lengths_numpy * coss_numpy)
        
#         plt.figure()
#         plt.xlabel('x')
#         plt.ylabel('y')
        
#         i = 0
#         check = False
#         labels = []
#         label_red_show = False
#         label_green_show = False
#         label_yellow_show = False
#         label_blue_show = False
#         lengths_tensor = torch.from_numpy(lengths_numpy)
#         coss_tensor = torch.from_numpy(coss_numpy)

#         for x, y in zip(lengths_tensor * coss_tensor, lengths_tensor * torch.sqrt(1 - coss_tensor ** 2)):
#             if i < num_malicious_clients:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         label = 'Tấn công nhưng được chọn'
#                         plt.plot([0, x], [0, y], 'r', linewidth=0.5, label=label if not label_red_show else None)
#                         label_red_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'green', linewidth=0.5, label="Tấn công và bị loại" if not label_green_show else None)
#                     label_green_show = True
#             else:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         plt.plot([0, x], [0, y], 'yellow', linewidth=0.5, label='Đúng và được chọn' if not label_yellow_show else None)
#                         label_yellow_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'b', linewidth=0.5, label='Đúng nhưng bị loại' if not label_blue_show else None)
#                     label_blue_show = True
#             check = False
#             i += 1
        
#         plt.legend()
#         plt.savefig('./' + args.save + '/' + str(epochs) + str(args.attack) + 'vu.pdf', format='pdf', bbox_inches='tight')

    
#     for i in range(len(benign_client)):
#         if benign_client[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     proportion_of_malicious_are_selected = args.wrong_mal/(num_malicious_clients*args.turn)
#     proportion_of_benign_are_selected = args.right_ben/(num_benign_clients*args.turn)
#     print('proportion of malicious are selected:',proportion_of_malicious_are_selected)
#     print('proportion of benign are selected:',proportion_of_benign_are_selected)
#     clip_value = np.median(norm_list)
#     for i in range(len(benign_client)):
#         gama = clip_value/norm_list[i]
#         if gama < 1:
#             for key in update_params[benign_client[i]]:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#                 update_params[benign_client[i]][key] *= gama
#     global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
#     return global_model, proportion_of_malicious_are_selected, proportion_of_benign_are_selected

# def Spectral_Clustering(local_model, update_params, global_model, args, epochs):
#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     for i in range(len(local_model_vector)):
#         cos_i = []
#         for j in range(len(local_model_vector)):
#             cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#             # cos_i.append(round(cos_ij.item(),4))
#             cos_i.append(cos_ij.item())
#         cos_list.append(cos_i)
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
    
#     clusterer = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0).fit(cos_list)
#     label_counts = np.bincount(clusterer.labels_)
#     count_label_0 = label_counts[0]
#     count_label_1 = label_counts[1]
#     benign_label = 1
#     if(count_label_0 >= count_label_1):
#         benign_label = 0
#     print("clusterer_labels:",clusterer.labels_)
#     benign_client = []
#     norm_list = np.array([])

#     max_num_in_cluster=0
#     max_cluster_index=0
    
#     if benign_label == 0:
#         benign_client = np.where(clusterer.labels_ == 0)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     elif benign_label == 1:
#         benign_client = np.where(clusterer.labels_ == 1)[0]
#         # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    
#     # if clusterer.labels_.max() < 1:
#     #     for i in range(len(local_model)):
#     #         benign_client.append(i)
#     #         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     # else:
#     #     for index_cluster in range(clusterer.labels_.max()+1):
#     #         if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
#     #             max_cluster_index = index_cluster
#     #             max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
#     #     for i in range(len(clusterer.labels_)):
#     #         if clusterer.labels_[i] == max_cluster_index:
#     #             benign_client.append(i)
#     if epochs % 10 == 0:
#         lengths = [torch.norm(vecto).cuda() for vecto in local_model_vector]
#         lengths_cuda = [tensor.cuda() for tensor in lengths]
#         print(lengths_cuda)
#         coss = []
#         weight_1 = torch.ones_like(local_model_vector[0]).cuda()
#         for i in range(len(local_model_vector)):
#             cos_0i = torch.cosine_similarity(weight_1.unsqueeze(0), local_model_vector[i].unsqueeze(0))
#             coss.append(cos_0i)
#         coss_cuda = [tensor.cuda() for tensor in coss]
#         print(coss_cuda)
        
#         lengths_numpy = torch.stack(lengths_cuda).flatten().cpu().numpy()
#         coss_numpy = torch.stack(coss_cuda).flatten().cpu().numpy()
#         print(lengths_numpy * coss_numpy)
        
#         plt.figure()
#         plt.xlabel('x')
#         plt.ylabel('y')
        
#         i = 0
#         check = False
#         labels = []
#         label_red_show = False
#         label_green_show = False
#         label_yellow_show = False
#         label_blue_show = False
#         lengths_tensor = torch.from_numpy(lengths_numpy)
#         coss_tensor = torch.from_numpy(coss_numpy)

#         for x, y in zip(lengths_tensor * coss_tensor, lengths_tensor * torch.sqrt(1 - coss_tensor ** 2)):
#             if i < num_malicious_clients:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         label = 'Tấn công nhưng được chọn'
#                         plt.plot([0, x], [0, y], 'r', linewidth=0.5, label=label if not label_red_show else None)
#                         label_red_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'green', linewidth=0.5, label="Tấn công và bị loại" if not label_green_show else None)
#                     label_green_show = True
#             else:
#                 for j in range(len(benign_client)):
#                     if benign_client[j] == i:
#                         plt.plot([0, x], [0, y], 'yellow', linewidth=0.5, label='Đúng và được chọn' if not label_yellow_show else None)
#                         label_yellow_show = True
#                         check = True
#                         break
#                 if not check:
#                     plt.plot([0, x], [0, y], 'b', linewidth=0.5, label='Đúng nhưng bị loại' if not label_blue_show else None)
#                     label_blue_show = True
#             check = False
#             i += 1
        
#         plt.legend()
#         plt.savefig('./' + args.save + '/' + str(epochs) + str(args.attack) + 'vu.pdf', format='pdf', bbox_inches='tight')

    
#     for i in range(len(benign_client)):
#         if benign_client[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     proportion_of_malicious_are_selected = args.wrong_mal/(num_malicious_clients*args.turn)
#     proportion_of_benign_are_selected = args.right_ben/(num_benign_clients*args.turn)
#     print('proportion of malicious are selected:',proportion_of_malicious_are_selected)
#     print('proportion of benign are selected:',proportion_of_benign_are_selected)
#     clip_value = np.median(norm_list)
#     for i in range(len(benign_client)):
#         gama = clip_value/norm_list[i]
#         if gama < 1:
#             for key in update_params[benign_client[i]]:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#                 update_params[benign_client[i]][key] *= gama
#     global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
#     return global_model, proportion_of_malicious_are_selected, proportion_of_benign_are_selected
