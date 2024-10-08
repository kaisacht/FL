from random import random
from models.Test import test_img
from models.FedAvg import FedAvg
from models.Net import CNN_MNIST, ResNet18, ResNet50, ResNet101, MobileNetV2, VGG
from models.MaliciousUpdate import LocalMaliciousUpdate
from models.Update import LocalUpdate
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sample import mnist_iid
from utils.defense import  multi_krum, get_update, RLR, flame, DABA, fltrust
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import random
import math
matplotlib.use('Agg')


def write_file(filename, accu_list, back_list, loss_list, args, analyse = False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    f.write('\n')
    f.write("loss_list=")
    f.write(str(loss_list))
    
    if args.defence == "krum":
        krum_file = filename+"_krum_dis"
        torch.save(args.krum_distance,krum_file)
    if analyse == True:
        need_length = len(accu_list)//10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc),2)
        average_back=round(np.mean(back),2)
        best_back=round(max(back),2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()


def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def caculatorLabel(dataset, dict_user):
    client  = [0,0,0,0,0,0,0,0,0,0]
    for j in range(len(dict_user)):
        my_list = list(dict_user)
        idx = my_list[j]
        label = dataset[idx][1]
        client[label] += 1
    client = normalized_data(client)
    return client

def attackCaculatorLabel():
    client = normalized_data([240,20,10,5,3,2,1,1,1,1])
    return client

def normalized_data(data):
    data = np.array(data)
    data = (data - min(data)) / (max(data) - min(data))
    return data  

def constrain_and_scale(w, w_glob, args):
    squared_sum = 0
    for key in w.keys():
        squared_sum += torch.sum(torch.pow(w[key] - w_glob[key], 2))
    model_norm = math.sqrt(squared_sum)
    if model_norm > args.s_norm:
        norm_scale = args.s_norm / model_norm
        for key in w.keys():
            w[key] = w_glob[key] + norm_scale * (w[key] - w_glob[key])
    return w

def pgd(w, w_glob, args):
    squared_sum = 0
    for key in w.keys():
        squared_sum += torch.sum(torch.pow(w[key] - w_glob[key], 2))
    model_norm = math.sqrt(squared_sum)
    if model_norm > args.s_norm:
        for key in w.keys():
            w[key] = w_glob[key] + args.malicious * (w[key] - w_glob[key])
    return w

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    test_mkdir('./'+args.save)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            if args.q_noniid == 0:
                dict_users = np.load('./data/non_iid_mnist_q0.npy', allow_pickle=True).item()
                print("load q0")
            elif args.q_noniid == 5:
                dict_users = np.load('./data/non_iid_mnist_q5.npy', allow_pickle=True).item()
                print("load q5")
            elif args.q_noniid == 15:
                dict_users = np.load('./data/non_iid_mnist_q15.npy', allow_pickle=True).item()
                print("load q15")
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            #dict_users = fashion_mnist_iid(dataset_train, args.num_users)
            dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            #dict_users = fashion_mnist_noniid( dataset_train, args.num_users)
            dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            #dict_users = cifar10_iid(dataset_train, args.num_users)
            dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            #dict_users = cifar10_noniid(dataset_train, args.num_users)
            dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == "cnn_mnist" or args.dataset == "mnist":
        net_glob = CNN_MNIST().to(args.device)
    elif args.model == "resnet18" and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == "resnet50" and args.dataset == 'cifar':
        net_glob = ResNet50().to(args.device)
    elif args.model == "resnet101" and args.dataset == 'cifar':
        net_glob = ResNet101().to(args.device)
    elif args.model == "mobilenetv2" and args.dataset == 'cifar':
        net_glob = MobileNetV2().to(args.device)
    elif args.dataset == 'fashion_mnist':
        net_glob = CNN_MNIST().to(args.device)

    else:
        exit('Error: unrecognized model')
    
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    
    if math.isclose(args.malicious, 0):
        backdoor_begin_acc = 100
    else:
        backdoor_begin_acc = args.attack_begin  # overtake backdoor_begin_acc then attack
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)
    base_info = get_base_info(args)
    filename = './save_result/cifar10/accuracy_file_{}.txt'.format(base_info)
    
    if args.init != 'None':
        param = torch.load(args.init)
        net_glob.load_state_dict(param)
        print("load init model")

        
    val_acc_list, net_list = [0], []
    backdoor_acculist = [0]
    loss_list = []
    
    args.attack_layers=[]
    if args.attack == "dba":
        args.dba_sign=0
    if args.defence == "krum":
        args.krum_distance=[]
        
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        random_array = np.random.uniform(-1, 1, 10)
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        list_label = []            
        if val_acc_list[-1] > backdoor_begin_acc:
            attack_number = int(args.malicious * m)
        else:
            attack_number = 0
        
        for num_turn, idx in enumerate(idxs_users):
            if attack_number > 0:
                attack = True
            else:
                attack = False
            if attack == True:
                idx = random.randint(0, int(args.num_users * args.malicious))
                if args.attack == "dba":
                    num_dba_attacker = int(args.num_users * args.malicious)
                    dba_group = num_dba_attacker/4
                    idx = args.dba_sign % (4*dba_group)
                    args.dba_sign+=1
                local = LocalMaliciousUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], order=idx)
                if args.attack == "layerattack_ER_his" or args.attack == "LFA" or args.attack == "LPA":
                    w, loss, args.attack_layers = local.train(
                        net=copy.deepcopy(net_glob).to(args.device), test_img = test_img)
                else:
                    w, loss = local.train(
                        net=copy.deepcopy(net_glob).to(args.device), test_img = test_img)
                    if args.type_poi == 'cs': # constrain and scale
                        w = constrain_and_scale(w, w_glob, args)
                    elif args.type_poi == 'pgd':
                        w = pgd(w, w_glob, args)
                print("client", idx, "--attack--")
                attack_number -= 1
                if args.style_send == "trust":
                    label_idx = caculatorLabel(dataset_train, dict_users[idx])
                    label_idx = label_idx + random_array
                    list_label.append(label_idx)
                elif args.style_send == "mistrust":
                    label_idx = attackCaculatorLabel()
                    label_idx = label_idx + random_array
                    list_label.append(label_idx)
            else:
                label_idx = caculatorLabel(dataset_train, dict_users[idx])
                label_idx = label_idx + random_array
                list_label.append(label_idx)
                
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(
                    net=copy.deepcopy(net_glob).to(args.device))
            
            w_updates.append(get_update(w, w_glob))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        if args.defence == 'avg':  # no defence
            w_glob = FedAvg(w_locals)
        elif args.defence == 'krum':  # single krum
            selected_client = multi_krum(w_updates, 1, args)
            # print(args.krum_distance)
            w_glob = w_locals[selected_client[0]]
            # w_glob = FedAvg([w_locals[i] for i in selected_clinet])
        elif args.defence == 'RLR':
            w_glob = RLR(copy.deepcopy(net_glob), w_updates, args)
        elif args.defence == 'flame':
            w_glob = flame(w_locals,w_updates,w_glob, args)
        elif args.defence == 'DABA':
            w_glob = DABA(copy.deepcopy(net_glob), w_updates, args, list_label, 3)
        elif args.defence == 'fltrust':
            local = LocalUpdate(args=args, dataset=dataset_test, idxs=central_dataset)
            fltrust_norm, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            fltrust_norm = get_update(fltrust_norm, w_glob)
            w_glob = fltrust(w_updates, fltrust_norm, w_glob, args)
        else:
            print("Wrong Defense Method")
            os._exit(0)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter % 1 == 0:
            acc_test, _, back_acc = test_img(
                net_glob, dataset_test, args, test_backdoor=True)
            print("Main accuracy: {:.2f}".format(acc_test))
            print("Backdoor accuracy: {:.2f}".format(back_acc))
            val_acc_list.append(acc_test.item())
            loss_list.append(loss_avg)
            backdoor_acculist.append(back_acc)
            write_file(filename, val_acc_list, backdoor_acculist, loss_list, args)
    
    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, loss_list, args, True)
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
   