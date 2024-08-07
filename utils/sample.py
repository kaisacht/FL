#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def group(datasets):
    dict_class = {}
    for iter in range(len(datasets)):
        label = datasets[iter][1]  # Lấy nhãn của mẫu dữ liệu thứ i
        if label not in dict_class:
            dict_class[label] = set()  # Khởi tạo tập hợp nếu nhãn chưa tồn tại trong dict_class
        dict_class[label].add(iter)  # Thêm chỉ mục i vào tập hợp tương ứng với nhãn
    return dict_class

def mnist_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# def mnist_noniid(datasets, number_user, number_class, q):
#     dict_user = {j: set() for j in range(number_user)}
#     num_per_user = len(datasets) // number_user
#     dict_class = group(datasets)
#     for ii in range(number_class):
#         num_per_user = min(num_per_user, len(dict_class[ii]) // number_class)
#     min_per_class = num_per_user // (q + number_class - 1)

#     for i in range(number_class):
#         min_index = 0
#         for j in range(number_user):
#             my_list = list(dict_class[i])
#             if (j - i) % number_class == 0:
#                 dict_user[j].update(set(my_list[k] for k in range(min_index, min_index + min_per_class * q)))
#                 min_index += min_per_class * q

#             else:
#                 num_elements = random.randint(1, min_per_class)
#                 if num_elements > min_per_class//2:
#                     num_elements -= min_per_class//2
#                 else:
#                     num_elements = 0
#                 selected_elements = random.sample(my_list[min_index:min_index + min_per_class], num_elements)
#                 dict_user[j].update(set(selected_elements))
#                 min_index += min_per_class
#     return dict_user


def cifar10_noniid(dataset, num_users):
    num_shards, num_imgs_per_class = 200, 250  # 10 classes x 30 images per class per shard
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs_per_class)
    labels = np.array(dataset.targets)

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs_per_class:(rand + 1) * num_imgs_per_class]), axis=0
            )
    return dict_users

def cifar10_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fashion_mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from Fashion MNIST dataset
    :param dataset: Fashion MNIST dataset
    :param num_users: Number of users
    :return: A dictionary containing non-I.I.D data for each user
    """
    num_shards, num_imgs_per_class = 200, 300  # 10 classes x 25 images per class per shard
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs_per_class)
    labels = np.array(dataset.targets)

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs_per_class:(rand + 1) * num_imgs_per_class]), axis=0
            )
    return dict_users


def fashion_mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def check_data_each_client(dataset_label, client_data_proportion, num_client, num_classes):
    for client in client_data_proportion.keys():
        client_data = dataset_label[list(client_data_proportion[client])]
        print('client', client, 'distribution information:')
        for i in range(num_classes):
            print('class ', i, ':', len(client_data[client_data==i])/len(client_data))


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    print(d)