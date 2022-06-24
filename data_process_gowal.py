#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2022/6/23 17:14
# @Author  : lqyang
# @File    : data_process_gowal.py
# Desc:
change data format to fit to lightgcn input
"""

import os
import json
import tqdm
import pickle
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(load_path):
    data = []
    with open(load_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                data.append([int(k) for k in l])
    return data


def precess(data, save_path):
    """
    save data by list type
    """
    data_dict = {}
    for line in data:
        uer_id, item_id, label = line[0], line[1], line[2]
        if label:
            if data_dict.get(uer_id):
                data_dict[uer_id].extend([item_id])
            else:
                data_dict[uer_id] = [item_id]
    user_list = list(data_dict.keys())
    item_click_list = list(data_dict.values())
    result = [[]+[u]+v for u, v in zip(user_list, item_click_list)]

    with open(save_path, 'wb') as f:
        pickle.dump(result,f)

    with open(save_path, 'rb') as f:  # read data
        data_ = pickle.load(f)
    logging.info("%s number of users: %d", save_path[-15:],(len(user_list)))

def data_main(load_path, save_path):
    data = load_data(load_path)
    precess(data, save_path)



if __name__ == "__main__":
    # os.chdir('../')
    dataset = 'book'
    for name in ["train", "eval", "test"]:
        for i in range(1, 6):
            data_path = f"./data/{dataset}/split/{name}_{i}.txt"
            save_path = f"./data/{dataset}/lightgcn/{name}_{i}_gcn.txt"
            data_main(data_path, save_path)

