import numpy as np
import os
import torch
from collections import defaultdict


def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    if dataset == 'ISIC2019':
        # Use the custom dataset class for on-the-fly augmentations
        import sys
        import importlib.util
        
        dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        module_path = os.path.join(dataset_path, 'isic2019_dataset.py')
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("isic2019_dataset", module_path)
        isic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isic_module)
        ISIC2019Dataset = isic_module.ISIC2019Dataset
        
        data = ISIC2019Dataset(client_id=idx, train=is_train, data_path=dataset_path)
        if is_train and few_shot > 0:
            # For few-shot, limit the dataset size
            from torch.utils.data import Subset
            indices = []
            class_counts = defaultdict(int)
            for i in range(len(data)):
                label = data.labels[i]  # Access labels directly without loading image
                if class_counts[label] < few_shot:
                    indices.append(i)
                    class_counts[label] += 1
            data = Subset(data, indices)
        return data
    else:
        # Original implementation for other datasets
        data = read_data(dataset, idx, is_train)
        if "News" in dataset:
            data_list = process_text(data)
        elif "Shakespeare" in dataset:
            data_list = process_Shakespeare(data)
        else:
            data_list = process_image(data, dataset)

        if is_train and few_shot > 0:
            shot_cnt_dict = defaultdict(int)
            data_list_new = []
            for data_item in data_list:
                label = data_item[1].item()
                if shot_cnt_dict[label] < few_shot:
                    data_list_new.append(data_item)
                    shot_cnt_dict[label] += 1
            data_list = data_list_new
        return data_list

def process_image(data, dataset=None):
    X = torch.Tensor(data['x']).type(torch.float32)
    # Transpose from (batch, H, W, C) to (batch, C, H, W) if needed
    if X.dim() == 4 and X.shape[-1] == 3:
        X = X.permute(0, 3, 1, 2)
    
    # Normalize for pretrained models (ImageNet stats)
    if dataset in ['ISIC2019', 'Cifar10', 'Cifar100', 'TinyImagenet', 'Flowers102', 'StanfordCars', 'kvasir', 'Camelyon17', 'iWildCam', 'Country211']:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        X = (X / 255.0 - mean) / std
    
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

