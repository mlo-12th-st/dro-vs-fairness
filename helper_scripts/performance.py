"""
performance.py

Evaluate the performance of a model using the following metrics:
    - Accuracy
    - Distributional robustness (worst-group accuracy)
    - Privacy ([insert measure here])
"""

import os
import sys
sys.path.append('../')
import utils
import models
import celebA_dataset

import argparse
import numpy as np
import pandas as pd
import torch


def main():
    os.chdir('../')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)              # model file
    parser.add_argument('-a', '--arch', default='resnet50')     # model architecture
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--target_attr', default='Blond_Hair')  # Target attribute
    parser.add_argument('--spur_attr', default='Male')          # Spurious attribute
    args = parser.parse_args()
    
    
    """ Option for GPU computing """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    
    """ Load model from file """
    model_path = './models/' + args.model
    checkpoint = torch.load(model_path)
    if args.arch.lower() == 'resnet50':
        model = models.ResNet50()
    elif args.arch.lower() == 'resnet18':
        model = models.ResNet18()
    elif args.arch.lower() == 'cnn':
        model = models.CNN(dim=args.image_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    """ Load dataset and dataloader """
    train_data, test_data, trainloader, testloader = celebA_dataset.load_data(batch_size=4,
                                                       image_size=args.image_size,
                                                       # train_test_split=2e-4,
                                                       train_test_split=.2,
                                                       target_attr=args.target_attr,
                                                       spur_attr=args.spur_attr,
                                                       dataset='celeba_subset_test')
    
    
    # accuracy
    acc = utils.accuracy(model, testloader, device)
    print('Accuracy: %.4f' % (acc))
    
    # worst-group accuracy
    worst_group_acc = utils.worst_group_acc(model, testloader, device)
    print('Worst-group accuracy: %.4f' % (worst_group_acc))
    
    # privacy metric
    
    

if __name__ == '__main__':
    main()