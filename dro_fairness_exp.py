"""
dro_fairness_exp.py

Wrapper code to run experiment for training
    model(s) using SGD, distributionally-robust
    training, fairness training
    
To run in Jupyter notebook cell/IPython console:
    runfile('dro_fairness_exp.py', args='--f1 arg1 --f2 arg2 ...')
"""

import celebA_dataset
import models
import loss
import train
import utils
import plots

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser()
    
    """ Command Line Args """
    
    # Model
    parser.add_argument('-m', '--model', default='resnet50')
    
    # Optimization
    parser.add_argument('-l', '--loss_fn', default='BCEWithLogitsLoss')
    parser.add_argument('-t', '--train_method', default='standard')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)     # learning rate
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    
    # Data
    parser.add_argument('-d', '--dataset', default='celebA')
    parser.add_argument('--train_test_split', type=float, default=0.8)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--target_attr', default='Blond_Hair')
    parser.add_argument('--spur_attr', default='Male')
    args = parser.parse_args()
    
    
    """ Print experiment parameters (for debugging purposes) """
    print('training method: ' + args.train_method)
    print('dataset: ' + args.dataset)
    print('epochs: ' + str(args.epochs))
    print('batch size: ' + str(args.batch_size))
    
    
    """ Load dataset and dataloader """
    if args.dataset == 'celebA':
        train_data, test_data, trainloader, testloader = celebA_dataset.load_data(batch_size=args.batch_size,
                                                       image_size=args.image_size,
                                                       train_test_split=args.train_test_split,
                                                       target_attr=args.target_attr,
                                                       spur_attr=args.spur_attr)
    
    """ Display example of CelebA dataset """
    # Display image and label
    train_features, target_attrs, spur_attrs = next(iter(trainloader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    plt.imshow(np.transpose(img[:64],(1,2,0)))
    # plt.imshow(np.transpose(vutils.make_grid(img[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.title('Target Attribute: %s, Spurious Attribute %s' % (target_attrs[0], spur_attrs[0]))
    plt.show()
    
    
    """ Model Selection """
    if args.model == 'ResNet50':
        #hello
        print('ResNet50')
    else:
        model = models.CNN(dim=args.image_size)
        
    """ Loss Function """
    if args.loss_fn == 'hinge':
        criterion = loss.hinge_loss
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    """ Optimizer """
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        
    """ Training Loop """
    if args.train_method == 'standard':
        train_acc, test_acc, \
        group_train_acc, group_test_acc = train.standard_train(model, criterion, optimizer, trainloader,
                                                  testloader, args.epochs, dro_flag=True)
    elif args.train_method == 'dro':
        train_acc, test_acc, \
        group_train_acc, group_test_acc = train.dro_train(model, criterion, optimizer, trainloader, 
                                              testloader, args.epochs, dro_flag=True)
    
    """ Plot Accuracy """
    plots.plot_acc([train_acc, test_acc], labels=['train', 'test'])
    
    """ Plot Group Accuracy """
    group_labels = ['Blond, male', 'Blond, female', 'Not blond, male', 'Not blond, female']
    plots.plot_group_acc(group_train_acc, group_test_acc, labels=group_labels)
    
    """ Test Performance """
    utils.print_metrics(model, trainloader, testloader)
    
if __name__ == '__main__':
    main()

