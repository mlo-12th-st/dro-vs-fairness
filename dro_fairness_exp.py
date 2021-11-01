"""
dro_fairness_exp.py

Wrapper code to run experiment for training
    model(s) using SGD, distributionally-robust
    training, fairness training
    
To run in Jupyter notebook cell/IPython console:
    runfile('dro_fairness_exp.py', args='--f1 arg1 --f2 arg2 ...')
"""

import celebA_dataset

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--train_method', default='SGD')
    parser.add_argument('-d', '--dataset', default='celebA')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-image_size', type=int, default=128)
    parser.add_argument('--target_attr', default='Blond_Hair')
    parser.add_argument('--spur_attr', default='Male')
    
    args = parser.parse_args()
    
    print('training method: ' + args.train_method)
    print('dataset: ' + args.dataset)
    print('epochs: ' + str(args.epochs))
    print('batch size: ' + str(args.batch_size))
    
    if args.dataset == 'celebA':
        dataset, dataloader = celebA_dataset.load_data(batch_size=args.batch_size,
                                                       image_size=args.image_size,
                                                       target_attr=args.target_attr,
                                                       spur_attr=args.spur_attr)
    
    
    # Display image and label
    train_features, target_attrs, spur_attrs = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    plt.imshow(np.transpose(img[:64],(1,2,0)))
    # plt.imshow(np.transpose(vutils.make_grid(img[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.title('Target Attribute: %s, Spurious Attribute %s' % (target_attrs[0], spur_attrs[0]))
    plt.show()
    
    
if __name__ == '__main__':
    main()

