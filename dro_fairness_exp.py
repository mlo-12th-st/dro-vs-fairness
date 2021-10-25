"""
dro_fairness_exp.py

Wrapper code to run experiment for training
    model(s) using SGD, distributionally-robust
    training, fairness training
    
To run in Jupyter notebook cell/IPython console:
    runfile('dro_fairness_exp.py', args='--f1 arg1 --f2 arg2 ...')
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--train_method', default='SGD')
    parser.add_argument('-d', '--dataset', default='celebA')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    print('training method: ' + args.train_method)
    print('dataset: ' + args.dataset)
    print('epochs: ' + str(args.epochs))
    
    
    
if __name__ == '__main__':
    main()

