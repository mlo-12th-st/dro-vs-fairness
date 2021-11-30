"""
acc_plots.py

Script to read in csv and generate accuracy plots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    args = parser.parse_args()
    
    """ Overall accuracy plot """
    df = pd.read_csv('../results/'+args.file, index_col=False)
    plt.plot(df['Epochs'], df['Overall- train'], label='train')
    plt.plot(df['Epochs'], df['Overall- test'], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    """ Group accuracy plot """
    plt.figure()
    plt.plot(df['Epochs'], df['Blond, male- train'], c='tab:blue', alpha=0.5)
    plt.plot(df['Epochs'], df['Blond, male- test'], label='Blond, male', c='tab:blue')
    plt.plot(df['Epochs'], df['Blond, female- train'], c='tab:orange', alpha=0.5)
    plt.plot(df['Epochs'], df['Blond, female- test'], label='Blond, female', c='tab:orange')
    plt.plot(df['Epochs'], df['Not blond, male- train'], c='tab:green', alpha=0.5)
    plt.plot(df['Epochs'], df['Not blond, male- test'], label='Not blond, male', c='tab:green')
    plt.plot(df['Epochs'], df['Not blond, female- train'], c='tab:red', alpha=0.5)
    plt.plot(df['Epochs'], df['Not blond, female- test'], label='Not blond, female', c='tab:red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

if __name__ == '__main__':
    main()