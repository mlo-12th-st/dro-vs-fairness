"""
plots.py

All helper functions for plotting
"""

import matplotlib.pyplot as plt

def plot_acc(acc_arr, labels=None, title='Accuracy Plot'):
    """
    Create plot of accuracy vs. epochs

    Parameters
    ----------
    acc_arr : list, np.ndarray
        Each item in the list stores a
        numpy array holding the accuracy
        at each epoch in training
        We assume each numpy array has
        the same length
    labels : list, str
        Labels for each line on plot
    title : str
        Plot title

    Returns
    -------
    None.

    """
    
    epochs = [i+1 for i in range(len(acc_arr[0]))]
    
    for i in range(len(acc_arr)):
        if labels==None:
            plt.plot(epochs, acc_arr[i])
        else:
            plt.plot(epochs, acc_arr[i], label=labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    if labels!=None:
        plt.legend()
    plt.show()