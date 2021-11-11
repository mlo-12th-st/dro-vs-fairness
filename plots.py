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
    
def plot_group_acc(group_train_acc, group_test_acc, labels=None, title='Group Accuracy'):
    """
    Create plot of accuracies for each subgroup of dataset
    (e.g. {blond, male}, {not blond, female})

    Parameters
    ----------
    group_train_acc : list, np.ndarray
    group_test_acc : list, np.ndarray
        list of length 4 with each element containing
        a numpy array holding the accuracy of the model
        on a specific subgroup at each epoch in training
    labels : str, optional
        list of labels for subgroups on plot. The default is None.
    title : str, optional
        Plot title. The default is 'Group Accuracy'.

    Returns
    -------
    None.

    """
    
    epochs = [i+1 for i in range(len(group_train_acc[0]))]
    if labels==None:
        labels=['y1=1, y2=1', 'y1=1, y2=0', 'y1=0, y2=1', 'y1=0, y2=0']
    
    plt.plot(epochs, group_train_acc[0], c='tab:blue', alpha=0.5)
    plt.plot(epochs, group_test_acc[0], label=labels[0], c='tab:blue')
    plt.plot(epochs, group_train_acc[1], c='tab:orange', alpha=0.5)
    plt.plot(epochs, group_test_acc[1], label=labels[1], c='tab:orange')
    plt.plot(epochs, group_train_acc[2], c='tab:green', alpha=0.5)
    plt.plot(epochs, group_test_acc[2], label=labels[2], c='tab:green')
    plt.plot(epochs, group_train_acc[3], c='tab:red', alpha=0.5)
    plt.plot(epochs, group_test_acc[3], label=labels[3], c='tab:red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()