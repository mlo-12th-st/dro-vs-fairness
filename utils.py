"""
utils.py

Helper functions
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report


def split_group_data(dataloader):
    """
    Split data from dataloader into 4 separate groups
    based on target and spurious attributes
        ( attr1,  attr2) - e.g. blond hair, male
        ( attr1, !attr2) - e.g. blond, female
        (!attr1,  attr2)
        (!attr1, !attr2)

    Parameters
    ----------
    dataloader : torch.data.utils.DataLoader

    Returns
    -------
    group1_data, group2_data
    group3_data, group4_data : list, torch.Tensor
        subsets of data from dataloader

    """
    
    group1_data = []
    group2_data = []
    group3_data = []
    group4_data = []
    
    for i, data in enumerate(dataloader, 0):
        inputs, t_labels, s_labels = data
        
        # iterate through mini-batch
        for (x,y1,y2) in zip(inputs, t_labels, s_labels):
            if (y1==1 and y2==1):
                group1_data.append([x,y1,y2])
            elif (y1==1 and y2==0):
                group2_data.append([x,y1,y2])
            elif (y1==0 and y2==1):
                group3_data.append([x,y1,y2])
            else:
                group4_data.append([x,y1,y2])
                
    return group1_data, group2_data, group3_data, group4_data
    


def accuracy(model, dataloader, device):
    """
    Calculate accuracy of model on data in dataloader

    Parameters
    ----------
    model : torch.nn.Module
    dataloader : torch.data.utils.DataLoader
        
    Returns
    -------
    acc : float
        Model accuracy

    """
    
    y_real = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            X_batch, y, spur_labels = data
            y_hat = model(X_batch.to(device))
            y_hat = torch.sigmoid(y_hat)
            y_hat = torch.round(y_hat)
            y_pred.append(y_hat.cpu().numpy())
            y_real.append(y.cpu().numpy())
    
    y_real = np.array([a.squeeze() for a in y_real]).flatten()
    y_pred = np.array([a.squeeze() for a in y_pred]).flatten()

    acc = 1-np.sum(np.abs(y_real-y_pred))/len(y_real)
    return acc
    

def group_accuracy(model, dataloader, device):
    """
    Calculate accuracy of model on each group
    in dataset:
        ( attr1,  attr2) - e.g. blond hair, male
        ( attr1, !attr2) - e.g. blond, female
        (!attr1,  attr2)
        (!attr1, !attr2)

    Parameters
    ----------
    model : torch.nn.Module
    dataloader : torch.data.utils.DataLoader
        
    Returns
    -------
    group1_acc, group2_acc
    group3_acc, group4_acc : float
        Model accuracies on each group
        
    """
    
    y_real = []
    y_pred = []
    y2_real = []    # spurious attribute (i.e. attr2)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            X_batch, y, y2 = data
            y_hat = model(X_batch.to(device))
            y_hat = torch.sigmoid(y_hat)
            y_hat = torch.round(y_hat)
            y_pred.append(y_hat.cpu().numpy())
            y_real.append(y.cpu().numpy())
            y2_real.append(y2.cpu().numpy())
    
    y_real = np.array([a.squeeze() for a in y_real]).flatten()
    y_pred = np.array([a.squeeze() for a in y_pred]).flatten()
    y2_real = np.array([a.squeeze() for a in y2_real]).flatten()
    
    # in each list, store 1 if correct prediction, 0 if incorrect
    group1_pred = []
    group2_pred = []
    group3_pred = []
    group4_pred = []
    
    for i in range(len(y_real)):
        if (y_real[i] == 1 and y2_real[i] == 1):
            group1_pred.append(1-np.abs(y_real[i]-y_pred[i]))
        elif (y_real[i] == 1 and y2_real[i] == 0):
            group2_pred.append(1-np.abs(y_real[i]-y_pred[i]))
        elif (y_real[i] == 0 and y2_real[i] == 1):
            group3_pred.append(1-np.abs(y_real[i]-y_pred[i]))
        else:
            group4_pred.append(1-np.abs(y_real[i]-y_pred[i]))

    group1_acc = np.sum(group1_pred)/len(group1_pred)
    group2_acc = np.sum(group2_pred)/len(group2_pred)
    group3_acc = np.sum(group3_pred)/len(group3_pred)
    group4_acc = np.sum(group4_pred)/len(group4_pred)

    return group1_acc, group2_acc, group3_acc, group4_acc


def worst_group_acc(model, dataloader, device):
    """
    Return the Worst-Group accuracy of the model 
    on the data

    Parameters
    ----------
    model : torch.nn.Module
    dataloader : torch.data.utils.DataLoader

    Returns
    -------
    worst_group_acc : float

    """
    
    a, b, c, d = group_accuracy(model, dataloader, device)
    worst_group_acc = np.min([a,b,c,d])
    return worst_group_acc


def print_metrics(model, trainloader, testloader, device):
    """
    Print performance metrics of model on training and test data
        Calculate train/test accuracy
        Calculate worst-group accuracy for train/test sets
        Display confusion matrix and classification report on test data

    Parameters
    ----------
    model : torch.nn.Module
    trainloader : torch.data.utils.DataLoader
    testloader : torch.data.utils.DataLoader

    Returns
    -------
    None.

    """
    
    # calculate train
    y_train = []
    y_pred_train = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            X_batch, y, spur_labels = data
            y_test_pred = model(X_batch.to(device))
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_train.append(y_pred_tag.cpu().numpy())
            y_train.append(y.cpu().numpy())
    
    y_train = np.array([a.squeeze() for a in y_train]).flatten()
    y_pred_train = np.array([a.squeeze() for a in y_pred_train]).flatten()
    
    # calculate test
    y_test = []
    y_pred_test = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            X_batch, y, spur_labels = data
            y_test_pred = model(X_batch.to(device))
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_test.append(y_pred_tag.cpu().numpy())
            y_test.append(y.cpu().numpy())
    
    y_test = np.array([a.squeeze() for a in y_test]).flatten()
    y_pred_test = np.array([a.squeeze() for a in y_pred_test]).flatten()
    
    print('Train accuracy: %.3f' % (1-np.sum(np.abs(y_train-y_pred_train))/len(y_train)))
    print(' Test accuracy: %.3f' % (1-np.sum(np.abs(y_test-y_pred_test))/len(y_test)))
    
    print('Training Data Worst-Group Accuracy: %.3f' % (worst_group_acc(model, trainloader, device)))
    print('Test Data Worst-Group Accuracy: %.3f' % (worst_group_acc(model, testloader, device)))
    
    print(confusion_matrix(y_test, y_pred_test))
    
    print(classification_report(y_test, y_pred_test))