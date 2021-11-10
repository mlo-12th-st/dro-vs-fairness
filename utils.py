"""
utils.py

Helper functions
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

def print_metrics(model, trainloader, testloader):
    """
    Print performance metrics of model on training and test data
        Calculate train/test accuracy
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
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_train.append(y_pred_tag.cpu().numpy())
            y_train.append(y.cpu().numpy())
    
    y_train = np.array([a.squeeze() for a in y_train]).flatten()
    y_pred_train = np.array([a.squeeze() for a in y_pred_train]).flatten()
    
    y_test = []
    y_pred_test = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            X_batch, y, spur_labels = data
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_test.append(y_pred_tag.cpu().numpy())
            y_test.append(y.cpu().numpy())
    
    y_test = np.array([a.squeeze() for a in y_test]).flatten()
    y_pred_test = np.array([a.squeeze() for a in y_pred_test]).flatten()
    
    print('Train accuracy: %.3f' % (1-np.sum(y_train-y_pred_train)/len(y_train)))
    print(' Test accuracy: %.3f' % (1-np.sum(y_test-y_pred_test)/len(y_test)))
    
    print(confusion_matrix(y_test, y_pred_test))
    
    print(classification_report(y_test, y_pred_test))