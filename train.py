"""
train.py

All training loop code
    Standard SGD
    Distributionally-robust training
    Fairness training
    DRO+fairness training
"""

import utils
import loss as Loss

import numpy as np
import torch

def standard_train(model, criterion, optimizer, trainloader,
                   testloader, num_epochs, device, dro_flag=False):
    
    """
    Train model using ERM

    Parameters
    ----------
    model : torch.nn.Module
    criterion : torch.nn.[Loss]
        loss function definition
    optimizer : torch.optim.[Optimizer]
    trainloader : torch.data.utils.DataLoader
    testloader : torch.data.utils.DataLoader
    num_epochs : int
    dro_flag : bool
        if True, keep track of accuracy on 
        different subsets of data

    Returns
    -------
    train_acc : np.ndarray
    test_acc : np.ndarray
        both hold accuracy with each epoch
        
    group_train_acc : list, np.ndarray
    group_test_acc : list, np.ndarray
        list of numpy arrays that hold
        accuracy on each group of data
        (e.g. blond hair, male)

    """
    
    train_acc = []
    test_acc = []
    group_train_acc = [[],[],[],[]]
    group_test_acc = [[],[],[],[]]
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels, spur_label]
            inputs, labels = data[0].to(device), data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[Epoch %d/%d, mini-batch %5d/%d] loss: %.3f' %
                      (epoch + 1, num_epochs, i + 1, len(trainloader), running_loss / 100))
                running_loss = 0.0
        
        train_acc.append(utils.accuracy(model, trainloader, device))
        test_acc.append(utils.accuracy(model, testloader, device))
        
        if (dro_flag):
            group1_acc, group2_acc, group3_acc, group4_acc = utils.group_accuracy(model, trainloader, device)
            group_train_acc[0].append(group1_acc)
            group_train_acc[1].append(group2_acc)
            group_train_acc[2].append(group3_acc)
            group_train_acc[3].append(group4_acc)
            
            group1_acc, group2_acc, group3_acc, group4_acc = utils.group_accuracy(model, testloader, device)
            group_test_acc[0].append(group1_acc)
            group_test_acc[1].append(group2_acc)
            group_test_acc[2].append(group3_acc)
            group_test_acc[3].append(group4_acc)
        
        model.train()
    
    print('Finished Training')
    if dro_flag:
        return np.array(train_acc), np.array(test_acc), group_train_acc, group_test_acc
    else:
        return np.array(train_acc), np.array(test_acc)



def dro_train(model, criterion, optimizer, trainloader,
                   testloader, num_epochs, device, dro_flag=True):
    
    """
    Train model using DRO
    (https://arxiv.org/pdf/1911.08731.pdf) - pg. 9

    Parameters
    ----------
    model : torch.nn.Module
    criterion : torch.nn.[Loss]
        loss function definition
    optimizer : torch.optim.[Optimizer]
    trainloader : torch.data.utils.DataLoader
    testloader : torch.data.utils.DataLoader
    num_epochs : int
    dro_flag : bool
        if True, keep track of accuracy on 
        different subsets of data

    Returns
    -------
    train_acc : np.ndarray
    test_acc : np.ndarray
        both hold accuracy with each epoch
        
    group_train_acc : list, np.ndarray
    group_test_acc : list, np.ndarray
        list of numpy arrays that hold
        accuracy on each group of data
        (e.g. blond hair, male)

    """
    
    train_acc = []
    test_acc = []
    group_train_acc = [[],[],[],[]]
    group_test_acc = [[],[],[],[]]
    
    group1_data, group2_data, \
    group3_data, group4_data = utils.split_group_data(trainloader)
    
    group_data = [group1_data, group2_data, group3_data, group4_data]
    group_order = np.array([0,1,2,3])
    q = 0.25*torch.ones(4)   # weighted dist. over groups - high mass on high-loss groups
    eta_q = 1e-3             # learning rate for q
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        # randomly choose order of group training
        np.random.shuffle(group_order)
        
        for i in group_order:
            np.random.shuffle(group_data[i])
            for j, data in enumerate(group_data[i], 0):
                # data = [inputs, target_labels, spur_labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward propagation
                outputs = model(torch.unsqueeze(inputs,0))
                loss = criterion(outputs, labels.float().unsqueeze(0).unsqueeze(0))
                
                # update weights for group g
                q[i] = q[i]*torch.exp(eta_q*loss)
                
                # renormalize q
                sum_ = torch.sum(q)
                for k in range(len(q)):
                    q[k] /= sum_
                
                # backpropagation
                loss.backward()
                
                # multiply learning rate by weight q and update model params
                # then restore orig. learning rate
                for g in optimizer.param_groups:
                    g['lr'] *= q[i]
                optimizer.step()
                for g in optimizer.param_groups:
                    g['lr'] /= q[i]
                
                # print statistics
                running_loss += loss.item()
                if j % 100 == 99:    # print every 100 mini-batches
                    print('[Epoch %d/%d, Group %d, mini-batch %5d/%d] loss: %.4f' %
                          (epoch + 1, num_epochs, i, j + 1, len(group_data[i]), running_loss / 100))
                    running_loss = 0.0
                
        
        train_acc.append(utils.accuracy(model, trainloader, device))
        test_acc.append(utils.accuracy(model, testloader, device))
        
        if (dro_flag):
            group1_acc, group2_acc, group3_acc, group4_acc = utils.group_accuracy(model, trainloader, device)
            group_train_acc[0].append(group1_acc)
            group_train_acc[1].append(group2_acc)
            group_train_acc[2].append(group3_acc)
            group_train_acc[3].append(group4_acc)
            
            group1_acc, group2_acc, group3_acc, group4_acc = utils.group_accuracy(model, testloader, device)
            group_test_acc[0].append(group1_acc)
            group_test_acc[1].append(group2_acc)
            group_test_acc[2].append(group3_acc)
            group_test_acc[3].append(group4_acc)
        
        model.train()
    
    print('Finished Training')
    if dro_flag:
        return np.array(train_acc), np.array(test_acc), group_train_acc, group_test_acc
    else:
        return np.array(train_acc), np.array(test_acc)