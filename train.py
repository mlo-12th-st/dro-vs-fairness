"""
train.py

All training loop code
    Standard SGD
    Distributionally-robust training
    Fairness training
    DRO+fairness training
"""

import utils
import numpy as np

def standard_train(model, criterion, optimizer, trainloader,
                   testloader, num_epochs):
    
    """
    Train model using ERM

    Parameters
    ----------
    model : torch.nn.Module
    criterion : torch.nn.[Loss]
        loss function definition
    optimizer : torch.optim.[Optimizer]
    trainloader : torch.utils.data.DataLoader
    testloader : torch.utils.data.DataLoader
    num_epochs : int

    Returns
    -------
    train_acc : np.ndarray
    test_acc : np.ndarray
        both hold accuracy with each epoch

    """
    
    train_acc = []
    test_acc = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, spur_labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[Epoch %d/%d, mini-batch %5d/%d] loss: %.3f' %
                      (epoch + 1, num_epochs, i + 1, len(trainloader), running_loss / 200))
                running_loss = 0.0
        
        train_acc.append(utils.accuracy(model, trainloader))
        test_acc.append(utils.accuracy(model, testloader))
        model.train()
    
    print('Finished Training')
    return np.array(train_acc), np.array(test_acc)
    