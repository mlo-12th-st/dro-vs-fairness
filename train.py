"""
train.py

All training loop code
    Standard SGD
    Distributionally-robust training
    Fairness training
    DRO+fairness training
"""

def standard_train(model, criterion, optimizer, trainloader,
                   num_epochs):
    
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
    
    print('Finished Training')
    