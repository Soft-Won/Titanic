# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import TitanDataset
import TitanModel



parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--train_dev', action='store_true')
parser.add_argument('--test', action='store_true')

# global data
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# DATA_PATH = './data'
MODEL_PATH = './titan_net.pth'
SUBMISSION_PATH = './submission.csv'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # CPU

#★1 Hyper Parameter
batch_size = 4
epochs = 10
dev_epoch = 1
learning_rate = 1e-4
momentum = 0.99

def main():
    args = parser.parse_args()
    
    train_dataset = TitanDataset.TitanDataset(purpose='train')
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)
    
    model = TitanModel._titanmodel().to(device) # ■1 new Train
    model = TitanModel._titanmodel(pretrained=True, path=MODEL_PATH).to(device) # ■2 Resume Train
    # criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.BCELoss().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # ■ Optimizer 1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # ■ Optimizer 2


    # Train
    if args.train :
        for epoch in range(epochs):
            train(train_dataloader, model, criterion, optimizer, epoch)
            torch.save(model.state_dict(), MODEL_PATH)

    # Train & Valid
    if args.train_dev :
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [791,100])
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
        for epoch in range(epochs):
            train_dev(train_dataloader,val_dataloader, model, criterion, optimizer, epoch, dev_epoch)
            torch.save(model.state_dict(), MODEL_PATH)

    # Test
    if args.test :
        model = TitanModel._titanmodel(pretrained=True, path=MODEL_PATH).to(device) # ■2 Resume Train
        test_dataset = TitanDataset.TitanDataset(purpose='test')
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
        test(test_dataloader,model)


def train(dataloader, model, criterion, optimizer, epoch):
    correct = 0; total = 0; running_loss = 0.0          #CPU
    printCount = int(100/batch_size)                    # ★2 어떠한 batch_size에도 대략 100번째 X마다 running_loss를 출력

    for i, data in enumerate(dataloader):
        X, Y = data[0].to(device), data[1].unsqueeze(1).to(device)   #CPU -> GPU
        
        logit = model(X)                                #GPU
        #loss = F.mse_loss(logit, Y.unsqueeze(1))       #GPU
        loss = F.mse_loss(logit, Y.float())             #GPU

        predicted = torch.round(logit)                  # Predict
        total += Y.size(0)                              # Total +
        correct += (predicted == Y).sum().item()        # Correct +

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item() #GPU -> CPU
        if i%printCount == printCount-1: #
            if(i == 5*printCount-1) : print('%.4f'%(running_loss/float(printCount)))
            running_loss = 0.0
    print('%.2f %%' % (100 * correct / total))

def train_dev(train_dataloader, dev_dataloader, model, criterion, optimizer, epoch, dev_epoch):
    model.train()
    correct = 0; total = 0; running_loss = 0.0          #CPU
    printCount = int(100/batch_size)                    # ★2 어떠한 batch_size에도 대략 100번째 X마다 running_loss를 출력

    # Training 1 Epoch
    for i, data in enumerate(train_dataloader):
        X, Y = data[0].to(device), data[1].unsqueeze(1).to(device)   #CPU -> GPU
        
        logit = model(X)                                #GPU
        #loss = F.mse_loss(logit, Y.unsqueeze(1))       #GPU
        loss = F.mse_loss(logit, Y.float())             #GPU

        predicted = torch.round(logit)                  # Predict
        total += Y.size(0)                              # Total +
        correct += (predicted == Y).sum().item()        # Correct +

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item() #GPU -> CPU
        if i%printCount == printCount-1: #
            if(i == 5*printCount-1) : print('%.4f'%(running_loss/float(printCount)))
            running_loss = 0.0
    print('Training : %.2f %%' % (100 * correct / total))

    # Validation 1 Epoch
    if(epoch%dev_epoch == dev_epoch-1):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dev_dataloader:
                X, Y = data[0].to(device), data[1].unsqueeze(1).to(device)   #CPU -> GPU
                logit = model(X)                                #GPU
                
                predicted = torch.round(logit)                  # Predict
                total += Y.size(0)                              # Total +
                correct += (predicted == Y).sum().item()        # Correct +
        print('Validation : %.2f %%' % (100 * correct / total))

def test(dataloader, model):
    import sys
    sys.stdout = open(SUBMISSION_PATH,'w')

    PassengerId = 891
    print('PassengerId,Survived')
    for i, data in enumerate(dataloader):
        X, _ = data[0].to(device), data[1].unsqueeze(1).to(device)   #CPU -> GPU
        
        logit = model(X)                                #GPU
        predicted = torch.round(logit)                  # Predict
        result = np.squeeze(predicted.detach().numpy().astype(np.int).reshape((1,-1)))
        
        for i in result:
            PassengerId = PassengerId+1
            print('%d,%d'%(PassengerId,i))

        
if __name__ == '__main__':
    main()
