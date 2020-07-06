# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import CifarNet as cfn

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--show_data', action='store_true')
parser.add_argument('--train_dev', action='store_true') #★4

# global data
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DATA_PATH = './data'
MODEL_PATH = './cifar_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#★1 Hyper Parameter
batch_size = 32
epochs = 40
dev_epoch = 1
learning_rate = 0.0001
momentum = 0.99

def main():
    args = parser.parse_args()
    if args.test:
        test_dataloader = cifar10_dataloader(train=False,
        batch_size=16, shuffle=True, num_workers=2)
        model = cfn._cifarnet(pretrained=args.test, path=MODEL_PATH).to(device)
        test(test_dataloader, model, args.show_data)
        return

    if args.show_data:
        dataloader = cifar10_dataloader(train=False, batch_size=batch_size, shuffle=True, num_workers=0)
        show_data(dataloader)
        return

    train_dataloader = cifar10_dataloader(train=True, batch_size=batch_size, shuffle=True, num_workers=2)
    #model = cfn._cifarnet(pretrained=True, path=MODEL_PATH).to(device) # ■ Resume Train
    model = cfn._cifarnet().to(device) # ■ new Train
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    #Train & Dev ★4
    if args.train_dev :
        dev_dataloader = cifar10_dataloader(train=False, batch_size=16, shuffle=True, num_workers=2)
        for epoch in range(epochs):
            train_dev(train_dataloader,dev_dataloader, model, criterion, optimizer, epoch, dev_epoch)
            torch.save(model.state_dict(), MODEL_PATH)
        return

    #Train
    for epoch in range(epochs):
        train(train_dataloader, model, criterion, optimizer, epoch)
        torch.save(model.state_dict(), MODEL_PATH)

def cifar10_dataloader(root=DATA_PATH, train=True, transform=None, shuffle=False, download=True, batch_size=batch_size, num_workers=0):
	if transform is None:
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

	dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return dataloader 

def train(dataloader, model, criterion, optimizer, epoch):
    running_loss = 0.0 #CPU
    printCount = int(10000/batch_size) # ★2 어떠한 batch_size에도 대략 10000번째 Image마다 running_loss를 출력
    for i, data in enumerate(dataloader):
        images, labels = data[0].to(device), data[1].to(device) #CPU -> GPU
        logit = model(images) #GPU
        loss = criterion(logit, labels) #GPU
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() #GPU -> CPU
        if i%printCount == printCount-1: # ★2 Print Count로 인해 1epoch당 5번 출력
            if(i == 5*printCount-1) : print('%.4f'%(running_loss/float(printCount))) # 기록을 위해 1epoch당 5번 출력 중 마지막만 출력
            running_loss = 0.0

# ★4
def train_dev(train_dataloader, dev_dataloader, model, criterion, optimizer, epoch, dev_epoch):
    running_loss = 0.0 #CPU
    model.train()
    correct = 0
    total = 0
    printCount = int(10000/batch_size) #어떠한 batch_size에도 대략 10000번째 Image마다 running_loss를 출력
    for i, data in enumerate(train_dataloader):
        images, labels = data[0].to(device), data[1].to(device)     #CPU -> GPU
        logit = model(images)                                       #GPU Forward
        loss = criterion(logit, labels)                             #GPU Loss

        _, predicted = torch.max(logit.data, 1)                     # Predict
        total += labels.size(0)                                     # Total +
        correct += (predicted == labels).sum().item()               # Correct +

        optimizer.zero_grad()                                       # Gradient Initialize
        loss.backward()                                             # Backward
        optimizer.step()                                            # Update & Optimize
        
        running_loss += loss.item()                                 #GPU -> CPU
        if i%printCount == printCount-1:                            #Print Count로 인해 1epoch당 5번 출력
            if(i == 5*printCount-1) : print('%.4f'%(running_loss/float(printCount))) # 기록을 위해 1epoch당 5번 출력 중 마지막만 출력
            running_loss = 0.0
    #print('Test : %.2f %%' % (100 * correct / total))
    print('%.2f %%' % (100 * correct / total))

    if(epoch%dev_epoch == dev_epoch-1):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dev_dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        #print('dev : %.2f %%' % (100 * correct / total))
        print('%.2f %%' % (100 * correct / total))


def test(dataloader, model, show_data):   
    if show_data:
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        # show images
        imshow(torchvision.utils.make_grid(images))  
        output = model(images.to(device))
        _, predicted = torch.max(output, 1)
        print('GT', ' '.join('%6s' % classes[labels[j]] for j in range(16)))
        print('PT', ' '.join('%6s' % classes[predicted[j]] for j in range(16)))
        print()
    
    model.eval() #★3
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(16):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print()
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def show_data(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%10s' % classes[labels[j]] for j in range(batch_size)))

def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np    

    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    main()
