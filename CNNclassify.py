"""
CNNclassify.py

CNN CIFAR-10 classifier

Created by Satish Kumar Anbalagan on 3/1/20.
Copyright © 2020 Satish Kumar Anbalagan. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

import os
from os import path

import argparse

from PIL import Image   # Loading the test image


DATA_DIR = './data'
MODEL_DIR = './model'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = 100  # end at epoch 2

# Data
#print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.convlayer = None
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=5, out_channels=out_channels, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        # output = self.conv(input)
        # output = self.bn(output)
        self.convlayer = self.conv(input)
        output = self.bn(self.convlayer)
        output = self.relu(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output, self.unit1.convlayer


net = SimpleNet(num_classes=10)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


# Training
def training():
    """
    Function to train the model.
    :rtype: training loss, training accuracy

    """
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, layer1 = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return (train_loss/(batch_idx+1)), (correct/total)


# Validate
def validate():
    """
    Function to validate the model.
    :rtype: test loss, test accuracy, first CONV layer

    """
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, layer1 = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return (test_loss/(batch_idx+1)), (correct/total), layer1


# Train
def train():
    """
    Function to train the model.
    :rtype: None

    """

    #print('==> Building model..')
    for epoch in range(start_epoch, start_epoch + end_epoch):
        train_loss, train_acc = training()
        test_loss, test_acc, layer1 = validate()

        if epoch % 10 == 0:
            print('epoch:', epoch + 1, '/', start_epoch + end_epoch , ', training loss:', train_loss, ', training accuracy:',
              100 * train_acc, '%, testing loss: ', test_loss, ', testing accuracy:', 100 * test_acc, '%')

    torch.save(net, MODEL_DIR + '/model.pt')
    print("\nModel successfully saved at {}".format(os.getcwd()+MODEL_DIR))


# Visualization
def visualize_layer(layer, n_filters= 32):
    """
    Function to visualize layers of the input image.
    :param: input image path layers
    :param: number of filters
    :rtype: None

    """

    fig = plt.figure(figsize=(32, 32))
    for i in range(n_filters):

        axis = fig.add_subplot(4, n_filters/4, i+1)
        # grab layer outputs
        axis.imshow(np.squeeze(layer[0,i].cpu().data.numpy()), cmap='gray')
        axis.set_title('Image%s' % str(i+1))

    fig.show()
    fig.savefig('./result/CONV_rslt.png')


# Test
def test(inputFilePath):
    """
    Function to test the input image.
    :param: input image path
    :rtype: None

    """
    #print('==> Testing model..')
    # Loading the model
    generatedModel = torch.load(MODEL_DIR + '/model.pt')

    inputImage = Image.open(inputFilePath).convert('RGB')

    inputImage = inputImage.resize((32, 32))

    with torch.no_grad():
        # Image Transformation
        imageTensor = transform_test(inputImage)
        singleBatchImage = imageTensor.unsqueeze(0)  # shape = [1, 3, 32, 32]

        outputs, layer1 = generatedModel(singleBatchImage)
        _, predicted = torch.max(outputs.data, 1)
        predictedClass = classes[predicted[0].item()]
        print("Prediction result : {}".format(predictedClass))

        # visualize the output of the convolutional layer
        visualize_layer(layer1)


# Main
def main():
    """
    main function
    :return: None
    """
    parser = argparse.ArgumentParser(description='Convolutional Neural network classifier')
    parser.add_argument('function', nargs='*', type=str,
                    help='train/test xxx.png, enter it to train/test the Neural network classifier model accordingly')

    args = parser.parse_args()
    if len(args.function) >= 1:
        if args.function[0] == 'train':
            train()
        elif args.function[0] == 'test':
            inputFile = args.function[1]
            if path.exists(inputFile):
                test(inputFile)
            else:
                print('\nInvalid input. Path does not exists')
        else:
            print('\nPlease enter a valid command')
            parser.print_help()
    else:
        print('\nPlease enter a valid command')
        parser.print_help()

    exit()


if __name__ == '__main__':
    main()

