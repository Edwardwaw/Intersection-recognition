import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model import EfficientNet
from Autoaug import ImageNetPolicy

import os
import argparse
import numpy as np
from datetime import datetime

# __INPUT__dataset path
DATA_TRAIN_PATH = './train_dataset'
DATA_TEST_PATH = './train_dataset'

# __TRAIN PARAMETERS__
EPOCHS = 200  # number of epochs
PRERPOCH = 0  # pretraining epoch number
BATCHSIZE = 6  # size of each image batch
WORKERS = 4  # number of workers for dataloader
INIT_LR = 0.005  # initial learning rate
SHUFFLE = True  # whether shuffle the dataset
VAL_SIZE = 0.9  # validation size of training dataset
WARM = 1

# mean and std of the dataset
TRAIN_MEAN = (0.422, 0.421, 0.409)
TRAIN_STD = (0.264, 0.261, 0.277)
####### data below is imagenet statics
# TRAIN_MEAN = (0.485, 0.456, 0.4063)
# TRAIN_STD = (0.229, 0.224, 0.22)

# __OUTPUT__directory to save weights file
CHECKPOINT_PATH = './checkpoint'
LR_MILESTONES = [13, 25, 35, 45]

# time of we run the script
TIME_NOW = datetime.now().isoformat()

# tensorboard log dir
LOG_DIR = 'runs'


device = torch.device('cuda:0')


def train(epoch, trainloader, net, loss_function, optimizer):
    net.train()
    corr1 = 0
    corr5 = 0
    total_num = 0
    for batch_index, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        total_num += images.shape[0]
        outputs = net(images)
        loss = loss_function(outputs, labels)

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        corr1 += acc1
        corr5 += acc5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            print('iter = ', batch_index, 'loss = ', loss.item(), 'acc1 = ', acc1 / images.shape[0], 'acc5 = ',
                  acc5 / images.shape[0])
    print('train epoch', epoch, 'grade is', corr1 / total_num, corr5 / total_num)


def eval_training(epoch, testloader, net, loss_function):
    net.eval()
    corr1 = 0
    corr5 = 0
    total_num = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            total_num += images.shape[0]
            outputs = net(images)
            loss = loss_function(outputs, labels)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            corr1 += acc1
            corr5 += acc5
        print(' val grade *Acc@1 {:.3f} Acc@5 {:.3f}'.format(corr1 / total_num, corr5 / total_num))

    return corr1 / total_num


def get_train_test_dataloader(mean, std, batch_size=1, num_workers=1, valsize=0.1):
    transform_train = transforms.Compose([
        transforms.Resize(480),  # Resize the smaller edge of PIL image to 600px
        # transforms.RandomCrop(32, padding=4),   #Crop the given PIL Image at a random location
        # transforms.RandomHorizontalFlip(),      #Horizontally flip the given PIL Image randomly with a probability of 0.5
        # transforms.RandomRotation(15),          #Rotate the image by angle.
        # transforms.ColorJitter(brightness=0.5, contrast=0.35, saturation=0.2), #Randomly change the brightness, contrast and saturation of an image.
        ImageNetPolicy(),
        # transforms.RandomCrop((480, 600)),
        transforms.CenterCrop((480,600)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop((480, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train = torchvision.datasets.ImageFolder(root=DATA_TRAIN_PATH, transform=transform_train)
    test = torchvision.datasets.ImageFolder(root=DATA_TRAIN_PATH, transform=transform_test)

    print(train.class_to_idx)

    # length of the whole training dataset
    dataset_len = len(train)
    # list of the whole training dataset
    dataset_list = list(range(dataset_len))

    # split the training deataset to train set and test set with ratio validsize
    split = int(np.floor(valsize * dataset_len))

    # random the dataset
    np.random.shuffle(dataset_list)

    # get the train and test set index
    train_index, test_index = dataset_list[:split], dataset_list[split:]
    # print(len(train_index))
    # print(len(test_index))
    # sampler the training set and test set
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)

    trainloader = DataLoader(
        train, sampler=train_sampler, num_workers=num_workers, batch_size=batch_size)
    testloader = DataLoader(
        test, sampler=test_sampler, num_workers=num_workers, batch_size=batch_size)

    return trainloader, testloader



def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdims=True).item()
            res.append(correct_k)
        return res


if __name__ == "__main__":

    # arguments setting
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="number of epochs")
    parser.add_argument("--pre_epoch", type=int, default=PRERPOCH, help='pretraining epoch number')
    parser.add_argument("--batch_size", type=int, default=BATCHSIZE, help="size of each image batch")
    parser.add_argument('--workers', type=int, default=WORKERS, help='number of workers for dataloader')
    parser.add_argument("--lr", type=float, default=INIT_LR, help='initial learning rate')
    parser.add_argument('--shuffle', type=bool, default=SHUFFLE, help='whether shuffle the dataset')
    parser.add_argument('--valsize', type=float, default=VAL_SIZE,
                        help='ratio of the size of test(val) set and train set')
    parser.add_argument('-warm', type=int, default=WARM, help='warm up training phase')
    args = parser.parse_args()

    # set the net model

    net = EfficientNet.from_pretrained('efficientnet-b0',num_classes=39)


    net = net.to(device)
    print("Start Training now")
    trainloader, testloader = get_train_test_dataloader(TRAIN_MEAN, TRAIN_STD, batch_size=args.batch_size,
                                                        num_workers=args.workers, valsize=args.valsize)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES,
                                                     gamma=0.2)  # learning rate decay

    checkpoint_path = os.path.join(CHECKPOINT_PATH, "EfficientNet", TIME_NOW)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0

    for epoch in range(args.epochs):
        train(epoch, trainloader, net, loss_function, optimizer)
        if epoch > args.warm:
            train_scheduler.step()

        acc = eval_training(epoch, testloader, net, loss_function)
        # start to save best performance model after learning rate decay to 0.01
        if best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net='EfficientNet', epoch=epoch, type='best'))
        torch.save(net.state_dict(), checkpoint_path.format(net='EfficientNet', epoch=epoch, type='regular'))
        best_acc = max(best_acc, acc)


