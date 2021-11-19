import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if (training == True):

        data_set = datasets.MNIST('.\data', train=True, download=True,
                                  transform=custom_transform)
    else:

        data_set = datasets.MNIST('.\data', train=False,
                                  transform=custom_transform)

    loader = torch.utils.data.DataLoader(data_set, batch_size=50)
    return loader


def build_model():
    train_set = get_data_loader()
    size = (train_set.dataset)
    # print(size)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10))

    return model


def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        prds = 0
        running_loss = 0
        accuracy = 0
        model.train()
        for i, labels in train_loader:
            opt.zero_grad()
            outputs = model(i)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            labels = labels.tolist()
            running_loss += loss.item()
            index = torch.argmax(outputs, dim=1)
            for j in range(len(index)):
                if index[j] == labels[j]:
                    prds += 1
            accuracy = (prds / 60000) * 100

        print("Train Epoch: " + str(epoch) + "   " + "Accuracy: " +
              str(prds) + "/" + str(60000) + "(" + str(round(accuracy, 2)) + "%)" + "    " + "Loss: " + str(
            running_loss / 60000))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    running_loss = 0.0
    accuracy = 0
    lenDataset = 0
    model.eval()
    with torch.no_grad():
        for i, labels in test_loader:
            outputs = model(i)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            labels = labels.tolist()
            index = torch.argmax(outputs, dim=1)
            for i in range(len(index)):
                lenDataset += 1
                if index[i] == labels[i]:
                    accuracy += 1

        print("Average Loss: " + str(round((running_loss / 60000), 4)) + "\n")
        print("Accuracy: " + str(round((accuracy / lenDataset) * 100, 2)))


def predict_label(model, test_images, index):
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for j, labels in test_images:
        outputs = model(j)
        prob_ = F.softmax(outputs, dim=1)
        image = list(prob_[index])
        prd = dict()
        for i in range(len(class_names)):
            # .
            prd[class_names[i]] = image[i]
        prd = sorted(prd.items(), key=lambda x: x[1], reverse=True)
        for i in range(3):
            (class_name, prdiction) = prd[i]
            print("{}: {:.2}f%".format(class_name, prdiction * 100))


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()




