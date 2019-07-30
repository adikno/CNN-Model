import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from gcommand_loader import GCommandLoader

# Hyperparameters
num_epochs = 15
batch_size = 100
learning_rate = 0.001

class FirstNet(nn.Module):

    def __init__(self):
        super(FirstNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(10, 15, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout2d()
        self.fc1 = nn.Linear(3600, 1000)
        nn.BatchNorm2d(1000)
        self.fc2 = nn.Linear(1000, 500)
        nn.BatchNorm2d(500)
        self.fc3 = nn.Linear(500, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.log_softmax(out,dim=1)
        return out


def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_validation(model, val_loader)


def test_validation(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        all = 0
        for data, labels in test_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all += labels.size(0)
            correct += (predicted == labels).sum().item()

def test(model, test_loader):
    model.eval()
    f = open("test_y", "w")
    for i, data, labels in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        output = model(data)
        # get the index of the max log-probability
        tmp, pred = output.data.max(1, keepdim=True)
        f.write(str(test_loader.dataset.spects[i][0].split("/")[4]) + ", " + str(pred[0].item()) + "\n")

def loadData(dir, batch_size, shuffle):
    dataset = GCommandLoader(dir)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=20, pin_memory=True, sampler=None)

    return loader

if __name__ == '__main__':

    train_loader = loadData('./data/train', batch_size, True)
    val_loader = loadData('./data/valid', batch_size, True)
    test_loader = loadData('./data/test', 1, False)

    if torch.cuda.is_available():
        model = FirstNet().cuda()
    else:
        model = FirstNet()

    train(model, train_loader)

    test(model, test_loader)
