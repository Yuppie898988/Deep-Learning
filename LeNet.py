import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.full_connect = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.full_connect(x)
        return x


def train_loop(data_iter, model, loss, trainer, batch_size):
    size = len(data_iter.dataset)
    for batch, (X, y) in enumerate(data_iter):
        X = X.to("cuda")
        y = y.to("cuda")
        y_hat = model(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        # if batch % 100 == 0:
        #     print(f"loss:{l:>.2f}, [{batch * batch_size} / {size}]")


def test_loop(data_iter, model):
    accuracy = 0.0
    size = len(data_iter.dataset)
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_iter):
            X = X.to("cuda")
            y = y.to("cuda")
            y_hat = model(X)
            l = loss(y_hat, y)
            accuracy += (y_hat.argmax(1) == y).type(y.dtype).sum().item()
    accuracy /= size
    return accuracy


trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
learning_rate = 0.01
epochs = 10
batch_size = 4
net = LeNet().to("cuda")
dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
train_iter = data.DataLoader(dataset_train, batch_size, True, num_workers=0)
test_iter = data.DataLoader(dataset_test, batch_size, False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()
for epoch in range(epochs):
    train_loop(train_iter, net, loss, trainer, batch_size)
    accuracy = test_loop(test_iter, net)
    print(f"第{epoch + 1}次训练，accuracy：{accuracy*100:>.2f}%")

