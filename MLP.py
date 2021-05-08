import torch
import torchvision
from torch.utils import data
from torchvision.transforms import ToTensor
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.L_set = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.L_set(x)
        return x


def train_loop(dataloader, model, loss, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        X = X.to("cuda")
        y = y.to("cuda")
        y_hat = model(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # if batch % 100 == 0:
        #     print(f"loss:{l:f}")


def test_loop(dataloader, model):
    accuracy = 0.0
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to("cuda")
            y = y.to("cuda")
            y_hat = model(X)
            accuracy += (y_hat.argmax(1) == y).type(torch.int64).sum().item()
    accuracy /= size
    return accuracy


mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=ToTensor(), download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=ToTensor(), download=True)
batch_size = 64
learning_rate = 0.2
epochs = 50
train_iter = data.DataLoader(mnist_train, batch_size, True)
test_iter = data.DataLoader(mnist_test, batch_size, False)
mlp = MLP().to("cuda")
trainer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()
for epoch in range(epochs):
    train_loop(train_iter, mlp, loss, trainer)
    accuracy = test_loop(test_iter, mlp)
    print(f"accuracy:{(100 * accuracy):>0.1f}%")
torch.save(mlp.state_dict(), 'mlp_weights.pth')
