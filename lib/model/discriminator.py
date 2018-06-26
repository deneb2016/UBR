import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * 2)
        self.fc2 = nn.Linear(in_dim * 2, in_dim * 2)
        self.fc3 = nn.Linear(in_dim * 2, 1)
        self.relu = nn.LeakyReLU()

        nn.init.xavier_normal(self.fc1.weight.data)
        self.fc1.bias.data.zero_()
        nn.init.xavier_normal(self.fc2.weight.data)
        self.fc2.bias.data.zero_()
        nn.init.xavier_normal(self.fc3.weight.data)
        self.fc3.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BoxDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super(BoxDiscriminator, self).__init__()
        self.fc1 = nn.Linear(in_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)
        self.relu = nn.LeakyReLU()

        nn.init.xavier_normal(self.fc1.weight.data)
        self.fc1.bias.data.zero_()
        nn.init.xavier_normal(self.fc2.weight.data)
        self.fc2.bias.data.zero_()
        nn.init.xavier_normal(self.fc3.weight.data)
        self.fc3.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
