import torch
import torch.nn.utils


IDENTITY = torch.nn.Sequential()


class ResBlock(torch.nn.Module):

    def __init__(self, module, shortcut=IDENTITY, activation=IDENTITY):
        super().__init__()
        self.net = module
        self.sht = shortcut
        self.act = activation

    def forward(self, X):
        return self.act(self.net(X) + self.sht(X))


class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Goal is to convert (N, 1, 128) noise into (N, 3, 200) trajectories.
        self.net = torch.nn.Sequential(
            # Expected input noise is length 128

        )

    def forward(self, z):
        return self.net(z)


class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Goal is to convert (N, 3, 200) trajectories into (N, 1) discriminations.
        self.net = torch.nn.Sequential(

        )

    def forward(self, X):
        return self.net(X)

    def loss(self, X, Xh):
        return self.forward(X).mean(axis=0) - self.forward(Xh).mean(axis=0)
