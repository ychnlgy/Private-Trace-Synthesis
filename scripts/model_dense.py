import torch

from model_simple import Lambda


class Generator(torch.nn.Module):

    def __init__(self, noise_size, hidden_size, max_traj_len):
        super().__init__()
        self.net = torch.nn.Sequential(
            # 32 -> 64
            torch.nn.Linear(noise_size, 64),
            torch.nn.LeakyReLU(),

            # 64 -> 128
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),

            # 128 -> max_traj_len
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, max_traj_len),
            torch.nn.LeakyReLU(),

            # (1, max_traj_len) -> (3, max_traj_len)
            Lambda(lambda X: X.unsqueeze(1)),

            torch.nn.BatchNorm1d(1),
            torch.nn.Conv1d(1, 3, 7, padding=3),
            torch.nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(torch.nn.Module):

    def __init__(self, array_length, hidden_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            # (3, array_length) -> (1, array_length//2)
            torch.nn.utils.spectral_norm(
                torch.nn.Conv1d(3, 1, 7, padding=3, stride=2)
            ),
            Lambda(lambda X: X.squeeze(1)),

            # array_length//2 -> 64
            torch.nn.LeakyReLU(),
            torch.nn.utils.spectral_norm(
                torch.nn.Linear(array_length//2, 64)
            ),

            # 64 -> 32
            torch.nn.LeakyReLU(),
            torch.nn.utils.spectral_norm(
                torch.nn.Linear(64, 32)
            ),

            # 32 -> 1
            torch.nn.LeakyReLU(),
            torch.nn.utils.spectral_norm(
                torch.nn.Linear(32, 1)
            )
        )

    def forward(self, X):
        return self.net(X)