import torch


class Lambda(torch.nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, X):
        return self.f(X)


class ResBlock(torch.nn.Module):

    def __init__(self, shortcut, act, layers):
        super().__init__()
        self.sht = shortcut
        self.act = act
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.act(self.sht(X) + self.net(X))


class Generator(torch.nn.Module):

    def __init__(self, noise_size, hidden_size, max_traj_len):
        super().__init__()
        assert max_traj_len % 4 == 0
        self.net = torch.nn.Sequential(
            torch.nn.Linear(noise_size, hidden_size * max_traj_len//4),

            Lambda(lambda X: X.view(-1, hidden_size, max_traj_len//4)),

            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(hidden_size, hidden_size, 8, stride=2, padding=3),

            ResBlock(
                shortcut=torch.nn.Conv1d(hidden_size, hidden_size, 1),
                act=torch.nn.ReLU(),
                layers=[
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden_size, hidden_size, 7, padding=3),

                    torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden_size, hidden_size, 7, padding=3)
                ]
            ),

            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(hidden_size, hidden_size, 8, stride=2, padding=3),

            ResBlock(
                shortcut=torch.nn.Conv1d(hidden_size, hidden_size, 1),
                act=torch.nn.ReLU(),
                layers=[
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden_size, hidden_size, 11, padding=5),

                    torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden_size, hidden_size, 11, padding=5)
                ]
            ),

            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_size, 3, 7, padding=3),

            torch.nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(torch.nn.Module):

    def __init__(self, array_length, hidden_size):
        super().__init__()
        assert array_length % 8 == 0
        self.net = torch.nn.Sequential(
            Lambda(lambda X: X.squeeze(1)),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv1d(3, hidden_size, 7, stride=2, padding=3)
            ),

            ResBlock(
                shortcut=torch.nn.utils.spectral_norm(
                    torch.nn.Conv1d(hidden_size, hidden_size, 1)
                ),
                act=torch.nn.LeakyReLU(),
                layers=[
                    torch.nn.LeakyReLU(),
                    torch.nn.utils.spectral_norm(
                        torch.nn.Conv1d(hidden_size, hidden_size, 11, padding=5)
                    ),

                    torch.nn.LeakyReLU(),
                    torch.nn.utils.spectral_norm(
                        torch.nn.Conv1d(hidden_size, hidden_size, 11, padding=5)
                    ),
                ]
            ),

            torch.nn.LeakyReLU(),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv1d(hidden_size, hidden_size, 7, stride=2, padding=3)
            ),

            ResBlock(
                shortcut=torch.nn.utils.spectral_norm(
                    torch.nn.Conv1d(hidden_size, hidden_size, 1)
                ),
                act=torch.nn.LeakyReLU(),
                layers=[
                    torch.nn.LeakyReLU(),
                    torch.nn.utils.spectral_norm(
                        torch.nn.Conv1d(hidden_size, hidden_size, 7, padding=3)
                    ),

                    torch.nn.LeakyReLU(),
                    torch.nn.utils.spectral_norm(
                        torch.nn.Conv1d(hidden_size, hidden_size, 7, padding=3)
                    )
                ]
            ),

            torch.nn.LeakyReLU(),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv1d(hidden_size, hidden_size, 7, stride=2, padding=3)
            ),

            Lambda(lambda X: X.view(-1, hidden_size * array_length//8)),
            torch.nn.LeakyReLU(),
            torch.nn.utils.spectral_norm(
                torch.nn.Linear(hidden_size * array_length//8, 1)
            )
        )

    def forward(self, X):
        return self.net(X)

    @staticmethod
    def loss(D, X, Xh):
        return D(X).mean(axis=0) - D(Xh).mean(axis=0)