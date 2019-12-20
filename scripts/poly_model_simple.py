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

    def __init__(self, noise_size, hidden_size, out_dim=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(noise_size, hidden_size),
			torch.nn.BatchNorm1d(hidden_size),
			torch.nn.ReLU(),

			torch.nn.Linear(hidden_size, 2 * hidden_size), 
			torch.nn.BatchNorm1d(2 * hidden_size),
			torch.nn.ReLU(),
			
			torch.nn.Linear(2 * hidden_size, 4 * hidden_size), 
			torch.nn.BatchNorm1d(4 * hidden_size),
			torch.nn.ReLU(),

			torch.nn.Linear(4 * hidden_size, out_dim), 
			torch.nn.BatchNorm1d(out_dim),
			torch.nn.ReLU(),

            torch.nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(torch.nn.Module):

    def __init__(self, array_length, hidden_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
				torch.nn.Linear(array_length, hidden_size)
            ),


			torch.nn.utils.spectral_norm(
				torch.nn.Linear(hidden_size, 2 * hidden_size)
			),
			torch.nn.LeakyReLU(),

			torch.nn.utils.spectral_norm(
				torch.nn.Linear(2* hidden_size, 4 * hidden_size)
			),
			torch.nn.LeakyReLU(),

			torch.nn.utils.spectral_norm(
				torch.nn.Linear(4 * hidden_size, hidden_size)
			),
			torch.nn.LeakyReLU(),


			torch.nn.utils.spectral_norm(
				torch.nn.Linear(hidden_size, 1)
			),
		)
        

    def forward(self, X):
        return self.net(X)

    @staticmethod
    def loss(D, X, Xh):
        return D(X).mean(axis=0) - D(Xh).mean(axis=0)
