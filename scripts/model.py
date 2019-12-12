import torch
import torch.nn.utils


# GAN architecture based off of BigGAN:
# https://arxiv.org/pdf/1809.11096.pdf


IDENTITY = torch.nn.Sequential()


class ResBlock(torch.nn.Module):

    def __init__(self, module, shortcut=IDENTITY):
        super().__init__()
        self.net = module
        self.sht = shortcut

    def forward(self, X):
        return self.net(X) + self.sht(X)


class GeneratorResBlock(torch.nn.Module):

    def __init__(self, noise_net1, noise_net2, module1, module2, shortcut):
        super().__init__()
        self.noz1 = noise_net1
        self.noz2 = noise_net2
        self.mod1 = module1
        self.mod2 = module2
        self.shtc = shortcut

    def forward(self, X, z):
        Xh = self._expand_vec_and_add(self.noz1, self.mod1, z, X)
        Xh = self._expand_vec_and_add(self.noz2, self.mod2, z, Xh)
        return Xh + self.shtc(X)

    def _expand_vec_and_add(self, net, mod, v, X):
        S = len(X.shape)
        vh = net(v)
        E = [1] * (S - len(vh.shape))
        Vh = vh.view(*vh.shape, *E)
        return mod(X + Vh)


class Generator(torch.nn.Module):

    def __init__(self, noise_size, hidden_size, output_array_length):
        super().__init__()

        assert noise_size % 4 == 0
        self.noise_part = noise_size // 4

        assert output_array_length % 8 == 0
        self.output_array_length = output_array_length // 8

        assert hidden_size % 4 == 0
        self.h = hidden_size

        self.lin = torch.nn.Linear(self.noise_part, 3 * self.output_array_length)

        # (N, 1, 3, L) -> (N, self.h, 3, 2L)
        self.bk1 = GeneratorResBlock(
            noise_net1=torch.nn.Linear(self.noise_part, self.h),
            noise_net2=torch.nn.Linear(self.noise_part, self.h),
            shortcut=torch.nn.ConvTranspose2d(1, self.h, (1, 2), stride=(1, 2)),
            module1=torch.nn.Sequential(
                torch.nn.BatchNorm2d(self.h),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(self.h, self.h, (3, 4), stride=(1, 2), padding=1)
            ),
            module2=torch.nn.Sequential(
                torch.nn.BatchNorm2d(self.h),
                torch.nn.ReLU(),
                torch.nn.Conv2d(self.h, self.h, 3, padding=1)
            )
        )

        # (N, self.h, 3, 2L) -> (N, self.h//2, 3, 4L)
        self.bk2 = GeneratorResBlock(
            noise_net1=torch.nn.Linear(self.noise_part, self.h),
            noise_net2=torch.nn.Linear(self.noise_part, self.h),
            shortcut=torch.nn.ConvTranspose2d(self.h, self.h//2, (1, 2), stride=(1, 2)),
            module1=torch.nn.Sequential(
                torch.nn.BatchNorm2d(self.h),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(self.h, self.h, (3, 4), stride=(1, 2), padding=1)
            ),
            module2=torch.nn.Sequential(
                torch.nn.BatchNorm2d(self.h),
                torch.nn.ReLU(),
                torch.nn.Conv2d(self.h, self.h//2, 3, padding=1)
            )
        )

        # (N, self.h//2, 3, 4L) -> (N, self.h//4, 3, 8L)
        self.bk3 = GeneratorResBlock(
            noise_net1=torch.nn.Linear(self.noise_part, self.h//2),
            noise_net2=torch.nn.Linear(self.noise_part, self.h//2),
            shortcut=torch.nn.ConvTranspose2d(self.h//2, self.h//4, (1, 2), stride=(1, 2)),
            module1=torch.nn.Sequential(
                torch.nn.BatchNorm2d(self.h//2),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(self.h//2, self.h//2, (3, 4), stride=(1, 2), padding=1)
            ),
            module2=torch.nn.Sequential(
                torch.nn.BatchNorm2d(self.h//2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(self.h//2, self.h//4, 3, padding=1)
            )
        )

        # (N, self.h//4, 3, 8L) -> (N, 1, 3, 8L)
        self.tail = torch.nn.Sequential(
            torch.nn.Conv2d(self.h//4, 1, 3, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, z):
        N = z.size(0)
        Z = z.view(N, 4, self.noise_part)
        z1 = Z[:, 0]
        z2 = Z[:, 1]
        z3 = Z[:, 2]
        z4 = Z[:, 3]

        X = self.lin(z1).view(N, 1, 3, self.output_array_length)
        X = self.bk1(X, z2)
        X = self.bk2(X, z3)
        X = self.bk3(X, z4)
        return self.tail(X)


class Discriminator(torch.nn.Module):

    def __init__(self, array_length, hidden_size):
        super().__init__()

        assert hidden_size % 4 == 0
        self.h = hidden_size

        # Goal is to convert (N, 1, 3, array_length) trajectories into (N, 1) discriminations.
        self.net = torch.nn.Sequential(

            # (N, 1, 3, L) -> (N, self.h//4, 3, L//2)
            ResBlock(
                module=torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(1, self.h//4, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.h//4, self.h//4, 3, padding=1),
                    torch.nn.AvgPool2d((1, 3), stride=(1, 2), padding=(0, 1))
                ),
                shortcut=torch.nn.Sequential(
                    torch.nn.Conv2d(1, self.h//4, 1),
                    torch.nn.AvgPool2d((1, 3), stride=(1, 2), padding=(0, 1))
                )
            ),

            # (N, self.h//4, 3, L//2) -> (N, self.h//2, 3, L//4)
            ResBlock(
                module=torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.h//4, self.h//2, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.h//2, self.h//2, 3, padding=1),
                    torch.nn.AvgPool2d((1, 3), stride=(1, 2), padding=(0, 1))
                ),
                shortcut=torch.nn.Sequential(
                    torch.nn.Conv2d(self.h//4, self.h//2, 1),
                    torch.nn.AvgPool2d((1, 3), stride=(1, 2), padding=(0, 1))
                )
            ),

            # (N, self.h//2, 3, L//4) -> (N, self.h, 3, L//8)
            ResBlock(
                module=torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.h//2, self.h, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.h, self.h, 3, padding=1),
                    torch.nn.AvgPool2d((1, 3), stride=(1, 2), padding=(0, 1))
                ),
                shortcut=torch.nn.Sequential(
                    torch.nn.Conv2d(self.h//2, self.h, 1),
                    torch.nn.AvgPool2d((1, 3), stride=(1, 2), padding=(0, 1))
                )
            ),

            torch.nn.Conv2d(self.h, self.h, 3),
            # (N, self.h, 1, ?) -> (N, self.h, 1, 1)
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        self.tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.h, 1)
        )

    def forward(self, X):
        N = X.size(0)
        return self.tail(self.net(X).view(N, self.h))

    def loss(self, X, Xh):
        return self(X).mean(axis=0) - self(Xh).mean(axis=0)