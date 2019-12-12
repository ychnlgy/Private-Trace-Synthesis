import os

import numpy
import torch
import torch.utils.data
import tqdm

from data import brinkhoff_parser, plot_trajectories
import model


MID_X = 12000
MAX_X = 15000

MID_Y = 15000
MAX_Y = 18000


MAX_TRAJ_LENGTH = 240


def create_dataset(
    fpath,
    batch_size,
    max_length,
    expected_length=20000,
):
    dataset = numpy.zeros(
        (expected_length, 3, max_length),
        dtype=numpy.float32
    )

    for i, traj in enumerate(brinkhoff_parser.iter_trajectories(fpath)):
        n = len(traj)
        dataset[i, 0, :n] = (traj.T[0] - MID_X) / MAX_X
        dataset[i, 1, :n] = (traj.T[1] - MID_Y) / MAX_Y
        dataset[i, 2, :n] = 1.0
        dataset[i, 2, n:] = -1.0

    adata = torch.from_numpy(dataset).unsqueeze(1)
    assert (adata.abs() <= 1.0).all()
    tdata = torch.utils.data.TensorDataset(adata)
    dload = torch.utils.data.DataLoader(
        tdata,
        batch_size=batch_size,
        shuffle=True
    )
    return dload


def plot_and_save(Xh, save_path):
    Xh = Xh.cpu().numpy()
    plot_trajectories.plot_trajectories(Xh, save_path, prob=1.0)

def train(
    fpath, batch_size, noise_size,
    hidden_size, epochs, n_critic,
    D_lr, G_lr,
    epoch_sample_cycle,
    epoch_sample_count,
    save_path
):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, "E%03d.png")

    dset = create_dataset(fpath, batch_size, MAX_TRAJ_LENGTH)

    device = ["cpu", "cuda"][torch.cuda.is_available()]
    print("Using: %s" % device)

    G = model.Generator(noise_size, hidden_size, MAX_TRAJ_LENGTH).to(device)
    D = model.Discriminator(MAX_TRAJ_LENGTH, hidden_size).to(device)

    D_optim = torch.optim.AdamW(D.parameters(), lr=D_lr, betas=(0, 0.999))
    G_optim = torch.optim.AdamW(G.parameters(), lr=G_lr, betas=(0, 0.999))

    for epoch in range(epochs):

        with tqdm.tqdm(enumerate(dset, 1), ncols=80) as bar:
            for i, (X,) in bar:

                z = torch.randn(batch_size, noise_size).to(device)

                if i % n_critic == 0 or i == epochs:

                    z = noiser(batch_size)

                    G.train()
                    Xh = G(z)

                    D.eval()
                    G_optim.zero_grad()
                    (-D.loss(X, Xh)).backward()
                    G_optim.step()

                with torch.no_grad():
                    G.eval()
                    Xh = G(z)

                D.train()
                loss = D.loss(X, Xh)
                D_optim.zero_grad()
                loss.backward()
                D_optim.step()

                bar.set_description("[E%03d] %.4f" % (epoch, loss.item()))

            if epoch % epoch_sample_cycle:

                with torch.no_grad():
                    z = torch.randn(epoch_sample_count, noise_size).to(device)
                    Xh = G(z)
                    plot_and_save(Xh, save_path % epoch)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", required=True)
    parser.add_argument("--epochs", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--noise_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_critic", type=int, default=4)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--G_lr", type=float, default=5e-5)
    parser.add_argument("--epoch_sample_cycle", type=int, default=5)
    parser.add_argument("--epoch_sample_count", type=int, default=400)
    parser.add_argument("--save_path", default="synthesis")

    args = parser.parse_args()

    train(**vars(args))