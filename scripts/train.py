import os

import numpy
import torch
import torch.utils.data
import tqdm

from data import brinkhoff_parser, plot_trajectories
import model_simple as model
import model_tiny


MID_X = 12000
MAX_X = 15000

MID_Y = 15000
MAX_Y = 18000


MAX_TRAJ_LENGTH = 240
DATASET_SIZE = 20000


def create_dataset(
    fpath,
    batch_size,
    max_length
):
    tdata = create_tensorset(fpath, max_length)
    dload = torch.utils.data.DataLoader(
        tdata,
        batch_size=batch_size,
        shuffle=True
    )
    return dload


def create_tensorset(fpath, max_length):
    dataset = numpy.zeros(
        (DATASET_SIZE, 3, max_length),
        dtype=numpy.float32
    )

    for i, traj in enumerate(brinkhoff_parser.iter_trajectories(fpath)):
        n = len(traj)
        dataset[i, 0, :n] = (traj.T[0] - MID_X) / MAX_X
        dataset[i, 1, :n] = (traj.T[1] - MID_Y) / MAX_Y
        dataset[i, 2, :n] = +0.5
        dataset[i, 2, n:] = -0.5

    adata = torch.from_numpy(dataset).unsqueeze(1)
    assert (adata.abs() <= 1.0).all()
    tdata = torch.utils.data.TensorDataset(adata)
    return tdata


def iter_valid_trajectories(Xh):
    Xh = Xh.squeeze(1).cpu().numpy()
    for row in Xh:
        idx = numpy.argmax(row[2] < 0)
        out = row[:2, :idx].T
        yield out

def plot_and_save(Xh, save_path):
    Xh = iter_valid_trajectories(Xh)
    plot_trajectories.plot_trajectories(Xh, save_path, prob=1.0, use_tqdm=True)

def train(
    fpath, batch_size, noise_size,
    hidden_size, epochs, n_critic,
    D_lr, G_lr,
    epoch_sample_cycle,
    epoch_sample_count,
    save_path,
    debug,
    tiny
):
    global model

    if tiny:
        model = model_tiny


    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    modo_path = os.path.join(save_path, "E%05d.pkl")
    save_path = os.path.join(save_path, "E%05d.png")

    if debug:
        dset = create_dataset(fpath, epoch_sample_count, MAX_TRAJ_LENGTH)
        (X,) = next(iter(dset))
        plot_and_save(X, save_path % 0)
        return

    dset = create_dataset(fpath, batch_size, MAX_TRAJ_LENGTH)

    device = ["cpu", "cuda"][torch.cuda.is_available()]
    print("Using: %s" % device)

    G = model.Generator(noise_size, hidden_size, MAX_TRAJ_LENGTH).to(device)
    D = model.Discriminator(MAX_TRAJ_LENGTH, hidden_size).to(device)

    if device == "cuda":
        D = torch.nn.DataParallel(D)

    D_optim = torch.optim.Adam(D.parameters(), lr=D_lr, betas=(0, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=G_lr, betas=(0, 0.999))

    for epoch in range(0, epochs + 1):

        with tqdm.tqdm(dset, ncols=80) as bar:
            for i, (X,) in enumerate(bar, 1):

                X = X.to(device)
                z = torch.randn(batch_size, noise_size).to(device)

                if i % n_critic == 0:

                    G.train()

                    D.eval()
                    G_optim.zero_grad()
                    (-D(G(z))).mean().backward()
                    G_optim.step()

                else:
                    with torch.no_grad():
                        G.eval()
                        Xh = G(z)

                    D.train()
                    loss = D(Xh).mean() - D(X).mean()
                    D_optim.zero_grad()
                    loss.backward()
                    D_optim.step()

                    bar.set_description("[E%05d] %.4f" % (epoch, loss.item()))

            if epoch % epoch_sample_cycle == 0 or epoch == epochs:

                with torch.no_grad():
                    G.eval()
                    z = torch.randn(epoch_sample_count, noise_size).to(device)
                    Xh = G(z)
                    plot_and_save(Xh, save_path % epoch)

                torch.save(G.state_dict(), modo_path % epoch)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", required=True)
    parser.add_argument("--epochs", type=int, default=20000)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--noise_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_critic", type=int, default=3)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--G_lr", type=float, default=5e-5)
    parser.add_argument("--epoch_sample_cycle", type=int, default=20)
    parser.add_argument("--epoch_sample_count", type=int, default=100)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--tiny", type=int, required=True)

    args = parser.parse_args()

    train(**vars(args))