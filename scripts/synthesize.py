import tqdm

import torch
import torch.utils.data

import model_simple as model

from train import MAX_TRAJ_LENGTH, iter_valid_trajectories, MID_X, MAX_X, MID_Y, MAX_Y


def traj_to_string(traj):
    points = []
    for x, y in traj:
        p = "%.1f,%.1f" % (x, y)
        points.append(p)
    line = ";".join(points)
    return ">0:%s;\n" % line


def main(save_dat, noise_size, hidden_size, model_path, dataset_size=20000, batch_size=250):

    device = ["cpu", "cuda"][torch.cuda.is_available()]

    G = model.Generator(noise_size, hidden_size, MAX_TRAJ_LENGTH).to(device)
    G.eval()

    Z = torch.randn(dataset_size, noise_size)

    with open(save_dat, "w") as f:
        with torch.no_grad():
            with tqdm.tqdm(total=dataset_size//batch_size, ncols=80) as bar:
                i = 0
                while i < dataset_size:
                    j = i + batch_size

                    z = Z[i:j].to(device)
                    Xh = G(z)

                    for k, traj in enumerate(iter_valid_trajectories(Xh)):
                        assert traj.shape[1] == 2
                        traj[:, 0] = traj[:, 0] * MAX_X + MID_X
                        traj[:, 1] = traj[:, 1] * MAX_Y + MID_Y

                        f.write("#%d:\n" % (i + k))
                        f.write(traj_to_string(traj))
                bar.update(1)

        i = j


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dat", required=True)
    parser.add_argument("--noise_size", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    main(**vars(args))