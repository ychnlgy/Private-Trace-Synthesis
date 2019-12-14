import random

import tqdm

from matplotlib import pyplot


def plot_trajectories(trajectories, save_path, prob=0.05):
    """Randomly plots <prob> trajectories.

    The full set of trajectories is messy and takes too long.
    """
    trajectories = list(trajectories)
    for traj in tqdm.tqdm(trajectories, ncols=80):
        if random.random() < prob:
            print(traj.shape)
            pyplot.plot(traj[:, 0], traj[:, 1], ":")
    pyplot.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--save", default="brinkhoff.png")

    args = parser.parse_args()

    from brinkhoff_parser import iter_trajectories
    plot_trajectories(
        iter_trajectories(args.file),
        args.save
    )