import random

import tqdm

from matplotlib import pyplot


def plot_trajectories(trajectories, save_path, prob=0.05, use_tqdm=True):
    """Randomly plots <prob> trajectories.

    The full set of trajectories is messy and takes too long.
    """
    trajectories = list(trajectories)
    if use_tqdm:
        trajectories = tqdm.tqdm(trajectories, ncols=80)
    for traj in trajectories:
        if random.random() < prob:
            pyplot.plot(traj[:, 0], traj[:, 1], ":")
    pyplot.savefig(save_path, bbox_inches="tight")
    pyplot.clf()


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