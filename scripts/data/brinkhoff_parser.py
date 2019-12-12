import numpy


def iter_trajectories(fpath):
    """Iterate array of shape (T, 2).

    T is the number of points in the trajectory.
    2 is the number of coordinate axes, x and y.

    To access the 4th (x, y) in the trajectory, do arr[3].
    To access the y coordinates of the trajectory, do arr[:, 1].
    """
    with open(fpath, "r") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if _is_trajectory_line(i):
                yield _check_trajectory(line)
            else:
                _check_id(line)

        assert _is_trajectory_line(i)

def _is_trajectory_line(i):
    return i % 2


def _check_id(line):
    assert line[0] == "#"
    assert line[-1] == ":"
    assert line[1:-1].isdigit()


def _check_trajectory(line):
    assert line[:3] == ">0:"
    lines = line[3:].rstrip(";")
    pieces = [p.split(",") for p in lines.split(";")]
    coords = [[float(x), float(y)] for x, y in pieces]
    return numpy.array(coords)


if __name__ == "__main__":
    tally = []
    for traj in iter_trajectories("brinkhoff.dat"):
        tally.append(len(traj))
    from matplotlib import pyplot

    pyplot.hist(tally, bins=100)
    pyplot.show()
