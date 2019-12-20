import numpy
import numpy as np

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
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--degree', '-d', type=int, default=5)
	parser.add_argument('--save', '-s', type=str, default='degree_')

	args = parser.parse_args()
	tally = []
	lines = []
	for traj in iter_trajectories("brinkhoff.dat"):
		poly_fit = np.polyfit(traj[:,0], traj[:,1], args.degree)
		lines.append(poly_fit)
	lines = np.asarray(lines)
	print(args.degree, lines.shape)
	np.save( args.save+str(args.degree)+'.npy', lines)
