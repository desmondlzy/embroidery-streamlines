import numpy as np

from embroidery.utils.generator import pairwise


def subdivide(points, num=None, spacing=10, closed=False):
	point_num = len(points)

	# input is a segment
	if point_num == 2 and not closed:
		if num is None:
			num = int(np.linalg.norm(points[1] - points[0]) / spacing) + 1
			num = max(num, 2)
		return np.linspace(points[0], points[1], num=num, endpoint=True)
	
	# segmentate the path into segments and recurse
	else:
		adjacent_indices = np.vstack([np.array((i, i + 1), dtype=int) for i in np.arange(point_num - int(not closed))])
		adjacent_indices[-1:] %= point_num

		return np.vstack([subdivide(points[pair], num=num, spacing=spacing, closed=False)[:-1] for pair in adjacent_indices] + ([points[-1]] if not closed else []))

def subsample(polyline, spacing):
	segments = []
	for p, q in pairwise(polyline):
		num = max(
			np.round(np.linalg.norm(q - p) // spacing).astype(int) + 1,
			2)
		segments.append(np.linspace(p, q, num, endpoint=True)[:-1])
	
	segments.append(polyline[-1])
	
	return np.row_stack(segments)
	

def zigzag(first_row, second_row, returns_indices=False):
	assert len(first_row) == len(second_row)

	c = len(first_row)

	if c % 2 == 0:
		indices = np.concatenate([np.array([0, c, c + 1, 1]) + offset for offset in np.arange(0, c, 2)])
	else:
		indices = np.concatenate(
			[np.array([0, c, c + 1, 1]) + offset for offset in np.arange(0, c - 1, 2)] + \
			[np.array([c - 1, 2 * c - 1])])
	
	if returns_indices:
		return indices
	else:
		return np.vstack((first_row, second_row))[indices]

def chessboard(lx, lr, rx, ry, num):
	assert num % 2 == 1, f"{num = } has to be an odd number"
	left_bottom = [lx, lr]
	left_top = [lx, ry]
	right_top = [rx, ry]
	right_bottom = [rx, lr]
	pattern = subdivide(np.row_stack((
		zigzag(np.linspace(left_bottom, right_bottom, num=num), np.linspace(left_top, right_top, num=num))[:-1],
		zigzag(np.linspace(right_top, right_bottom, num=num), np.linspace(left_top, left_bottom, num=11)),
	)), spacing=15)

	return pattern
		
def double(points):
	return np.vstack((points, np.flip(points, axis=0)))


def triple(points):
	return np.vstack((points, np.flip(points, axis=0), points))


def triple_bounce(points):
	return np.vstack((
		points[0],
		*[np.array([q, p, q]) for p, q in pairwise(points)],
	))


def segment_length(points):
	return np.array([np.linalg.norm(a - b) for a, b in pairwise(points)])
