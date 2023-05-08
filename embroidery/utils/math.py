import numpy as np
import scipy
from scipy import optimize
from numpy.polynomial.legendre import leggauss
from numba import vectorize, float64, njit

from .generator import pairwise
from . import rng

def apply_rowwise(func, matrix):
	return np.apply_along_axis(func, axis=1, arr=matrix)

def apply_colwise(func, matrix):
	return np.apply_along_axis(func, axis=0, arr=matrix)


def rotation_matrix_2d(deg):
	rad = np.deg2rad(deg)
	return np.array([
		[np.cos(rad), -np.sin(rad)],
		[np.sin(rad), np.cos(rad)],
	])


def rowwise_rotate(matrix, deg, around=None):
	around = np.array([0, 0]) if around is None else around

	rot = rotation_matrix_2d(deg)
	centered = matrix - around
	rotated = apply_rowwise(lambda r: rot @ r, centered)

	return rotated + around

def map_range(val, from_left, from_right, to_left, to_right):
	return (val - from_left) / (from_right - from_left) * (to_right - to_left) + to_left

def gradient_fd(xk, f):
	idim = xk.size
	gradient = optimize.approx_fprime(xk, f, epsilon=1e-9)
	return gradient


def jacobian_fd(xk, f, dim=None):
	"""
	finite difference divergence for a vector function f
	"""
	if dim is None:
		dim = f(xk).size
	
	return np.column_stack([
		scipy.optimize.approx_fprime(xk, lambda x: f(x)[i], epsilon=1e-11)
		for i in range(dim)
	])


def divergence_fd(xk, f, dim=None):
	return np.trace(jacobian_fd(xk, f, dim))


def circumcenter(a, b, c):
	ax, ay = a
	bx, by = b
	cx, cy = c
	d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
	ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
	uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
	return np.array([ux, uy])


def circumradius(a, b, c):
	return np.linalg.norm(circumcenter(a, b, c) - a)


def rotate90(vector):
	if vector.ndim == 1:
		return np.array([vector[1], -vector[0]])

	elif vector.ndim == 2:
		return np.column_stack([vector[:, 1], -vector[:, 0]])

	

def gauss_legendre_integration_on_grid(f, xa, xb, ya, yb, deg):
	p, weights = leggauss(deg)
	p = (p + 1) / 2  # normalized to [0, 1]
	weights /= 2 # normalized to 1

	x_samples = p * (xb - xa) + xa
	y_samples = p * (yb - ya) + ya
	samples_with_weights = np.vstack([[x, y, xw * yw] 
		for x, xw in zip(x_samples, weights) 
		for y, yw in zip(y_samples, weights)])
	
	values = [f(s[:2]) * s[2] for s in samples_with_weights]

	# only for grid
	return sum(values) * (xb - xa) * (yb - ya)


def normalized(vec):
	return vec / np.linalg.norm(vec)


def winding_number(boundary_points, query):
	ps = boundary_points
	qs = np.roll(boundary_points, -1, axis=0)
	number = signed_angle(ps, qs, query[np.newaxis, :]).sum()
		
	return number


@njit
def inside_boundary(boundary_points, queries, strict=True):
	windings = np.zeros((len(queries), ))
	len_boundary = len(boundary_points)

	for i, query in enumerate(queries):
		for pi, p in enumerate(boundary_points):
			q = boundary_points[(pi + 1) % len_boundary]
			windings[i] += signed_angle(p, q, query);
		
		# assert not strict or np.isclose([1, 0, -1], windings[i]).any()

	return np.where(np.isclose(windings, 0), False, True)


def periodic_wrapping(func, start, end):
	"""
	return a periodic function defined on R, with repeated value from [start, end]
	"""
	assert end > start
	window_size = end - start
	def wrapped(xk):
		wrapped_xk = np.remainder(xk - start, window_size)  # this and next line: mod into [0, end - start]
		wrapped_xk += start  # back to [start, end]
		return func(wrapped_xk)

	return wrapped


def find_root(func, low, high):
	spaces, step = np.linspace(low, high, num=500, retstep=True)
	roots = set()

	for a, b in pairwise(spaces, cyclic=True):
		sign_a, sign_b = np.sign(func(a)), np.sign(func(b))
		if sign_a != sign_b:
			roots.add(a + step * np.abs(a) / (np.abs(a) + np.abs(b)))

	res = np.fmod(np.fromiter(roots, dtype=float), high - low) + low
	return np.sort(res)


def segment_lengths(polyline):
	lengths = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
	return lengths


def total_length(polyline):
	l = np.sum(segment_lengths(polyline))
	return l


def arclengths(polyline):
	lengths = segment_lengths(polyline)
	arclen = np.concatenate([[0], np.cumsum(lengths)])
	return arclen


def arclength_parametrization(polyline):
	arclen = arclengths(polyline)
	func = scipy.interpolate.interp1d(arclen, polyline, axis=0)
	return func


def periodic_arclength_parametrization(points):
	func = arclength_parametrization(np.row_stack((points, points[0])))
	length = total_length(np.row_stack((points, points[0])))

	def _periodic_func(ts):
		rem_ts = np.remainder(ts, length)
		return func(rem_ts)

	return _periodic_func, length

def consecutive(data, stepsize=1):
	# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def curve_normal(curve_f, ts):
	eps = 1e-6
	p_0 = curve_f(ts - eps)
	p_2 = curve_f(ts + eps)

	tg = (p_2 - p_0) / (2 * eps)

	if tg.ndim == 1:
		return rotate90(tg)
	else:
		return rotate90(tg) 


def curve_unit_normal(curve_f, ts):
	eps = 1e-6
	p_0 = curve_f(ts - eps)
	p_2 = curve_f(ts + eps)

	tg = (p_2 - p_0) / (2 * eps)

	if tg.ndim == 1:
		return rotate90(tg) / np.linalg.norm(tg)
	else:
		return rotate90(tg) / np.linalg.norm(tg, axis=1)[:, np.newaxis]

@vectorize([float64(float64, float64, float64, float64, float64, float64)])
def signed_angle_ufunc(px, py, qx, qy, ox, oy):
	ax = ox - px
	ay = oy - py
	bx = ox - qx
	by = oy - qy

	norm_a = np.sqrt(ax * ax)
	norm_b = np.sqrt(bx * bx)

	if norm_a != 0:
		ax /= norm_a
		ay /= norm_a
	if norm_b != 0:
		bx /= norm_b
		by /= norm_b

	return -np.arctan2(bx * ay - by * ax, bx * ax + by * ay) / (2 * np.pi)

@njit
def signed_angle(ps, qs, os):
	assert ps.ndim == 2 and qs.ndim == 2, os.ndim == 2
	res = signed_angle_ufunc(ps[:, 0], ps[:, 1], qs[:, 0], qs[:, 1], os[:, 0], os[:, 1])
	return res


def is_counterclockwise(curve):
	# query has to be inside of the polygon
	# use a random method to get away with it
	ps = curve
	qs = np.roll(curve, -1, axis=0)

	bbox_min = curve.min(axis=0)
	bbox_max = curve.max(axis=0)

	pts = np.row_stack((
		curve.mean(axis=0),
		rng.random((1000, 2)) * (bbox_max - bbox_min) + bbox_min,
	))
	for i, query in enumerate(pts):
		sum_angle = signed_angle(ps, qs, query[np.newaxis, :]).sum()
		if np.isclose(sum_angle, 0, atol=1e-4):
			continue

		print(f"terminate at {i = }")
		if sum_angle >= 0:
			return False
		else: 
			return True
	
	print(f"didn't find the point inside")
	return True



def ensure_counterclockwise(curve):
	"""
	returns a copy of curve so that it goes counterclockwise.
	the curve needs to be closed
	"""
	if is_counterclockwise(curve):
		return curve
	else:
		return np.flipud(curve)
	


def nearest_point_on_curve(pt, curve):
	ind = np.argmin(np.linalg.norm(curve - pt, axis=1))
	return curve[ind], ind


def perturb_curve_along_normal(boundary, perturb_dist=0.005):
	boundary_interpolant, boundary_length = periodic_arclength_parametrization(boundary)
	_ts = np.linspace(0, boundary_length, 1000, endpoint=False)
	resampled_boundary = boundary_interpolant(_ts)
	resampled_curve_normal = curve_normal(boundary_interpolant, _ts)
	inflated_boundary = resampled_boundary - resampled_curve_normal * perturb_dist
	return inflated_boundary