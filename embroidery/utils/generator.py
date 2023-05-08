import itertools

def pairwise(seq, cyclic=False):
	"""
	generator
	"""
	a, b = itertools.tee(seq if not cyclic else itertools.chain(seq, [seq[0]]))
	next(b, None)
	return zip(a, b)
