import unittest

from embroidery.utils import generator

class testGenerator(unittest.TestCase):
	def test_take_two(self):
		seq = range(4)
		gen_seq = tuple(generator.take_two(seq))
		answer = ((0, 1), (1, 2), (2, 3))
		self.assertEqual(gen_seq, answer)

		gen_seq = tuple(generator.take_two(seq, cyclic=True))
		answer = ((0, 1), (1, 2), (2, 3), (3, 0))