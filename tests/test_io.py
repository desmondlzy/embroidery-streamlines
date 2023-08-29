import unittest
import os

from embroidery.utils import io

class testIo(unittest.TestCase):
	def test_convert_exp_as_bundle(self):
		src = "tests/data/flower.exp"
		targets = ["tests/data/flower.png", "tests/data/flower.csv"]
		
		for target in targets:
			if os.path.exists(target):
				os.remove(target)

		io.convert_exp_as_bundle(src)

		for target in targets:
			self.assertTrue(os.path.exists(target))
