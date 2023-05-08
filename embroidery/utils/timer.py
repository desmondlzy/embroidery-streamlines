from time import perf_counter
class catchtime:
	def __init__(self, msg) -> None:
		self.msg = msg

	def __enter__(self):
		self.time = perf_counter()
		return self

	def __exit__(self, type, value, traceback):
		self.time = perf_counter() - self.time
		self.readout = f'[[[{self.msg}]]]: {self.time:.4f} seconds'
		print(self.readout)