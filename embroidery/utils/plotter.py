import os
from pathlib import Path
import matplotlib.pyplot as plt


def savefig_tight(path: Path, **kwargs):
	# create the folder if it doesn't exist
	path = Path(path)
	if not path.parent.exists():
		os.makedirs(os.path.dirname(path))

	plt.savefig(path, bbox_inches="tight", pad_inches=0, **kwargs)


class Plotter():
	save_folder : str = None
	show_figure : bool = False

	def __init__(self, save_folder=None, show_figure=False):
		self.show_figure = show_figure
		self.save_folder = save_folder

		assert not (save_folder is not None and show_figure), "save_folder and show_figure cannot both be True"
		
		if save_folder is not None:
			if not os.path.exists(save_folder):
				os.makedirs(save_folder)
	

	def show_or_save(self, name):
		if self.show_figure:
			plt.show()

		if self.save_folder is not None:
			savefig_tight(f"{self.save_folder}/{name}.svg")
			savefig_tight(f"{self.save_folder}/{name}.png")
		
