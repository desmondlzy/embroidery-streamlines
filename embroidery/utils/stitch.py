import numpy as np
from pyembroidery import EmbPattern, EmbThread
from pyembroidery.EmbConstant import STITCH, SEQUENCE_BREAK, COLOR_BREAK



def stitch_over_path(pattern: EmbPattern, points, closed=False, sequence_breaks=True, color_breaks=False):
	rounded_points = np.around(points, decimals=1)
	for x, y in rounded_points:
		pattern.add_stitch_absolute(STITCH, x, y)

	if closed:
		pattern.add_stitch_absolute(STITCH, *rounded_points[0])
	
	if sequence_breaks:
		pattern.add_command(SEQUENCE_BREAK)
	
	if color_breaks:
		pattern.add_command(COLOR_BREAK)
	

def summarize_pattern(pattern: EmbPattern):
	stitches = pattern.stitches
	stitch_positions = np.array([(x, y) for x, y, cmd in stitches if cmd == STITCH])

	bounding_box = np.min(stitch_positions, axis=0), np.max(stitch_positions, axis=0)
	width, height = bounding_box[1] - bounding_box[0]

	print(f"width: {width * 0.1:.1f}mm, height: {height * 0.1:.1f}mm")


def add_threads(pattern: EmbPattern, hex_colors, names=None):
	if names is not None:
		assert len(names) == len(hex_colors)
	for i, hc in enumerate(hex_colors):
		assert hc[0] == "#" 
		assert len(hc) == 7, f"{hc = }"
		thread_attr = {
			"hex": hc,
			"color": int(hc[1:], base=16),
		}
		if names is not None:
			thread_attr["name"] = names[i]

		pattern.add_thread(thread_attr)

