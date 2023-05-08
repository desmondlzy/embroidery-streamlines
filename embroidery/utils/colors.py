import numpy as np
from sklearn.decomposition import PCA

def rgb2cmy(img): 
	rgb = img.reshape(-1, 3)
	cmy = np.column_stack((
		1 - rgb[:, 0],
		1 - rgb[:, 1],
		1 - rgb[:, 2],
	))
	return cmy.reshape(*img.shape)

def cmy2rgb(img): 
	cmy = img.reshape(-1, 3)
	rgb = np.column_stack((
		1 - cmy[:, 0],
		1 - cmy[:, 1],
		1 - cmy[:, 2],
	))
	return rgb.reshape(*img.shape)


def pca_flatten(data_points):
	assert data_points.shape[1] == 3
	center = data_points.mean(axis=0)
	centered_data = data_points - center

	pca = PCA(n_components=1)
	line_values = pca.fit_transform(centered_data)

	return pca.inverse_transform(line_values) + center
	