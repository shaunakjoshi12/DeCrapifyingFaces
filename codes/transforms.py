import torchvision.transforms as transform
from skimage.transform import rescale, rotate
import numpy as np

def transforms(scale=None, flip_prob=0.5, angle=30):
	transforms_list = []
	if np.random.random() < 0.7:
		transforms_list.append(Scale(scale))
	if np.random.random() < 0.6:
		transforms_list.append(Flip(flip_prob))

	return transform.Compose(transforms_list)

class Scale(object):
	def __init__(self, scale):
		self.scale = scale

	def __call__(self, sample):
		correct_img, degrad_img = sample
		self.scale = np.random.uniform(low = 0.8*self.scale, high = 1.2*self.scale)
		correct_img_size, degrad_img_size = correct_img.shape[0], degrad_img.shape[0]
		correct_img = rescale(correct_img, (self.scale, self.scale), preserve_range=True , multichannel=True, mode='constant')
		degrad_img = rescale(degrad_img, (self.scale, self.scale), preserve_range=True , multichannel=True, mode='constant')

		if self.scale < 1.0:
			diff = (correct_img_size - correct_img.shape[0]) / 2.0
			padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0,0),)

			correct_img = np.pad(correct_img, padding, mode='constant', constant_values = 0)      
			degrad_img = np.pad(degrad_img, padding, mode='constant', constant_values = 0)

		else:
			xmin = (correct_img.shape[0] - correct_img_size) // 2
			xmax = xmin + correct_img_size
			correct_img = correct_img[xmin:xmax, xmin:xmax, :]      
			degrad_img = degrad_img[xmin:xmax, xmin:xmax, :] 

		return  correct_img, degrad_img

class Flip(object):
	def __init__(self, flip_prob):
		self.flip_prob = flip_prob

	def __call__(self, sample):
		correct_img, degrad_img = sample
		if np.random.random() < self.flip_prob:
			return correct_img, degrad_img 
		
		correct_img = np.fliplr(correct_img).copy()
		degrad_img = np.fliplr(degrad_img).copy()

		return  correct_img, degrad_img 



