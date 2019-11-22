from torch.utils.data import Dataset, DataLoader
from skimage import img_as_float32
from skimage.io import imread, imsave
from skimage.transform import resize



max_ = 1
min_ = -1

def norm_(img):
	img = img_as_float32(img)


	for i in range(3):
		img[:,:,i] = (max_- min_)*((img[:, :, i] - np.min(img[:, :, i]))/(np.max(img[:, :, i]) - np.min(img[:, :, i]))) + min_

	return img.astype(np.float32)



class PersonDataset(Dataset):
	def __init__(self, base_path, mode ,transforms=None):
		self.base_path = base_path
		self.transforms = transforms
		self.mode = mode
		self.correct_images = sorted(glob.glob(os.path.join(self.base_path,'correct/'+self.mode+'/*/*')))
		self.degraded_images = sorted(glob.glob(os.path.join(self.base_path,'degraded/'+self.mode+'/*/*')))
		self.classes = {'degraded':0, 'correct':1}

	def __getitem__(self, index):
		correct_img, degraded_img = imread(self.correct_images[index], plugin='imageio'), imread(self.degraded_images[index], plugin='imageio') 
		if self.transforms is not None:
			scale = np.random.randint(1, 3)
			transform = self.transforms(scale=scale)
			correct_img, degraded_img = transform((correct_img, degraded_img))
			correct_img, degraded_img = resize(correct_img, (224, 224), preserve_range=True), resize(degraded_img, (224, 224), preserve_range=True)
		
			correct_img, degraded_img = norm_(correct_img), norm_(degraded_img)
			correct_img, degraded_img = correct_img.transpose(2,0,1), degraded_img.transpose(2,0,1)
			return correct_img, degraded_img, self.classes['correct'], self.classes['degraded']

		correct_img, degraded_img = resize(correct_img, (224, 224), preserve_range=True), resize(degraded_img, (224, 224), preserve_range=True)
		
		correct_img, degraded_img = norm_(correct_img), norm_(degraded_img)
		correct_img, degraded_img = correct_img.transpose(2,0,1), degraded_img.transpose(2,0,1)
		
		return correct_img, degraded_img, self.classes['correct'], self.classes['degraded']

	def __len__(self):
		return len(self.correct_images)

if __name__=='__main__':
	persontraindataset = PersonDataset('./rephrase-pubfig831', mode='train', transforms=transforms)
	persontraindataloader = DataLoader(persontraindataset, batch_size=batch_size, shuffle=True)
	for i,(correct_img, degraded_img, _, _) in tqdm(enumerate(persontraindataloader)):
		correct_img = correct_img.cpu().numpy()
		degraded_img = degraded_img.cpu().numpy()
		print('correct dtype: ',correct_img.dtype)
		print('degraded dtype: ',degraded_img.dtype)
		print('correct max: ',np.max(correct_img))
		print('correct min: ',np.min(correct_img))
		print('correct mean: ',np.mean(correct_img))
		print('degraded max: ',np.max(degraded_img))
		print('degraded min: ',np.min(degraded_img))
		print('degraded mean: ',np.mean(degraded_img))
		break  


