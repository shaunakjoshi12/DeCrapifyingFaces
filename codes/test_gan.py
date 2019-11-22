import os
import random
from glob import glob
from dataset import norm_
from skimage.io import imsave

import cv2
import matplotlib.pyplot as plt
import numpy as np

def denorm_(img):
  for i in range(3):
    img[:,:,i] = (255-0)*((img[:, :, i] - np.min(img[:, :, i]))/(np.max(img[:, :, i]) - np.min(img[:, :, i])))
  #img = 
  return img.astype(np.float32)

def fix(image: np.ndarray) -> np.ndarray:

    image = image[:,:,::-1]
    image = resize(image, (224, 224), preserve_range=True)

    image = norm_(image)

    image = image.transpose(2,0,1)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image_fixed = generator(image.cuda())
    image_out = denorm_(resize(image_fixed[0].cpu().detach().numpy().transpose(1,2,0), (250, 250),preserve_range=True).astype(np.float32)).astype(np.uint8)


    return image_out[:,:,::-1]

NUM_DISPLAY = 5

files = glob('../datasets/celeb_dataset/correct/test/*/*')
grid = []

for path in random.sample(files, NUM_DISPLAY):
  correct = cv2.imread(path)

  split = path.split('/')
  degraded = cv2.imread('/'.join([*split[:3], 'degraded', *split[4:]]))

#  fixed = fix(degraded)

  #grid.append(np.column_stack([degraded, fixed, correct]))
  img = np.column_stack([degraded, correct])
  imsave('../Plots/'+path.split('/')[-2]+'_'+path.split('/')[-1], img[:, :, ::-1])
