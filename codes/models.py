##################################################################Generator Model##################################################################################
import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

def conv3x3(in_, out):
	return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
	def __init__(self, in_, out):
		super(ConvRelu, self).__init__()
		self.conv = conv3x3(in_, out)
		self.activation = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.activation(x)
		return x


class DecoderBlock(nn.Module):
	
	def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
		super(DecoderBlock, self).__init__()
		self.in_channels = in_channels

		if is_deconv:
			self.block = nn.Sequential(
				ConvRelu(in_channels, middle_channels),
				nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
								   padding=1),
				nn.ReLU(inplace=True)
			)
		else:
			self.block = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear'),
				ConvRelu(in_channels, middle_channels),
				ConvRelu(middle_channels, out_channels),
			)

	def forward(self, x):
		return self.block(x)

class UNet11(nn.Module):
	def __init__(self, num_filters=32, pretrained=False):
		super(UNet11,self).__init__()
		self.pool = nn.MaxPool2d(2, 2)
		

		if pretrained == 'vgg':
			self.encoder = models.vgg11(pretrained=True).features
		else:
			self.encoder = models.vgg11(pretrained=False).features

		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Sequential(self.encoder[0],
								   self.relu)

		self.conv2 = nn.Sequential(self.encoder[3],
								   self.relu)

		self.conv3 = nn.Sequential(
			self.encoder[6],
			nn.BatchNorm2d(self.encoder[6].out_channels),
			self.relu,
			self.encoder[8],
			nn.BatchNorm2d(self.encoder[8].out_channels),
			self.relu,
		)
		self.conv4 = nn.Sequential(
			self.encoder[11],
			nn.BatchNorm2d(self.encoder[11].out_channels),
			self.relu,
			self.encoder[13],
			nn.BatchNorm2d(self.encoder[13].out_channels),
			self.relu,
		)

		self.conv5 = nn.Sequential(
			self.encoder[16],
			nn.BatchNorm2d(self.encoder[16].out_channels),
			self.relu,
			self.encoder[18],
			nn.BatchNorm2d(self.encoder[18].out_channels),
			self.relu,
		)

		self.center = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
		self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
		self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=True)
		self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
		self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
		self.dec1 = ConvRelu(64 + num_filters, num_filters)
		self.final = nn.Conv2d(num_filters, 3, kernel_size=1)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(self.pool(conv1))
		conv3 = self.conv3(self.pool(conv2))
		conv4 = self.conv4(self.pool(conv3))
		conv5 = self.conv5(self.pool(conv4))
		center = self.center(self.pool(conv5))

		dec5 = self.dec5(torch.cat([center, conv5], 1))
		dec4 = self.dec4(torch.cat([dec5, conv4], 1))
		dec3 = self.dec3(torch.cat([dec4, conv3], 1))
		dec2 = self.dec2(torch.cat([dec3, conv2], 1))
		dec1 = self.dec1(torch.cat([dec2, conv1], 1))
		#import pdb;pdb.set_trace()
		final = torch.tanh(self.final(dec1))
		return final

############################################################Discriminator model finetune#############################################################



def finetune_resnet(resnet, num_classes):
  resnet.fc = nn.Linear(in_features = resnet.fc.in_features, out_features = num_classes)
  ct = 0
  for child in resnet.children():
    ct+=1
    if ct < 7:
      for param in child.parameters():
        param.requires_grad=False
    
  return resnet  
