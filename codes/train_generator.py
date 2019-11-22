import torch
import torch.nn as nn
from models import UNet11
import torch.optim as optim
from dataset import PersonDataset
from torch.utils.data import DataLoader



lr = 0.001
batch_size = 30
num_epochs = 4

persontraindataset = PersonDataset('../datasets/celeb_dataset', mode='train', transforms=transforms)
persontraindataloader = DataLoader(persontraindataset, batch_size=batch_size, shuffle=True)

personvaldataset = PersonDataset('../datasets/celeb_dataset', mode='test', transforms=None)
personvaldataloader = DataLoader(personvaldataset, batch_size=batch_size, shuffle=True)

generator = UNet11(pretrained='vgg')

generator.cuda()

image_loss  = nn.MSELoss()

optim_generator = optim.SGD(generator.parameters(), lr=lr, momentum=0.9)
losses_generator_train = []
losses_generator_val = []

for epoch in range(num_epochs): 
  loss_batch_train = 0.0
  loss_batch_val = 0.0
  generator.train()
  print('\n Epoch:{}'.format(epoch+1))
  for i,(correct_img, degraded_img, _, _) in tqdm(enumerate(persontraindataloader)):
    correct_img = correct_img.cuda()
    degraded_img = degraded_img.cuda()
    
    corrected_fake = generator(degraded_img)
    

    

    optim_generator.zero_grad()
    loss = image_loss(corrected_fake.view(correct_img.size()[0],-1),correct_img.view(correct_img.size()[0],-1))

    if (i+1)%5==0:
      print('running train loss: ',loss.item())
    #print('corrected ',corrected_fake.size(),' correct ',correct_img.size())
    losses_generator_train.append(loss.item())
    loss_batch_train+=loss.item()
    loss.backward()

    optim_generator.step()
    

  loss_per_epoch_train = loss_batch_train/len(persontraindataloader)
  print('Loss:{:.2f}'.format(loss_per_epoch_train))

  generator.eval()
  for i,(correct_img, degraded_img, _, _) in tqdm(enumerate(personvaldataloader)):
    correct_img = correct_img.cuda()
    degraded_img = degraded_img.cuda()
    corrected_fake = generator(degraded_img.cuda())
    

    loss = image_loss(corrected_fake.view(correct_img.size()[0],-1),correct_img.view(correct_img.size()[0],-1))    
    if (i+1)%5==0:
      print('running val loss: ',loss.item())
    losses_generator_val.append(loss.item())
    loss_batch_val+=loss.item()

  loss_per_epoch_val = loss_batch_val/len(personvaldataloader)
  print('Loss:{:.2f}'.format(loss_per_epoch_val))
  
torch.save(generator.state_dict(), '../Weights/generator_early_trained.pth')
  



