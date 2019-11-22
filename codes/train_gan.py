import os
import glob
import torch
from tqdm import tqdm
import torch.nn as nn
from models import UNet11
import torch.optim as optim
from dataset import PersonDataset
import torchvision.models as models


lr = 0.001
batch_size = 30
num_epochs = 20

generator = UNet11()
generator.load_state_dict(torch.load('generator_early_trained.pth'))
generator.cuda()

discriminator = torch.load('discriminator_early_trained.pth')
discriminator.cuda()

persontraindataset = PersonDataset('../datasets/celeb_dataset', mode='train', transforms=transforms)
persontraindataloader = DataLoader(persontraindataset, batch_size=batch_size, shuffle=True)

personvaldataset = PersonDataset('../datasets/celeb_dataset', mode='test', transforms=None)
personvaldataloader = DataLoader(personvaldataset, batch_size=batch_size, shuffle=True)

optim_discriminator=optim.Adam(discriminator.parameters(), lr = 0.2*lr, betas = (0.5,0.999))
optim_generator=optim.Adam(generator.parameters(), lr = lr*0.2, betas = (0.5, 0.999))

losses_adver_train=[]
losses_adver_val=[]


for epoch in range(num_epochs):
  print('\n Epoch:{}'.format(epoch+1))
  disc_loss_batch_train = 0.0
  disc_loss_batch_val = 0.0
  
  image_content_loss_batch_train = 0.0
  adversarial_loss_batch_train = 0.0

  image_content_loss_batch_val = 0.0
  adversarial_loss_batch_val = 0.0  

  

  
  total_loss_batch_train = 0.0
  total_loss_batch_val = 0.0

  losses_disc_train = []
  losses_disc_val = []

  losses_image_train =[]
  losses_image_val = []



  
  losses_total_val=[]

  discriminator.train()
  generator.train()
  
############################################TRAIN######################################################  
  for i,(correct_img, degraded_img, target_correct, target_degraded) in tqdm(enumerate(persontraindataloader)):
    correct_img = correct_img.cuda()
    degraded_img = degraded_img.cuda()
    target_correct = target_correct.unsqueeze(1).cuda().float()
    target_degraded = target_degraded.unsqueeze(1).cuda().float()

    discriminator.zero_grad()
    corrected_fake = generator(degraded_img)
    torch.cuda.empty_cache()
    discriminator_out_correct = torch.sigmoid(discriminator(correct_img))
    discriminator_out_degraded = torch.sigmoid(discriminator(corrected_fake.detach()))
    torch.cuda.empty_cache()

    optim_discriminator.zero_grad()
    loss = discriminator_loss(discriminator_out_correct, target_correct) + discriminator_loss(discriminator_out_degraded, target_degraded)
    
    
    

    loss.backward(retain_graph= True)
    torch.cuda.empty_cache()
    optim_discriminator.step()

    generator.zero_grad()

    discriminator_out_degraded_adv = torch.sigmoid(discriminator(corrected_fake))
    adversarial_loss = discriminator_loss(discriminator_out_degraded_adv, target_correct)
    adversarial_loss_batch_train+=adversarial_loss.item()
    


    optim_generator.zero_grad()
    adversarial_loss.backward()
    

    
    
    torch.cuda.empty_cache()
    optim_generator.step()

    disc_loss_batch_train+=loss.item()
    losses_disc_train.append(loss.item())

    if (i+1)%5 == 0:
      print('Running Adversarial train loss:{:.4f}'.format(adversarial_loss.item()))      
      print('Running Discriminator train loss:{:.4f}'.format(loss.item()))
      

  losses_adver_train.append((adversarial_loss_batch_train/len(persontraindataloader)))
  print('Adversarial train loss:{:.4f}'.format(adversarial_loss_batch_train/len(persontraindataloader)))
  print('Discriminator train loss:{:.4f}'.format(disc_loss_batch_train/len(persontraindataloader)))
  

###################################VALIDATION###########################
  discriminator.eval()
  generator.eval()
  
  for i,(correct_img, degraded_img, target_correct, target_degraded) in tqdm(enumerate(personvaldataloader)):
    correct_img = correct_img.cuda()
    degraded_img = degraded_img.cuda()
    target_correct = target_correct.unsqueeze(1).cuda().float()
    target_degraded = target_degraded.unsqueeze(1).cuda().float()    

    discriminator.zero_grad()
    corrected_fake = generator(degraded_img)
    torch.cuda.empty_cache()

    discriminator_out_correct = torch.sigmoid(discriminator(correct_img))
    discriminator_out_degraded = torch.sigmoid(discriminator(corrected_fake.detach()))
    torch.cuda.empty_cache()

    loss = discriminator_loss(discriminator_out_correct, target_correct) + discriminator_loss(discriminator_out_degraded, target_degraded)
    
  
    generator.zero_grad()
    
    discriminator_out_degraded_adv = torch.sigmoid(discriminator(corrected_fake))
    adversarial_loss = discriminator_loss(discriminator_out_degraded_adv, target_correct)
    adversarial_loss_batch_val+=adversarial_loss.item()
   
    disc_loss_batch_val+=loss.item()
    losses_disc_val.append(loss.item())
    if (i+1)%5 == 0:
      print('Running Adversarial val loss:{:.4f}'.format(adversarial_loss.item()))   
      print('Running Discriminator val loss:{:.4f}'.format(loss.item()))
     

  losses_adver_val.append((adversarial_loss_batch_val/len(personvaldataloader)))
  print('Adversarial val loss:{:.4f}'.format(adversarial_loss_batch_val/len(personvaldataloader)))  
  print('Discriminator val loss:{:.4f}'.format(disc_loss_batch_val/len(personvaldataloader)))
  
  torch.save(generator.state_dict(),'../Weights/gan_model/generator_newly_trained_{}_ep.pth'.format(epoch+1))



