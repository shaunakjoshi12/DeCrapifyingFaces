import torch
import torch.optim as optim
from dataset import PersonDataset
from models import finetune_resnet
from torch.utils.data import DataLoader


resnet18 = models.resnet18(pretrained=True)
discriminator = finetune_resnet(resnet18,1)
discriminator.cuda()

lr = 0.001
batch_size = 30
num_epochs = 4

persontraindataset = PersonDataset('../datasets/celeb_dataset', mode='train', transforms=transforms)
persontraindataloader = DataLoader(persontraindataset, batch_size=batch_size, shuffle=True)

personvaldataset = PersonDataset('../datasets/celeb_dataset', mode='test', transforms=None)
personvaldataloader = DataLoader(personvaldataset, batch_size=batch_size, shuffle=True)

discriminator_loss = nn.BCELoss()

optim_discriminator = optim.SGD(discriminator.parameters(), lr=lr, momentum=0.9)

for epoch in range(num_epochs):
  loss_batch_train = 0.0
  loss_batch_val = 0.0
  discriminator.train()
  print('\n Epoch:{}'.format(epoch+1))
  for i,(correct_img, degraded_img, target_correct, target_degraded) in tqdm(enumerate(persontraindataloader)):
    correct_img = correct_img.cuda()
    degraded_img = degraded_img.cuda()
    target_correct = target_correct.unsqueeze(1).cuda().float()
    target_degraded = target_degraded.unsqueeze(1).cuda().float()

    discriminator_out_correct = torch.sigmoid(discriminator(correct_img))
    discriminator_out_degraded = torch.sigmoid(discriminator(degraded_img))    

    optim_discriminator.zero_grad()
    loss = discriminator_loss(discriminator_out_correct, target_correct) + discriminator_loss(discriminator_out_degraded, target_degraded)

    loss.backward()
    optim_discriminator.step()

    if (i+1)%5==0:
      print('running train loss: ',loss.item())
    #print('corrected ',corrected_fake.size(),' correct ',correct_img.size())
    
    loss_batch_train+=loss.item()


  loss_per_epoch_train = loss_batch_train/len(persontraindataloader)
  print('Loss:{:.2f}'.format(loss_per_epoch_train))

  discriminator.eval()
  for i,(correct_img, degraded_img, target_correct, target_degraded) in tqdm(enumerate(personvaldataloader)):
    correct_img = correct_img.cuda()
    degraded_img = degraded_img.cuda()
    target_correct = target_correct.unsqueeze(1).cuda().float()
    target_degraded = target_degraded.unsqueeze(1).cuda().float()

    discriminator_out_correct = torch.sigmoid(discriminator(correct_img))
    discriminator_out_degraded = torch.sigmoid(discriminator(degraded_img))    


    loss = discriminator_loss(discriminator_out_correct, target_correct) + discriminator_loss(discriminator_out_degraded, target_degraded)



    if (i+1)%5==0:
      print('running val loss: ',loss.item())
    #print('corrected ',corrected_fake.size(),' correct ',correct_img.size())
    
    loss_batch_val+=loss.item()

  loss_per_epoch_val = loss_batch_val/len(personvaldataloader)
  print('Loss:{:.2f}'.format(loss_per_epoch_val))  

torch.save(discriminator, '../Weights/discriminator_early_trained.pth')