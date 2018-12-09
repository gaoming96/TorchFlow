from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import models
import torch
from torch import nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils


""" gpu """
gpu_id = [0]
utils.cuda_devices(gpu_id)


""" param """
epochs = 200
batch_size = 1
lr = 0.0002
dataset_dir = 'datasets/apple2orange'


""" data """
load_size = 286
crop_size = 256

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Scale(load_size),
     transforms.RandomCrop(crop_size),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

a_data = dsets.ImageFolder(dataset_dir+'/trainA', transform=transform)
b_data = dsets.ImageFolder(dataset_dir+'/trainB', transform=transform)
a_test_data = dsets.ImageFolder(dataset_dir+'/testA', transform=transform)
b_test_data = dsets.ImageFolder(dataset_dir+'/testB', transform=transform)
a_loader = torch.utils.data.DataLoader(a_data, batch_size=batch_size, shuffle=True, num_workers=0)
b_loader = torch.utils.data.DataLoader(b_data, batch_size=batch_size, shuffle=True, num_workers=0)
a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=1, shuffle=True, num_workers=0)
b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=1, shuffle=True, num_workers=0)

a_fake_pool = utils.ItemPool()
b_fake_pool = utils.ItemPool()


""" model """
Da = models.Discriminator()
Db = models.Discriminator()
Ga = models.Generator()
Gb = models.Generator()
MSE = nn.MSELoss()
L1 = nn.L1Loss()
utils.cuda([Da, Db, Ga, Gb])

da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(0.5, 0.999))
db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(Ga.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))


""" run """
a_real_test = iter(a_test_loader).next()[0]
b_real_test = iter(b_test_loader).next()[0]
a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])

#a_real_test.requires_grad

for epoch in range(0, epochs):
    for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
        # step
        step = epoch * min(len(a_loader), len(b_loader)) + i + 1

        # set train for batchnorm
        Ga.train()
        Gb.train()

        # leaves
        a_real = a_real[0]
        b_real = b_real[0]
        a_real, b_real = utils.cuda([a_real, b_real])

        # train G
        a_fake = Ga(b_real)
        b_fake = Gb(a_real)

        a_rec = Ga(b_fake)
        b_rec = Gb(a_fake)

        # gen losses
        a_f_dis = Da(a_fake)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(torch.ones(a_f_dis.size()))
        a_gen_loss = MSE(a_f_dis, r_label)
        b_gen_loss = MSE(b_f_dis, r_label)

        # rec losses
        a_rec_loss = L1(a_rec, a_real)
        b_rec_loss = L1(b_rec, b_real)

        # g loss
        g_loss = a_gen_loss + b_gen_loss + a_rec_loss * 10.0 + b_rec_loss * 10.0

        # backward
        Ga.zero_grad()
        Gb.zero_grad()
        g_loss.backward()
        ga_optimizer.step()
        gb_optimizer.step()

        # leaves
        a_fake = torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0])
        b_fake = torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0])
        a_fake, b_fake = utils.cuda([a_fake, b_fake])

        # train D
        a_r_dis = Da(a_real)
        a_f_dis = Da(a_fake)
        b_r_dis = Db(b_real)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(torch.ones(a_f_dis.size()))
        f_label = utils.cuda(torch.zeros(a_f_dis.size()))

        # d loss
        a_d_r_loss = MSE(a_r_dis, r_label)
        a_d_f_loss = MSE(a_f_dis, f_label)
        b_d_r_loss = MSE(b_r_dis, r_label)
        b_d_f_loss = MSE(b_f_dis, f_label)

        a_d_loss = a_d_r_loss + a_d_f_loss
        b_d_loss = b_d_r_loss + b_d_f_loss

        # backward
        Da.zero_grad()
        Db.zero_grad()
        a_d_loss.backward()
        b_d_loss.backward()
        da_optimizer.step()
        db_optimizer.step()

        if (i + 1) % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d), (gloss,dloss): (%5.4f,%5.4f)" \
                  % (epoch, i + 1, min(len(a_loader), len(b_loader)),g_loss.item(),a_d_loss.item()+b_d_loss.item()))

        if (i + 1) % 50 == 0:
            Ga.eval()
            Gb.eval()
            
            # train G
            a_fake = Ga(b_real_test)
            b_fake = Gb(a_real_test)

            pic = (torch.cat([a_real, b_fake,  b_real, a_fake], dim=0).data + 1) / 2.0

            save_dir = './sample_images_while_training'
            utils.mkdir(save_dir)
            torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, min(len(a_loader), len(b_loader))), nrow=2)

            