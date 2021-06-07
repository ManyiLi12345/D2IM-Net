import os
import numpy as np 
import torch 
import torch.nn as nn

from datasets.CamDataset import CamDataset
from cam_est.CamNet import Net
import utils.NetHelper as NetHelper

def test(config):
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
    model = Net(config)
    if(config.cuda):
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cam_lr, betas=(config.beta1, 0.999), weight_decay=1e-5)
    
    testset = CamDataset(config, 'test')
    test_iter = torch.utils.data.DataLoader(testset, batch_size=config.cam_batch_size, shuffle=False)
    epoch, model, optimizer = NetHelper.load_checkpoint(config.model_dir+'/model.pt.tar',model, optimizer)
    output_dir = config.output_dir+str(epoch)+'_epoch/'
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    for batch_idx, batch in enumerate(test_iter):
        [images, cams, sampling] = batch
        if(config.cuda):
            images, cams, sampling = images.cuda(), cams.cuda(), sampling.cuda()
        pred_cams, _ = model(images, cams, sampling)
        # output

def train(config):
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
    
    model = Net(config)
    mseloss = nn.MSELoss()
    if(config.cuda):
        model.cuda()
        mseloss.cuda()
    
    trainset = CamDataset(config, 'train')
    train_iter = torch.utils.data.DataLoader(trainset, batch_size=config.cam_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.cam_lr, betas=(config.beta1, 0.999))
    epoch = 0

    if(config.load_pretrain and os.path.exists(config.model_dir+'/model.pt.tar')):
        epoch, model, optimizer = NetHelper.load_checkpoint(config.model_dir+'/model.pt.tar',model, optimizer)
    else:
        f = open(config.log, 'w')
        f.write('')
        f.close()

    # train
    while(epoch<config.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_iter):
            [images, cams, sampling] = batch
            if(config.cuda):
                images, cams, sampling = images.cuda(), cams.cuda(), sampling.cuda()
            _, loss = model(images, cams, sampling)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach()

            # print
            logline = 'epoch:%d//%d, batch:%d//%d, loss:%f' % (epoch+1, config.epochs, batch_idx+1, len(train_iter), loss.item())
            print(logline)
            NetHelper.print_log(config.log, logline)

        NetHelper.save_checkpoint(epoch, model, optimizer, config.model_dir+'/model.pt.tar')
        if((epoch+1)%config.save_every_epoch==0):
            NetHelper.save_checkpoint(epoch, model, optimizer, config.model_dir+'/model'+str(epoch+1)+'.pt.tar')
        
        epoch += 1
    
if __name__ == "__main__":
    config = NetHelper.get_args()
    train(config)
    #test(config)