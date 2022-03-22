import time
import os
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rtpose_vgg import get_model, use_vgg
from datasets import trainset, valset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                           

def get_loss(keypoints1, keypoints2, keypoints_target1, keypoints_target2, matrix_1, matrix_2):

    criterion = nn.MSELoss(reduction='mean').to(device)
    total_loss = 0

    index1 = keypoints_target1[:,:,-1].unsqueeze(-1).int()
    loss1_1 = criterion((keypoints1[:,:,:-1] * index1).float(), (keypoints_target1[:,:,:-1] * index1).float())
    index2 = keypoints_target2[:,:,-1].unsqueeze(-1).int()
    loss1_2 = criterion((keypoints2[:,:,:-1] * index2).float(), (keypoints_target2[:,:,:-1] * index2).float())

    loss1 = loss1_1 + loss1_2
    loss2 = criterion(matrix_1.float(), matrix_2.float())

    total_loss += loss1
    total_loss += loss2

    return total_loss.float()

def get_loss_val(keypoint, keypoints_target):

    criterion = nn.MSELoss(reduction='mean').to(device)

    index = keypoints_target[:,:,-1].unsqueeze(-1).int()
    loss = criterion((keypoints[:,:,:-1] * index).float(), (keypoints_target[:,:,:-1] * index).float())

    return loss.float()
         

def train(train_loader, model, optimizer, epoch):
    
    # switch to train mode
    model.train()

    for i, (img1, img2, keypoints_target1, keypoints_target2) in enumerate(train_loader):
        # measure data loading time
        #writer.add_text('Text', 'text logged at step:' + str(i), i)
        
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(),i)        

        img1 = img1.to(device)
        img2 = img2.to(device)
        
        keypoints_target1 = keypoints_target1.to(device)
        keypoints_target2 = keypoints_target2.to(device)

        # compute output
        keypoints1, matrix1 = model(img1)
        keypoints2, matrix2 = model(img2)
        
        total_loss = get_loss(keypoints1, keypoints2, keypoints_target1, keypoints_target2, matrix1, matrix2)
        total_loss.float()
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)))
            print('Loss {loss:.4f}'.format(loss=total_loss))

    return total_loss
        
        
def validate(val_loader, model, epoch):

    # switch to train mode
    model.eval()

    for i, (img, keypoints_target) in enumerate(val_loader):
        # measure data loading time
        img = img.to(device)
        keypoints_target = keypoints_target.to(device)

        
        # compute output
        keypoints, _ = model(img)
        
        total_loss = get_loss_val(keypoints, keypoints_target)
               

        # measure elapsed time
        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader)))
            print('Loss {loss:.4f}'.format(loss=total_loss))
                
    return total_loss



# dataset
train_data  = trainset('/home/wang/cube_experiment/NEW_CUBE_DATASET/train/cube_rotation1', '/home/wang/cube_experiment/NEW_CUBE_DATASET/train/cube_rotation2',
                        '/home/wang/cube_experiment/NEW_CUBE_DATASET/train/cube_location1', '/home/wang/cube_experiment/NEW_CUBE_DATASET/train/cube_location2')
trainloader = DataLoader(train_data, batch_size=8, shuffle=False)

val_data = valset('E:/NEW_CUBE_DATASET/val/cube_rotation', 'E:/NEW_CUBE_DATASET/val/cube_location')
valloader = DataLoader(val_data, batch_size=8, shuffle=False)

print("Data ready")

# model
model = get_model(trunk='vgg19')
model = torch.nn.DataParallel(model).to(device)
# load pretrained
use_vgg(model)


print("Model ready")

# Fix the VGG weights first, and then the weights will be released
for i in range(20):
    for param in model.module.model0[i].parameters():
        param.requires_grad = False

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=1e-3,
                           momentum=0.9,
                           weight_decay=0.001,
                           nesterov=True)     
                                                                                          
for epoch in range(5):
    # train for one epoch
    print('epoch: {0}'.format(epoch))
    train_loss = train(trainloader, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(valloader, model, epoch)  
                                            
# Release all weights                                   
for param in model.module.parameters():
    param.requires_grad = True

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=1e-3,
                           momentum=0.9,
                           weight_decay=0.001,
                           nesterov=True)          
                                                    
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

best_val_loss = np.inf


model_save_filename = './best_model.pth'
for epoch in range(5, 75):
    print('epoch: {0}'.format(epoch))

    # train for one epoch
    train_loss = train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_loader, model, epoch)   
    
    lr_scheduler.step(val_loss)                        
    
    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    if is_best:
        torch.save(model.state_dict(), model_save_filename)      
          
