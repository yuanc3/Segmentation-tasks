import os 
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils.metrics import f_score
# file_path='datasets//Testing-L'
# files = os.listdir(file_path)
# mini=512
# minj=512
# maxi=0
# maxj=0
# print(len(files))
# with tqdm(total=len(files),postfix=dict,mininterval=0.3) as pbar:
#     for fi in files:
#         fi_d = os.path.join(file_path, fi)
#         img=Image.open(fi_d)
#         img=np.array(img)
#         for i in range(len(img)):
#             for j in range(len(img[0])):
#                 if(img[i][j]!=0):
#                     mini=min(mini,i)
#                     maxi=max(maxi,i)
#                     minj=min(minj,j)
#                     maxj=max(maxj,j)
#         pbar.set_postfix(
#             **{
#                 'mini': mini,
#                 'maxi': maxi,
#                 'minj': minj,
#                 'maxj': maxj
#             })
#         pbar.update(1)

# print(mini,maxi,minj,maxj)

import torch
from torch import nn
import torch.nn.functional as F
class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss
class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss


criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)
# a = torch.Tensor([[[[1, 2], [3, 4],[5, 6], [7, 8]]]])
# b = torch.Tensor([[[[1, 2], [3, 4],[5, 6], [7, 8]]]])
a=torch.zeros([2,2,2,2])
b=torch.zeros([2,2,2,2])
a[1][1][1][1]=1
pad = torch.nn.ZeroPad2d(padding=(5, 5, 5, 5))
a=pad(a)
b=b.permute(0,3,1,2)
b=pad(b)
b=b.permute(0,2,3,1)
print(criterion._show_dice(a, b.float()))

