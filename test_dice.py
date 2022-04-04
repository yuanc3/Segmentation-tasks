from utils.metrics import f_score
import torch
from nets.unet import Unet
import numpy as np
import time
from torch import nn
import time

import numpy as np
import torch
from tqdm import tqdm

from nets.unet import Unet
from nets.attU_net import AttU_Net
from nets.unet_training import CE_Loss, Dice_loss
from utils.metrics import f_score
from utils.dataloader1 import DeeplabDataset, deeplab_dataset_collate
from torch.utils.data import DataLoader
import os

def fit_one_epoch(net, epoch_size_val, name):
    dataset_path = "datasets//aa"
    Batch_size = 4
    val_lines =os.listdir(os.path.join(dataset_path, "Training"))
    list=[]
    for x in val_lines:
        if x.find(name)!=-1:
            list.append(x)
    length=len(list)
    val_dataset = DeeplabDataset(True,list, inputs_size, NUM_CLASSES,
                                     False, dataset_path)
    
    genval = DataLoader(val_dataset,
                             batch_size=Batch_size,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True,
                             collate_fn=deeplab_dataset_collate)
    net = net.train()


    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    val_loss = 0
    print('Start Validation')
    with tqdm(total=length//Batch_size,
              postfix=dict,
              mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                imgs = imgs.cuda()
                pngs = pngs.cuda()
                labels = labels.cuda()

                outputs = net(imgs)

                # pad = nn.ZeroPad2d(padding=(125, 131, 125, 131))
                # outputs=pad(outputs)
                
                # labels=labels.permute(0,3,1,2)
                # labels=pad(labels)
                # labels=labels.permute(0,2,3,1)

                main_dice = Dice_loss(outputs, labels)
                val_loss = val_loss + main_dice
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()

            pbar.set_postfix(
                **{
                    'total_loss': val_toal_loss / (iteration + 1),
                    'f_score': val_total_f_score / (iteration + 1),
                    'val_loss': val_loss / (iteration + 1)
                })
            pbar.update(1)

    print('Finish Validation')
    print('Total f: %.4f' %
          (val_total_f_score / (length//Batch_size)))

    return val_loss

if __name__ == "__main__":
    Batch_size = 4
    NUM_CLASSES=3
    inputs_size = [256, 256, 3]
    pretrained = False

    model = Unet(num_classes=NUM_CLASSES,
                in_channels=inputs_size[-1],
                pretrained=pretrained).eval()
    model_path = r"Best_Model_State_256.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if np.shape(model_dict[k]) == np.shape(v)
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    model.cuda()

    dataset_path = "datasets//aa"
    val_lines =os.listdir(os.path.join(dataset_path, "Training"))
    epoch_size_val = len(val_lines) // Batch_size

    val_loss = fit_one_epoch(model, epoch_size_val,"1705R-Neg")

    # print(val_loss.item())