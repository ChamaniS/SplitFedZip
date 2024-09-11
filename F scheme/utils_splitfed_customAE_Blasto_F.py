import torch
import torchvision
from dataset import EmbryoDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
DEVICE = "cuda"
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from entropy_codecs.huffman_codec import huffman
import numpy as np
import math
import torch.nn.functional as F

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=False

):
    train_ds = EmbryoDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_ds = EmbryoDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

def get_loaders_test(
        test_dir,
        test_maskdir,
        test_transform
):
    test_ds = EmbryoDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_ds,
    )
    return test_loader

def eval(loader, local_model1,local_model2,local_model3,comp_model1, comp_model2, loss_fn,folder):
    #local_model1.eval()
    #local_model2.eval()
    #local_model3.eval()
    val_running_loss = 0.0
    valid_iou_score_class0 = 0.0
    valid_iou_score_class1 = 0.0
    valid_iou_score_class2 = 0.0
    valid_iou_score_class3 = 0.0
    valid_iou_score_class4 = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    val_running_totrateloss = 0.0
    val_running_rateloss1 = 0.0
    val_running_rateloss2 = 0.0
    val_running_mseloss1 = 0.0
    val_running_mseloss2 = 0.0
    val_running_correct = 0.0
    val_running_totdiceloss = 0.0
    lambda1=1
    with torch.no_grad():
        for idx,(x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.type(torch.LongTensor).to(device=DEVICE)
            enc1, predictions1 = local_model1(x)           
            x_hat1, y_likelihoods1 = comp_model1(predictions1)
            # bitrate of the quantized latent
            N, C, H, W = x.size()
            bpp_loss1 = torch.log(y_likelihoods1).sum() / (-math.log(2) * N*H*W)
            mse_loss1 = F.mse_loss(predictions1, x_hat1)
            S1_totloss = bpp_loss1  + lambda1*(mse_loss1)
            
            predictions2 = local_model2(x_hat1)
          
            x_hat2, y_likelihoods2 = comp_model2(predictions2)
            # bitrate of the quantized latent
            N, C, H, W = x.size()
            bpp_loss2 = torch.log(y_likelihoods2).sum() / (-math.log(2) * N*H*W)
            mse_loss2= F.mse_loss(predictions2, x_hat2)
            S2_totloss = bpp_loss2   + lambda1*(mse_loss2)
            
            predictions3 = local_model3(enc1, x_hat2)
            dice_loss3 = loss_fn(predictions3, y)
            loss = S1_totloss+S2_totloss + lambda1*dice_loss3
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            val_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            val_running_loss += loss.item()
            val_running_mseloss1 += mse_loss1.item()
            val_running_mseloss2 += mse_loss2.item()
            val_running_rateloss1 += bpp_loss1
            val_running_rateloss2 += bpp_loss2
            val_running_totrateloss += bpp_loss1 + bpp_loss2
            val_running_totdiceloss +=    dice_loss3.item()
            
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            valid_iou_score_class0 += iou_sklearn[0]
            valid_iou_score_class1 += iou_sklearn[1]
            valid_iou_score_class2 += iou_sklearn[2]
            valid_iou_score_class3 += iou_sklearn[3]
            valid_iou_score_class4 += iou_sklearn[4]
            torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0, scale_each=True,normalize=True)
    epoch_loss = val_running_loss / len(loader.dataset)
    epoch_totrateloss = val_running_totrateloss / len(loader.dataset)
    epoch_rateloss1 = val_running_rateloss1 / len(loader.dataset)
    epoch_rateloss2 = val_running_rateloss2 / len(loader.dataset)
    epoch_mseloss1 = val_running_mseloss1 / len(loader.dataset)
    epoch_mseloss2 = val_running_mseloss2 / len(loader.dataset)
    epoch_totdiceloss = val_running_totdiceloss / len(loader.dataset)
    epoch_acc = 100. * (val_running_correct / len(loader.dataset))
    epoch_iou_class0 = (valid_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (valid_iou_score_class1 / len(loader.dataset))
    epoch_iou_class2 = (valid_iou_score_class2 / len(loader.dataset))
    epoch_iou_class3 = (valid_iou_score_class3 / len(loader.dataset))
    epoch_iou_class4 = (valid_iou_score_class4 / len(loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 5
    epoch_iou_nobackground = (epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 4
    print("V epoch mse_loss1:", epoch_mseloss1)
    print("V epoch mse_loss2:", epoch_mseloss2)
    print("V epoch bpp_loss1:", epoch_rateloss1)
    print("V epoch bpp_loss2:", epoch_rateloss2)
    print("V epoch total bpp_loss:", epoch_totrateloss)
    print("V epoch total dice_loss:", epoch_totdiceloss)
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1, epoch_iou_class2, epoch_iou_class3, epoch_iou_class4
   

def test(loader, modelclientFE,modelserver,modelclientBE,comp_model1,comp_model2,loss_fn,folder):
    #modelclientFE.eval()
    #modelserver.eval()
    #modelclientBE.eval()
    test_running_loss = 0.0
    test_iou_score_class0 = 0.0
    test_iou_score_class1 = 0.0
    test_iou_score_class2 = 0.0
    test_iou_score_class3 = 0.0
    test_iou_score_class4 = 0.0
    test_running_totrateloss = 0.0
    test_running_rateloss1 = 0.0
    test_running_rateloss2 = 0.0
    test_running_mseloss1 = 0.0
    test_running_mseloss2 = 0.0
    test_running_correct = 0.0
    test_running_totdiceloss = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    lambda1 =1
    with torch.no_grad():
        for idx,(x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.type(torch.LongTensor).to(device=DEVICE)
            enc1, predictions1 = modelclientFE(x)
            comp_model1.entropy_bottleneck.update(force=True)           
            
            #------------Theoretical bitrate----------------------
            x_hat1, y_likelihoods1 = comp_model1(predictions1)
            N, C, H, W = x.size()
            #num_pixels = N * C*H * W 
            bpp_loss1 = torch.log(y_likelihoods1).sum() / (-math.log(2) * N*H*W)
            mse_loss1 = F.mse_loss(predictions1, x_hat1)
            S1_totloss = bpp_loss1  + lambda1*(mse_loss1)
            #----------------------------------------------------
            predictions2 = modelserver(x_hat1)
            #------------Theoretical bitrate----------------------
            x_hat2, y_likelihoods2 = comp_model2(predictions2)
            N, C, H, W = x.size()
            #num_pixels = N * C*H * W
            bpp_loss2 = torch.log(y_likelihoods2).sum() / (-math.log(2) * N*H*W)
            mse_loss2 = F.mse_loss(predictions2, x_hat2)
            S2_totloss = bpp_loss2  + lambda1*(mse_loss2)
            #----------------------------------------------------
                       
            predictions3 = modelclientBE(enc1, x_hat2)
            dice_loss3 = loss_fn(predictions3, y)
            loss = S1_totloss+S2_totloss+ lambda1*dice_loss3
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            test_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            test_running_loss += loss.item()
            test_running_mseloss1 += mse_loss1.item()
            test_running_mseloss2 += mse_loss2.item()
            test_running_rateloss1 += bpp_loss1
            test_running_rateloss2 += bpp_loss2
            test_running_totrateloss += bpp_loss1 + bpp_loss2
            test_running_totdiceloss +=    dice_loss3.item()
            
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            test_iou_score_class0 += iou_sklearn[0]
            test_iou_score_class1 += iou_sklearn[1]
            test_iou_score_class2 += iou_sklearn[2]
            test_iou_score_class3 += iou_sklearn[3]
            test_iou_score_class4 += iou_sklearn[4]
            torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0, scale_each=True,normalize=True)

    epoch_loss = test_running_loss / len(loader.dataset)
    epoch_totrateloss = test_running_totrateloss / len(loader.dataset)
    epoch_rateloss1 = test_running_rateloss1 / len(loader.dataset)
    epoch_rateloss2 = test_running_rateloss2 / len(loader.dataset)
    epoch_mseloss1 = test_running_mseloss1 / len(loader.dataset)
    epoch_mseloss2 = test_running_mseloss2 / len(loader.dataset)
    epoch_totdiceloss = test_running_totdiceloss / len(loader.dataset)
    epoch_acc = 100. * (test_running_correct / len(loader.dataset))
    epoch_iou_class0 = (test_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (test_iou_score_class1 / len(loader.dataset))
    epoch_iou_class2 = (test_iou_score_class2 / len(loader.dataset))
    epoch_iou_class3 = (test_iou_score_class3 / len(loader.dataset))
    epoch_iou_class4 = (test_iou_score_class4 / len(loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 5
    epoch_iou_nobackground = (epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 4
    print("Testing accuracy score:",epoch_acc)
    print("Testing mean IoU withbackground:",epoch_iou_withbackground)
    print("Testing mean IoU withoutbackground:",epoch_iou_nobackground) #needed_prediction
    print ("IoU of Background:", epoch_iou_class0)
    print ("IoU of ZP:", epoch_iou_class1)
    print ("IoU of TE:", epoch_iou_class2)
    print ("IoU of ICM:", epoch_iou_class3)
    print ("IoU of Blastocoel:", epoch_iou_class4)
    print("TESTING epoch mse_loss1:", epoch_mseloss1)
    print("TESTING epoch mse_loss2:", epoch_mseloss2)
    print("TESTING epoch bpp_loss1:", epoch_rateloss1)
    print("TESTINGepoch bpp_loss2:", epoch_rateloss2)
    print("TESTINGepoch total bpp_loss:", epoch_totrateloss)
    print("TESTING epoch total dice_loss:", epoch_totdiceloss)
    return epoch_loss, epoch_acc, epoch_iou_withbackground,epoch_iou_nobackground
  