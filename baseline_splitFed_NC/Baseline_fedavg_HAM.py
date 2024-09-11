import copy
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

torch.cuda.empty_cache()
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from models.clientmodel_FE import UNET_FE
from models.clientmodel_BE import UNET_BE
from models.servermodel import UNET_server
from baselines_main.utils_splitfed_HAM import (get_loaders, eval, get_loaders_test, test)
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score
from options import args_parser
from agg.Fed_Avg import fedAvg
import numpy as np
import pandas as pd
import random
import time

CUDA_LAUNCH_BLOCKING = 1

# Hyperparameters
LEARNING_RATE = 0.0001
device = "cuda"
NUM_WORKERS = 1
SHUFFLE = False
NUM_CLASSES = 5
PIN_MEMORY = False

dataDir = LOCATION_TO_DATA
parentF = LOCATION_TO_PARENT_FOLDER
# client 1
TRAIN_IMG_DIR_C1 = dataDir + "./client1/train_imgs/"
TRAIN_MASK_DIR_C1 = dataDir + "./client1/train_masks/"
VAL_IMG_DIR_C1 = dataDir + "./client1/val_imgs/"
VAL_MASK_DIR_C1 = dataDir + "./client1/val_masks/"

# client 2
TRAIN_IMG_DIR_C2 = dataDir + "./client2/train_imgs/"
TRAIN_MASK_DIR_C2 = dataDir + "./client2/train_masks/"
VAL_IMG_DIR_C2 = dataDir + "./client2/val_imgs/"
VAL_MASK_DIR_C2 = dataDir + "./client2/val_masks/"

# client 3
TRAIN_IMG_DIR_C3 = dataDir + "./client3/train_imgs/"
TRAIN_MASK_DIR_C3 = dataDir + "./client3/train_masks/"
VAL_IMG_DIR_C3 = dataDir + "./client3/val_imgs/"
VAL_MASK_DIR_C3 = dataDir + "./client3/val_masks/"

# client 4
TRAIN_IMG_DIR_C4 = dataDir + "./client4/train_imgs/"
TRAIN_MASK_DIR_C4 = dataDir + "./client4/train_masks/"
VAL_IMG_DIR_C4 = dataDir + "./client4/val_imgs/"
VAL_MASK_DIR_C4 = dataDir + "./client4/val_masks/"

# client 5
TRAIN_IMG_DIR_C5 = dataDir + "./client5/train_imgs/"
TRAIN_MASK_DIR_C5 = dataDir + "./client5/train_masks/"
VAL_IMG_DIR_C5 = dataDir + "./client5/val_imgs/"
VAL_MASK_DIR_C5 = dataDir + "./client5/val_masks/"

TEST_IMG_DIR = dataDir + "./test_imgs/"
TEST_MASK_DIR = dataDir + "./test_masks/"


# 1. Train function
def train(train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3, loss_fn, PL, T):
    loop = tqdm(train_loader)
    train_running_loss = 0.0
    train_running_correct = 0.0
    train_iou_score = 0.0
    train_iou_score_class0 = 0.0
    train_iou_score_class1 = 0.0
    PL_uplink1 = PL[0]
    PL_downlink1 = PL[1]
    PL_uplink2 = PL[2]
    PL_downlink2 = PL[3]
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.type(torch.LongTensor).to(device)
        predictions1 = local_model1(data)
        lostpredictions1 = PL_uplink1 * predictions1
        predictions2 = local_model2(lostpredictions1)
        lostpredictions2 = PL_downlink1 * predictions2
        predictions3 = local_model3(lostpredictions2)
        loss = loss_fn(predictions3, targets)
        preds = torch.argmax(predictions3, dim=1)
        equals = preds == targets
        train_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
        train_running_loss += loss.item()
        train_iou_score += jaccard_score(targets.cpu().flatten(), preds.cpu().flatten(), average='micro')
        iou_sklearn = jaccard_score(targets.cpu().flatten(), preds.cpu().flatten(), average=None)
        train_iou_score_class0 += iou_sklearn[0]
        train_iou_score_class1 += iou_sklearn[1]
        loss.backward(retain_graph=True)
        optimizer3.step()
        optimizer3.zero_grad()
        mygrad3 = grads3
        mygrad3lost = PL_uplink2 * mygrad3
        predictions2.backward(mygrad3lost, retain_graph=True)
        optimizer2.step()
        optimizer2.zero_grad()
        mygrad2 = grads2
        mygrad2lost = PL_downlink2 * mygrad2
        predictions1.backward(mygrad2lost)
        optimizer1.step()
        optimizer1.zero_grad()
        loop.set_postfix(loss=loss.item())
    epoch_loss = train_running_loss / len(train_loader.dataset)
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    epoch_iou_class0 = (train_iou_score_class0 / len(train_loader.dataset))
    epoch_iou_class1 = (train_iou_score_class1 / len(train_loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1) / 2
    epoch_iou_nobackground = epoch_iou_class1
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1


# 2. Main function
def main():
    args = args_parser()
    start_time = time.time()
    train_transform = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader_C1, val_loader_C1 = get_loaders(
        TRAIN_IMG_DIR_C1,
        TRAIN_MASK_DIR_C1,
        VAL_IMG_DIR_C1,
        VAL_MASK_DIR_C1,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY

    )

    train_loader_C2, val_loader_C2 = get_loaders(
        TRAIN_IMG_DIR_C2,
        TRAIN_MASK_DIR_C2,
        VAL_IMG_DIR_C2,
        VAL_MASK_DIR_C2,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY

    )

    train_loader_C3, val_loader_C3 = get_loaders(
        TRAIN_IMG_DIR_C3,
        TRAIN_MASK_DIR_C3,
        VAL_IMG_DIR_C3,
        VAL_MASK_DIR_C3,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C4, val_loader_C4 = get_loaders(
        TRAIN_IMG_DIR_C4,
        TRAIN_MASK_DIR_C4,
        VAL_IMG_DIR_C4,
        VAL_MASK_DIR_C4,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C5, val_loader_C5 = get_loaders(
        TRAIN_IMG_DIR_C5,
        TRAIN_MASK_DIR_C5,
        VAL_IMG_DIR_C5,
        VAL_MASK_DIR_C5,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    test_loader = get_loaders_test(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        train_transform
    )

    global_model1_fed = UNET_FE(in_channels=3).to(device)
    global_model2_fed = UNET_server(in_channels=32).to(device)
    global_model3_fed = UNET_BE(out_channels=NUM_CLASSES).to(device)

    # global round
    w_locals_model1, w_locals_model2, w_locals_model3 = [], [], []
    C1time, C2time, C3time, C4time, C5time = 0, 0, 0, 0, 0
    client1_train_acc, client1_train_loss, client1_train_withbackiou, client1_train_nobackiou, client1_val_acc, client1_val_loss, client1_val_withbackiou, client1_val_nobackiou, client1_g_val_acc, client1_g_val_loss, client1_g_val_iouwithback, client1_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client2_train_acc, client2_train_loss, client2_train_withbackiou, client2_train_nobackiou, client2_val_acc, client2_val_loss, client2_val_withbackiou, client2_val_nobackiou, client2_g_val_acc, client2_g_val_loss, client2_g_val_iouwithback, client2_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client3_train_acc, client3_train_loss, client3_train_withbackiou, client3_train_nobackiou, client3_val_acc, client3_val_loss, client3_val_withbackiou, client3_val_nobackiou, client3_g_val_acc, client3_g_val_loss, client3_g_val_iouwithback, client3_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client4_train_acc, client4_train_loss, client4_train_withbackiou, client4_train_nobackiou, client4_val_acc, client4_val_loss, client4_val_withbackiou, client4_val_nobackiou, client4_g_val_acc, client4_g_val_loss, client4_g_val_iouwithback, client4_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client5_train_acc, client5_train_loss, client5_train_withbackiou, client5_train_nobackiou, client5_val_acc, client5_val_loss, client5_val_withbackiou, client5_val_nobackiou, client5_g_val_acc, client5_g_val_loss, client5_g_val_iouwithback, client5_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    test_Acc, test_Iou_withback, test_Iou_noback, test_Loss = [], [], [], []
    least_lossg = 100000000;

    # generating packet loss
    C1_PL, C2_PL, C3_PL, C4_PL, C5_PL, C1_EPL, C2_EPL, C3_EPL, C4_EPL, C5_EPL = [], [], [], [], [], [], [], [], [], []
    C1T, C2T, C3T, C4T, C5T = [], [], [], [], []

    imgsize = 256

    for c in range(1, 6):
        if (c == 1):  # C1------------------------------------------------
            c1ch_list1, c1ch_list2, c1ch_list3, c1ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c1 = torch.ones(imgsize, imgsize)
                num_zero_rows1c1 = math.floor(args.pack_prob_up_C1 * rows1c1)
                x1 = set(random.sample(list(first), num_zero_rows1c1))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c1[i] = 0 * random_tensor1c1[i]
                output_tensor1c1 = random_tensor1c1
                c1ch_list1.append(output_tensor1c1)
            pack_loss_uplink1C1 = torch.stack(c1ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c1 = torch.ones(imgsize, imgsize)
                num_zero_rows2c1 = math.floor(args.pack_prob_down_C1 * rows2c1)
                x2 = set(random.sample(list(first), num_zero_rows2c1))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c1[i] = 0 * random_tensor2c1[i]
                output_tensor2c1 = random_tensor2c1
                c1ch_list2.append(output_tensor2c1)
            pack_loss_downlink1C1 = torch.stack(c1ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c1 = torch.ones(imgsize, imgsize)
                num_zero_rows3c1 = math.floor(args.pack_prob_up_C1 * rows3c1)
                x3 = set(random.sample(list(first), num_zero_rows3c1))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c1[i] = 0 * random_tensor3c1[i]
                output_tensor3c1 = random_tensor3c1
                c1ch_list3.append(output_tensor3c1)
            pack_loss_uplink2C1 = torch.stack(c1ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c1 = torch.ones(imgsize, imgsize)
                num_zero_rows4c1 = math.floor(args.pack_prob_down_C1 * rows4c1)
                x4 = set(random.sample(list(first), num_zero_rows4c1))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c1[i] = 0 * random_tensor4c1[i]
                output_tensor4c1 = random_tensor4c1
                c1ch_list4.append(output_tensor4c1)
            pack_loss_downlink2C1 = torch.stack(c1ch_list4, dim=0).to(device)

            C1_PL.extend([pack_loss_uplink1C1, pack_loss_downlink1C1, pack_loss_uplink2C1, pack_loss_downlink2C1])
            C1_EPL.extend([pack_loss_uplink1C1, pack_loss_downlink1C1])

        elif (c == 2):  # C2------------------------------------------------
            c2ch_list1, c2ch_list2, c2ch_list3, c2ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c2 = torch.ones(imgsize, imgsize)
                num_zero_rows1c2 = math.floor(args.pack_prob_up_C2 * rows1c2)
                x1 = set(random.sample(list(first), num_zero_rows1c2))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c2[i] = 0 * random_tensor1c2[i]
                output_tensor1c2 = random_tensor1c2
                c2ch_list1.append(output_tensor1c2)
            pack_loss_uplink1C2 = torch.stack(c2ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c2 = torch.ones(imgsize, imgsize)
                num_zero_rows2c2 = math.floor(args.pack_prob_down_C2 * rows2c2)
                x2 = set(random.sample(list(first), num_zero_rows2c2))

                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c2[i] = 0 * random_tensor2c2[i]
                output_tensor2c2 = random_tensor2c2
                c2ch_list2.append(output_tensor2c2)
            pack_loss_downlink1C2 = torch.stack(c2ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c2 = torch.ones(imgsize, imgsize)
                num_zero_rows3c2 = math.floor(args.pack_prob_up_C2 * rows3c2)
                x3 = set(random.sample(list(first), num_zero_rows3c2))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c2[i] = 0 * random_tensor3c2[i]
                output_tensor3c2 = random_tensor3c2
                c2ch_list3.append(output_tensor3c2)
            pack_loss_uplink2C2 = torch.stack(c2ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c2 = torch.ones(imgsize, imgsize)
                num_zero_rows4c2 = math.floor(args.pack_prob_down_C2 * rows4c2)
                x4 = set(random.sample(list(first), num_zero_rows4c2))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c2[i] = 0 * random_tensor4c2[i]
                output_tensor4c2 = random_tensor4c2
                c2ch_list4.append(output_tensor4c2)
            pack_loss_downlink2C2 = torch.stack(c2ch_list4, dim=0).to(device)
            C2_PL.extend(
                [pack_loss_uplink1C2, pack_loss_downlink1C2, pack_loss_uplink2C2, pack_loss_downlink2C2])
            C2_EPL.extend([pack_loss_uplink1C2, pack_loss_downlink1C2])

        elif (c == 3):  # C3------------------------------------------------
            c3ch_list1, c3ch_list2, c3ch_list3, c3ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c3 = torch.ones(imgsize, imgsize)
                num_zero_rows1c3 = math.floor(args.pack_prob_up_C3 * rows1c3)
                x1 = set(random.sample(list(first), num_zero_rows1c3))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c3[i] = 0 * random_tensor1c3[i]
                output_tensor1c3 = random_tensor1c3
                c3ch_list1.append(output_tensor1c3)
            pack_loss_uplink1C3 = torch.stack(c3ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c3 = torch.ones(imgsize, imgsize)
                num_zero_rows2c3 = math.floor(args.pack_prob_down_C3 * rows2c3)
                x2 = set(random.sample(list(first), num_zero_rows2c3))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c3[i] = 0 * random_tensor2c3[i]
                output_tensor2c3 = random_tensor2c3
                c3ch_list2.append(output_tensor2c3)
            pack_loss_downlink1C3 = torch.stack(c3ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c3 = torch.ones(imgsize, imgsize)
                num_zero_rows3c3 = math.floor(args.pack_prob_up_C3 * rows3c3)
                x3 = set(random.sample(list(first), num_zero_rows3c3))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c3[i] = 0 * random_tensor3c3[i]
                output_tensor3c3 = random_tensor3c3
                c3ch_list3.append(output_tensor3c3)
            pack_loss_uplink2C3 = torch.stack(c3ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c3 = torch.ones(imgsize, imgsize)
                num_zero_rows4c3 = math.floor(args.pack_prob_down_C3 * rows4c3)
                x4 = set(random.sample(list(first), num_zero_rows4c3))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c3[i] = 0 * random_tensor4c3[i]
                output_tensor4c3 = random_tensor4c3
                c3ch_list4.append(output_tensor4c3)
            pack_loss_downlink2C3 = torch.stack(c3ch_list4, dim=0).to(device)

            C3_PL.extend([pack_loss_uplink1C3, pack_loss_downlink1C3, pack_loss_uplink2C3, pack_loss_downlink2C3])
            C3_EPL.extend([pack_loss_uplink1C3, pack_loss_downlink1C3])

        elif (c == 4):  # C4------------------------------------------------
            c4ch_list1, c4ch_list2, c4ch_list3, c4ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c4 = torch.ones(imgsize, imgsize)
                num_zero_rows1c4 = math.floor(args.pack_prob_up_C4 * rows1c4)
                x1 = set(random.sample(list(first), num_zero_rows1c4))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c4[i] = 0 * random_tensor1c4[i]
                output_tensor1c4 = random_tensor1c4
                c4ch_list1.append(output_tensor1c4)
            pack_loss_uplink1C4 = torch.stack(c4ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c4 = torch.ones(imgsize, imgsize)
                num_zero_rows2c4 = math.floor(args.pack_prob_down_C4 * rows2c4)
                x2 = set(random.sample(list(first), num_zero_rows2c4))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c4[i] = 0 * random_tensor2c4[i]
                output_tensor2c4 = random_tensor2c4
                c4ch_list2.append(output_tensor2c4)
            pack_loss_downlink1C4 = torch.stack(c4ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c4 = torch.ones(imgsize, imgsize)
                num_zero_rows3c4 = math.floor(args.pack_prob_up_C4 * rows3c4)
                x3 = set(random.sample(list(first), num_zero_rows3c4))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c4[i] = 0 * random_tensor3c4[i]
                output_tensor3c4 = random_tensor3c4
                c4ch_list3.append(output_tensor3c4)
            pack_loss_uplink2C4 = torch.stack(c4ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c4 = torch.ones(imgsize, imgsize)
                num_zero_rows4c4 = math.floor(args.pack_prob_down_C4 * rows4c4)
                x4 = set(random.sample(list(first), num_zero_rows4c4))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c4[i] = 0 * random_tensor4c4[i]
                output_tensor4c4 = random_tensor4c4
                c4ch_list4.append(output_tensor4c4)
            pack_loss_downlink2C4 = torch.stack(c4ch_list4, dim=0).to(device)

            C4_PL.extend(
                [pack_loss_uplink1C4, pack_loss_downlink1C4, pack_loss_uplink2C4, pack_loss_downlink2C4])
            C4_EPL.extend([pack_loss_uplink1C4, pack_loss_downlink1C4])

        else:  # C5------------------------------------------------
            c5ch_list1, c5ch_list2, c5ch_list3, c5ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c5 = torch.ones(imgsize, imgsize)
                num_zero_rows1c5 = math.floor(args.pack_prob_up_C5 * rows1c5)
                x1 = set(random.sample(list(first), num_zero_rows1c5))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c5[i] = 0 * random_tensor1c5[i]
                output_tensor1c5 = random_tensor1c5
                c5ch_list1.append(output_tensor1c5)
            pack_loss_uplink1C5 = torch.stack(c5ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c5 = torch.ones(imgsize, imgsize)
                num_zero_rows2c5 = math.floor(args.pack_prob_down_C5 * rows2c5)
                x2 = set(random.sample(list(first), num_zero_rows2c5))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c5[i] = 0 * random_tensor2c5[i]
                output_tensor2c5 = random_tensor2c5
                c5ch_list2.append(output_tensor2c5)
            pack_loss_downlink1C5 = torch.stack(c5ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c5 = torch.ones(imgsize, imgsize)
                num_zero_rows3c5 = math.floor(args.pack_prob_up_C5 * rows3c5)
                x3 = set(random.sample(list(first), num_zero_rows3c5))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c5[i] = 0 * random_tensor3c5[i]
                output_tensor3c5 = random_tensor3c5
                c5ch_list3.append(output_tensor3c5)
            pack_loss_uplink2C5 = torch.stack(c5ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c5 = torch.ones(imgsize, imgsize)
                num_zero_rows4c5 = math.floor(args.pack_prob_down_C5 * rows4c5)
                x4 = set(random.sample(list(first), num_zero_rows4c5))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c5[i] = 0 * random_tensor4c5[i]
                output_tensor4c5 = random_tensor4c5
                c5ch_list4.append(output_tensor4c5)
            pack_loss_downlink2C5 = torch.stack(c5ch_list4, dim=0).to(device)

            C5_PL.extend(
                [pack_loss_uplink1C5, pack_loss_downlink1C5, pack_loss_uplink2C5, pack_loss_downlink2C5])
            C5_EPL.extend([pack_loss_uplink1C5, pack_loss_downlink1C5])

    for com_round in (range(args.rounds)):
        local_weights1, local_weights2, local_weights3 = [], [], []
        least_lossC1, least_lossC2, least_lossC3, least_lossC4, least_lossC5 = 100000000, 100000000, 100000000, 100000000, 100000000;
        # Getting global model params
        round_idx = com_round + 1

        # --------------------------------------LOCAL TRAINING & VALIDATING---------------------------------------------------------------------------
        print(f'\n | Global Training Round : {round_idx} |\n')
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local_model1 = copy.deepcopy(global_model1_fed)
            local_model2 = copy.deepcopy(global_model2_fed)
            local_model3 = copy.deepcopy(global_model3_fed)

            loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
            optimizer1 = optim.Adam(local_model1.parameters(), lr=LEARNING_RATE)
            optimizer2 = optim.Adam(local_model2.parameters(), lr=LEARNING_RATE)
            optimizer3 = optim.Adam(local_model3.parameters(), lr=LEARNING_RATE)

            # Backward hook for modelclientBE
            grads3 = 0

            def grad_hook1(model, grad_input, grad_output):
                global grads3
                grads3 = grad_input[0].clone().detach()

            local_model3.decoder1.register_full_backward_hook(grad_hook1)

            # Backward hook for modelserver
            grads2 = 0

            def grad_hook2(model, grad_input, grad_output):
                global grads2
                grads2 = grad_input[0].clone().detach()

            local_model2.encoder1.register_full_backward_hook(grad_hook2)

            cl_idx = idx + 1
            print("Selected client:", cl_idx)
            if cl_idx == 1:
                train_loader = train_loader_C1
                val_loader = val_loader_C1
                folder = parentF + "Baseline/noisycom/Fed_Avg/Saved/local_models/client1"
            elif cl_idx == 2:
                train_loader = train_loader_C2
                val_loader = val_loader_C2
                folder = parentF + "Baseline/noisycom/Fed_Avg/Saved/local_models/client2"
            elif cl_idx == 3:
                train_loader = train_loader_C3
                val_loader = val_loader_C3
                folder = parentF + "Baseline/noisycom/Fed_Avg/Saved/local_models/client3"
            elif cl_idx == 4:
                train_loader = train_loader_C4
                val_loader = val_loader_C4
                folder = parentF + "Baseline/noisycom/Fed_Avg/Saved/local_models/client4"
            elif cl_idx == 5:
                train_loader = train_loader_C5
                val_loader = val_loader_C5
                folder = parentF + "Baseline/noisycom/Fed_Avg/Saved/local_models/client5"

                # local epoch
            for epoch in range(args.local_ep):
                print(f"[INFO]: Epoch {epoch + 1} of {args.local_ep}")
                print("Client", cl_idx, " training.........")
                if cl_idx == 1:  # C1---------------------------------------------------------------C1 local training & validation--------------------------------------------------------------------------------------------------------------------
                    start_timec1 = time.time()
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1 = train(
                        train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3,
                        loss_fn, C1_PL, C1T)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1 = eval(
                        val_loader, local_model1, local_model2, local_model3, loss_fn, folder, C1_EPL, C1T)
                    client1_train_acc.append(train_epoch_acc)
                    client1_train_loss.append(train_epoch_loss)
                    client1_train_withbackiou.append(trainepoch_iou_withbackground)
                    client1_train_nobackiou.append(trainepoch_iou_nobackground)
                    client1_val_acc.append(val_epoch_acc)
                    client1_val_loss.append(val_epoch_loss)
                    client1_val_withbackiou.append(valepoch_iou_withbackground)
                    client1_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC1 > val_epoch_loss:
                        least_lossC1 = val_epoch_loss
                        torch.save(local_model1.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpoint.pth')
                        print('C1localmodel saved')
                    end_timec1 = time.time()
                    c1t = end_timec1 - start_timec1
                    C1time = C1time + c1t
                    print("C1 cumulative time:", C1time)

                if cl_idx == 2:  # C2--------------------------------------------------------------C2 local training & validation--------------------------------------------------------------------------------------------------------------------
                    start_timec2 = time.time()
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1 = train(
                        train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3,
                        loss_fn, C2_PL, C2T)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1 = eval(
                        val_loader, local_model1, local_model2, local_model3, loss_fn, folder, C2_EPL, C2T)
                    client2_train_acc.append(train_epoch_acc)
                    client2_train_loss.append(train_epoch_loss)
                    client2_train_withbackiou.append(trainepoch_iou_withbackground)
                    client2_train_nobackiou.append(trainepoch_iou_nobackground)
                    client2_val_acc.append(val_epoch_acc)
                    client2_val_loss.append(val_epoch_loss)
                    client2_val_withbackiou.append(valepoch_iou_withbackground)
                    client2_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC2 > val_epoch_loss:
                        least_lossC2 = val_epoch_loss
                        torch.save(local_model1.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C2M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C2M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C2M3_localcheckpoint.pth')
                        print('C2localmodel saved')
                    end_timec2 = time.time()
                    c2t = end_timec2 - start_timec2
                    C2time = C2time + c2t
                    print("C2 cumulative time:", C2time)

                if cl_idx == 3:  # C3--------------------------------------------------------------C3 local training & validation-----------------------------------------------------------------------------------------------------------
                    start_timec3 = time.time()
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1 = train(
                        train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3,
                        loss_fn, C3_PL, C3T)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1 = eval(
                        val_loader, local_model1, local_model2, local_model3, loss_fn, folder, C3_EPL, C3T)
                    client3_train_acc.append(train_epoch_acc)
                    client3_train_loss.append(train_epoch_loss)
                    client3_train_withbackiou.append(trainepoch_iou_withbackground)
                    client3_train_nobackiou.append(trainepoch_iou_nobackground)
                    client3_val_acc.append(val_epoch_acc)
                    client3_val_loss.append(val_epoch_loss)
                    client3_val_withbackiou.append(valepoch_iou_withbackground)
                    client3_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC3 > val_epoch_loss:
                        least_lossC3 = val_epoch_loss
                        torch.save(local_model1.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C3M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C3M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C3M3_localcheckpoint.pth')
                        print('C3localmodel saved')
                    end_timec3 = time.time()
                    c3t = end_timec3 - start_timec3
                    C3time = C3time + c3t
                    print("C3 cumulative time:", C3time)

                if cl_idx == 4:  # C4--------------------------------------------------------------C4 local training & validation-----------------------------------------------------------------------------------------------------------
                    start_timec4 = time.time()
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1 = train(
                        train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3,
                        loss_fn, C4_PL, C4T)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1 = eval(
                        val_loader, local_model1, local_model2, local_model3, loss_fn, folder, C4_EPL, C4T)
                    client4_train_acc.append(train_epoch_acc)
                    client4_train_loss.append(train_epoch_loss)
                    client4_train_withbackiou.append(trainepoch_iou_withbackground)
                    client4_train_nobackiou.append(trainepoch_iou_nobackground)
                    client4_val_acc.append(val_epoch_acc)
                    client4_val_loss.append(val_epoch_loss)
                    client4_val_withbackiou.append(valepoch_iou_withbackground)
                    client4_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC4 > val_epoch_loss:
                        least_lossC4 = val_epoch_loss
                        torch.save(local_model1.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C4M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C4M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C4M3_localcheckpoint.pth')
                        print('C4localmodel saved')
                    end_timec4 = time.time()
                    c4t = end_timec4 - start_timec4
                    C4time = C4time + c4t
                    print("C4 cumulative time:", C4time)

                if cl_idx == 5:  # C5--------------------------------------------------------------C5 local training & validation-----------------------------------------------------------------------------------------------------------
                    start_timec5 = time.time()
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1 = train(
                        train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3,
                        loss_fn, C5_PL, C5T)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1 = eval(
                        val_loader, local_model1, local_model2, local_model3, loss_fn, folder, C5_EPL, C5T)
                    client5_train_acc.append(train_epoch_acc)
                    client5_train_loss.append(train_epoch_loss)
                    client5_train_withbackiou.append(trainepoch_iou_withbackground)
                    client5_train_nobackiou.append(trainepoch_iou_nobackground)
                    client5_val_acc.append(val_epoch_acc)
                    client5_val_loss.append(val_epoch_loss)
                    client5_val_withbackiou.append(valepoch_iou_withbackground)
                    client5_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC5 > val_epoch_loss:
                        least_lossC5 = val_epoch_loss
                        torch.save(local_model1.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C5M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C5M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),
                                   parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C5M3_localcheckpoint.pth')
                        print('C5localmodel saved')
                    end_timec5 = time.time()
                    c5t = end_timec5 - start_timec5
                    C5time = C5time + c5t
                    print("C5 cumulative time:", C5time)

                print(
                    f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                print("\n Training IoUs Client:", cl_idx)
                print("T: Background:", trainepoch_iou_class0)
                print("T: Tumor:", trainepoch_iou_class1)

                print(
                    f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                print("\n Validating IoUs Client:", cl_idx)
                print("V: Background:", valepoch_iou_class0)
                print("V: Tumor:", valepoch_iou_class1)

        C1M1localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpoint.pth')
        C1M2localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpoint.pth')
        C1M3localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpoint.pth')
        C2M1localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C2M1_localcheckpoint.pth')
        C2M2localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C2M2_localcheckpoint.pth')
        C2M3localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C2M3_localcheckpoint.pth')
        C3M1localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C3M1_localcheckpoint.pth')
        C3M2localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C3M2_localcheckpoint.pth')
        C3M3localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C3M3_localcheckpoint.pth')
        C4M1localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C4M1_localcheckpoint.pth')
        C4M2localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C4M2_localcheckpoint.pth')
        C4M3localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C4M3_localcheckpoint.pth')
        C5M1localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C5M1_localcheckpoint.pth')
        C5M2localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C5M2_localcheckpoint.pth')
        C5M3localbest = torch.load(parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/C5M3_localcheckpoint.pth')

        tot_loader = len(train_loader_C1) + len(train_loader_C2) + len(train_loader_C3) + len(train_loader_C4) + len(
            train_loader_C5)
        D1 = len(train_loader_C1) / tot_loader;
        D2 = len(train_loader_C2) / tot_loader;
        D3 = len(train_loader_C3) / tot_loader;
        D4 = len(train_loader_C4) / tot_loader;
        D5 = len(train_loader_C5) / tot_loader;

        # updated parameters
        C1M1localbest.update((x, y * D1) for x, y in C1M1localbest.items())
        C1M2localbest.update((x, y * D1) for x, y in C1M2localbest.items())
        C1M3localbest.update((x, y * D1) for x, y in C1M3localbest.items())
        C2M1localbest.update((x, y * D2) for x, y in C2M1localbest.items())
        C2M2localbest.update((x, y * D2) for x, y in C2M2localbest.items())
        C2M3localbest.update((x, y * D2) for x, y in C2M3localbest.items())
        C3M1localbest.update((x, y * D3) for x, y in C3M1localbest.items())
        C3M2localbest.update((x, y * D3) for x, y in C3M2localbest.items())
        C3M3localbest.update((x, y * D3) for x, y in C3M3localbest.items())
        C4M1localbest.update((x, y * D4) for x, y in C4M1localbest.items())
        C4M2localbest.update((x, y * D4) for x, y in C4M2localbest.items())
        C4M3localbest.update((x, y * D4) for x, y in C4M3localbest.items())
        C5M1localbest.update((x, y * D5) for x, y in C5M1localbest.items())
        C5M2localbest.update((x, y * D5) for x, y in C5M2localbest.items())
        C5M3localbest.update((x, y * D5) for x, y in C5M3localbest.items())

        # Model1Averaging
        M1dict = [C1M1localbest, C2M1localbest, C3M1localbest, C4M1localbest, C5M1localbest]
        M2dict = [C1M2localbest, C2M2localbest, C3M2localbest, C4M2localbest, C5M2localbest]
        M3dict = [C1M3localbest, C2M3localbest, C3M3localbest, C4M3localbest, C5M3localbest]
        local_weights1.extend(M1dict)
        local_weights2.extend(M2dict)
        local_weights3.extend(M3dict)

        # averaging parameters
        global_fed_weights1 = fedAvg(local_weights1)
        global_fed_weights2 = fedAvg(local_weights2)
        global_fed_weights3 = fedAvg(local_weights3)

        # load the new parameters - FedAvg
        global_model1_fed.load_state_dict(global_fed_weights1)
        global_model2_fed.load_state_dict(global_fed_weights2)
        global_model3_fed.load_state_dict(global_fed_weights3)

        print("Weights averaged, loaded new weights")

        # ------------------------------------------VALIDATING USING THE GLOBAL MODEL-----------------------------------------------------------------------
        # Validating using the global model
        m1 = max(int(args.frac * args.num_users), 1)
        idxs_users1 = np.random.choice(range(args.num_users), m1, replace=False)
        for idx in idxs_users1:

            cl_idx = idx + 1
            print("Selected client:", cl_idx)
            if cl_idx == 1:
                val_loader = val_loader_C1
                folder = parentF + "/Baseline/noisycom/Fed_Avg/Saved/global_model/val/client1"
            elif cl_idx == 2:
                val_loader = val_loader_C2
                folder = parentF + "/Baseline/noisycom/Fed_Avg/Saved/global_model/val/client2"
            elif cl_idx == 3:
                val_loader = val_loader_C3
                folder = parentF + "/Baseline/noisycom/Fed_Avg/Saved/global_model/val/client3"
            elif cl_idx == 4:
                val_loader = val_loader_C4
                folder = parentF + "/Baseline/noisycom/Fed_Avg/Saved/global_model/val/client4"
            elif cl_idx == 5:
                val_loader = val_loader_C5
                folder = parentF + "/Baseline/noisycom/Fed_Avg/Saved/global_model/val/client5"

            best_epoch = 0
            for epoch in range(args.val_global_ep):
                print(f"[INFO]: Epoch {epoch + 1} of {args.val_global_ep}")
                print("Client", cl_idx, " validating.........")
                if cl_idx == 1:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed, loss_fn, folder, C1_EPL,
                        C1T)
                    client1_g_val_acc.append(g_val_epoch_acc)
                    client1_g_val_loss.append(g_val_epoch_loss)
                    client1_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client1_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 2:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed, loss_fn, folder, C2_EPL,
                        C2T)
                    client2_g_val_acc.append(g_val_epoch_acc)
                    client2_g_val_loss.append(g_val_epoch_loss)
                    client2_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client2_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 3:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed, loss_fn, folder, C3_EPL,
                        C3T)
                    client3_g_val_acc.append(g_val_epoch_acc)
                    client3_g_val_loss.append(g_val_epoch_loss)
                    client3_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client3_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 4:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed, loss_fn, folder, C4_EPL,
                        C4T)
                    client4_g_val_acc.append(g_val_epoch_acc)
                    client4_g_val_loss.append(g_val_epoch_loss)
                    client4_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client4_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 5:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed, loss_fn, folder, C5_EPL,
                        C5T)
                    client5_g_val_acc.append(g_val_epoch_acc)
                    client5_g_val_loss.append(g_val_epoch_loss)
                    client5_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client5_g_val_iounoback.append(g_val_epoch_iounoback)

                print(
                    f"Global Validating dice loss: {g_val_epoch_loss:.3f}, Global Validating accuracy: {g_val_epoch_acc:.3f},Global Validating iou Score with background: {g_val_epoch_iouwithback:.3f},Global Validating iou Score without background: {g_val_epoch_iounoback:.3f}")
                print("\n Global Validating IoUs Client:", cl_idx)
                print("GV: Background:", g_valepoch_iou_class0)
                print("GV: Tumor:", g_valepoch_iou_class1)

        tot_gloss = client1_g_val_loss[0] + client2_g_val_loss[0] + client3_g_val_loss[0] + client4_g_val_loss[0] + \
                    client5_g_val_loss[0]
        avg_g_val_loss = tot_gloss / 5;

        if least_lossg > avg_g_val_loss:
            least_lossg = avg_g_val_loss
            best_epoch = epoch
            torch.save(global_model1_fed.state_dict(),
                       parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/M1_globalcheckpoint.pth')
            torch.save(global_model2_fed.state_dict(),
                       parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/M2_globalcheckpoint.pth')
            torch.save(global_model3_fed.state_dict(),
                       parentF + 'Baseline/noisycom/Fed_Avg/Checkpoints/M3_globalcheckpoint.pth')
            print('Global best model saved')
            print('-' * 50)

        # ------------------------------------------TESTING USING THE GLOBAL MODEL-----------------------------------------------------------------------

        test_folder = parentF + "/Baseline/noisycom/Fed_Avg/testingsaved"

        for epoch in range(args.val_global_ep):
            print("Global testing.........")
            test_epoch_loss, test_epoch_acc, test_epoch_accwithback, test_epoch_accnoback = test(test_loader,
                                                                                                 global_model1_fed,
                                                                                                 global_model2_fed,
                                                                                                 global_model3_fed,
                                                                                                 loss_fn, test_folder)
            print('\n')
            print(
                f"Testing dice loss: {test_epoch_loss:.3f}, Testing accuracy: {test_epoch_acc:.3f},Testing iou Score with background: {test_epoch_accwithback:.3f},Testing iou Score without background: {test_epoch_accnoback:.3f}")
            test_Acc.append(test_epoch_acc)
            test_Iou_withback.append(test_epoch_accwithback)
            test_Iou_noback.append(test_epoch_accnoback)
            test_Loss.append(test_epoch_loss)

        # Training time per each client--------------------------------------
        print("Each client's cumulative training time")
        print("C1 cum time:", C1time)
        print("C2 cum time:", C2time)
        print("C3 cum time:", C3time)
        print("C4 cum time:", C4time)
        print("C5 cum time:", C5time)

        # -------------------------------------------------PLOTTING RESULTS-----------------------------------------------------------------------

        # local training accuracy plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_acc, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_acc, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_acc, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_acc, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_acc, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/train accuracy.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_loss, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_loss, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_loss, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_loss, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_loss, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/train Dice loss.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_withbackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_withbackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_withbackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_withbackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_withbackiou, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/train iou with background.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_nobackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_nobackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_nobackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_nobackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_nobackiou, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train IoU: Without Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/train iou without background.jpg')

        # -----------------------------------------------------------------------------------------------------------------------
        # local validation accuracy plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_acc, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_acc, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_acc, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_acc, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_acc, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('val Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/val accuracy.jpg')

        # global validation dice loss plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_loss, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_loss, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_loss, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_loss, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_loss, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('val Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/val Dice loss.jpg')

        # global validation iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_withbackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_withbackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_withbackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_withbackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_withbackiou, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Validation IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/val iou with background.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_nobackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_nobackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_nobackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_nobackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_nobackiou, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Validation IoU: Without Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/val iou without background.jpg')

        # -----------------------------------------------------------------------------------------------------------------------
        # global validation accuracy plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_acc, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_acc, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_acc, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_acc, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_acc, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/global val accuracy.jpg')

        # global validation dice loss plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_loss, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_loss, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_loss, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_loss, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_loss, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/global val Dice loss.jpg')

        # global validation iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_iouwithback, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_iouwithback, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_iouwithback, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_iouwithback, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_iouwithback, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/global val iou with background.jpg')

        # global validation iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_iounoback, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_iounoback, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_iounoback, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_iounoback, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_iounoback, color='black', linestyle='-', label='Client 5')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val IoU: Without Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/global val iou without background.jpg')

        # -------------------------------------------------------------------------------------

        # Global model testing performance
        plt.figure(figsize=(20, 5))
        plt.plot(test_Acc, color='black', linestyle='-', label='Global testing accuracy')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/Testing accuracy.jpg')

        # Global model testing dice loss plots
        plt.figure(figsize=(20, 5))
        plt.plot(test_Loss, color='black', linestyle='-', label='Global testing loss')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/Testing Dice loss.jpg')

        # Global model testing iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(test_Iou_withback, color='black', linestyle='-', label='Global testing IoU')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing IoU: With background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/Testing iou withbackground.jpg')

        # Global model testing iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(test_Iou_noback, color='black', linestyle='-', label='Global testing IoU')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing IoU: Without background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/Testing iou without background.jpg')

        alltest_acc, alltest_iouwithback, alltest_iounoback, alltest_loss = [], [], [], []
        alltest_acc.append(test_Acc)
        alltest_loss.append(test_Loss)
        alltest_iouwithback.append(test_Iou_withback)
        alltest_iounoback.append(test_Iou_noback)
        alltest_acc = pd.DataFrame(alltest_acc)
        alltest_loss = pd.DataFrame(alltest_loss)
        alltest_iouwithback = pd.DataFrame(alltest_iouwithback)
        alltest_iounoback = pd.DataFrame(alltest_iounoback)

        alltest_acc.to_csv(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/alltest_acc.csv')
        alltest_loss.to_csv(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/alltest_loss.csv')
        alltest_iouwithback.to_csv(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/alltest_iouwithback.csv')
        alltest_iounoback.to_csv(parentF + 'Baseline/noisycom/Fed_Avg/Outputs/alltest_iouwithoutback.csv')

        # -------------------------------------------------------------------------------------

    print('TRAINING COMPLETE')
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


if __name__ == "__main__":
    main()
