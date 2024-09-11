import torch
import torchvision
from dataset import EmbryoDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
DEVICE = "cuda"
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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


def eval_performance(loader, modelclientFE,modelserver,modelclientBE,loss_fn,folder):
    modelclientFE.eval()
    modelserver.eval()
    modelclientBE.eval()
    val_running_loss = 0.0
    valid_running_correct = 0.0
    valid_iou_score = 0.0
    valid_iou_score_class0 = 0.0
    valid_iou_score_class1 = 0.0
    valid_iou_score_class2 = 0.0
    valid_iou_score_class3 = 0.0
    valid_iou_score_class4 = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    with torch.no_grad():
        for idx,(x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.type(torch.LongTensor).to(device=DEVICE)
            predictions1 = modelclientFE(x)
            predictions2 = modelserver(predictions1)
            predictions3 = modelclientBE(predictions2)
            loss = loss_fn(predictions3, y)

            # calculate the testing accuracy
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            valid_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()

            #  Validation loss
            val_running_loss += loss.item()

            # iou score
            valid_iou_score += jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            valid_iou_score_class0 +=  iou_sklearn[0]
            valid_iou_score_class1 += iou_sklearn[1]
            valid_iou_score_class2 +=  iou_sklearn[2]
            valid_iou_score_class3 += iou_sklearn[3]
            valid_iou_score_class4 += iou_sklearn[4]

            #for t, p in zip(y.view(-1), preds.view(-1)):confusion_matrix[t.long(), p.long()] += 1

            torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0,scale_each=True,normalize=True)

    #print(confusion_matrix)
    epoch_loss = val_running_loss / len(loader.dataset)
    epoch_acc = 100. * (valid_running_correct / len(loader.dataset))
    epoch_iou = (valid_iou_score / len(loader.dataset))
    epoch_iou_class0 = (valid_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (valid_iou_score_class1 / len(loader.dataset))
    epoch_iou_class2 = (valid_iou_score_class2 / len(loader.dataset))
    epoch_iou_class3 = (valid_iou_score_class3 / len(loader.dataset))
    epoch_iou_class4 = (valid_iou_score_class4 / len(loader.dataset))
    return epoch_loss, epoch_acc, epoch_iou, epoch_iou_class0, epoch_iou_class1, epoch_iou_class2, epoch_iou_class3, epoch_iou_class4
    model.train()

