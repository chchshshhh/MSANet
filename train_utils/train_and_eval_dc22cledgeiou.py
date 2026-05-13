import torch
from torch import nn
import train_utils.distributed_utils as utils
import torch.nn.functional as F
import numpy as np
import math
import os
import csv
import cv2
import torchvision.transforms as transforms
import torch_dct as DCT
import pytorch_ssim
# from loss.DenseContrastiveLossV2_ms import DenseContrastiveLossV2_ms as v2ms
from loss.DenseContrastiveLossV2_ms import DenseContrastiveLossV2_ms as v2ms
from sklearn.metrics import average_precision_score
transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])

def compute_IoU(pred, gt, threshold=0.5, eps=1e-6):
    pred = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred)).to(pred.device)
    intersection = (pred * gt).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3]) - intersection
    return (intersection / (union + eps)).mean().item()

def calculate_pixel_f1(pd, gt):
    # if np.max(pd) == np.max(gt) and np.max(pd) == 0:
    #     f1, iou = 1.0, 1.0
    #     return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    iou=true_pos/(true_pos+false_pos+false_neg+1e-6)
    return f1, precision, recall,iou
def criterion4(inputs, target):
    losses = {}
    target0=target.unsqueeze(1)
    ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()

    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充

        losses[name] = bcefLoss(x, target) + diceloss(x, target)+1 - ssim_loss(x,target0)

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +losses['x4']+losses['x3']+losses['x2']



def calculate_pixel_f1(pd, gt):
    # if np.max(pd) == np.max(gt) and np.max(pd) == 0:
    #     f1, iou = 1.0, 1.0
    #     return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    iou=true_pos/(true_pos+false_pos+false_neg+1e-6)
    return f1, precision, recall,iou

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets=targets.unsqueeze(1)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs.long(), targets.long(), reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



# def criterion(inputs, target):
#     losses = {}
#     for name, x in inputs.items():
#         # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
#         losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
#
#     if len(losses) == 1:
#         return losses['out']
#
#     return losses['out'] + 0.5 * losses['aux']

def criterion(inputs, target):
    losses = {}
    bcefLoss=FocalLoss()
    diceloss=SoftDiceLoss()

    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] = bcefLoss(x, target)+diceloss(x,target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def criterion3cl(inputs, target,edgemask):
    losses = {}
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()
    dcv2=v2ms()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name=='edge':
            losses[name]=bcefLoss(x, edgemask)+diceloss(x, edgemask)
        elif name!='feature':
            losses[name] = bcefLoss(x, target) + diceloss(x, target)
        else:
            losses[name]=0.3*(dcv2(target,x))

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +losses['x4']+losses['x3']+losses['x2']+losses['feature']+losses['edge']
def criterion3clz3(inputs, target,edgemask):
    losses = {}
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()
    dcv2=v2ms()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name=='edge' or name=='edge2':
            losses[name]=bcefLoss(x, edgemask)+diceloss(x, edgemask)
        elif name!='feature':
            losses[name] = bcefLoss(x, target) + diceloss(x, target)
        else:
            losses[name]=0.3*(dcv2(target,x))

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +losses['x4']+losses['x3']+losses['feature']+losses['edge']+losses['edge2']
def criterion3clgraph3(inputs, target,edgemask):
    losses = {}
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()
    dcv2=v2ms()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name=='edge':
            losses[name]=bcefLoss(x, edgemask)+diceloss(x, edgemask)
        elif name!='feature':
            losses[name] = bcefLoss(x, target) + diceloss(x, target)
        else:
            losses[name]=0.3*(dcv2(target,x))

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +losses['region']+losses['feature']+losses['edge']
def criterion3clz2(inputs, target,edgemask):
    losses = {}
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()
    dcv2=v2ms()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name=='edge':
            losses[name]=bcefLoss(x, edgemask)+diceloss(x, edgemask)
        elif name!='feature':
            losses[name] = bcefLoss(x, target) + diceloss(x, target)
        else:
            losses[name]=0.3*(dcv2(target,x))

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +losses['x3']+losses['feature']+losses['edge']

def criterion3clz4(inputs, target,edgemask):
    losses = {}
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()
    dcv2=v2ms()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name=='edge' or name=='edge2':
            losses[name]=bcefLoss(x, edgemask)+diceloss(x, edgemask)
        elif name!='feature':
            losses[name] = bcefLoss(x, target) + diceloss(x, target)
        else:
            losses[name]=0.3*(dcv2(target,x))

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +losses['x3']+losses['x4']+losses['feature']+losses['edge']+losses['edge2']
def criterion3clzgai(inputs, target,edgemask):
    losses = {}
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()
    dcv2=v2ms()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name=='x13':
            losses[name]=bcefLoss(x, edgemask)+diceloss(x, edgemask)
        elif name!='feature':
            losses[name] = bcefLoss(x, target) + diceloss(x, target)
        else:
            losses[name]=0.3*(dcv2(target,x))

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +losses['feature']+losses['x13']+losses['x24']
def criterion3cl2(inputs, target):
    losses = {}
    bcefLoss = FocalLoss()
    diceloss = SoftDiceLoss()
    dcv2=v2ms()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name!='feature':
            losses[name] = bcefLoss(x, target) + diceloss(x, target)
        else:
            losses[name]=0.3*(dcv2(target,x))

    # if len(losses) == 1:
        # return losses['out']

    return losses['out'] +0.3*(losses['x4']+losses['x3']+losses['x2'])+losses['feature']

def criterion2(inputs, target):
    losses = {}
    diceloss=SoftDiceLoss()
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] =diceloss(x,target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    # confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    f1s = []
    aps=[]
    ious=[]
    # image_list = []
    # # 掩码列表
    # mask_list = []
    # image_base = '/disk/csh/segdataset/CASIA 1.0 dataset'
    # mask_base = '/disk/csh/segdataset/casia1groundtruth-master/CASIA 1.0 groundtruth'
    # with open(os.path.join(mask_base, 'imgmask_pair_1.0.txt')) as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         img_realpath = os.path.join(image_base, 'Tp', row[0])
    #         mask_realpath = os.path.join(mask_base, row[1])
    #         image_list.append(img_realpath)
    #         mask_list.append(mask_realpath)
    #
    # i=0

    with torch.no_grad():
        for image1,image2,image3,target,h,w,edgemask in metric_logger.log_every(data_loader, 1000, header):
            image1,image2,image3,  target,edgemask = image1.to(device),image2.to(device),image3.to(device), target.to(device),edgemask.to(device)
            # num_batchsize = ycrcb.shape[0]
            # size = ycrcb.shape[2]
            target[target >= 0.5] = 1
            target[target < 0.5] = 0
            # ycbcr_image = ycrcb.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
            # ycbcr_image = DCT.dct_2d(ycbcr_image, norm='ortho')
            # ycbcr_image = ycbcr_image.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)

            # output1 = model(image,h[0],w[0])
            output1 = model(image1,image2,image3,h[0],w[0],t=False)
            output1 = output1['out']
            output1=F.sigmoid(output1)
            iouu = compute_IoU(output1, target.unsqueeze(1))


            # output2 = model(torch.flip(image, dims=[2]),h,w)
            # output2 = output2['out']
            # output2 = F.sigmoid(output2)
            # output2 = torch.flip(output2, dims=[2])
            #
            # output3 = model(torch.flip(image, dims=[3]),h,w)
            # output3 = output3['out']
            # output3 = F.sigmoid(output3)
            # output3 = torch.flip(output3, dims=[3])

            # output = model(image)
            # output = output['out']
            # output = F.sigmoid(output)
            # output = (output1 + output2 + output3) / 3.0
            # output=np.array(output)
            output=output1.data.cpu().numpy()


            # output = output1.data.cpu()
            #
            # seg=np.array(transform_pil(output[0][0]))
            # # seg = seg[0]
            # save_seg_path = os.path.join('/disk/csh/evaluator_onelast/scnoisee/CASIAv1/best_model1/pred', image_list[i][45:-4] + '.png')
            # os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
            # cv2.imwrite(save_seg_path, seg.astype(np.uint8))
            # # progbar.add(1, values=[('path', save_seg_path), ])
            # i=i+1
            #
            # output=output.numpy()
            target=target.cpu().numpy()
            ap = average_precision_score(target.flatten().astype('int'), output.flatten())
            output = (output > 0.5).astype(np.float).squeeze(1).astype('int64')

            f1, p, r,iou = calculate_pixel_f1(output.flatten(), target.flatten())
            z = f1
            z2 = ap
            z3 = iouu
            f1s.append(z)
            aps.append(z2)
            ious.append(z3)
        meanf1 = np.mean(f1s)
        meanap = np.mean(aps)
        meaniou = np.mean(ious)

        # confmat.update(target.flatten(), output.argmax(1).flatten())

    return meanf1, meanap, meaniou


# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#
#     for image, target in metric_logger.log_every(data_loader, print_freq, header):
#         image, target = image.to(device), target.to(device)
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             output = model(image)
#             loss = criterion(output, target)
#
#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()
#
#         lr_scheduler.step()
#
#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(loss=loss.item(), lr=lr)
#
#     return metric_logger.meters["loss"].global_avg, lr


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    f1t=[]

    for (image1,image2,image3, target,edgemask) in metric_logger.log_every(data_loader, print_freq, header):
        image1, image2, image3,  target,edgemask = image1.to(device), image2.to(device), image3.to(device), target.to(device),edgemask.to(device)
        # num_batchsize = ycrcb.shape[0]
        # size = ycrcb.shape[2]

        # ycbcr_image = ycrcb.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
        # ycbcr_image = DCT.dct_2d(ycbcr_image, norm='ortho')
        # ycbcr_image = ycbcr_image.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output= model(image1,image2,image3)
            # loss = criterion3clzgai(output, target,edgemask)
            loss = criterion3clz4(output, target,edgemask)
            # loss = criterion3cl(output, target, edgemask)
            # loss = criterion3cl2(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
        for i in range(output['out'].shape[0]):
            output11 = F.sigmoid(output['out'][i])
            output11 = output11.data.cpu().numpy()

            target11 = target[i].cpu().numpy()
            output11 = (output11 > 0.5).astype(np.float).astype('int64')

            f1, p, r, iou = calculate_pixel_f1(output11.flatten(), target11.flatten())
            z = f1
            f1t.append(z)
    meanf1 = np.mean(f1t)

    return metric_logger.meters["loss"].global_avg, lr,meanf1

def train_one_epoch_swin(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)



    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def create_lr_scheduler2(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-7):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



def create_lr_scheduler_multipoly(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            # return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
            current_step = (x - warmup_epochs * num_step)
            decay_steps = num_step + num_step * math.floor(current_step / num_step)
            return pow((1 - current_step / decay_steps), 0.9) *(1-1e-7)+ 1e-7


    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)