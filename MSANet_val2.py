import os
import time
import datetime
import numpy as np
import torch
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(666)
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from src.fcnconvnext import FcnNet
from src.model.xin.MSANet import FcnNet
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# from src.swintransformer_model import fcn_transformer
from train_utils.train_and_eval_dc22cledgeiou import train_one_epoch, evaluate, create_lr_scheduler
from dataset.my_dataset_irl384gaiedge import CasiaSegmentation,CasiaSegmentation2,CasiaSegmentation3,CasiaSegmentation4,CasiaSegmentation5,CasiaSegmentation6,CasiaSegmentation7,CasiaSegmentation8



# class SegmentationPresetTrain:
#     def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
#                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         min_size = 512
#         max_size = 512
#
#         trans = [T.RandomResize(min_size, max_size)]
#         if hflip_prob > 0:
#             trans.append(T.RandomHorizontalFlip(hflip_prob))
#         if vflip_prob > 0:
#             trans.append(T.RandomVerticalFlip(vflip_prob))
#         trans.extend([
#             # T.RandomCrop(crop_size),
#             T.colorfilter(),
#             # A.OneOf([
#             #     A.VerticalFlip(p=0.5),
#             #     A.RandomRotate90(p=0.5),
#             #     A.RandomBrightnessContrast(p=0.5),
#             #     A.HueSaturationValue(p=0.5),
#             #     A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
#             #     # A.CoarseDropout(p=0.2),
#             #     A.Transpose(p=0.5)
#             # ]),
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#         ])
#         self.transforms = T.Compose(trans)
#
#     def __call__(self, img, target):
#         return self.transforms(img, target)

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = 256
        max_size = 256

        self.transformspre = A.Compose([
            # reszie
            A.Resize(384, 384),
            # # A.RandomCrop(256, 256),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            # A.OneOf([
            #     A.RandomContrast(),
            #     A.RandomGamma(),
            #     A.RandomBrightness(),
            # ], p=0.5),
            # A.OneOf([
            #     A.MotionBlur(blur_limit=5),
            #     A.MedianBlur(blur_limit=5),
            #     A.GaussianBlur(blur_limit=5),
            #     A.GaussNoise(var_limit=(5.0, 20.0)),
            # ], p=0.5),
            # A.Resize(288, 288),
            # A.RandomCrop(256, 256),
            # A.RandomRotate90(p=0.5),
            # A.Transpose(p=0.5),
        ], )

        self.transforms1 = A.Compose([
            # reszie

            # A.HorizontalFlip(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),

        ], )

        self.transforms2 =  A.Compose([
            # reszie

            A.HorizontalFlip(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),

        ],)
        self.transforms3 = A.Compose([
            # reszie

            # A.VerticalFlip(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],)



    def __call__(self, img, target):
        out=self.transformspre(image=np.array(img),mask=np.array(target))
        out1= self.transforms1(image=out['image'],mask=out['mask'])
        out2 =self.transforms2(image=out['image'],mask=out['mask'])
        out3 =self.transforms3(image=out['image'],mask=out['mask'])
        img1=out1['image']
        img2=out2['image']
        img3=out3['image']
        # ycrcb=out['image4']
        mask=out1['mask']
        return (img1,img2,img3,mask)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.transforms1 = A.Compose([
            # reszie
            A.Resize(384, 384),
            # A.HorizontalFlip(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),

        ], )

        self.transforms2 = A.Compose([
            # reszie
            A.Resize(384, 384),
            A.HorizontalFlip(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),

        ], )
        self.transforms3 = A.Compose([
            # reszie
            A.Resize(384, 384),
            # A.VerticalFlip(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], )


        # self.transforms2 = A.Compose([
        #     # reszie
        #     # A.Resize(256, 256),
        #     A.Normalize(mean=mean, std=std),
        #     ToTensorV2(),
        #
        # ])



    def __call__(self, img,target,h,w):
        # h,w= img2.size
        out1 = self.transforms1(image=np.array(img), mask=np.array(target))
        out2 = self.transforms2(image=np.array(img), mask=np.array(target))
        out3 = self.transforms3(image=np.array(img), mask=np.array(target))
        img1 = out1['image']
        img2 = out2['image']
        img3 = out3['image']
        # ycrcb=out['image4']
        mask = out1['mask']
        # target = out['mask']
        # out=out+(h,)
        # out=out+(w,)
        return (img1,img2,img3,mask,h,w)

def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 512
    crop_size = 512

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


# def create_model(num_classes):
#     model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
#     return model
def create_model(aux, num_classes, pretrain=False):
    # model = deeplabv3_convnext(aux=aux, num_classes=num_classes,pretrain_backbone=True)
    # model=fcn_transformer(num_classes=1,pretrain_backbone=True)
    # model = C_UDNet(num_classes=num_classes, pretrain_backbone=True)
    model = FcnNet(num_classes=num_classes, pretrain_backbone=True)
    if pretrain:
        weights_dict = torch.load("./deeplabv3_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes

    # using compute_mean_std.py
    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)
    # mean = (0.442, 0.439, 0.395)
    # std = (0.231, 0.224, 0.240)

    # 用来保存训练以及验证过程中信息
    results_file = "fcnscresults{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # train_dataset = CasiaSegmentation(transforms=get_transform(train=True, mean=mean, std=std))
    #
    # val_dataset1 = CasiaSegmentation2(transforms=get_transform(train=False, mean=mean, std=std))
    # val_dataset2 = CasiaSegmentation3(transforms=get_transform(train=False, mean=mean, std=std))
    # val_dataset3 = CasiaSegmentation4(transforms=get_transform(train=False, mean=mean, std=std))
    train_dataset = CasiaSegmentation(transforms=get_transform(train=True))

    val_dataset1 = CasiaSegmentation2(transforms=get_transform(train=False))
    val_dataset2 = CasiaSegmentation3(transforms=get_transform(train=False))
    val_dataset3 = CasiaSegmentation4(transforms=get_transform(train=False))
    val_dataset4 = CasiaSegmentation5(transforms=get_transform(train=False))
    val_dataset5 = CasiaSegmentation6(transforms=get_transform(train=False))
    val_dataset6 = CasiaSegmentation7(transforms=get_transform(train=False))
    val_dataset7 = CasiaSegmentation8(transforms=get_transform(train=False))
    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    num_workers = 16

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               worker_init_fn=seed_worker,
                                               collate_fn=train_dataset.collate_fn)

    val_loader1 = torch.utils.data.DataLoader(val_dataset1,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              collate_fn=val_dataset1.collate_fn)

    val_loader2 = torch.utils.data.DataLoader(val_dataset2,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              collate_fn=val_dataset2.collate_fn)

    val_loader3 = torch.utils.data.DataLoader(val_dataset3,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              collate_fn=val_dataset3.collate_fn)
    val_loader4 = torch.utils.data.DataLoader(val_dataset4,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              collate_fn=val_dataset4.collate_fn)
    val_loader5 = torch.utils.data.DataLoader(val_dataset5,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              collate_fn=val_dataset5.collate_fn)
    val_loader6 = torch.utils.data.DataLoader(val_dataset6,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              collate_fn=val_dataset6.collate_fn)
    val_loader7 = torch.utils.data.DataLoader(val_dataset7,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              collate_fn=val_dataset7.collate_fn)

    model = create_model(num_classes=num_classes,aux=args.aux)
    model.to(device)

    # params_to_optimize = [
    #     {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
    #     {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    # ]
    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]}
    ]
    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    # )
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=5E-2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=False)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1
        # if args.amp:
        #     scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    best_dice2 = 0.
    best_dice3 = 0.
    best_eval = 0.
    best_eval2 = 0.
    start_time = time.time()
    setup_seed(666)
    save_file = {"model": model.state_dict(),
                 }
    # torch.save(save_file, "/root/data/IRLpeng2/pretrain_weight/zoomnofrecl03zz23.pth")
    for epoch in range(args.start_epoch, args.epochs):
        # mean_loss, lr,meanf1 = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
        #                                 lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # print(meanf1)
        # f1 = evaluate(model, val_loader2, device=device, num_classes=num_classes)
        # print(f"f1: {f1:.5f}")

        # if epoch >= 29:
        # f1,ap,iou = evaluate(model, val_loader2, device=device, num_classes=num_classes)
        # print('test')
        # print(f"f1: {f1:.5f}")
        # print(f"ap: {ap:.5f}")
        # print(f"iou: {iou:.5f}")
        # f1, ap, iou = evaluate(model, val_loader3, device=device, num_classes=num_classes)
        # print('hadobe5k')
        # print(f"f1: {f1:.5f}")
        # print(f"ap: {ap:.5f}")
        # print(f"iou: {iou:.5f}")
        # f1, ap, iou = evaluate(model, val_loader4, device=device, num_classes=num_classes)
        # print('HCOCO')
        # print(f"f1: {f1:.5f}")
        # print(f"ap: {ap:.5f}")
        # print(f"iou: {iou:.5f}")
        # f1, ap, iou = evaluate(model, val_loader5, device=device, num_classes=num_classes)
        # print('hday2night')
        # print(f"f1: {f1:.5f}")
        # print(f"ap: {ap:.5f}")
        # print(f"iou: {iou:.5f}")
        # f1, ap, iou = evaluate(model, val_loader6, device=device, num_classes=num_classes)
        # print('HFlickr')
        # print(f"f1: {f1:.5f}")
        # print(f"ap: {ap:.5f}")
        # print(f"iou: {iou:.5f}")

        f1, ap, iou = evaluate(model, val_loader7, device=device, num_classes=num_classes)
        print('mul')
        print(f"f1: {f1:.5f}")
        print(f"ap: {ap:.5f}")
        print(f"iou: {iou:.5f}")
        # # f2 = evaluate(model, val_loader2, device=device, num_classes=num_classes)
        # # print(f"f1: {f2:.3f}")
        # # # print(f'auc: {auc:.3f}')
        # f3 = evaluate(model, val_loader3, device=device, num_classes=num_classes)
        # print(f"f1: {f3:.3f}")
        # ave= (f1+f2)/2
        # print(f"eval: {ave:.3f}")
        # ave2= (f1+f2+f3)/2
        # print(f"eval2: {ave2:.3f}")
        # write into txt
        # with open(results_file, "a") as f:
        #     # 记录每个epoch对应的train_loss、lr以及验证集各指标
        #     train_info = f"[epoch: {epoch}]\n" \
        #                  f"train_loss: {mean_loss:.4f}\n" \
        #                  f"lr: {lr:.6f}\n" \
        #                  f"casiaf1: {f1:.3f}\n" \
        #                  f"coverf1: {f2:.3f}\n" \
        #                  f"colunmbiaf1: {f3:.3f}\n"
        #     f.write(train_info  + "\n\n")

        save_file = {"model": model.state_dict(),
                }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # if args.save_best is True:
        #     if best_dice < f1:
        #         best_dice = f1
        #         torch.save(save_file, "save_weights_fcnnoisefredct13/best_model1.pth")
        #     if best_dice2 < f2:
        #         best_dice2 = f2
        #         torch.save(save_file, "save_weights_fcnnoisefredct13/best_model2.pth")
        #     if best_dice3 < f3:
        #         best_dice3 = f3
        #         torch.save(save_file, "save_weights_fcnnoisefredct13/best_model3.pth")
        #     if best_eval < ave:
        #         best_eval = ave
        #         torch.save(save_file, "save_weights_fcnnoisefredct13/best_modeleval.pth")
        #     if best_eval2 < ave2:
        #         best_eval2 = ave2
        #         torch.save(save_file, "save_weights_fcnnoisefredct13/best_modeleval2.pth")
        #
        #     else:
        #         continue


        #
        # if args.save_best is True:
        #     torch.save(save_file, "save_weights_convext_du3/best_model.pth")
        # else:
        #     torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
        # torch.save(save_file, "dd/model-{}.pth".format(epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import argparse

    parser = argparse.ArgumentParser(description="pytorch u-net training")

    # parser.add_argument("--data-path", default="/disk/csh/tianchi/tianchidata/", help="DRIVE root")
    # parser.add_argument("--data-path", default="/disk1/csh/tianchi/tianchidata/", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    # parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    # parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=35, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default='/raid/csh/peng555/duo23/model-34.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    import random

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(666)
    args = parse_args()

    if not os.path.exists("./dd"):
        os.mkdir("./dd")

    main(args)