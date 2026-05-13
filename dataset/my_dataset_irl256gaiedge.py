import os

import torch.utils.data as data
from PIL import Image
import csv
import numpy as np
import torch
import random
from scipy.ndimage.morphology import distance_transform_edt
def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.float32)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        # dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.float32)
    return edgemap

def rand_bbox(size):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat_w = random.random() * 0.2 + 0.05
    cut_rat_h = random.random() * 0.2 + 0.05

    cut_w = int(W * cut_rat_w)
    cut_h = int(H * cut_rat_h)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def copy_move(img: np.array, msk: np.array):


    img = torch.from_numpy(img).permute(2, 0, 1)
    msk = torch.from_numpy(msk)
    size = img.size()
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception


    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size())

    x_move = random.randrange(-bbx1, (W - bbx2))
    y_move = random.randrange(-bby1, (H - bby2))

    img[:, bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = img[:, bbx1:bbx2, bby1:bby2]
    try:
        msk[bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = torch.ones_like(msk[bbx1:bbx2, bby1:bby2])
    except:
        print('aaa')
    # img = cv2.rectangle(img.numpy().transpose(1, 2, 0), pt1=(bby1 + y_move, bbx1 + x_move),
    #                     pt2=(bby2 + y_move, bbx2 + x_move), color=(255, 0, 0), thickness=5)
    img = img.numpy()
    img = np.transpose(img, axes=[1, 2, 0])

    img = Image.fromarray(img)
    msk= msk.numpy()# 自动转换为0-255
    msk=Image.fromarray(msk)
    return img, msk


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)



class CasiaSegmentation(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation, self).__init__()
        # self.image_base = '/raid/csh/segdataset/CASIA 2.0/Tp'
        # self.mask_base = '/raid/csh/segdataset/casia2groundtruth-master'
        # self.image_list = []
        # self.mask_list=[]
        # with open(os.path.join(self.mask_base, 'Notes', 'modify_pair_list.txt')) as f:
        #     reader = csv.reader(f)
        #
        #     for row in reader:
        #         image_realpath = os.path.join(self.image_base, row[0])
        #         mask_realpath = os.path.join(self.mask_base, 'CASIA 2 Groundtruth', row[1])
        #         self.image_list.append(image_realpath)
        #         self.mask_list.append(mask_realpath)
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/sydata/le50_train2.txt').readlines())
        data = []
        for line in lines:
            temp = line.split('/')

            datafile,_ , name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line)
            # if line == 'Hday2night/composite_images/d21901-20130410-181833_1_9.jpg':
            #     print('aa')
            names=name.split('_')
            name1,name2,_=names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + datafile+'/masks/'+name1+'_'+name2+'.png')

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        # if img.size!=target.size:
        #     print('aaaaa')
        target = np.array(target) / 255
        target[target>=0.5]=1
        target[target<0.5]=0
        target = Image.fromarray(target)
        if self.transforms is not None:

            # if random.random() < 0.5 and img.size==target.size:
            # # if img.size == target.size:
            #     img = np.array(img)
            #     target = np.array(target)
            #     img, target = copy_move(img, target)



            img1,img2,img3,target = self.transforms(img, target)

            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask=edgemask.squeeze(0)
            # edgemask = edgemask.unsqueeze(0)
            # edgemask = F.interpolate(edgemask, size=(128, 128))
            # edgemask = edgemask.squeeze(0)
            a=np.unique(target)
            if str(a)!='[0. 1.]':
                print(a)

        return img1,img2,img3,target,edgemask

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets,edgemasks = list(zip(*batch))
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edges = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets,batched_edges


class CasiaSegmentation2(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation2, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/sydata/le50_val.txt').readlines())
        data = []
        for line in lines:
            temp = line.split('/')

            datafile, _, name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line)
            names = name.split('_')
            name1, name2, _ = names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + datafile + '/masks/' + name1 + '_' + name2 + '.png')

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        target = np.array(target) / 255

        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            # target = target.resize((256, 256), Image.NEAREST)
            img1,img2,img3, target, h, w = self.transforms(img, target, 256, 256)
            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask = edgemask.squeeze(0)

        return img1,img2,img3, target, h, w,edgemask


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w ,edgemasks= list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3,  batched_targets, h, w,batched_edgemasks


class CasiaSegmentation3(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation3, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/sydata/le50_test.txt').readlines())
        data = []
        for line in lines:
            temp = line.split('/')

            datafile, _, name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line)
            names = name.split('_')
            name1, name2, _ = names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + datafile + '/masks/' + name1 + '_' + name2 + '.png')

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # if self.image_list[index]=='/raid/csh/segdataset/sydata/HAdobe5k/composite_images/a3364_1_1.jpg':
        #     print('zhaodao')
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        if img.size!=target.size:
            img = img.resize(target.size)
        target = np.array(target) / 255
        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            target = target.resize((256, 256), Image.NEAREST)
            img1, img2, img3, target, h, w = self.transforms(img, target, 256, 256)
            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask = edgemask.squeeze(0)
        return img1, img2, img3, target, h, w,edgemask


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w, edgemasks = list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets, h, w, batched_edgemasks

class CasiaSegmentation4(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation4, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/sydata/HCOCOt.txt').readlines())
        data = []
        for line in lines:
            temp = line.split('/')

            datafile, _, name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line)
            names = name.split('_')
            name1, name2, _ = names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + datafile + '/masks/' + name1 + '_' + name2 + '.png')

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        if img.size != target.size:
            img = img.resize(target.size)
        target = np.array(target) / 255
        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            target = target.resize((256, 256), Image.NEAREST)
            img1, img2, img3, target, h, w = self.transforms(img, target, 256, 256)
            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask = edgemask.squeeze(0)
        return img1, img2, img3, target, h, w, edgemask


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w, edgemasks = list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets, h, w, batched_edgemasks

class CasiaSegmentation5(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation5, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/sydata/hadobe5kt.txt').readlines())
        data = []
        for line in lines:
            temp = line.split('/')

            datafile, _, name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line)
            names = name.split('_')
            name1, name2, _ = names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + datafile + '/masks/' + name1 + '_' + name2 + '.png')

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        if img.size != target.size:
            img = img.resize(target.size)
        target = np.array(target) / 255
        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            target = target.resize((256, 256), Image.NEAREST)
            img1, img2, img3, target, h, w = self.transforms(img, target, 256, 256)
            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask = edgemask.squeeze(0)
        return img1, img2, img3, target, h, w, edgemask


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w, edgemasks = list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets, h, w, batched_edgemasks

class CasiaSegmentation6(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation6, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/sydata/hday2nightt.txt').readlines())
        data = []
        for line in lines:
            temp = line.split('/')

            datafile, _, name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line)
            names = name.split('_')
            name1, name2, _ = names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + datafile + '/masks/' + name1 + '_' + name2 + '.png')

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        if img.size != target.size:
            img = img.resize(target.size)
        target = np.array(target) / 255
        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            target = target.resize((256, 256), Image.NEAREST)
            img1, img2, img3, target, h, w = self.transforms(img, target, 256, 256)
            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask = edgemask.squeeze(0)
        return img1, img2, img3, target, h, w, edgemask


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w, edgemasks = list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets, h, w, batched_edgemasks

class CasiaSegmentation7(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation7, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/sydata/HFlickrt.txt').readlines())
        data = []
        for line in lines:
            temp = line.split('/')

            datafile, _, name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line)
            names = name.split('_')
            name1, name2, _ = names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + datafile + '/masks/' + name1 + '_' + name2 + '.png')

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        if img.size != target.size:
            img = img.resize(target.size)
        target = np.array(target) / 255
        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            target = target.resize((256, 256), Image.NEAREST)
            img1, img2, img3, target, h, w = self.transforms(img, target, 256, 256)
            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask = edgemask.squeeze(0)
        return img1, img2, img3, target, h, w, edgemask


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w, edgemasks = list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets, h, w, batched_edgemasks


class CasiaSegmentation8(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentation8, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines1 = map(str.strip, open('/raid/csh/segdataset/sydata/mulimage.txt').readlines())
        lines2 = map(str.strip, open('/raid/csh/segdataset/sydata/mulmask.txt').readlines())
        data = []
        for line1,line2 in zip(lines1,lines2):
            temp = line1.split('/')

            # datafile, _, name = temp

            self.image_list.append('/raid/csh/segdataset/sydata/' + line1)
            # names = name.split('_')
            # name1, name2, _ = names
            self.mask_list.append('/raid/csh/segdataset/sydata/' + line2)

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        target = Image.open(self.mask_list[index]).convert('L')
        if img.size != target.size:
            img = img.resize(target.size)
        target = np.array(target) / 255
        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            target = target.resize((256, 256), Image.NEAREST)
            img1, img2, img3, target, h, w = self.transforms(img, target, 256, 256)
            _edgemap = np.array(target).astype(np.float32)
            _edgemap = mask_to_onehot(_edgemap, 2)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)
            edgemask = torch.from_numpy(_edgemap).float()
            edgemask = edgemask.squeeze(0)
        return img1, img2, img3, target, h, w, edgemask


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w, edgemasks = list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets, h, w, batched_edgemasks


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



class CasiaSegmentationnisttrain(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentationnisttrain, self).__init__()
        # self.image_base = '/raid/csh/segdataset/CASIA 2.0/Tp'
        # self.mask_base = '/raid/csh/segdataset/casia2groundtruth-master'
        # self.image_list = []
        # self.mask_list=[]
        # with open(os.path.join(self.mask_base, 'Notes', 'modify_pair_list.txt')) as f:
        #     reader = csv.reader(f)
        #
        #     for row in reader:
        #         image_realpath = os.path.join(self.image_base, row[0])
        #         mask_realpath = os.path.join(self.mask_base, 'CASIA 2 Groundtruth', row[1])
        #         self.image_list.append(image_realpath)
        #         self.mask_list.append(mask_realpath)
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/mvss_dataset/nisttrain16.txt').readlines())
        data = []
        for line in lines:
            temp = line.split()

            sample_path, mask_path, label = temp
            label = int(int(label) > 0)
            self.image_list.append('/raid/csh/segdataset/mvss_dataset' + sample_path[1:])
            self.mask_list.append('/raid/csh/segdataset/mvss_dataset' + mask_path[1:])

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')
        img_dct = np.array(img).copy()
        target = Image.open(self.mask_list[index]).convert('L')
        # if img.size!=target.size:
        #     print('aaaaa')
        target = np.array(target) / 255
        target[target>=0.5]=1
        target[target<0.5]=0
        target = Image.fromarray(target)
        if self.transforms is not None:

            if random.random() < 0.5 and img.size==target.size:
            # if img.size == target.size:
                img = np.array(img)

                target = np.array(target)
                img, target = copy_move(img, target)



            img, target = self.transforms(img, target)

            # transform2 = td.Compose([
            #     td.Resize((1024,1024)),
            #     td.RandomHorizontalFlip(),
            #     td.Upscale(upscale_factor=2),
            #     td.TransformUpscaledDCT(),
            #     td.ToTensorDCT(),
            #     td.SubsetDCT(channels='48'),
            #     td.Aggregate(),
            #     td.NormalizeDCT(
            #         td.train_upscaled_static_mean,
            #         td.train_upscaled_static_std,
            #         channels='48'
            #     )
            # ])
            # img_dct,_,_=transform2(img_dct)

            a=np.unique(target)
            if str(a)!='[0. 1.]':
                print(a)

        return img, target

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images1, images2, images3, targets, h, w, edgemasks = list(zip(*batch))
        # batched_imgs = cat_list(images, fill_value=0)
        batched_imgs1 = cat_list(images1, fill_value=0)
        batched_imgs2 = cat_list(images2, fill_value=0)
        batched_imgs3 = cat_list(images3, fill_value=0)
        # batched_ycrcbs = cat_list(ycrcbs, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_edgemasks = cat_list(edgemasks, fill_value=255)
        return batched_imgs1, batched_imgs2, batched_imgs3, batched_targets, h, w, batched_edgemasks
class CasiaSegmentationnisttest(data.Dataset):
    def __init__(self,  transforms=None, txt_name: str = "train.txt"):
        super(CasiaSegmentationnisttest, self).__init__()
        self.image_list = []
        # 掩码列表
        self.mask_list = []
        lines = map(str.strip, open('/raid/csh/segdataset/mvss_dataset/nisttest16.txt').readlines())
        data = []
        for line in lines:
            temp = line.split()

            sample_path, mask_path, label = temp
            label = int(int(label) > 0)
            self.image_list.append('/raid/csh/segdataset/mvss_dataset' + sample_path[1:])
            self.mask_list.append('/raid/csh/segdataset/mvss_dataset' + mask_path[1:])
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_list[index]).convert('RGB')

        target = Image.open(self.mask_list[index]).convert('L')
        if img.size!=target.size:
            img = img.resize(target.size)
        target = np.array(target) / 255
        target = Image.fromarray(target)
        h, w = img.size
        if self.transforms is not None:
            img, target, h, w = self.transforms(img, target, h, w)
        # transform2 = td.Compose([
        #     td.Resize((1024, 1024)),
        #     # td.RandomHorizontalFlip(),
        #     td.Upscale(upscale_factor=2),
        #     td.TransformUpscaledDCT(),
        #     td.ToTensorDCT(),
        #     td.SubsetDCT(channels='48'),
        #     td.Aggregate(),
        #     td.NormalizeDCT(
        #         td.train_upscaled_static_mean,
        #         td.train_upscaled_static_std,
        #         channels='48'
        #     )
        # ])
        # img_dct, _, _ = transform2(img_dct)
        return img, target, h, w


    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images, targets,h,w = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)

        return batched_imgs, batched_targets,h,w


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
