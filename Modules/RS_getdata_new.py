import os
import torch.utils.data as data
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import copy
import cv2 as cv

IMAGE_SIZE = 256

dataTransform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


class RS_dataset(data.Dataset):
    def __init__(self, dir, transform_change=False, transform=dataTransform, image_size=256, part=None):
        self.list_img = []
        self.list_label = []
        self.list_img_now = []
        self.list_label_now = []
        self.data_size = 0
        self.img_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        self.transform_change = transform_change
        self.part = part
        if self.transform_change:
            self.transform = transform
        for folder in os.listdir(dir):
            if self.part is not None:
                for images in os.listdir(dir + '/' + folder):
                    folder_index = int(folder) - 1
                    self.list_img_now.append(dir + '/' + folder + '/' + images)
                    self.list_label_now.append(folder_index)
                all_len = len(self.list_img_now)
                part_len = all_len / self.part
                part_len = int(part_len)
                start_pos_upper_limit = all_len - part_len
                start_pos = random.randint(0, start_pos_upper_limit)
                self.list_img_now = self.list_img_now[start_pos:start_pos + part_len]
                self.list_label_now = self.list_label_now[start_pos:start_pos + part_len]
                self.list_img += self.list_img_now
                self.list_label += self.list_label_now
                self.data_size += len(self.list_img_now)
                self.list_img_now = []
                self.list_label_now = []
            else:
                for images in os.listdir(dir + '/' + folder):
                    self.list_img.append(dir + '/' + folder + '/' + images)
                    folder_index = int(folder) - 1
                    self.data_size += 1
                    self.list_label.append(folder_index)

    def __getitem__(self, item):
        img = Image.open(self.list_img[item])
        label = self.list_label[item]
        return self.transform(img), torch.LongTensor([label])

    def __len__(self):
        return self.data_size

class RS_dataset_trinity(data.Dataset):
    def __init__(self, dir, transform_change=True, transform=dataTransform, image_size=256,
                 part=None):
        self.list_img = []
        self.list_label = []
        self.list_img_now = []
        self.list_label_now = []
        self.data_size = 0
        self.img_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        self.transform_change = transform_change
        self.part = part
        if self.transform_change:
            self.transform = transform
        dir_list = [_ for _ in os.listdir(dir)]  # 01~15
        img = []
        for i in dir_list:
            dir_1 = dir + '/' + i  # i is floder (01 for instance)
            if self.part is not None:
                for img_now_1 in os.listdir(dir_1):
                    img_now_1_idx = img_now_1

                    img_now_1 = dir_1 + '/' + img_now_1
                    img.append(img_now_1)

                    dir_1_list = os.listdir(dir_1)  # all imgs index in one list
                    dir_1_list.remove(img_now_1_idx)  # all imgs index except img_now_1 in one list
                    dir_1_list_remove = dir_1_list  # all imgs index except img_now_1 in one list
                    img_now_2 = random.choice(dir_1_list_remove)  # img index NOT FULLY DIR
                    img_now_2 = dir_1 + '/' + img_now_2  # fully directory
                    img.append(img_now_2)

                    dir_remove = copy.copy(dir_list)  # 01~15
                    dir_remove.remove(i)  # 01~15 except i
                    dir_2 = random.choice(dir_remove)
                    dir_2 = dir + '/' + dir_2
                    dir_2_list = os.listdir(dir_2)  # all imgs index in one list
                    img_now_3 = random.choice(dir_2_list)
                    img_now_3 = dir_2 + '/' + img_now_3
                    img.append(img_now_3)
                    self.list_img_now.append(img)
                    self.list_label_now.append(int(i) - 1)
                    img = []
                all_len = len(self.list_img_now)  #
                part_len = all_len / self.part  #
                part_len = int(part_len)
                start_pos_upper_limit = all_len - part_len  #
                start_pos = random.randint(0, start_pos_upper_limit)  #
                self.list_img_now = self.list_img_now[start_pos:start_pos + part_len]  #
                self.list_label_now = self.list_label_now[start_pos:start_pos + part_len]  #
                self.list_img += self.list_img_now
                self.list_label += self.list_label_now
                self.data_size += len(self.list_img_now)
                self.list_img_now = []
                self.list_label_now = []
            else:
                for img_now_1 in os.listdir(dir_1):
                    img_now_1_idx = img_now_1

                    img_now_1 = dir_1 + '/' + img_now_1
                    img.append(img_now_1)

                    dir_1_list = os.listdir(dir_1)  # all imgs index in one list
                    dir_1_list.remove(img_now_1_idx)  # all imgs index except img_now_1 in one list
                    dir_1_list_remove = dir_1_list  # all imgs index except img_now_1 in one list
                    img_now_2 = random.choice(dir_1_list_remove)  # img index NOT FULLY DIR
                    img_now_2 = dir_1 + '/' + img_now_2  # fully directory
                    img.append(img_now_2)

                    dir_remove = copy.copy(dir_list)  # 01~15
                    dir_remove.remove(i)  # 01~15 except i
                    dir_2 = random.choice(dir_remove)  #
                    dir_2 = dir + '/' + dir_2  #
                    dir_2_list = os.listdir(dir_2)  # all imgs index in one list
                    img_now_3 = random.choice(dir_2_list)  #
                    img_now_3 = dir_2 + '/' + img_now_3  #
                    img.append(img_now_3)
                    self.list_img.append(img)
                    self.data_size += 1
                    img = []
                    self.list_label.append(int(i) - 1)

    def __getitem__(self, item):
        img = self.list_img[item]
        img0 = Image.open(img[0])
        img0 = dataTransform(img0)
        img1 = Image.open(img[1])
        img1 = dataTransform(img1)
        img2 = Image.open(img[2])
        img2 = dataTransform(img2)
        img_trinity = torch.cat((img0, img1, img2), dim=0)  #9*224*224
        label = self.list_label[item]
        return img_trinity, torch.LongTensor([label])

    def __len__(self):
        return self.data_size


class RS_dataset_single_class(data.Dataset):
    def __init__(self, dir, transform_change=False, transform=dataTransform,
                 image_size=256):
        self.list_img = []
        self.list_label = []
        self.data_size = 0  #
        self.img_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        self.transform_change = transform_change
        if self.transform_change:
            self.transform = transform
        for images in os.listdir(dir):
            self.list_img.append(dir + '/' + images)
            dir_index = dir.split('/')[-1]
            folder_index = int(dir_index) - 1
            self.data_size += 1
            self.list_label.append(folder_index)

    def __getitem__(self, item):
        img = Image.open(self.list_img[item])
        label = self.list_label[item]
        return self.transform(img), torch.LongTensor([label])

    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    single_class_path = '/home/yrk/remote sensing dataset/Remote Sensing Object Detection/NWPU VHR-10 dataset/NWPU ' \
                        'VHR-10 dataset/NWPU_class_intersection/15 '
    single_class_dataset = RS_dataset_single_class(dir=single_class_path)

    img = single_class_dataset[123][0]
    label = single_class_dataset[123][1].item()
    img = img.numpy().transpose(1, 2, 0)
    label_dict = np.load('/home/yrk/remote sensing dataset/Remote Sensing Object Detection/label_dic.npy',
                         allow_pickle=True).item()
    label = label_dict['0' + str(label + 1) if label + 1 < 10 else str(label + 1)]
    cv.imshow('{}'.format(label), img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print('Single class img number is {}'.format(len(single_class_dataset)))
