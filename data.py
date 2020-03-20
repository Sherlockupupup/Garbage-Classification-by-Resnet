from PIL import Image
import torch
import os
import json


#class for main
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datajson, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(datajson, 'r')
        load_dict = json.load(fh)
        imgs = []
        for line in load_dict:
            imgs.append((line['name'], int(line['label'])))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(os.path.join(self.root, fn)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
