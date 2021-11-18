import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(buffer):
    return Image.frombuffer(buffer).convert('RGB')


class Dataset(data.Dataset):
    def __init__(self, img: Image, loadSize, fineSize, test=False, video=False):
        super(Dataset, self).__init__()
        self.img = img
        if not test:
            self.transform = transforms.Compose([
                transforms.Resize(fineSize),
                transforms.RandomCrop(fineSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(fineSize),
                transforms.ToTensor()])

        self.test = test

    def __getitem__(self, index):
        ImgA = self.transform(self.img)
        return ImgA, 'upload_img'

    def __len__(self):
        return 1
